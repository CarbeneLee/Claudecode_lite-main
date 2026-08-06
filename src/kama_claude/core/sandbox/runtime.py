from __future__ import annotations

import asyncio
import logging
import os
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kama_claude.core.sandbox.errors import (
    ContainerNotReadyError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    classify_cli_error,
)
from kama_claude.core.sandbox.executors import ExecResult

_LOGGER = logging.getLogger(__name__)

_CONTAINER_MOUNT = "/workspace"
_CONTAINER_NAME = "kama-sandbox"


# 终止进程组并完成 reap；docker CLI 的子进程（如 sleep）可能残留，必须整组清理
async def _kill_and_reap_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("failed to kill docker subprocess group during cleanup")
    try:
        await proc.communicate()
    except (Exception, asyncio.CancelledError):
        _LOGGER.exception("failed to reap docker subprocess during cleanup")


class ContainerRuntime(ABC):
    # 沙箱容器生命周期与执行接口：确保运行、执行命令、关闭清理
    @abstractmethod
    async def ensure_running(self) -> None: ...

    # 在容器内 cwd 执行命令并返回合并输出；超时抛 SandboxTimeoutError
    @abstractmethod
    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult: ...

    # 幂等关闭：删除容器并释放资源
    @abstractmethod
    async def close(self) -> None: ...


class DockerCliRuntime(ContainerRuntime):
    # 用 docker CLI 子进程管理 sandbox 容器；argv 参数数组调用，绝不经过宿主 shell
    def __init__(
        self,
        *,
        image: str,
        workspace_root: Path,
        network: bool = True,
        docker_executable: str = "docker",
        container_name: str = _CONTAINER_NAME,
        container_mount: str = _CONTAINER_MOUNT,
    ) -> None:
        self._image = image
        self._workspace_root = workspace_root
        self._network = network
        self._docker = docker_executable
        self._name = container_name
        self._mount = container_mount
        self._container_id: str | None = None

    # 探活容器；running 则返回，否则幂等创建（先清理同名残留）
    async def ensure_running(self) -> None:
        if await self._inspect_running():
            return
        await self._create()

    # 查询容器状态并同步容器 ID；daemon 不可用抛 SandboxUnavailableError
    async def _inspect_running(self) -> bool:
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            "inspect",
            "-f",
            "{{.Id}} {{.State.Running}}",
            self._name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode == 0:
            # 容器已存在（如 daemon 重启后残留）时同步 ID，保证 exec 可用
            parts = out.decode("utf-8", errors="replace").split()
            if len(parts) == 2 and parts[1] == "true":
                self._container_id = parts[0]
                return True
        if proc.returncode != 0:
            stderr = err.decode("utf-8", errors="replace")
            if "cannot connect to the docker daemon" in stderr.lower():
                raise SandboxUnavailableError(
                    "docker daemon unavailable", detail=stderr
                )
        return False

    # 创建并启动 sandbox 容器；镜像相关故障与创建故障按 stderr 关键词分类
    async def _create(self) -> None:
        # 幂等：清理可能残留的同名容器，忽略"容器不存在"
        await self._run_cli("rm", "-f", self._name)

        args: list[Any] = [
            self._docker,
            "run",
            "--detach",
            "--name",
            self._name,
            "--volume",
            f"{self._workspace_root}:{self._mount}",
            "--workdir",
            self._mount,
        ]
        # git 状态所有权归宿主运行时：.git 以只读子挂载覆盖主挂载（更深路径者生效），
        # 容器内可读不可写，杜绝 agent 在沙箱里篡改 git 元数据
        git_dir = self._workspace_root / ".git"
        if git_dir.is_dir():
            args += ["--volume", f"{git_dir}:{self._mount}/.git:ro"]
        if not self._network:
            args += ["--network", "none"]
        args += [self._image, "sleep", "infinity"]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise classify_cli_error(err.decode("utf-8", errors="replace"))
        self._container_id = out.decode("utf-8").strip()

    # 在容器内 cwd 执行命令；command 作为独立 argv 原样透传给容器内 sh
    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        if self._container_id is None:
            raise ContainerNotReadyError("sandbox container not ready")
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            "exec",
            "--workdir",
            cwd,
            self._container_id,
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _kill_and_reap_group(proc)
            raise SandboxTimeoutError("sandbox exec timed out") from None
        except asyncio.CancelledError:
            await _kill_and_reap_group(proc)
            raise
        return ExecResult(
            output=out,
            returncode=proc.returncode or 0,
            timed_out=False,
        )

    # 幂等关闭：删除容器；容器已删或从未创建时直接返回
    async def close(self) -> None:
        if self._container_id is None:
            return
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            "rm",
            "-f",
            self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        self._container_id = None

    # 执行一次 docker CLI 并丢弃输出；调用方自行处理返回码
    async def _run_cli(self, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
