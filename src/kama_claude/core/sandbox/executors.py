from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kama_claude.core.sandbox.manager import SandboxManager

_LOGGER = logging.getLogger(__name__)

_CONTAINER_MOUNT = "/workspace"


@dataclass(frozen=True)
class ExecResult:
    output: bytes  # stdout+stderr 合并后的原始输出
    returncode: int
    timed_out: bool


class CommandExecutor(ABC):
    # 在指定 cwd 执行命令并返回合并输出；超时以 timed_out 表达，不抛异常
    @abstractmethod
    async def exec(self, command: str, *, cwd: Path, timeout: float) -> ExecResult: ...


# 装配工厂：注入 manager 选容器执行器，否则宿主执行器（单一选型决策点）
def build_executor(
    manager: SandboxManager | None,
    *,
    workspace_root: Path,
) -> CommandExecutor:
    if manager is None:
        return HostExecutor()
    return ContainerExecutor(manager, workspace_root=workspace_root)


# 终止仍在运行的子进程并完成 reap；清理失败记日志但不覆盖调用方原始异常
async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("failed to terminate subprocess during cleanup")
    try:
        await proc.communicate()
    except (Exception, asyncio.CancelledError):
        _LOGGER.exception("failed to reap subprocess during cleanup")


class HostExecutor(CommandExecutor):
    # 在宿主 cwd 直接执行 shell 命令，合并 stdout/stderr；超时与取消均先清理子进程
    async def exec(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _kill_and_reap(proc)
            return ExecResult(output=b"", returncode=-1, timed_out=True)
        except asyncio.CancelledError:
            await _kill_and_reap(proc)
            raise
        except Exception:
            await _kill_and_reap(proc)
            raise
        return ExecResult(
            output=stdout_bytes,
            returncode=proc.returncode or 0,
            timed_out=False,
        )


class ContainerExecutor(CommandExecutor):
    # 绑定 SandboxManager 与挂载映射，把宿主 cwd 映射为容器内路径后转发
    def __init__(
        self,
        manager: SandboxManager,
        *,
        workspace_root: Path,
        container_mount: str = _CONTAINER_MOUNT,
    ) -> None:
        self._manager = manager
        self._workspace_root = workspace_root
        self._container_mount = container_mount

    # 将宿主 cwd 映射为容器内路径并转发给 manager；沙箱异常与取消原样传播
    async def exec(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        container_cwd = self._map_cwd(cwd)
        return await self._manager.exec(
            command, cwd=container_cwd, timeout=timeout
        )

    # 宿主 cwd → 容器内路径；workspace 外路径拒绝
    def _map_cwd(self, cwd: Path) -> str:
        try:
            rel = cwd.resolve().relative_to(self._workspace_root)
        except ValueError:
            raise ValueError(f"cwd outside workspace: {cwd}") from None
        if rel == Path("."):
            return self._container_mount
        return f"{self._container_mount}/{rel}"
