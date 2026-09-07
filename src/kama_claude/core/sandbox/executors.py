from __future__ import annotations

import asyncio
import logging
import os
import signal
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
    raw_truncated: bool = False
    original_size: int | None = None
    captured_size: int | None = None


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
    # 终止进程组后逐块 drain，避免 communicate() 再次无界缓存 stdout
    if proc.returncode is None:
        try:
            pid = getattr(proc, "pid", None)
            if pid is None:
                proc.kill()
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        except (Exception, asyncio.CancelledError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except (Exception, asyncio.CancelledError):
                _LOGGER.exception("failed to terminate subprocess during cleanup")
    try:
        await _drain_process_output(proc, max_bytes=_PRODUCER_OUTPUT_CAP)
        wait = getattr(proc, "wait", None)
        if callable(wait):
            await wait()
    except (Exception, asyncio.CancelledError):
        _LOGGER.exception("failed to reap subprocess during cleanup")


_PROCESS_CHUNK_BYTES = 64 * 1024
_PRODUCER_OUTPUT_CAP = 1 * 1024 * 1024


# 逐块读取进程输出并在 producer cap 超限时终止整个进程组
async def _drain_process_output(
    proc: asyncio.subprocess.Process,
    *,
    max_bytes: int,
) -> tuple[bytes, int | None, bool]:
    stdout = getattr(proc, "stdout", None)
    if stdout is None:
        communicate = getattr(proc, "communicate", None)
        if not callable(communicate):
            return b"", 0, False
        output, _ = await communicate()
        if not isinstance(output, bytes):
            output = bytes(output or b"")
        captured_bytes = output[:max_bytes]
        truncated = len(output) > max_bytes
        return captured_bytes, (None if truncated else len(output)), truncated
    captured = bytearray()
    original_size = 0
    truncated = False
    while True:
        chunk = await stdout.read(_PROCESS_CHUNK_BYTES)
        if not chunk:
            break
        original_size += len(chunk)
        remaining = max(0, max_bytes - len(captured))
        if remaining:
            captured.extend(chunk[:remaining])
        if len(captured) >= max_bytes and original_size > max_bytes and not truncated:
            truncated = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
    return bytes(captured), (original_size if not truncated else None), truncated


class HostExecutor(CommandExecutor):
    # 在宿主 cwd 直接执行 shell 命令，合并 stdout/stderr；超时与取消均先清理子进程
    async def exec(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_bytes, original_size, raw_truncated = await asyncio.wait_for(
                _drain_process_output(proc, max_bytes=_PRODUCER_OUTPUT_CAP),
                timeout=timeout,
            )
            await proc.wait()
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
            returncode=proc.returncode if proc.returncode is not None else 0,
            timed_out=False,
            raw_truncated=raw_truncated,
            original_size=original_size,
            captured_size=len(stdout_bytes),
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
