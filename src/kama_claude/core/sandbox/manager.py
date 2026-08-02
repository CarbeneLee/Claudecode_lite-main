from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.errors import (
    SandboxCreationFailedError,
    SandboxError,
)
from kama_claude.core.sandbox.executors import ExecResult
from kama_claude.core.sandbox.runtime import ContainerRuntime, DockerCliRuntime

_LOGGER = logging.getLogger(__name__)


class SandboxManager:
    # 沙箱生命周期状态机：idle→creating→ready→closed；failed 非终态，故障恢复后可重试
    IDLE = "idle"
    CREATING = "creating"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"

    def __init__(
        self,
        *,
        config: SandboxConfig,
        workspace_root: Path,
        runtime: ContainerRuntime | None = None,
    ) -> None:
        self._config = config
        self._state = self.IDLE
        self._last_error: str | None = None
        self._creation_lock = asyncio.Lock()
        self._runtime = runtime or DockerCliRuntime(
            image=config.image,
            workspace_root=workspace_root,
            network=config.network,
        )

    @property
    def state(self) -> str:
        # 只读状态，供日志与测试断言状态机
        return self._state

    # 在容器内 cwd 执行命令；懒创建 + 并发保护 + fail closed
    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        await self._ensure_running()
        return await self._runtime.exec(command, cwd=cwd, timeout=timeout)

    # 幂等关闭：只转发一次容器清理，失败仅记日志不中断 daemon 退出
    async def close(self) -> None:
        if self._state == self.CLOSED:
            return
        self._state = self.CLOSED
        try:
            await self._runtime.close()
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("sandbox runtime close failed")

    # 确保容器就绪；lock 内 double-check 防并发重复创建，创建故障一律 fail closed
    async def _ensure_running(self) -> None:
        if self._state == self.CLOSED:
            raise RuntimeError("sandbox manager closed")
        async with self._creation_lock:
            if self._state == self.READY:
                return
            self._state = self.CREATING
            try:
                await self._runtime.ensure_running()
            except asyncio.CancelledError:
                self._state = self.FAILED
                raise
            except SandboxError as exc:
                self._state = self.FAILED
                self._last_error = str(exc)
                raise
            except Exception as exc:
                self._state = self.FAILED
                self._last_error = str(exc)
                _LOGGER.exception("unexpected sandbox runtime failure")
                raise SandboxCreationFailedError(
                    "sandbox runtime failure", detail=str(exc)
                ) from exc
            self._state = self.READY
            self._last_error = None
