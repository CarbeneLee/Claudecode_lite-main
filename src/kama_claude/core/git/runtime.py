from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from kama_claude.core.git.errors import GitUnavailableError, classify_cli_error
from kama_claude.core.sandbox.executors import ExecResult

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0


# 终止进程组并完成 reap；git 子进程可能派生子进程，必须整组清理
async def _kill_and_reap_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("failed to kill git subprocess group during cleanup")
    try:
        await proc.communicate()
    except (Exception, asyncio.CancelledError):
        _LOGGER.exception("failed to reap git subprocess during cleanup")


class GitCliRuntime:
    # 用 git CLI 子进程执行仓库操作；argv 参数数组调用，绝不经过宿主 shell
    def __init__(
        self,
        *,
        workspace_root: Path,
        git_executable: str = "git",
    ) -> None:
        self._workspace_root = workspace_root
        self._git = git_executable

    @property
    def workspace_root(self) -> Path:
        # git 命令只在 workspace 内执行的边界锚点
        return self._workspace_root

    # 执行一次 git 命令，返回合并输出与退出码；超时/CLI 缺失抛 GitUnavailableError
    async def run(
        self,
        args: list[str],
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._git,
                "-C",
                str(self._workspace_root),
                *args,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise GitUnavailableError(
                "git CLI unavailable", detail=f"executable not found: {self._git}"
            ) from None
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await _kill_and_reap_group(proc)
            raise GitUnavailableError("git command timed out") from None
        except asyncio.CancelledError:
            await _kill_and_reap_group(proc)
            raise
        return ExecResult(
            output=out,
            returncode=proc.returncode or 0,
            timed_out=False,
        )

    # 执行一次 git 命令并断言成功；非零退出按 stderr 关键词分类抛 GitError
    async def run_check(
        self,
        args: list[str],
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        result = await self.run(args, timeout=timeout, env=env)
        if result.returncode != 0:
            stderr = result.output.decode("utf-8", errors="replace")
            raise classify_cli_error(stderr)
        return result
