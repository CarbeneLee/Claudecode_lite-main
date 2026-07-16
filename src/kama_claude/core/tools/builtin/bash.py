from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
_DEFAULT_TIMEOUT = 60
_LOGGER = logging.getLogger(__name__)


# 终止仍在运行的子进程并完成 reap；清理失败记日志但不覆盖调用方原始异常
async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("failed to terminate bash subprocess during cleanup")
    try:
        await proc.communicate()
    except (Exception, asyncio.CancelledError):
        _LOGGER.exception("failed to reap bash subprocess during cleanup")


class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)


class BashTool(BaseTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a shell command and return its output (stdout + stderr combined). "
        "Non-interactive only — commands requiring user input will hang and time out. "
        "Prefer short, focused commands. Output is truncated at 64 KB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
        },
        "required": ["command"],
    }

    # 绑定 shell 子进程的 canonical 启动 workspace
    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = WorkspacePathResolver(workspace_root).root

    # 从 workspace 启动 shell，合并 stdout/stderr，处理超时与非零退出码
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = BashParams.model_validate(params)
        command = p.command
        timeout = p.timeout

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._workspace_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            await _kill_and_reap(proc)
            return ToolResult(
                content=f"[timeout after {timeout}s]",
                is_error=True,
                error_type="timeout",
            )
        except asyncio.CancelledError:
            await _kill_and_reap(proc)
            raise
        except Exception:
            await _kill_and_reap(proc)
            raise

        output = stdout_bytes.decode("utf-8", errors="replace")
        truncated = len(stdout_bytes) > _MAX_OUTPUT_BYTES
        if truncated:
            output = output[:_MAX_OUTPUT_BYTES] + "\n[truncated]"

        returncode = proc.returncode or 0
        if returncode != 0:
            return ToolResult(
                content=f"[exit {returncode}]\n{output}",
                is_error=True,
                error_type="command_failed",
            )
        return ToolResult(content=output or "[no output]")
