from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.sandbox.executors import CommandExecutor
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.evidence import ToolEvidenceBudget, bound_tool_output
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_DURABLE_OUTPUT_BYTES = 1 * 1024 * 1024
_MODEL_OUTPUT_CHARS = 8_000
_DEFAULT_TIMEOUT = 60


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
        "Prefer short, focused commands. Raw output is bounded and the model "
        "receives a head/tail receipt."
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

    # 绑定执行器（宿主或容器）与 canonical 启动 workspace；执行细节全部委托
    def __init__(self, executor: CommandExecutor, *, workspace_root: Path) -> None:
        self._executor = executor
        self._workspace_root = WorkspacePathResolver(workspace_root).root

    # 委托 executor 执行，只保留展示逻辑：截断、超时与非零退出码分类
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = BashParams.model_validate(params)
        result = await self._executor.exec(
            p.command, cwd=self._workspace_root, timeout=p.timeout
        )

        if result.timed_out:
            return ToolResult(
                content=f"[timeout after {p.timeout}s]",
                is_error=True,
                error_type="timeout",
            )

        receipt = bound_tool_output(
            result.output,
            budget=ToolEvidenceBudget(
                producer_max_bytes=_DURABLE_OUTPUT_BYTES,
                durable_max_bytes=_DURABLE_OUTPUT_BYTES,
                model_max_chars=_MODEL_OUTPUT_CHARS,
            ),
            original_size=result.original_size,
            raw_truncated=result.raw_truncated,
        )
        truncated = receipt.raw_truncated
        display_truncated = truncated or receipt.model_truncated
        output = receipt.to_model_text()
        if display_truncated:
            output += "\n[truncated]"

        if result.returncode != 0:
            return ToolResult(
                content=f"[exit {result.returncode}]\n{output}",
                is_error=True,
                error_type="command_failed",
                raw_truncated=truncated,
                original_size=(
                    result.original_size
                    if result.original_size is not None
                    else receipt.original_size
                ),
                captured_size=(
                    result.captured_size
                    if result.captured_size is not None
                    else receipt.captured_size
                ),
            )
        return ToolResult(
            content=output or "[no output]",
            raw_truncated=truncated,
            original_size=(
                result.original_size
                if result.original_size is not None
                else receipt.original_size
            ),
            captured_size=(
                result.captured_size
                if result.captured_size is not None
                else receipt.captured_size
            ),
        )
