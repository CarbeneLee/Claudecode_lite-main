from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.evidence import ToolEvidenceBudget, bound_tool_output
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_MAX_BYTES = 512 * 1024  # 512 KB


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Path must be relative to the session workspace. "
        "Files larger than 512 KB are truncated."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the session workspace.",
            }
        },
        "required": ["path"],
    }

    # 注入 workspace 路径解析器与敏感路径策略
    def __init__(
        self,
        resolver: WorkspacePathResolver,
        access_policy: WorkspaceAccessPolicy,
    ) -> None:
        self._resolver = resolver
        self._access_policy = access_policy

    # 读取 workspace 内文件内容并保持 512KB 截断行为
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = ReadFileParams.model_validate(params).path
        path = self._resolver.resolve_existing(path_str)
        self._access_policy.ensure_allowed(path_str, path)
        # 先读取文件大小，再只把 durable cap 内的前缀载入内存；原文件仍可按 path 重新读取。
        original_size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(_MAX_BYTES)
        receipt = bound_tool_output(
            raw,
            budget=ToolEvidenceBudget(
                producer_max_bytes=_MAX_BYTES,
                durable_max_bytes=_MAX_BYTES,
                model_max_chars=_MAX_BYTES,
            ),
            original_size=original_size,
            raw_truncated=original_size > len(raw),
        )
        truncated = receipt.raw_truncated
        text = receipt.preview
        if truncated:
            text += "\n[truncated]"

        return ToolResult(
            content=text,
            raw_truncated=truncated,
            original_size=receipt.original_size,
            captured_size=receipt.captured_size,
        )
