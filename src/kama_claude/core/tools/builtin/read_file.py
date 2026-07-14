from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kama_claude.core.tools.base import BaseTool, ToolResult
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
        raw = path.read_bytes()  # raises FileNotFoundError if absent
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"

        return ToolResult(content=text)
