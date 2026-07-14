from __future__ import annotations

from pathlib import Path, PurePath

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.workspace.errors import InvalidWorkspacePathError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_MAX_DEPTH = 4
_MAX_ENTRIES = 200


class ListDirParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=_MAX_DEPTH)


class ListDirTool(BaseTool):
    params_model = ListDirParams
    name = "list_dir"
    description = (
        "List the contents of a directory as a tree. "
        "Path must be relative to the session workspace. "
        "Hidden entries (starting with .) are included. "
        f"Maximum depth is {_MAX_DEPTH}, maximum total entries is {_MAX_ENTRIES}."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the directory (default '.').",
            },
            "max_depth": {
                "type": "integer",
                "description": f"How many levels deep to recurse (default 2, max {_MAX_DEPTH}).",
            },
        },
        "required": [],
    }

    # 注入 workspace 路径解析器与敏感路径策略
    def __init__(
        self,
        resolver: WorkspacePathResolver,
        access_policy: WorkspaceAccessPolicy,
    ) -> None:
        self._resolver = resolver
        self._access_policy = access_policy

    # 安全列出 workspace 内目录树，过滤敏感、逃逸和无法解析的 child
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = ListDirParams.model_validate(params)
        path_str = p.path
        max_depth = p.max_depth

        root = self._resolver.resolve_existing(path_str)
        self._access_policy.ensure_allowed(path_str, root)
        if not root.is_dir():
            raise NotADirectoryError("workspace path is not a directory")

        lines: list[str] = [path_str + "/"]
        count = 0

        # 递归收集已通过 resolver/policy 的目录项
        def _walk(
            directory: Path,
            logical_directory: PurePath,
            depth: int,
            prefix: str,
        ) -> None:
            nonlocal count
            if depth > max_depth or count >= _MAX_ENTRIES:
                return
            entries: list[tuple[str, PurePath, Path, bool]] = []
            for entry in sorted(directory.iterdir(), key=lambda child: child.name):
                logical_child = logical_directory / entry.name
                try:
                    resolved_child = self._resolver.resolve_existing(str(logical_child))
                    self._access_policy.ensure_allowed(
                        str(logical_child),
                        resolved_child,
                    )
                    is_dir = resolved_child.is_dir()
                except (OSError, RuntimeError, InvalidWorkspacePathError):
                    continue
                entries.append(
                    (
                        entry.name,
                        logical_child,
                        resolved_child,
                        is_dir,
                    )
                )

            for i, (name, logical_child, resolved_child, is_dir) in enumerate(entries):
                if count >= _MAX_ENTRIES:
                    lines.append(f"{prefix}... (truncated)")
                    return
                connector = "└── " if i == len(entries) - 1 else "├── "
                suffix = "/" if is_dir else ""
                lines.append(f"{prefix}{connector}{name}{suffix}")
                count += 1
                if is_dir and depth < max_depth:
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(
                        resolved_child,
                        logical_child,
                        depth + 1,
                        prefix + extension,
                    )

        _walk(root, PurePath().joinpath(path_str), 1, "")
        return ToolResult(content="\n".join(lines))
