from __future__ import annotations

from pathlib import Path

from kama_claude.core.workspace.errors import InvalidWorkspaceError


# 校验客户端提供的 workspace 并返回已存在目录的 canonical Path
def validate_workspace_root(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise InvalidWorkspaceError("not_absolute")
    if not path.exists():
        raise InvalidWorkspaceError("not_found")
    if not path.is_dir():
        raise InvalidWorkspaceError("not_directory")
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise InvalidWorkspaceError("not_found") from exc
