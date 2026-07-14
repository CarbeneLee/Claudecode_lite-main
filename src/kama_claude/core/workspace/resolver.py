from __future__ import annotations

from pathlib import Path

from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    WorkspaceEscapeError,
)


class WorkspacePathResolver:
    # 校验并保存 canonical workspace root
    def __init__(self, workspace_root: Path) -> None:
        try:
            root = workspace_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidWorkspacePathError(
                "workspace root must be an existing directory"
            ) from exc
        if not root.is_dir():
            raise InvalidWorkspacePathError(
                "workspace root must be an existing directory"
            )
        self._root = root

    @property
    # 返回 canonical workspace root
    def root(self) -> Path:
        return self._root

    # 解析已存在的 workspace-relative 路径并执行 containment 检查
    def resolve_existing(self, relative_path: str) -> Path:
        logical_path = self._relative_path(relative_path)
        try:
            candidate = (self._root / logical_path).resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError("workspace path does not exist") from None
        self._ensure_contained(candidate)
        return candidate

    # 解析可写的 workspace-relative 路径，不要求目标或父目录已存在
    def resolve_for_write(self, relative_path: str) -> Path:
        logical_path = self._relative_path(relative_path)
        candidate = (self._root / logical_path).resolve(strict=False)
        self._ensure_contained(candidate)
        return candidate

    # 将工具参数转换为相对 Path 并拒绝绝对路径
    def _relative_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise InvalidWorkspacePathError("path must be relative to workspace")
        return path

    # 确认 canonical candidate 仍位于 workspace root 内
    def _ensure_contained(self, candidate: Path) -> None:
        if not candidate.is_relative_to(self._root):
            raise WorkspaceEscapeError("path escapes workspace")
