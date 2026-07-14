from __future__ import annotations

from typing import Literal

INVALID_WORKSPACE = -32013

type InvalidWorkspaceReason = Literal["not_absolute", "not_found", "not_directory"]


class InvalidWorkspaceError(ValueError):
    # 初始化带稳定原因标识的 workspace 校验错误
    def __init__(self, reason: InvalidWorkspaceReason) -> None:
        super().__init__(reason)
        self.reason = reason


class InvalidWorkspacePathError(ValueError):
    pass


class WorkspaceEscapeError(PermissionError):
    pass


class SensitivePathError(PermissionError):
    pass
