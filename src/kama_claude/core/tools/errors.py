from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)

_LOGGER = logging.getLogger(__name__)

_STABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "unknown_tool",
        "schema_error",
        "permission_denied",
        "timeout",
        "not_found",
        "invalid_path",
        "sensitive_path",
        "permission_error",
        "is_directory",
        "not_directory",
        "invalid_input",
        "command_failed",
        "execution_error",
        "transient_error",
        "rate_limited",
    }
)
RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {"transient_error", "rate_limited"}
)
_RETRYABLE_ERROR_MESSAGES: dict[str, str] = {
    "transient_error": "temporary tool failure",
    "rate_limited": "tool rate limit exceeded",
}


class RateLimitedError(Exception):
    """Raised by a tool when the upstream service is rate-limiting the request."""


class TransientToolError(Exception):
    """Raised when a tool explicitly reports a temporary execution failure."""


# 将工具异常映射为稳定错误类型与安全摘要，并保持取消信号传播
def classify_tool_exception(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, ValidationError):
        return "schema_error", "tool input validation failed"
    if isinstance(exc, FileNotFoundError):
        return "not_found", "requested path was not found"
    if isinstance(exc, InvalidWorkspacePathError):
        return "invalid_path", "invalid workspace path"
    if isinstance(exc, WorkspaceEscapeError):
        return "invalid_path", "path escapes workspace"
    if isinstance(exc, SensitivePathError):
        return "sensitive_path", "sensitive workspace path is blocked"
    if isinstance(exc, IsADirectoryError):
        return "is_directory", "path is a directory"
    if isinstance(exc, NotADirectoryError):
        return "not_directory", "path component is not a directory"
    if isinstance(exc, PermissionError):
        return "permission_error", "tool lacks permission for this operation"
    if isinstance(exc, TransientToolError):
        return "transient_error", "temporary tool failure"
    if isinstance(exc, RateLimitedError):
        return "rate_limited", "tool rate limit exceeded"
    if isinstance(exc, TimeoutError):
        return "timeout", "tool execution timed out"

    _LOGGER.exception(
        "unexpected tool execution failure",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return "execution_error", "tool execution failed"


# 归一化工具返回的错误类型，并净化 legacy、未知和瞬态错误内容
def normalize_tool_error(error_type: str | None, content: str) -> tuple[str, str]:
    if error_type not in _STABLE_ERROR_TYPES:
        return "execution_error", "tool execution failed"
    if error_type in _RETRYABLE_ERROR_MESSAGES:
        return error_type, _RETRYABLE_ERROR_MESSAGES[error_type]
    return error_type, content
