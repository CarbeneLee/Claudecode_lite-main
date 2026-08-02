from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, ValidationError

from kama_claude.core.sandbox.errors import (
    ContainerNotReadyError,
    SandboxCreationFailedError,
    SandboxImageError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
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
        "sandbox_unavailable",
        "sandbox_image_error",
        "sandbox_creation_failed",
        "container_not_ready",
        "sandbox_timeout",
    }
)
RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset(
    {"transient_error", "rate_limited", "container_not_ready"}
)
_SAFE_ERROR_MESSAGES: dict[str, str] = {
    "execution_error": "tool execution failed",
    "transient_error": "temporary tool failure",
    "rate_limited": "tool rate limit exceeded",
}
_MAX_VALIDATION_ERRORS = 5
_MAX_VALIDATION_MESSAGE_CHARS = 512
_MAX_SCHEMA_REF_HOPS = 64


class RateLimitedError(Exception):
    """Raised by a tool when the upstream service is rate-limiting the request."""


class TransientToolError(Exception):
    """Raised when a tool explicitly reports a temporary execution failure."""


# 解析 model_json_schema 中的本地引用，供 loc 与 schema 逐段同行
def _resolve_schema_ref(node: object, root: dict[str, object]) -> object:
    current = node
    seen: set[str] = set()
    for _ in range(_MAX_SCHEMA_REF_HOPS):
        if not isinstance(current, dict):
            break
        ref = current.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen:
            break
        seen.add(ref)
        target: object = root
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return current
            target = target[part]
        current = target
    return current


# 展开 anyOf/oneOf 分支，但不读取任何 ValidationError input 或 ctx
def _schema_options(
    nodes: list[object],
    root: dict[str, object],
) -> list[dict[str, object]]:
    pending = list(nodes)
    options: list[dict[str, object]] = []
    while pending:
        candidate = pending.pop()
        node = _resolve_schema_ref(candidate, root)
        if not isinstance(node, dict):
            continue
        variants: list[object] = []
        for keyword in ("anyOf", "oneOf"):
            value = node.get(keyword)
            if isinstance(value, list):
                variants.extend(value)
        if variants:
            pending.extend(variants)
        else:
            options.append(node)
    return options


# 仅在正确的 object/array 节点保留声明字段或索引，动态 mapping 键统一净化
def _format_validation_loc(
    loc: tuple[object, ...],
    schema: dict[str, object],
) -> str:
    nodes: list[object] = [schema]
    rendered: list[str] = []
    for part in loc:
        declared_nodes: list[object] = []
        indexed_nodes: list[object] = []
        dynamic_nodes: list[object] = []
        for node in _schema_options(nodes, schema):
            properties = node.get("properties")
            if (
                isinstance(part, str)
                and isinstance(properties, dict)
                and part in properties
            ):
                declared_nodes.append(properties[part])
                continue

            prefix_items = node.get("prefixItems")
            if (
                isinstance(part, int)
                and isinstance(prefix_items, list)
                and 0 <= part < len(prefix_items)
            ):
                indexed_nodes.append(prefix_items[part])
                continue

            items = node.get("items")
            if isinstance(part, int) and isinstance(items, dict):
                indexed_nodes.append(items)
                continue

            if "additionalProperties" in node:
                additional = node["additionalProperties"]
                dynamic_nodes.append(additional if isinstance(additional, dict) else {})

        if dynamic_nodes:
            rendered.append("<key>")
            nodes = dynamic_nodes
        elif declared_nodes:
            rendered.append(str(part))
            nodes = declared_nodes
        elif indexed_nodes:
            rendered.append(str(part))
            nodes = indexed_nodes
        else:
            rendered.append("<key>")
            nodes = []
    return ".".join(rendered) or "<root>"


# 从静态 Pydantic schema 构造 loc 过滤上下文；失败时退化为全量键净化
def _validation_schema(model: type[BaseModel] | None) -> dict[str, object]:
    if model is None:
        return {}
    try:
        return model.model_json_schema()
    except Exception:
        return {}


# 将 Pydantic 校验错误压缩为只含安全字段路径和错误类型的摘要
def format_validation_error(
    exc: ValidationError,
    model: type[BaseModel] | None = None,
) -> str:
    errors = exc.errors(include_input=False, include_url=False)
    schema = _validation_schema(model)
    summaries = [
        f'{_format_validation_loc(error["loc"], schema)} '
        f'[{error["type"]}]'
        for error in errors[:_MAX_VALIDATION_ERRORS]
    ]
    message = "invalid tool input"
    if summaries:
        message += ": " + "; ".join(summaries)
    remaining = len(errors) - len(summaries)
    if remaining > 0:
        message += f"; ... and {remaining} more"
    if len(message) > _MAX_VALIDATION_MESSAGE_CHARS:
        message = message[: _MAX_VALIDATION_MESSAGE_CHARS - len("...")] + "..."
    return message


# 将工具异常映射为稳定错误类型与安全摘要，并保持取消信号传播
def classify_tool_exception(
    exc: BaseException,
    *,
    validation_model: type[BaseModel] | None = None,
) -> tuple[str, str]:
    if isinstance(exc, asyncio.CancelledError):
        raise exc
    if isinstance(exc, ValidationError):
        return "schema_error", format_validation_error(exc, validation_model)
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
    if isinstance(exc, SandboxUnavailableError):
        return "sandbox_unavailable", "sandbox is unavailable"
    if isinstance(exc, SandboxImageError):
        return "sandbox_image_error", "sandbox image unavailable"
    if isinstance(exc, SandboxCreationFailedError):
        return "sandbox_creation_failed", "sandbox container creation failed"
    if isinstance(exc, ContainerNotReadyError):
        return "container_not_ready", "sandbox container not ready"
    if isinstance(exc, SandboxTimeoutError):
        return "sandbox_timeout", "sandbox command timed out"
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
    if error_type in _SAFE_ERROR_MESSAGES:
        return error_type, _SAFE_ERROR_MESSAGES[error_type]
    return error_type, content
