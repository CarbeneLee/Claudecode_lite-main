from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import BaseModel, ValidationError

from kama_claude.core.tools.errors import (
    RateLimitedError,
    TransientToolError,
    classify_tool_exception,
)
from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)


class _ValidatedParams(BaseModel):
    count: int


@pytest.mark.parametrize(
    ("exc", "expected_error_type"),
    [
        (FileNotFoundError("missing"), "not_found"),
        (InvalidWorkspacePathError("invalid"), "invalid_path"),
        (WorkspaceEscapeError("escape"), "invalid_path"),
        (SensitivePathError("sensitive"), "sensitive_path"),
        (PermissionError("denied"), "permission_error"),
        (IsADirectoryError("directory"), "is_directory"),
        (NotADirectoryError("not-directory"), "not_directory"),
        (TimeoutError("timed-out"), "timeout"),
    ],
)
# 功能：验证文件与 workspace domain 异常按最具体类型映射为稳定 error_type
# 设计：参数化覆盖父子类易混淆分支，确保 PermissionError 不会抢先吞掉 workspace 特例
def test_classifies_filesystem_and_workspace_errors(
    exc: Exception,
    expected_error_type: str,
) -> None:
    error_type, message = classify_tool_exception(exc)

    assert error_type == expected_error_type
    assert message


# 功能：验证 Pydantic ValidationError 被统一分类为 schema_error
# 设计：通过真实 model_validate 生成 ValidationError，避免手工构造偏离 Pydantic v2 的异常形状
def test_classifies_validation_error_as_schema_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _ValidatedParams.model_validate({"count": "not-an-int"})

    error_type, message = classify_tool_exception(exc_info.value)

    assert error_type == "schema_error"
    assert message


# 功能：验证未知异常统一变成 execution_error 且返回固定安全摘要
# 设计：异常文本模拟绝对路径、token 与 payload，直接断言分类结果不会把任何敏感原文返回给调用方
def test_unknown_exception_uses_safe_execution_error_message() -> None:
    secret = "/private/workspace/.env token=secret raw-payload"

    error_type, message = classify_tool_exception(RuntimeError(secret))

    assert error_type == "execution_error"
    assert message == "tool execution failed"
    assert secret not in message


# 功能：验证未知异常由 logger.exception 记录完整 traceback
# 设计：在活动 except 上下文调用分类器并检查 caplog 的 exc_info，证明诊断细节只进入日志而非返回消息
def test_unknown_exception_logs_traceback(caplog: pytest.LogCaptureFixture) -> None:
    try:
        raise RuntimeError("diagnostic-only-payload")
    except RuntimeError as caught:
        exc = caught

    with caplog.at_level(logging.ERROR, logger="kama_claude.core.tools.errors"):
        error_type, message = classify_tool_exception(exc)

    assert error_type == "execution_error"
    assert message == "tool execution failed"
    record = caplog.records[-1]
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert record.exc_info[1] is exc
    assert record.exc_info[2] is exc.__traceback__
    assert "Traceback (most recent call last)" in caplog.text


# 功能：验证 asyncio cancellation 不会被分类成普通 ToolResult 错误
# 设计：直接传入 CancelledError 并要求原对象上抛，锁定取消信号的身份与传播语义
def test_cancelled_error_is_reraised() -> None:
    exc = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        classify_tool_exception(exc)

    assert exc_info.value is exc


@pytest.mark.parametrize(
    ("exc", "expected_error_type", "expected_message"),
    [
        (
            TransientToolError("provider payload"),
            "transient_error",
            "temporary tool failure",
        ),
        (
            RateLimitedError("429 vendor payload"),
            "rate_limited",
            "tool rate limit exceeded",
        ),
    ],
)
# 功能：验证显式瞬态异常只映射到冻结的两个 retryable error_type
# 设计：同时覆盖 TransientToolError 与 RateLimitedError，并断言供应商异常文本不会成为稳定协议内容
def test_classifies_explicit_retryable_errors_with_stable_messages(
    exc: Exception,
    expected_error_type: str,
    expected_message: str,
) -> None:
    error_type, message = classify_tool_exception(exc)

    assert error_type == expected_error_type
    assert message == expected_message
    assert str(exc) not in message
