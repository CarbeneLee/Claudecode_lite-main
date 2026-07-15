from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import BaseModel, Field, ValidationError

import kama_claude.core.tools.errors as error_mod
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


class _ConstrainedParams(BaseModel):
    timeout: int = Field(le=10)


class _NestedConfig(BaseModel):
    timeout: int


class _NestedParams(BaseModel):
    config: _NestedConfig


class _ManyErrorsParams(BaseModel):
    first: int
    second: int
    third: int
    fourth: int
    fifth: int
    sixth: int


class _MappingParams(BaseModel):
    mapping: dict[str, int]


class _IntegerMappingParams(BaseModel):
    mapping: dict[int, int]


class _MappingCollisionParams(BaseModel):
    timeout: int = 1
    mapping: dict[str, int]


# 功能：验证缺字段摘要只包含字段路径和 missing 类型
# 设计：从真实 Pydantic ValidationError 格式化，断言精确可操作文本且不带 input/msg/url
def test_format_validation_error_reports_missing_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _ValidatedParams.model_validate({})

    message = error_mod.format_validation_error(exc_info.value, _ValidatedParams)

    assert message == "invalid tool input: count [missing]"


# 功能：验证约束错误摘要包含字段名和错误 type，但不包含原始字段值
# 设计：用超出上限的整数触发 less_than_equal，证明 formatter 不输出 input 或 ctx 中的限制详情
def test_format_validation_error_reports_type_without_input() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _ConstrainedParams.model_validate({"timeout": 999_999})

    message = error_mod.format_validation_error(exc_info.value, _ConstrainedParams)

    assert message == "invalid tool input: timeout [less_than_equal]"
    assert "999999" not in message
    assert "999_999" not in message


# 功能：验证嵌套 ValidationError loc 使用点号连接
# 设计：让 config 内部缺少 timeout，直接锁定 config.timeout 的稳定路径表达
def test_format_validation_error_joins_nested_location() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _NestedParams.model_validate({"config": {}})

    message = error_mod.format_validation_error(exc_info.value, _NestedParams)

    assert message == "invalid tool input: config.timeout [missing]"


# 功能：验证 validation feedback 最多列出五条并报告剩余数量
# 设计：一次制造六个带敏感输入的类型错误，断言第六字段被省略且所有原始值均未泄露
def test_format_validation_error_limits_output_to_five_errors() -> None:
    raw = {
        "first": "secret-1",
        "second": "secret-2",
        "third": "secret-3",
        "fourth": "secret-4",
        "fifth": "secret-5",
        "sixth": "secret-6",
    }
    with pytest.raises(ValidationError) as exc_info:
        _ManyErrorsParams.model_validate(raw)

    message = error_mod.format_validation_error(exc_info.value, _ManyErrorsParams)

    assert message == (
        "invalid tool input: first [int_parsing]; second [int_parsing]; "
        "third [int_parsing]; fourth [int_parsing]; fifth [int_parsing]; "
        "... and 1 more"
    )
    assert "sixth" not in message
    assert "secret" not in message


# 功能：验证 mapping 的用户键不会通过 ValidationError loc 泄露
# 设计：用包含 token 的动态字典键触发 value 校验错误，只保留公开 schema 字段和安全占位符
def test_format_validation_error_redacts_user_controlled_mapping_key() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _MappingParams.model_validate({"mapping": {"token=secret": "raw-value"}})

    message = error_mod.format_validation_error(exc_info.value, _MappingParams)

    assert message == "invalid tool input: mapping.<key> [int_parsing]"
    assert "token=secret" not in message
    assert "raw-value" not in message


# 功能：验证整数 mapping 键不会被误当作安全数组索引
# 设计：动态整数键只允许在 additionalProperties 节点显示占位符，原始键和值均不可见
def test_format_validation_error_redacts_integer_mapping_key() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _IntegerMappingParams.model_validate({"mapping": {8_675_309: "raw-value"}})

    message = error_mod.format_validation_error(exc_info.value, _IntegerMappingParams)

    assert message == "invalid tool input: mapping.<key> [int_parsing]"
    assert "8675309" not in message
    assert "raw-value" not in message


# 功能：验证动态键即使与其他公开 schema 字段同名也会被净化
# 设计：在 mapping 节点使用 timeout 键，锁定字段判断必须依赖当前位置而不是全局字段名集合
def test_format_validation_error_redacts_mapping_key_that_matches_other_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _MappingCollisionParams.model_validate({"mapping": {"timeout": "raw-value"}})

    message = error_mod.format_validation_error(
        exc_info.value,
        _MappingCollisionParams,
    )

    assert message == "invalid tool input: mapping.<key> [int_parsing]"
    assert "mapping.timeout" not in message
    assert "raw-value" not in message


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
        _ValidatedParams.model_validate({})

    error_type, message = classify_tool_exception(
        exc_info.value,
        validation_model=_ValidatedParams,
    )

    assert error_type == "schema_error"
    assert message == "invalid tool input: count [missing]"


# 功能：验证未知异常统一变成 execution_error 且返回固定安全摘要
# 设计：异常文本模拟绝对路径、token 与 payload，直接断言分类结果不会把任何敏感原文返回给调用方
def test_unknown_exception_uses_safe_execution_error_message() -> None:
    secret = "/private/workspace/.env token=secret raw-payload"

    error_type, message = classify_tool_exception(RuntimeError(secret))

    assert error_type == "execution_error"
    assert message == "tool execution failed"
    assert secret not in message


# 功能：验证显式 execution_error ToolResult 内容始终归一化为固定安全摘要
# 设计：直接调用 central normalizer，使用包含绝对路径和 token 的 content 锁定无泄露输出
def test_normalize_execution_error_uses_safe_message() -> None:
    error_type, message = error_mod.normalize_tool_error(
        "execution_error",
        "/private/.env token=secret",
    )

    assert error_type == "execution_error"
    assert message == "tool execution failed"
    assert "secret" not in message


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
