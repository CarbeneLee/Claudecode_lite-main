from __future__ import annotations

import asyncio
import logging
from typing import cast

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

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


class _SevenErrorsParams(BaseModel):
    first: int
    second: int
    third: int
    fourth: int
    fifth: int
    sixth: int
    seventh: int


class _NestedItem(BaseModel):
    count: int


class _CollectionParams(BaseModel):
    items: list[_NestedItem]
    pair: tuple[int, int]


class _NestedTupleParams(BaseModel):
    pair: tuple[_NestedItem, _NestedItem]


class _OptionalParams(BaseModel):
    count: int | None


class _UnionParams(BaseModel):
    value: int | bool


class _DeclaredPayload(BaseModel):
    timeout: int


class _UnionMappingParams(BaseModel):
    payload: _DeclaredPayload | dict[str, int]


class _StaticPayloadParams(BaseModel):
    payload: _DeclaredPayload


class _SchemaProbeParams(BaseModel):
    payload: dict[str, int]


class _NestedMappingParams(BaseModel):
    mapping: dict[str, _NestedItem]


class _RootValidatedParams(BaseModel):
    count: int

    @model_validator(mode="after")
    # 拒绝整个模型以稳定生成空 loc 的 root ValidationError
    def reject_model(self) -> _RootValidatedParams:
        raise ValueError("root token=secret")


class _SchemaFailureParams(BaseModel):
    # 模拟 schema 生成阶段发生非校验异常
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("schema token=secret")


class _BrokenRefParams(BaseModel):
    # 返回含失效本地引用的确定性测试 schema
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/missing"}},
        }


class _CyclicRefParams(BaseModel):
    # 返回两个本地引用相互循环的确定性测试 schema
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/a"}},
            "$defs": {
                "a": {"$ref": "#/$defs/b"},
                "b": {"$ref": "#/$defs/a"},
            },
        }


class _EscapedRefParams(BaseModel):
    # 返回引用名含 JSON Pointer 转义字符的确定性测试 schema
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"payload": {"$ref": "#/$defs/a~1b~0c"}},
            "$defs": {
                "a/b~c": {
                    "type": "object",
                    "properties": {"timeout": {"type": "integer"}},
                }
            },
        }


class _MixedSchemaOptionsParams(BaseModel):
    # 返回同时含合法与非字典分支的 schema 以验证解析器跳过坏分支
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"timeout": {"type": "integer"}},
                        },
                        None,
                    ]
                }
            },
        }


class _OneOfSchemaParams(BaseModel):
    # 返回 oneOf 静态分支以验证非 Pydantic 默认 union 关键字
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "payload": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"timeout": {"type": "integer"}},
                        }
                    ]
                }
            },
        }


class _OneOfMappingCollisionParams(BaseModel):
    # 让 declared 分支先被遍历，再用 dynamic 分支强制 fail closed
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "payload": {
                    "oneOf": [
                        {"type": "object", "additionalProperties": {"type": "integer"}},
                        {
                            "type": "object",
                            "properties": {"timeout": {"type": "integer"}},
                        },
                    ]
                }
            },
        }


class _TupleMappingCollisionParams(BaseModel):
    # 返回 fixed tuple 与整数 mapping 歧义 schema 以验证 dynamic 分支优先
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        nested = {"type": "object", "properties": {"count": {"type": "integer"}}}
        return {
            "type": "object",
            "properties": {
                "payload": {
                    "oneOf": [
                        {"type": "object", "additionalProperties": nested},
                        {"type": "array", "prefixItems": [nested]},
                    ]
                }
            },
        }


class _ListMappingCollisionParams(BaseModel):
    # 返回 list 与整数 mapping 歧义 schema 以验证 dynamic 分支优先
    @classmethod
    def model_json_schema(cls, **kwargs: object) -> dict[str, object]:
        nested = {"type": "object", "properties": {"count": {"type": "integer"}}}
        return {
            "type": "object",
            "properties": {
                "payload": {
                    "oneOf": [
                        {"type": "object", "additionalProperties": nested},
                        {"type": "array", "items": nested},
                    ]
                }
            },
        }


class _ValidationErrorsProbe:
    # 保存公开错误 loc 与类型
    def __init__(
        self,
        loc: tuple[object, ...] = ("count",),
        error_type: str = "missing",
    ) -> None:
        self._loc = loc
        self._error_type = error_type

    # 记录 formatter 是否显式关闭 ValidationError 的 input 与 URL 展开
    def errors(
        self,
        *,
        include_input: bool = True,
        include_url: bool = True,
    ) -> list[dict[str, object]]:
        assert include_input is False
        assert include_url is False
        return [{"loc": self._loc, "type": self._error_type}]


class _MappingParams(BaseModel):
    mapping: dict[str, int]


class _IntegerMappingParams(BaseModel):
    mapping: dict[int, int]


class _MappingCollisionParams(BaseModel):
    timeout: int = 1
    mapping: dict[str, int]


_LONG_FIELD = "trusted_" + ("segment_" * 80)
_LongPathParams = create_model(
    "_LongPathParams",
    **{_LONG_FIELD: (dict[str, int], ...)},
)
_EXACT_LIMIT_FIELD = "x" * (512 - len("invalid tool input:  [missing]"))
_ExactLimitParams = create_model(
    "_ExactLimitParams",
    **{_EXACT_LIMIT_FIELD: (int, ...)},
)


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


@pytest.mark.parametrize(
    ("error_count", "expected_visible", "expected_more"),
    [
        (4, 4, None),
        (5, 5, None),
        (6, 5, "... and 1 more"),
        (7, 5, "... and 2 more"),
    ],
)
# 功能：验证 validation feedback 在 4/5/6/7 条错误时恰好执行五条上限与剩余计数
# 设计：参数化同一七字段模型，只改变无效字段数量，统一杀死 slice、比较符与 remaining 的边界变异
def test_format_validation_error_enforces_count_boundary(
    error_count: int,
    expected_visible: int,
    expected_more: str | None,
) -> None:
    field_names = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
    raw = {
        name: f"secret-{index}" if index <= error_count else index
        for index, name in enumerate(field_names, start=1)
    }
    with pytest.raises(ValidationError) as exc_info:
        _SevenErrorsParams.model_validate(raw)

    message = error_mod.format_validation_error(exc_info.value, _SevenErrorsParams)

    assert message.count("[int_parsing]") == expected_visible
    if expected_more is None:
        assert "more" not in message
    else:
        assert expected_more in message
    assert ("sixth [int_parsing]" in message) is (expected_visible >= 6)
    assert "secret" not in message


# 功能：验证 list 的第零与非零索引、nested 字段名都会作为安全静态路径保留
# 设计：同一真实 Pydantic list 同时制造两个元素错误，精确断言索引路径且排除两份原始值
def test_format_validation_error_preserves_list_indices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _CollectionParams.model_validate(
            {
                "items": [{"count": "secret-zero"}, {"count": "secret-one"}],
                "pair": [1, 2],
            }
        )

    message = error_mod.format_validation_error(exc_info.value, _CollectionParams)

    assert message == (
        "invalid tool input: items.0.count [int_parsing]; "
        "items.1.count [int_parsing]"
    )
    assert "secret" not in message


# 功能：验证 fixed tuple 的首尾合法索引都会作为安全路径保留
# 设计：让二元 tuple 两项同时失败，覆盖 prefixItems 的下界与上界内侧而不直接测试 helper
def test_format_validation_error_preserves_fixed_tuple_indices() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _CollectionParams.model_validate(
            {
                "items": [],
                "pair": ["secret-first", "secret-last"],
            }
        )

    message = error_mod.format_validation_error(exc_info.value, _CollectionParams)

    assert message == (
        "invalid tool input: pair.0 [int_parsing]; pair.1 [int_parsing]"
    )
    assert "secret" not in message


# 功能：验证 fixed tuple 索引后仍能保留 nested 静态字段
# 设计：第二个 tuple 元素使用真实 nested model，区分保留 item schema 与仅保留索引文本两种实现
def test_format_validation_error_preserves_nested_tuple_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _NestedTupleParams.model_validate(
            {"pair": [{"count": 1}, {"count": "token=secret"}]}
        )

    message = error_mod.format_validation_error(exc_info.value, _NestedTupleParams)

    assert message == "invalid tool input: pair.1.count [int_parsing]"
    assert "secret" not in message


# 功能：验证 Optional 与普通 Union 只保留可信外层字段且不回显原始输入
# 设计：参数化两类 anyOf schema，以安全不变量断言避免依赖 Pydantic 的 union branch label 文本
@pytest.mark.parametrize(
    ("model", "raw", "trusted_field"),
    [
        (_OptionalParams, {"count": "token=optional-secret"}, "count"),
        (_UnionParams, {"value": {"token": "union-secret"}}, "value"),
    ],
)
def test_format_validation_error_handles_optional_and_union_fail_closed(
    model: type[BaseModel],
    raw: dict[str, object],
    trusted_field: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(raw)

    message = error_mod.format_validation_error(exc_info.value, model)

    assert f"invalid tool input: {trusted_field}" in message
    assert "secret" not in message
    assert "token" not in message


# 功能：验证 declared model 与动态 mapping union 冲突时动态键不会因同名字段而泄露
# 设计：让 timeout 同时命中 declared 分支名称和 mapping 用户键，断言只保留 payload 并对歧义路径 fail closed
def test_format_validation_error_redacts_union_mapping_collision() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _UnionMappingParams.model_validate(
            {"payload": {"timeout": "token=secret"}}
        )

    message = error_mod.format_validation_error(exc_info.value, _UnionMappingParams)

    assert "invalid tool input: payload" in message
    assert ".timeout" not in message
    assert "token=secret" not in message


# 功能：验证 oneOf 静态分支能解析并保留可信 nested 字段
# 设计：用真实 payload.timeout missing 错误配合确定性 oneOf schema，避免只覆盖 Pydantic 默认 anyOf
def test_format_validation_error_supports_one_of_schema() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _StaticPayloadParams.model_validate({"payload": {}})

    message = error_mod.format_validation_error(exc_info.value, _OneOfSchemaParams)

    assert message == "invalid tool input: payload.timeout [missing]"


# 功能：验证 oneOf 中 declared 与 dynamic mapping 同名冲突时仍净化用户键
# 设计：人为控制分支遍历顺序让 declared 先命中，确保后续 dynamic 分支仍可提升为 fail-closed 占位符
def test_format_validation_error_one_of_mapping_collision_fails_closed() -> None:
    probe = cast(
        ValidationError,
        _ValidationErrorsProbe(("payload", "timeout"), "int_parsing"),
    )

    message = error_mod.format_validation_error(
        probe,
        _OneOfMappingCollisionParams,
    )

    assert message == "invalid tool input: payload.<key> [int_parsing]"
    assert ".timeout" not in message


@pytest.mark.parametrize(
    "schema_model",
    [_TupleMappingCollisionParams, _ListMappingCollisionParams],
)
# 功能：验证 tuple/list 索引与 dynamic mapping 整数键歧义时始终按动态键净化
# 设计：参数化 prefixItems/items 两条分支并让 indexed 先命中，锁定后续 dynamic 分支的安全优先级
def test_format_validation_error_index_mapping_collision_fails_closed(
    schema_model: type[BaseModel],
) -> None:
    probe = cast(
        ValidationError,
        _ValidationErrorsProbe(("payload", 0, "count"), "int_parsing"),
    )

    message = error_mod.format_validation_error(probe, schema_model)

    assert message == "invalid tool input: payload.<key>.count [int_parsing]"


# 功能：验证动态 mapping 键之后的 nested 静态字段仍可安全保留
# 设计：真实 dict value 使用 NestedItem，联合断言用户键净化与 count 字段可操作性
def test_format_validation_error_preserves_field_after_dynamic_mapping_key() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _NestedMappingParams.model_validate(
            {"mapping": {"token=secret-key": {"count": "raw-secret-value"}}}
        )

    message = error_mod.format_validation_error(exc_info.value, _NestedMappingParams)

    assert message == "invalid tool input: mapping.<key>.count [int_parsing]"
    assert "secret" not in message


@pytest.mark.parametrize(
    "schema_model",
    [None, _SchemaFailureParams, _BrokenRefParams, _CyclicRefParams],
)
# 功能：验证缺失、异常、断裂与循环 schema resolution 均安全降级且不会泄露动态键值
# 设计：复用真实 ValidationError 并只替换公开 formatter 的 schema model，覆盖失败模式而不测试私有中间状态
def test_format_validation_error_schema_resolution_fails_closed(
    schema_model: type[BaseModel] | None,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _SchemaProbeParams.model_validate(
            {"payload": {"token=secret-key": "raw-secret-value"}}
        )

    message = error_mod.format_validation_error(exc_info.value, schema_model)

    if schema_model in (_BrokenRefParams, _CyclicRefParams):
        assert message.startswith("invalid tool input: payload.<key>")
    else:
        assert message.startswith("invalid tool input: <key>.<key>")
    assert "token=secret-key" not in message
    assert "raw-secret-value" not in message


# 功能：验证合法的 ~0 与 ~1 JSON Pointer 转义引用仍保留静态字段并净化动态键
# 设计：自定义确定性 schema 指向含斜杠与波浪号的 definition，走公开 formatter 验证 ref 解码结果
def test_format_validation_error_resolves_escaped_json_pointer() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _StaticPayloadParams.model_validate({"payload": {}})

    message = error_mod.format_validation_error(exc_info.value, _EscapedRefParams)

    assert message == "invalid tool input: payload.timeout [missing]"


# 功能：验证 schema union 中的坏分支不会阻止合法静态分支继续提供安全路径
# 设计：让 non-dict 分支最后入栈先被访问，公开 formatter 必须跳过它并继续保留 payload.timeout
def test_format_validation_error_skips_invalid_schema_option() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _StaticPayloadParams.model_validate({"payload": {}})

    message = error_mod.format_validation_error(
        exc_info.value,
        _MixedSchemaOptionsParams,
    )

    assert message == "invalid tool input: payload.timeout [missing]"


# 功能：验证 formatter 请求 ValidationError 明细时显式禁止 input 与 URL 展开
# 设计：使用只实现公开 errors 协议的探针在参数被省略或改写时立即失败，锁定敏感数据不被读取
def test_format_validation_error_does_not_request_input_or_url() -> None:
    probe = cast(ValidationError, _ValidationErrorsProbe())

    message = error_mod.format_validation_error(probe, _ValidatedParams)

    assert message == "invalid tool input: count [missing]"


# 功能：验证模型级 ValidationError 的空 loc 使用稳定 root 占位符且不泄露 ctx
# 设计：真实 after model_validator 产生 root error，精确断言公开摘要只包含 <root> 与错误类型
def test_format_validation_error_uses_root_placeholder_for_empty_loc() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _RootValidatedParams.model_validate({"count": 1})

    message = error_mod.format_validation_error(exc_info.value, _RootValidatedParams)

    assert message == "invalid tool input: <root> [value_error]"
    assert "root token=secret" not in message


# 功能：验证 ValidationError 安全摘要在极长静态路径下仍不超过总字符上限
# 设计：用超长公开字段包裹动态 mapping 错误，确保截断发生在路径净化后且稳定保留省略后缀
def test_format_validation_error_caps_total_message_length() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _LongPathParams.model_validate(
            {_LONG_FIELD: {"token=secret-key": "raw-secret-value"}}
        )

    message = error_mod.format_validation_error(exc_info.value, _LongPathParams)

    assert len(message) <= 512
    assert message.endswith("...")
    assert "token=secret-key" not in message
    assert "raw-secret-value" not in message


# 功能：验证恰好等于总字符上限的安全摘要保持逐字不变
# 设计：首次 mutation 已证明 > 到 >= 会存活，因此动态计算字段长度锁定唯一有意义的比较边界
def test_format_validation_error_preserves_message_at_exact_limit() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _ExactLimitParams.model_validate({})

    message = error_mod.format_validation_error(exc_info.value, _ExactLimitParams)

    assert len(message) == 512
    assert message == f"invalid tool input: {_EXACT_LIMIT_FIELD} [missing]"
    assert not message.endswith("...")


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
    ("exc", "expected_error_type", "expected_message"),
    [
        (FileNotFoundError("missing"), "not_found", "requested path was not found"),
        (InvalidWorkspacePathError("invalid"), "invalid_path", "invalid workspace path"),
        (WorkspaceEscapeError("escape"), "invalid_path", "path escapes workspace"),
        (
            SensitivePathError("sensitive"),
            "sensitive_path",
            "sensitive workspace path is blocked",
        ),
        (
            PermissionError("denied"),
            "permission_error",
            "tool lacks permission for this operation",
        ),
        (IsADirectoryError("directory"), "is_directory", "path is a directory"),
        (
            NotADirectoryError("not-directory"),
            "not_directory",
            "path component is not a directory",
        ),
        (TimeoutError("timed-out"), "timeout", "tool execution timed out"),
    ],
)
# 功能：验证文件与 workspace domain 异常按最具体类型映射为稳定 error_type
# 设计：参数化覆盖父子类易混淆分支，确保 PermissionError 不会抢先吞掉 workspace 特例
def test_classifies_filesystem_and_workspace_errors(
    exc: Exception,
    expected_error_type: str,
    expected_message: str,
) -> None:
    error_type, message = classify_tool_exception(exc)

    assert error_type == expected_error_type
    assert message == expected_message


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
    assert record.getMessage() == "unexpected tool execution failure"
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
