from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

import kama_claude.core.tools.invocation as inv_mod
from kama_claude.core.bus.events import (
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

# --- stub tools --------------------------------------------------------------


class _EchoParams(BaseModel):
    msg: str


class _ExplodingParams(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    # 通过真实 Pydantic validator 抛出未知异常
    def reject_value(cls, value: int) -> int:
        raise RuntimeError("/private/workspace/.env token=validator-secret")


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the msg param"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    params_model = _EchoParams

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


class _TrackedSchemaTool(BaseTool):
    name = "tracked_schema"
    description = "Tracks whether schema-invalid calls reach invoke"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }
    params_model = _EchoParams

    # 初始化真实工具调用计数
    def __init__(self) -> None:
        self.calls = 0

    # 记录调用并返回消息内容
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(content=str(params["msg"]))


class _MappingParams(BaseModel):
    mapping: dict[str, int]


class _IntegerMappingParams(BaseModel):
    mapping: dict[int, int]


class _TrackedMappingTool(BaseTool):
    name = "tracked_mapping"
    description = "Tracks whether mapping-invalid calls reach invoke"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            }
        },
        "required": ["mapping"],
    }
    params_model = _MappingParams

    # 初始化真实工具调用计数
    def __init__(self) -> None:
        self.calls = 0

    # 记录调用以证明 schema_error 不会进入工具执行
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(content="ok")


class _TrackedIntegerMappingTool(BaseTool):
    name = "tracked_integer_mapping"
    description = "Tracks whether integer-mapping-invalid calls reach invoke"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "mapping": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            }
        },
        "required": ["mapping"],
    }
    params_model = _IntegerMappingParams

    # 初始化真实工具调用计数
    def __init__(self) -> None:
        self.calls = 0

    # 记录调用以证明整数 mapping schema_error 不会进入工具执行
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(content="ok")


class _SlowTool(BaseTool):
    name = "slow"
    description = "Sleeps forever"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        await asyncio.sleep(0.2)
        return ToolResult(content="done")


class _BrokenTool(BaseTool):
    name = "broken"
    description = "Always raises"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("boom")


class _CancelledTool(BaseTool):
    name = "cancelled"
    description = "Always cancels"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 抛出取消信号以验证 invocation lifecycle 原样传播
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise asyncio.CancelledError


class _ValidatorBrokenTool(BaseTool):
    name = "validator_broken"
    description = "Has a validator that raises an unknown exception"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    params_model = _ExplodingParams

    # 初始化真实工具调用计数
    def __init__(self) -> None:
        self.calls = 0

    # validator 失败时不应到达真实工具执行
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        raise AssertionError("tool invoke should not run")


class _StubPermissionManager:
    # 保存权限决策并初始化接收参数
    def __init__(self, allowed: bool, decision: str, *, emit_requested: bool) -> None:
        self._allowed = allowed
        self._decision = decision
        self._emit_requested = emit_requested
        self.received: dict[str, object] | None = None

    # 记录 invocation 传参并按配置模拟 ask 或自动权限决策
    async def check_and_wait(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> tuple[bool, str]:
        self.received = {
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "params": params,
            "session_id": session_id,
        }
        if self._emit_requested:
            await event_emitter(
                {
                    "type": "permission.requested",
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "params": params,
                    "param_preview": "msg=hi",
                    "session_id": session_id,
                    "ts": "2026-07-15T00:00:00+00:00",
                }
            )
        return self._allowed, self._decision


# --- helpers -----------------------------------------------------------------


def _call(name: str, inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {})


async def _run(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    timeout: float = 5.0,
    *,
    permission_manager: _StubPermissionManager | None = None,
    session_id: str | None = None,
) -> tuple[ToolResult, list[BaseModel]]:
    bus = EventBus()
    events: list[BaseModel] = []

    async def _collect(e: BaseModel) -> None:
        events.append(e)

    bus.subscribe(_collect)
    if session_id is None:
        result = await invoke_tool(
            registry,
            tool_call,
            bus,
            run_id="r1",
            timeout=timeout,
            permission_manager=permission_manager,  # type: ignore[arg-type]
        )
    else:
        result = await invoke_tool(
            registry,
            tool_call,
            bus,
            run_id="r1",
            timeout=timeout,
            permission_manager=permission_manager,  # type: ignore[arg-type]
            session_id=session_id,
        )
    return result, events


# --- tests -------------------------------------------------------------------


# 功能：验证正常调用时返回工具内容且发布 started + finished 事件
# 设计：同时检查返回值和事件序列，因为 invoke_tool 的双重职责是"返回结果 + 发布事件"，缺一不可
async def test_success_returns_content_and_finished_event() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result, events = await _run(registry, _call("echo", {"msg": "hi"}))
    assert not result.is_error
    assert result.content == "hi"
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert types[0] == "tool.call_started"
    assert "tool.call_finished" in types
    assert "tool.call_failed" not in types
    finished = [event for event in events if isinstance(event, ToolCallFinishedEvent)]
    assert [event.output for event in finished] == ["hi"]
    assert all(datetime.fromisoformat(event.ts).utcoffset() is not None for event in events)
    assert all(
        datetime.fromisoformat(event.ts).utcoffset().total_seconds() == 0  # type: ignore[union-attr]
        for event in events
    )


# 功能：验证调用不存在的工具时返回 unknown_tool 并发布 failed 事件而非 finished
# 设计：传入空 registry，确认稳定 error_type 和事件类型同时正确，排除未知工具被误当 execution_error
async def test_unknown_tool_returns_unknown_tool() -> None:
    result, events = await _run(ToolRegistry(), _call("nonexistent"))
    assert result.is_error
    assert result.error_type == "unknown_tool"
    assert result.content == "unknown tool: nonexistent"
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "tool.call_started" in types
    assert "tool.call_failed" in types
    assert "tool.call_finished" not in types
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_message for event in failed] == ["unknown tool: nonexistent"]


# 功能：验证 elapsed_ms 使用 monotonic 秒差并精确换算为毫秒
# 设计：固定起止时间相差一秒走 unknown_tool 快速路径，杀死加减号、乘除号与倍率变异
async def test_elapsed_ms_uses_monotonic_millisecond_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([10.0, 11.0])
    monkeypatch.setattr(
        inv_mod,
        "time",
        SimpleNamespace(monotonic=lambda: next(readings)),
    )

    result, events = await _run(ToolRegistry(), _call("nonexistent"))

    assert result.is_error
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.elapsed_ms for event in failed] == [1000]


@pytest.mark.parametrize(
    ("decision", "emit_requested", "expected_permission_types", "session_id"),
    [
        (
            "allow_once",
            True,
            [PermissionRequestedEvent, PermissionGrantedEvent],
            "session-1",
        ),
        ("auto_allow", False, [], None),
    ],
)
# 功能：验证 ask 与自动允许决策都传递完整上下文，且只在 ask 路径发布 permission 事件
# 设计：参数化 allow_once/auto_allow 并检查 manager 入参、事件字段、默认 session 与最终工具成功
async def test_permission_allow_lifecycle(
    decision: str,
    emit_requested: bool,
    expected_permission_types: list[type[BaseModel]],
    session_id: str | None,
) -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    manager = _StubPermissionManager(True, decision, emit_requested=emit_requested)

    result, events = await _run(
        registry,
        _call("echo", {"msg": "hi"}),
        permission_manager=manager,
        session_id=session_id,
    )

    assert result == ToolResult(content="hi")
    assert manager.received == {
        "tool_use_id": "t1",
        "tool_name": "echo",
        "params": {"msg": "hi"},
        "session_id": session_id or "",
    }
    permission_events = [
        event
        for event in events
        if isinstance(
            event,
            (PermissionRequestedEvent, PermissionGrantedEvent, PermissionDeniedEvent),
        )
    ]
    assert [type(event) for event in permission_events] == expected_permission_types
    if emit_requested:
        requested = permission_events[0]
        granted = permission_events[1]
        assert isinstance(requested, PermissionRequestedEvent)
        assert requested.model_dump() == {
            "type": "permission.requested",
            "run_id": "r1",
            "tool_use_id": "t1",
            "tool_name": "echo",
            "params": {"msg": "hi"},
            "param_preview": "msg=hi",
            "session_id": "session-1",
            "ts": "2026-07-15T00:00:00+00:00",
        }
        assert isinstance(granted, PermissionGrantedEvent)
        assert granted.run_id == "r1"
        assert granted.tool_use_id == "t1"
        assert granted.decision == "allow_once"
    assert isinstance(events[-1], ToolCallFinishedEvent)


@pytest.mark.parametrize(
    ("decision", "emit_requested", "expected_permission_types"),
    [
        ("deny_once", True, [PermissionRequestedEvent, PermissionDeniedEvent]),
        ("auto_deny", False, []),
    ],
)
# 功能：验证 ask 与自动拒绝都阻止工具执行并返回固定 permission_denied 结果
# 设计：参数化 deny_once/auto_deny，联合断言权限事件差异、失败 attempt、消息和无 finished 生命周期
async def test_permission_deny_lifecycle(
    decision: str,
    emit_requested: bool,
    expected_permission_types: list[type[BaseModel]],
) -> None:
    registry = ToolRegistry()
    tool = _TrackedSchemaTool()
    registry.register(tool)
    manager = _StubPermissionManager(False, decision, emit_requested=emit_requested)

    result, events = await _run(
        registry,
        _call("tracked_schema", {"msg": "hi"}),
        permission_manager=manager,
        session_id="session-1",
    )

    expected_message = (
        "Permission denied by user. You may not execute this command. "
        "Try an alternative approach or ask the user what to do."
    )
    assert tool.calls == 0
    assert result == ToolResult(
        content=expected_message,
        is_error=True,
        error_type="permission_denied",
    )
    permission_events = [
        event
        for event in events
        if isinstance(
            event,
            (PermissionRequestedEvent, PermissionGrantedEvent, PermissionDeniedEvent),
        )
    ]
    assert [type(event) for event in permission_events] == expected_permission_types
    if emit_requested:
        denied = permission_events[-1]
        assert isinstance(denied, PermissionDeniedEvent)
        assert denied.run_id == "r1"
        assert denied.tool_use_id == "t1"
        assert denied.decision == "deny_once"
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_class for event in failed] == ["permission_denied"]
    assert [event.error_message for event in failed] == [expected_message]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证缺少必填参数时返回 schema_error 而非 runtime_error
# 设计：注册需要 msg 参数的 EchoTool 但传空 input，确认错误分类准确，schema 错误与运行时错误对 S4 重试策略有不同影响
async def test_missing_required_param_gives_schema_error() -> None:
    registry = ToolRegistry()
    tool = _TrackedSchemaTool()
    registry.register(tool)
    result, events = await _run(registry, _call("tracked_schema", {}))
    assert result.is_error
    assert result.error_type == "schema_error"
    assert result.content == "invalid tool input: msg [missing]"
    assert tool.calls == 0
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_message for event in failed] == [
        "invalid tool input: msg [missing]"
    ]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 invoke_tool 返回的 schema_error 不泄露 mapping 用户键和值
# 设计：通过真实参数校验管线触发动态键错误，并断言工具未执行且 failed event 同步使用安全摘要
async def test_mapping_validation_error_redacts_user_controlled_key() -> None:
    registry = ToolRegistry()
    tool = _TrackedMappingTool()
    registry.register(tool)

    result, events = await _run(
        registry,
        _call(
            "tracked_mapping",
            {"mapping": {"token=secret": "raw-value"}},
        ),
    )

    assert result.is_error
    assert result.error_type == "schema_error"
    assert result.content == "invalid tool input: mapping.<key> [int_parsing]"
    assert "token=secret" not in result.content
    assert "raw-value" not in result.content
    assert tool.calls == 0
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_message for event in failed] == [result.content]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 invoke_tool 不会把整数 mapping 键作为数组索引回显
# 设计：真实管线传入动态整数键，断言结果与 failed event 均使用 <key> 且工具未执行
async def test_mapping_validation_error_redacts_integer_key() -> None:
    registry = ToolRegistry()
    tool = _TrackedIntegerMappingTool()
    registry.register(tool)

    result, events = await _run(
        registry,
        _call(
            "tracked_integer_mapping",
            {"mapping": {8_675_309: "raw-value"}},
        ),
    )

    assert result.is_error
    assert result.error_type == "schema_error"
    assert result.content == "invalid tool input: mapping.<key> [int_parsing]"
    assert "8675309" not in result.content
    assert "raw-value" not in result.content
    assert tool.calls == 0
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_message for event in failed] == [result.content]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证工具执行超时时返回 timeout 类型错误而非 runtime_error
# 设计：使用永久 sleep 的 SlowTool + 极短超时（50ms），测试 asyncio.wait_for 的超时路径，确认超时被正确分类
async def test_timeout_gives_timeout_error() -> None:
    registry = ToolRegistry()
    registry.register(_SlowTool())
    result, events = await _run(registry, _call("slow"), timeout=0.01)
    assert result.is_error
    assert result.error_type == "timeout"
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "tool.call_failed" in types


# 功能：验证未知工具异常转为 execution_error 且不向 LLM 暴露原始异常文本
# 设计：工具直接抛带敏感文本的 RuntimeError，同时断言稳定摘要与 failed event 内容均已净化
async def test_unknown_exception_gives_safe_execution_error() -> None:
    registry = ToolRegistry()
    registry.register(_BrokenTool())
    result, events = await _run(registry, _call("broken"))
    assert result.is_error
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert "boom" not in result.content
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_message for event in failed] == ["tool execution failed"]


# 功能：验证参数 validator 抛未知异常时仍转为安全 execution_error
# 设计：使用真实 Pydantic field_validator 抛含敏感文本的 RuntimeError，锁定 pre-invoke 分类与 failed event
async def test_validator_unknown_exception_gives_safe_execution_error() -> None:
    registry = ToolRegistry()
    tool = _ValidatorBrokenTool()
    registry.register(tool)

    result, events = await _run(
        registry,
        _call("validator_broken", {"value": 1}),
    )

    assert result.is_error
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert "validator-secret" not in result.content
    assert tool.calls == 0
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_class for event in failed] == ["execution_error"]
    assert [event.error_message for event in failed] == ["tool execution failed"]
    assert [event.attempt for event in failed] == [1]
    assert all("validator-secret" not in event.error_message for event in failed)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 tool.call_started 始终是第一个被发布的事件，即使工具调用最终失败
# 设计：用不存在的工具触发失败路径，确认即使失败也先发布 started，保证事件流的时序可观测性
async def test_started_event_always_first() -> None:
    result, events = await _run(ToolRegistry(), _call("nonexistent"))
    assert events[0].type == "tool.call_started"  # type: ignore[attr-defined]


# 功能：验证工具执行期间的 CancelledError 原样传播且不伪造普通失败事件
# 设计：捕获取消信号后检查唯一事件是 started，锁定 cancellation 不重试、不 failed、不 finished 的语义
async def test_cancelled_error_propagates_without_failed_event() -> None:
    registry = ToolRegistry()
    registry.register(_CancelledTool())
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集取消发生前发布的事件
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    with pytest.raises(asyncio.CancelledError):
        await invoke_tool(registry, _call("cancelled"), bus, run_id="r1")

    assert [type(event) for event in events] == [ToolCallStartedEvent]


# 功能：验证 tool started/failed/finished 事件字段形状保持 wire schema 不变
# 设计：直接锁定三个 Pydantic model 的字段名集合，防止 taxonomy 改造意外新增或删除 wire 字段
def test_tool_event_field_shapes_are_unchanged() -> None:
    assert set(ToolCallStartedEvent.model_fields) == {
        "type",
        "run_id",
        "tool_use_id",
        "tool_name",
        "params",
        "ts",
    }
    assert set(ToolCallFailedEvent.model_fields) == {
        "type",
        "run_id",
        "tool_use_id",
        "tool_name",
        "error_class",
        "error_message",
        "elapsed_ms",
        "attempt",
        "ts",
    }
    assert set(ToolCallFinishedEvent.model_fields) == {
        "type",
        "run_id",
        "tool_use_id",
        "tool_name",
        "elapsed_ms",
        "output",
        "ts",
    }
