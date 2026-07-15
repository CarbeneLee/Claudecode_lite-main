from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, field_validator

from kama_claude.core.bus.events import (
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


class _SlowTool(BaseTool):
    name = "slow"
    description = "Sleeps forever"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        await asyncio.sleep(60)
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

    # validator 失败时不应到达真实工具执行
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise AssertionError("tool invoke should not run")


# --- helpers -----------------------------------------------------------------


def _call(name: str, inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {})


async def _run(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    timeout: float = 5.0,
) -> tuple[ToolResult, list[BaseModel]]:
    bus = EventBus()
    events: list[BaseModel] = []

    async def _collect(e: BaseModel) -> None:
        events.append(e)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, tool_call, bus, run_id="r1", timeout=timeout)
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


# 功能：验证调用不存在的工具时返回 unknown_tool 并发布 failed 事件而非 finished
# 设计：传入空 registry，确认稳定 error_type 和事件类型同时正确，排除未知工具被误当 execution_error
async def test_unknown_tool_returns_unknown_tool() -> None:
    result, events = await _run(ToolRegistry(), _call("nonexistent"))
    assert result.is_error
    assert result.error_type == "unknown_tool"
    assert "unknown tool" in result.content
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "tool.call_started" in types
    assert "tool.call_failed" in types
    assert "tool.call_finished" not in types


# 功能：验证缺少必填参数时返回 schema_error 而非 runtime_error
# 设计：注册需要 msg 参数的 EchoTool 但传空 input，确认错误分类准确，schema 错误与运行时错误对 S4 重试策略有不同影响
async def test_missing_required_param_gives_schema_error() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result, events = await _run(registry, _call("echo", {}))  # "msg" is required
    assert result.is_error
    assert result.error_type == "schema_error"
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "tool.call_failed" in types


# 功能：验证工具执行超时时返回 timeout 类型错误而非 runtime_error
# 设计：使用永久 sleep 的 SlowTool + 极短超时（50ms），测试 asyncio.wait_for 的超时路径，确认超时被正确分类
async def test_timeout_gives_timeout_error() -> None:
    registry = ToolRegistry()
    registry.register(_SlowTool())
    result, events = await _run(registry, _call("slow"), timeout=0.05)
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
    registry.register(_ValidatorBrokenTool())

    result, events = await _run(
        registry,
        _call("validator_broken", {"value": 1}),
    )

    assert result.is_error
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert "validator-secret" not in result.content
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_class for event in failed] == ["execution_error"]
    assert [event.error_message for event in failed] == ["tool execution failed"]


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
