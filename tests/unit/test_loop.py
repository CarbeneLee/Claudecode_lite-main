from __future__ import annotations

import asyncio
import hashlib

import pytest
from pydantic import BaseModel

from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.registry import ToolRegistry

_DEFAULT_BASE_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer "
    "and do not call any more tools."
)
_REQUIREMENT_CONTRACT = (
    "Before changing the workspace, create a concise requirement contract from every "
    "explicit acceptance criterion. For each item, record the required observable "
    "behavior, relevant failure or invalid-input behavior, any side-effect or state "
    "invariant, and the evidence you plan to use for verification. Keep this checklist "
    "visible in the conversation as you work, and update each item as implemented, "
    "verified, or unchecked. Before finishing, review every item. Do not assume unchecked "
    "items are complete: verify them when possible, otherwise clearly report the "
    "limitation. Keep the contract brief and auditable; do not expose private "
    "chain-of-thought or force any particular tool."
)
_STATE_TRANSITION_PROTOCOL = (
    "When a task changes persistent or shared state through multiple operations, briefly "
    "map the pre-state, each mutation point, every later operation that can fail, and the "
    "required post-state after success or failure. Before finishing, exercise at least "
    "one failure after an earlier mutation succeeds, and verify that rollback or "
    "compensation preserves the stated invariant. Do not apply this protocol to tasks "
    "without multi-step side effects."
)
_PHASE9B_PROMPT_HASH = (
    "b248587ef77d172cefb5e7b777a1523cf50978d6d273b466a8b6eb37349621eb"
)
_REQUIREMENT_CONTRACT_HASH = (
    "90fc45897aa8323175ceb0e5be6af3561ae44b2d0ca9bcb64244411d65561812"
)
_STATE_TRANSITION_PROTOCOL_HASH = (
    "310b308df8cad96fc20ff63b7f50f799ed717a16a65f67db392ca4f6eb7762a3"
)

# --- stubs -------------------------------------------------------------------


class _MockProvider:
    """Returns canned responses in order; raises exc immediately if given."""

    def __init__(
        self,
        responses: list[LlmResponse],
        exc: BaseException | None = None,
    ) -> None:
        self._responses = iter(responses)
        self._exc = exc
        self.seen_messages: list[list[dict[str, object]]] = []
        self.seen_systems: list[str | None] = []
        self.seen_tool_schemas: list[list[dict[str, object]]] = []

    # 捕获每次模型调用收到的消息、系统提示和有序工具 schema
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self.seen_messages.append([dict(message) for message in messages])
        self.seen_systems.append(system)
        self.seen_tool_schemas.append([dict(schema) for schema in tool_schemas])
        if self._exc is not None:
            raise self._exc
        return next(self._responses)


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes msg"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


class _FailTool(BaseTool):
    name = "fail"
    description = "Always raises"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("tool error")


class _CancelledTool(BaseTool):
    name = "cancelled"
    description = "Always cancels"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 抛出取消信号以验证工具阶段能中断 AgentLoop
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise asyncio.CancelledError


# --- helpers -----------------------------------------------------------------


def _ctx(max_steps: int = 5) -> ExecutionContext:
    return ExecutionContext(run_id="r1", goal="test goal", max_steps=max_steps)


def _tc(name: str = "echo", inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {"msg": "hi"})


def _make_loop(
    provider: _MockProvider,
    registry: ToolRegistry | None = None,
    bus: EventBus | None = None,
) -> tuple[AgentLoop, EventBus]:
    b = bus or EventBus()
    return AgentLoop(provider, registry or ToolRegistry(), b), b  # type: ignore[arg-type]


async def _events(bus: EventBus) -> list[BaseModel]:
    collected: list[BaseModel] = []

    async def _h(e: BaseModel) -> None:
        collected.append(e)

    bus.subscribe(_h)
    return collected


# --- tests -------------------------------------------------------------------


# 功能：验证 LLM 返回 end_turn 时 loop 将 context 标记为 success
# 设计：单步 provider 直接返回 end_turn，最简正常路径，确认 loop 的基本终止逻辑
async def test_end_turn_marks_success() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="done")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 1


# 功能：验证达到 max_steps 时 loop 以 exceeded_max_steps 原因将 context 标记为 failed
# 设计：设置 max_steps=2 + 无限 tool_use provider，同时验证 step 数量和失败原因，确认计数器与终止逻辑联动正确
async def test_max_steps_marks_failed() -> None:
    tc = _tc("unknown", {})
    provider = _MockProvider([LlmResponse(stop_reason="tool_use", tool_calls=[tc])] * 10)
    loop, _ = _make_loop(provider)
    ctx = _ctx(max_steps=2)
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "exceeded_max_steps"
    assert ctx.step == 2


# 功能：验证"调工具 → end_turn"的两步路径最终标记为 success
# 设计：provider 返回 [tool_use, end_turn] 序列，注册真实 EchoTool，覆盖最常见的正常工作路径
async def test_tool_use_then_end_turn_marks_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn", text="summary"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具结果按 Anthropic 格式（tool_result user 消息）追加到消息历史
# 设计：检查 messages[2]（tool_result 所在位置），断言 tool_use_id 和 content，确认 loop 正确调用了 context.add_tool_result
async def test_tool_result_appended_to_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc(inp={"msg": "hello"})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    # messages: [goal, assistant(tool_use), user(tool_result), assistant(end_turn)]
    tool_result_msg = ctx.messages[2]
    assert tool_result_msg["role"] == "user"
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "hello"


# 功能：验证工具失败时 loop 不终止，而是将错误追加上下文让 LLM 重新决策
# 设计：工具始终 raise + provider 第二步返回 end_turn，确认 loop 最终到达 success；这是 agent 区别于普通脚本的核心特性
async def test_tool_failure_loop_continues_to_success() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn", text="handled error"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "success"
    assert ctx.step == 2


# 功能：验证工具失败的错误信息以 is_error=True 追加进上下文，让 LLM 能感知工具调用失败
# 设计：检查 tool_result block 中的 is_error 标记，与 test_tool_failure_loop_continues_to_success 互补
async def test_tool_failure_result_is_error_in_context() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()
    await loop.run(ctx)
    tool_result_msg = ctx.messages[2]
    block = tool_result_msg["content"][0]  # type: ignore[index]
    assert block.get("is_error") is True


# 功能：验证未知工具异常的稳定安全摘要会进入下一轮 LLM context
# 设计：记录 provider 每轮收到的 messages，直接检查第二轮 tool_result 的 content 与 is_error，不依赖最终 context 的事后状态
async def test_safe_tool_error_is_visible_to_next_llm_turn() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("fail", {})]),
        LlmResponse(stop_reason="end_turn", text="handled safely"),
    ])
    registry = ToolRegistry()
    registry.register(_FailTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()

    await loop.run(ctx)

    tool_result_message = provider.seen_messages[1][2]
    content = tool_result_message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    assert block["content"] == "tool execution failed"
    assert block["is_error"] is True
    assert "tool error" not in str(block["content"])


# 功能：验证工具执行期间的 CancelledError 可穿过 invoke_tool 中断 AgentLoop
# 设计：让 provider 发起真实 tool_use、工具随后取消，断言 loop 原样上抛且不会继续请求下一轮 LLM
async def test_tool_cancelled_error_interrupts_agent_loop() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc("cancelled", {})]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_CancelledTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx()

    with pytest.raises(asyncio.CancelledError):
        await loop.run(ctx)

    assert len(provider.seen_messages) == 1


# 功能：验证收到 CancelledError 时 loop 将 context 标记为 cancelled 后继续上抛 CancelledError
# 设计：用 pytest.raises 捕获 CancelledError，同时检查 context.status，确认优雅退出行为：先记录状态，再传播取消信号
async def test_cancelled_error_marks_failed_and_reraises() -> None:
    provider = _MockProvider([], exc=asyncio.CancelledError())
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    with pytest.raises(asyncio.CancelledError):
        await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "cancelled"


# 功能：验证 LLM 调用异常被捕获并标记为 llm_error，不向上传播
# 设计：provider 抛 RuntimeError，确认 loop 不崩溃、context 状态为 failed/llm_error，异常被正确吸收
async def test_llm_api_error_marks_failed() -> None:
    provider = _MockProvider([], exc=RuntimeError("api error"))
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assert ctx.status == "failed"
    assert ctx.reason == "llm_error"


# 功能：验证每个步骤都发布 step.started 和 step.finished 事件
# 设计：注入 bus + 事件收集器，检查事件类型集合，确认步骤级事件的可观测性（S2 TUI 依赖这两个事件显示进度）
async def test_step_started_and_finished_events_published() -> None:
    bus = EventBus()
    events = await _events(bus)
    provider = _MockProvider([LlmResponse(stop_reason="end_turn")])
    loop, _ = _make_loop(provider, bus=bus)
    ctx = _ctx()
    await loop.run(ctx)
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "step.started" in types
    assert "step.finished" in types


# 功能：验证多步执行后 step 计数器正确累积到步数总量
# 设计：三步序列 [tool_use, tool_use, end_turn]，确认 step==3，排除计数器初始化错误或某步未递增的情况
async def test_step_counter_increments_across_steps() -> None:
    provider = _MockProvider([
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
        LlmResponse(stop_reason="end_turn"),
    ])
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)
    ctx = _ctx(max_steps=10)
    await loop.run(ctx)
    assert ctx.step == 3
    assert ctx.status == "success"


# 功能：验证 LLM 文本响应以正确的 content block 格式追加到消息历史
# 设计：检查 messages[1] 的 role 和 content block 结构，确认 loop 构造的 assistant 消息符合 Anthropic 格式
async def test_assistant_message_blocks_added_to_context() -> None:
    provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="answer")])
    loop, _ = _make_loop(provider)
    ctx = _ctx()
    await loop.run(ctx)
    assistant_msg = ctx.messages[1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    assert blocks[0]["type"] == "text"  # type: ignore[index]
    assert blocks[0]["text"] == "answer"  # type: ignore[index]


# 功能：验证 Phase 9B 的 base 与 v1 字节身份在追加 v2 前保持冻结
# 设计：直接对冻结文本做词数、字节数和 SHA-256 检查，避免未来 prompt 变化改写历史 control
def test_phase9b_prompt_identity_remains_frozen() -> None:
    prompt = _DEFAULT_BASE_PROMPT + "\n\n" + _REQUIREMENT_CONTRACT

    assert len(_REQUIREMENT_CONTRACT.split()) == 97
    assert len(_REQUIREMENT_CONTRACT.encode("utf-8")) == 676
    assert (
        hashlib.sha256(_REQUIREMENT_CONTRACT.encode("utf-8")).hexdigest()
        == _REQUIREMENT_CONTRACT_HASH
    )
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == _PHASE9B_PROMPT_HASH


# 功能：验证冻结的 state-transition protocol 文本满足精确 addition 身份
# 设计：同时锁定词数、UTF-8 字节数和完整 SHA-256，防止摘要中的截断 hash 或同义改写进入实现
def test_state_transition_protocol_addition_identity_is_frozen() -> None:
    assert len(_STATE_TRANSITION_PROTOCOL.split()) == 65
    assert len(_STATE_TRANSITION_PROTOCOL.encode("utf-8")) == 440
    assert (
        hashlib.sha256(_STATE_TRANSITION_PROTOCOL.encode("utf-8")).hexdigest()
        == _STATE_TRANSITION_PROTOCOL_HASH
    )


# 功能：验证每次默认模型调用精确包含一次冻结的 v1 与 v2 指导
# 设计：使用两步 scripted provider 捕获完整 system，锁定 base、双换行、v1、双换行、v2 的字节级组合
async def test_default_prompt_contains_v1_and_v2_once_per_call() -> None:
    provider = _MockProvider(
        [
            LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
            LlmResponse(stop_reason="end_turn", text="done"),
        ]
    )
    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop, _ = _make_loop(provider, registry)

    await loop.run(_ctx())

    expected = (
        _DEFAULT_BASE_PROMPT
        + "\n\n"
        + _REQUIREMENT_CONTRACT
        + "\n\n"
        + _STATE_TRANSITION_PROTOCOL
    )
    for system in provider.seen_systems:
        assert system is not None
        assert system.count(_REQUIREMENT_CONTRACT) == 1
        assert system.count(_STATE_TRANSITION_PROTOCOL) == 1
        assert system.index(_REQUIREMENT_CONTRACT) < system.index(
            _STATE_TRANSITION_PROTOCOL
        )
    assert provider.seen_systems == [expected, expected]


# 功能：验证新增 contract 不改写消息、上下文顺序、工具顺序、步数、停止状态或事件序列
# 设计：联合真实工具调用与三层 context，比较 provider 捕获值和完整 step/tool lifecycle
async def test_requirement_contract_preserves_runtime_inputs_and_lifecycle() -> None:
    provider = _MockProvider(
        [
            LlmResponse(stop_reason="tool_use", tool_calls=[_tc()]),
            LlmResponse(stop_reason="end_turn", text="done"),
        ]
    )
    registry = ToolRegistry()
    registry.register(_EchoTool())
    registry.register(_FailTool())
    expected_tools = registry.tool_schemas()
    bus = EventBus()
    events = await _events(bus)
    loop, _ = _make_loop(provider, registry, bus)
    ctx = ExecutionContext(
        run_id="r1",
        goal="Implement behavior A and preserve invariant B.",
        max_steps=5,
        global_context="global-marker",
        project_context="project-marker",
        session_notes="session-marker",
    )

    await loop.run(ctx)

    assert provider.seen_messages[0] == [
        {"role": "user", "content": "Implement behavior A and preserve invariant B."}
    ]
    assert provider.seen_tool_schemas == [expected_tools, expected_tools]
    assert ctx.max_steps == 5
    assert ctx.step == 2
    assert ctx.status == "success"
    assert ctx.result == "done"
    systems = provider.seen_systems
    assert all(system is not None for system in systems)
    for system in systems:
        assert system is not None
        assert system.count(_REQUIREMENT_CONTRACT) == 1
        assert system.count(_STATE_TRANSITION_PROTOCOL) == 1
        assert system.index(_REQUIREMENT_CONTRACT) < system.index(_STATE_TRANSITION_PROTOCOL)
        assert system.index(_STATE_TRANSITION_PROTOCOL) < system.index("## Global Context")
        assert system.index("## Global Context") < system.index("## Project Context")
        assert system.index("## Project Context") < system.index("## Session Notes")
        assert "global-marker" in system
        assert "project-marker" in system
        assert "session-marker" in system
    assert [event.type for event in events] == [  # type: ignore[attr-defined]
        "step.started",
        "tool.call_started",
        "tool.call_finished",
        "step.finished",
        "step.started",
        "step.finished",
    ]


# 功能：验证 stateful 与 non-stateful goal 使用相同 prompt、工具和 loop 生命周期
# 设计：用同一 scripted provider 分别执行两类 goal，比较除 user goal 外的全部 runtime 观测值
async def test_stateful_and_non_stateful_goals_share_runtime_control_flow() -> None:
    observations: list[tuple[str | None, list[dict[str, object]], list[str], str]] = []
    for goal in (
        "Explain how the parser handles quoted input.",
        "Import several orders atomically and roll back on failure.",
    ):
        provider = _MockProvider([LlmResponse(stop_reason="end_turn", text="done")])
        registry = ToolRegistry()
        registry.register(_EchoTool())
        bus = EventBus()
        events = await _events(bus)
        loop, _ = _make_loop(provider, registry, bus)
        ctx = ExecutionContext(run_id="r1", goal=goal, max_steps=5)

        await loop.run(ctx)

        observations.append(
            (
                provider.seen_systems[0],
                provider.seen_tool_schemas[0],
                [event.type for event in events],  # type: ignore[attr-defined]
                ctx.status,
            )
        )

    assert observations[0] == observations[1]
