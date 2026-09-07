from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kama_claude.core import app as app_module
from kama_claude.core import compact as compact_module
from kama_claude.core import runner as runner_module
from kama_claude.core.bus.events import RunFinishedEvent
from kama_claude.core.compact.compactor import CompactionConflict, Compactor
from kama_claude.core.compact.protocol import (
    CompactionCheckpointEnvelope,
    CompactionCheckpointPayload,
)
from kama_claude.core.config import KamaConfig, _apply_toml
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.gateway import ProviderRequestGateway
from kama_claude.core.llm.types import (
    DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
    LlmResponse,
    ProviderContinuationState,
    ToolCallBlock,
    UsageStats,
)
from kama_claude.core.llm.usage import ProviderUsageNormalizer, RawUsageEnvelope
from kama_claude.core.loop import AgentLoop
from kama_claude.core.mcp.client import McpClient, McpToolDef
from kama_claude.core.mcp.tool import McpTool
from kama_claude.core.runner import AgentRunner
from kama_claude.core.sandbox.executors import HostExecutor
from kama_claude.core.session.manager import SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool
from kama_claude.core.task_contract import TaskContractRecord
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.invocation import DirectToolInvoker
from kama_claude.core.tools.registry import ToolRegistry


class _FiniteProvider:
    """为 closeout 测试提供有限且可审计的 provider 调用序列。"""

    model = "deepseek-v4-flash"
    protocol = "anthropic"
    usage_schema = "deepseek_anthropic_messages_v1"
    route_identity = "deepseek-v4-flash-test"
    context_window = 10_000
    max_output_tokens = 64
    output_budget_semantics = "inclusive_total"

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.requests: list[list[dict[str, object]]] = []

    # 按预先声明的序列返回响应或异常，意外调用立即失败
    async def chat(
        self,
        messages: list[dict[str, object]],
        _tool_schemas: list[dict[str, object]],
        _bus: EventBus,
        _run_id: str,
        **_: object,
    ) -> LlmResponse:
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("unexpected provider call")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, LlmResponse)
        return response


class _BlockingSummaryProvider(_FiniteProvider):
    """在 maintenance summary 返回前暂停，制造真实 surface CAS race。"""

    def __init__(self, responses: list[object]) -> None:
        super().__init__(responses)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    # 先通知测试已捕获 snapshot，再等待并发 mutation 完成后继续 provider 调用
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        **kwargs: object,
    ) -> LlmResponse:
        self.entered.set()
        await self.release.wait()
        return await super().chat(messages, tool_schemas, bus, run_id, **kwargs)


class _RecordingInvoker:
    """记录工具执行次数并返回固定的有界 receipt。"""

    def __init__(self, result: ToolResult | None = None) -> None:
        self.calls: list[ToolCallBlock] = []
        self.result = result or ToolResult(content="ok")

    # 提供最小 schema，避免测试越过 AgentLoop 的真实 invocation boundary
    def tool_schemas(self) -> list[dict[str, object]]:
        return [
            {
                "name": "write_file",
                "description": "test tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    # 记录实际执行；调用次数超出测试期望由断言捕获
    async def invoke(self, tool_call: ToolCallBlock) -> ToolResult:
        self.calls.append(tool_call)
        return self.result

    # 测试工具没有硬终止语义
    def terminal_reason(self) -> str | None:
        return None


class _RecordingCompactor:
    """在 admission 压力下把 surface 降到单条 user 消息。"""

    def __init__(self) -> None:
        self.calls = 0

    # 记录主动压缩并执行最小可逆 surface reduction
    async def compact(self, context: ExecutionContext, _provider: object, **_: object) -> object:
        self.calls += 1
        context.messages = [
            {"role": "user", "content": "compacted context"},
        ]
        context._rebuild_inference_units()
        context.refresh_surface_state()
        return object()


def _context(*, tool_result_limit: int = 40_000) -> ExecutionContext:
    """构造 closeout 测试共享的执行上下文。"""
    return ExecutionContext(
        run_id="closeout-run",
        goal="test goal",
        max_steps=3,
        tool_result_limit=tool_result_limit,
        tool_result_keep=max(1, min(tool_result_limit, 20_000)),
    )


# 功能：验证 usage normalizer 按 wire/API schema 解析不同 vendor/协议的 cache 与 reasoning 字段
# 设计：同一 normalizer 运行四种实际 payload fixture，断言 cached tokens 始终计入 occupancy 且不重复相加
@pytest.mark.parametrize(
    ("schema", "payload", "total", "hit", "miss", "reasoning"),
    [
        (
            "deepseek_responses_v1",
            {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 40}, "output_tokens_details": {"reasoning_tokens": 5}},
            100,
            40,
            60,
            5,
        ),
        (
            "native_anthropic_v1",
            {"input_tokens": 50, "cache_read_input_tokens": 30, "cache_creation_input_tokens": 10, "output_tokens": 8},
            90,
            30,
            60,
            0,
        ),
        (
            "openai_chat_completions_v1",
            {"prompt_tokens": 90, "prompt_tokens_details": {"cached_tokens": 25}, "completion_tokens": 7},
            90,
            25,
            65,
            0,
        ),
        (
            "openai_responses_v1",
            {"input_tokens": 80, "input_tokens_details": {"cached_tokens": 20}, "output_tokens": 9},
            80,
            20,
            60,
            0,
        ),
    ],
)
def test_usage_normalizer_protocol_fixtures(
    schema: str,
    payload: dict[str, object],
    total: int,
    hit: int,
    miss: int,
    reasoning: int,
) -> None:
    result = ProviderUsageNormalizer().normalize(
        RawUsageEnvelope(usage_schema=schema, payload=payload, route="deepseek-v4-flash")
    )
    assert result.total_input_tokens == total
    assert result.cache_hit_tokens == hit
    assert result.cache_miss_tokens == miss
    assert result.reasoning_output_tokens == reasoning


_VALID_SUMMARY = (
    "## 1. Original Goal\nTest\n"
    "## 2. Completed Steps\n- done\n"
    "## 3. Key Constraints & Discoveries\n- none\n"
    "## 4. Current File State\n- unchanged\n"
    "## 5. Remaining TODOs\n- none\n"
    "## 6. Critical Data\n- none"
)


# 功能：验证主动压缩依据 canonical next request 的 admission，而不是旧响应 context_pct
# 设计：第一步返回低 context_pct 但制造高成本 tool receipt，只有 pre-step admission 才能触发压缩
@pytest.mark.asyncio
async def test_proactive_compaction_uses_next_request_admission() -> None:
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="t1", name="write_file", input={})],
                usage=UsageStats(input_tokens=1, output_tokens=1, context_pct=0.0),
            ),
            LlmResponse(stop_reason="end_turn", text="done"),
        ]
    )
    invoker = _RecordingInvoker(ToolResult(content="x" * 7_000))
    compactor = _RecordingCompactor()
    context = _context()
    context.messages[0]["content"] = "initial request"

    await AgentLoop(
        provider,
        invoker,
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_threshold=0.80,
    ).run(context)

    assert context.status == "success"
    assert compactor.calls == 1
    assert provider.calls == 2


# 功能：验证未知模型容量 fail-closed 且 DeepSeek V4 保留 1M capacity
# 设计：显式未知 route 不允许静默 200K，已知 DeepSeek route 必须暴露 1_000_000
def test_unknown_capacity_fails_closed_and_deepseek_v4_is_one_million(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown = _FiniteProvider([])
    unknown.model = "mystery-model"
    unknown.context_window = None
    with pytest.raises(ValueError, match="context capacity"):
        ProviderRequestGateway(unknown)

    known = _FiniteProvider([])
    known.context_window = None
    assert ProviderRequestGateway(known).context_window == 1_000_000

    from kama_claude.core.llm.provider import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    known_anthropic = AnthropicProvider(model="claude-sonnet-4-6", client=object())
    assert ProviderRequestGateway(known_anthropic).context_window == 200_000
    explicit = _FiniteProvider([])
    explicit.model = "mystery-model"
    explicit.context_window = None
    assert ProviderRequestGateway(explicit, context_window=12_345).context_window == 12_345


# 功能：验证 automatic compaction 默认启用、legacy zero threshold 保持 opt-out 且 target 必须低于 trigger
# 设计：分别应用 fresh、legacy、显式 opt-in 和非法 TOML 配置，锁定迁移语义而不启动 daemon
def test_compaction_config_migration_and_ratio_validation() -> None:
    fresh = KamaConfig()
    assert fresh.compaction.auto_compact is True
    assert fresh.compaction.effective_threshold == 0.80

    legacy = KamaConfig()
    _apply_toml(legacy, {"compaction": {"auto_threshold": 0.0}})
    assert legacy.compaction.auto_compact is False
    assert legacy.compaction.effective_threshold == 0.0

    opted_in = KamaConfig()
    _apply_toml(opted_in, {"compaction": {"auto_threshold": 0.0, "auto_compact": True}})
    assert opted_in.compaction.effective_threshold == opted_in.compaction.soft_trigger_ratio

    disabled = KamaConfig()
    disabled.compaction.auto_compact = False
    disabled.compaction.auto_threshold = 0.9
    assert disabled.compaction.effective_threshold == 0.0

    with pytest.raises(SystemExit, match="below soft_trigger_ratio"):
        _apply_toml(
            KamaConfig(),
            {"compaction": {"soft_trigger_ratio": 0.7, "target_ratio": 0.8}},
        )


# 功能：验证 legacy auto_threshold 会实际接入 AgentLoop 使用的 soft trigger
# 设计：替换 loop 构造器记录 runner wiring，避免 provider 调用掩盖配置字段是否真正生效
@pytest.mark.asyncio
async def test_legacy_threshold_reaches_agent_loop_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = KamaConfig()
    _apply_toml(config, {"compaction": {"auto_threshold": 0.9}})
    provider = _FiniteProvider([])
    captured: dict[str, object] = {}

    class _RecordingLoop:
        # 记录 runner 传给 AgentLoop 的 compaction policy
        def __init__(self, _provider: object, _invoker: object, _bus: object, **kwargs: object) -> None:
            captured.update(kwargs)

        # 让 runner 完成终局 barrier 而不发起额外 provider 请求
        async def run(self, context: ExecutionContext) -> None:
            context.mark_success()
            context.result = "done"

    monkeypatch.setattr(runner_module, "AgentLoop", _RecordingLoop)
    runner = AgentRunner(
        config,
        workspace_root=tmp_path,
        provider=provider,  # type: ignore[arg-type]
        runs_dir=tmp_path / "runs",
    )
    outcome = await runner.run_and_capture("legacy threshold goal")

    assert outcome.status == "success"
    assert captured["soft_trigger_ratio"] == 0.9
    assert captured["compact_threshold"] == 1.0


# 功能：验证手动 compaction 使用与自动路径相同的 CompactionConfig policy
# 设计：只准备一条可压缩 durable assistant row，并包裹真实构造器记录所有 policy 参数
@pytest.mark.asyncio
async def test_manual_compaction_uses_shared_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = KamaConfig()
    config.compaction.summary_max_tokens = 123
    config.compaction.tool_result_limit = 456
    config.compaction.tool_result_keep = 234
    config.compaction.recent_tail_ratio = 0.07
    config.compaction.recent_tail_max_tokens = 789
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="end_turn",
                text=_VALID_SUMMARY,
                usage=UsageStats(input_tokens=1, output_tokens=2),
            )
        ]
    )
    captured: dict[str, object] = {}
    original_init = compact_module.Compactor.__init__

    def recording_init(self: Compactor, *args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(compact_module.Compactor, "__init__", recording_init)
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(
        store,
        lambda _root: AgentRunner(config, workspace_root=tmp_path, provider=provider),
        EventBus(),
        provider=provider,  # type: ignore[arg-type]
        compaction_config=config.compaction,
    )
    session = await manager.create("chat", workspace_root=tmp_path.resolve())
    store.append_message(session.id, "assistant", "durable history " * 20)

    result = await manager.compact(session.id)

    assert result.summary_tokens == 2
    assert captured == {
        "summary_max_tokens": 123,
        "tool_result_limit": 456,
        "tool_result_keep": 234,
        "recent_tail_ratio": 0.07,
        "recent_tail_max_tokens": 789,
    }


# 功能：验证 provider 真实 context overflow 会转为 typed retry path
# 设计：第一调用抛 overflow，compactor 只允许一次 reduction，第二调用正常结束且不记录失败响应
@pytest.mark.asyncio
async def test_provider_context_overflow_retries_once_after_reduction() -> None:
    provider = _FiniteProvider(
        [RuntimeError("maximum context length exceeded"), LlmResponse(stop_reason="end_turn", text="ok")]
    )
    compactor = _RecordingCompactor()
    context = _context()

    await AgentLoop(
        provider,
        _RecordingInvoker(),
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_threshold=0.0,
    ).run(context)

    assert context.status == "success"
    assert provider.calls == 2
    assert compactor.calls == 1
    assistant_messages = [
        message for message in context.messages if message.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].get("content") == [{"type": "text", "text": "ok"}]


# 功能：验证 provider overflow 在一次 bounded reduction 后再次 overflow 会稳定终止且不无限重试
# 设计：有限 provider 只允许两次请求，断言 compactor 只调用一次并保留 CONTEXT_WINDOW_EXCEEDED 终态
@pytest.mark.asyncio
async def test_provider_context_overflow_stops_after_one_retry() -> None:
    provider = _FiniteProvider(
        [RuntimeError("prompt is too long"), RuntimeError("context_length_exceeded")]
    )
    compactor = _RecordingCompactor()
    context = _context()

    await AgentLoop(
        provider,
        _RecordingInvoker(),
        EventBus(),
        compactor=compactor,  # type: ignore[arg-type]
        compact_threshold=0.0,
    ).run(context)

    assert context.status == "failed"
    assert context.reason == "CONTEXT_WINDOW_EXCEEDED"
    assert provider.calls == 2
    assert compactor.calls == 1


# 功能：验证 maintenance generation 收到 tool call 时在 gateway 层立即拒绝
# 设计：使用有限 provider 只返回一次 tool_use，断言不会把维护响应交给执行器或继续请求
@pytest.mark.asyncio
async def test_maintenance_gateway_rejects_tool_call_before_execution() -> None:
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="maintenance-tool", name="bash", input={})],
            )
        ]
    )
    gateway = ProviderRequestGateway(provider, context_window=100_000, safety_margin=0)

    with pytest.raises(RuntimeError, match="unexpected tool call"):
        await gateway.chat(
            messages=[{"role": "user", "content": "summarize"}],
            tool_schemas=[{"name": "bash"}],
            bus=EventBus(),
            run_id="maintenance-tool-call",
            maintenance=True,
        )
    assert provider.calls == 1


# 功能：验证 max_tokens 截断的工具调用不会执行或进入 active surface
# 设计：用真实 AgentLoop + invoker 记录调用，并检查 provider messages 不含未完成 tool_use
@pytest.mark.asyncio
async def test_incomplete_tool_call_is_not_executed_or_recorded() -> None:
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="max_tokens",
                text="partial",
                tool_calls=[ToolCallBlock(id="partial-1", name="write_file", input={})],
            )
        ]
    )
    invoker = _RecordingInvoker()
    context = _context()

    await AgentLoop(provider, invoker, EventBus(), compact_threshold=0.0).run(context)

    assert context.status == "failed"
    assert invoker.calls == []
    assert all(
        not (
            message.get("role") == "assistant"
            and any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in message.get("content", [])
            )
        )
        for message in context.messages
    )


# 功能：验证 DeepSeek thinking+tools 在两次工具步、同路由压缩和后续请求中按 unit 精确 replay
# 设计：有限 provider 固定返回 R1/T1、R2/T2、summary 三次响应，压缩只 shadow 第一 unit 并断言 R2 保留而 R1 离开 active surface
@pytest.mark.asyncio
async def test_deepseek_thinking_tools_replay_across_compaction(tmp_path: Path) -> None:
    state_1 = ProviderContinuationState.from_thinking_blocks(
        [{"type": "thinking", "thinking": "R1", "signature": "sig-1"}],
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
        route_identity="deepseek-v4-flash-test",
        protocol="anthropic",
    )
    state_2 = ProviderContinuationState.from_thinking_blocks(
        [{"type": "thinking", "thinking": "R2", "signature": "sig-2"}],
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
        route_identity="deepseek-v4-flash-test",
        protocol="anthropic",
    )
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="t1", name="write_file", input={})],
                continuation_state=state_1,
            ),
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="t2", name="write_file", input={})],
                continuation_state=state_2,
            ),
            LlmResponse(stop_reason="end_turn", text=_VALID_SUMMARY, usage=UsageStats(input_tokens=1, output_tokens=30)),
            LlmResponse(stop_reason="end_turn", text="next request accepted"),
        ]
    )
    provider.context_window = 15_000
    gateway = ProviderRequestGateway(provider, context_window=15_000, safety_margin=0)
    invoker = _RecordingInvoker(ToolResult(content="x" * 5_000))
    context = _context(tool_result_limit=6_000)
    context.max_steps = 2

    await AgentLoop(gateway, invoker, EventBus(), compact_threshold=0.0).run(context)
    assert [call.id for call in invoker.calls] == ["t1", "t2"]
    active_before = context.provider_messages()
    assert any(
        block.get("thinking") == "R1"
        for message in active_before
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict)
    )
    assert any(
        block.get("thinking") == "R2"
        for message in active_before
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict)
    )

    compactor = Compactor(EventBus(), tmp_path / "summary", "deepseek-replay")
    result = await compactor.compact(
        context,
        gateway,
        same_route=True,
        system="base",
        immutable_system="base",
        tool_schemas=invoker.tool_schemas(),
    )

    assert result is not None
    maintenance_request = provider.requests[2]
    assert any(
        block.get("thinking") == "R1"
        for message in maintenance_request
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict)
    )
    active_after = context.provider_messages()
    thinking_after = [
        block.get("thinking")
        for message in active_after
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    assert "R1" not in thinking_after
    assert thinking_after.count("R2") == 1
    assert all(
        "thinking" not in str(message.get("content", ""))
        or "R1" not in str(message.get("content", ""))
        for message in active_after
    )
    await gateway.chat(
        messages=active_after,
        tool_schemas=invoker.tool_schemas(),
        bus=EventBus(),
        run_id=context.run_id,
        system="base",
        immutable_system="base",
        surface_revision=context.surface_state.surface_revision,
    )
    next_request = provider.requests[3]
    assert any(
        block.get("thinking") == "R2"
        for message in next_request
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict)
    )
    assert all(
        block.get("thinking") != "R1"
        for message in next_request
        if message.get("role") == "assistant"
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "thinking"
    )
    assert provider.calls == 4


# 功能：验证 summarizer 在真实异步执行期间遇到 user surface mutation 会返回 CONFLICT 且丢弃旧摘要
# 设计：用 entered/release barrier 让 summary provider 阻塞，再由并发 coroutine 追加消息后释放并提交 CAS
@pytest.mark.asyncio
async def test_async_compaction_conflict_discards_stale_summary(tmp_path: Path) -> None:
    provider = _BlockingSummaryProvider([LlmResponse(stop_reason="end_turn", text=_VALID_SUMMARY)])
    gateway = ProviderRequestGateway(provider, context_window=100_000, safety_margin=0)
    context = _context()
    context.messages[0]["content"] = "x" * 10_000
    context.refresh_surface_state()
    compactor = Compactor(EventBus(), tmp_path / "summary", "cas-race")
    task = asyncio.create_task(
        compactor.compact(
            context,
            gateway,
            same_route=True,
            system="base",
            immutable_system="base",
            tool_schemas=[],
        )
    )

    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    context.messages.append(
        {"role": "user", "content": "new directive during summary", "_message_id": "m-race"}
    )
    context._rebuild_inference_units()
    context.refresh_surface_state()
    provider.release.set()

    with pytest.raises(CompactionConflict):
        await task
    assert context.checkpoint_envelope is None
    assert any(message.get("_message_id") == "m-race" for message in context.messages)


# 功能：验证 compaction summary 拒绝空输出与缺失必需结构
# 设计：参数化 provider 响应，确保维护 generation 不接受任意自然语言作为 checkpoint
@pytest.mark.asyncio
@pytest.mark.parametrize("summary", ["", "not a checkpoint"])
async def test_summary_acceptance_rejects_empty_or_invalid_structure(
    tmp_path: Path,
    summary: str,
) -> None:
    provider = _FiniteProvider([LlmResponse(stop_reason="end_turn", text=summary)])
    compactor = Compactor(EventBus(), tmp_path, "summary-closeout")

    result = await compactor.compact_messages(
        [{"role": "user", "content": "x" * 5_000}, {"role": "assistant", "content": "y" * 5_000}],
        ProviderRequestGateway(provider, context_window=50_000, safety_margin=0),
    )

    assert result is None


# 功能：验证 structured checkpoint 缺少语义字段时同样被拒绝，不因 JSON 形状合法而绕过 strict acceptance
# 设计：直接调用生产 parser 的 strict structured path，覆盖与 markdown rejection 不同的输入通道
def test_structured_summary_acceptance_requires_all_semantic_fields() -> None:
    from kama_claude.core.compact.protocol import accept_semantic_payload

    assert (
        accept_semantic_payload(
            "",
            {"progress": "done"},
            strict=True,
        )
        is None
    )


# 功能：验证 compaction summary 不比替换 span 更大时被拒绝
# 设计：provider usage 直接报告超大输出，覆盖 runtime 的 non-smaller acceptance guard
@pytest.mark.asyncio
async def test_summary_acceptance_rejects_non_smaller_replacement(tmp_path: Path) -> None:
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="end_turn",
                text=_VALID_SUMMARY,
                usage=UsageStats(input_tokens=10, output_tokens=50_000),
            )
        ]
    )
    result = await Compactor(EventBus(), tmp_path, "summary-closeout").compact_messages(
        [
            {"role": "user", "content": "x" * 20_000},
            {"role": "assistant", "content": "y" * 20_000},
        ],
        ProviderRequestGateway(provider, context_window=100_000, safety_margin=0),
    )

    assert result is None


# 功能：验证 checkpoint action fields 不得违反当前 TaskContract prohibition
# 设计：有效六段摘要只把违规动作放进 next step，确保不是结构校验误报
@pytest.mark.asyncio
async def test_summary_acceptance_rejects_stale_forbidden_action(tmp_path: Path) -> None:
    summary = _VALID_SUMMARY.replace("## 5. Remaining TODOs\n- none", "## 5. Remaining TODOs\n- modify foo.py")
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="end_turn",
                text=summary,
                usage=UsageStats(input_tokens=10, output_tokens=20),
            )
        ]
    )
    contract = TaskContractRecord.create(
        version=2,
        goal="finish the task",
        prohibitions=("do not modify foo.py",),
    )
    result = await Compactor(EventBus(), tmp_path, "summary-closeout").compact_messages(
        [{"role": "user", "content": "x" * 20_000}, {"role": "assistant", "content": "y" * 20_000}],
        ProviderRequestGateway(provider, context_window=100_000, safety_margin=0),
        task_contract=contract,
    )

    assert result is None


# 功能：验证 session thread/surface 已持久化后才发布 run.finished
# 设计：在真实 AgentRunner 的 terminal handler 内读取磁盘，捕获事件先于 store 写入的竞态
@pytest.mark.asyncio
async def test_run_finished_is_published_after_session_persistence(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-order",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
    )
    store.write_meta(session)
    observed: list[list[dict[str, Any]]] = []

    async def inspect_finished(event: object) -> None:
        if isinstance(event, RunFinishedEvent):
            observed.append(store.read_messages(session.id))

    runner = AgentRunner(
        KamaConfig(),
        workspace_root=tmp_path.resolve(),
        provider=_FiniteProvider([LlmResponse(stop_reason="end_turn", text="done")]),  # type: ignore[arg-type]
        extra_handlers=[inspect_finished],
        runs_dir=tmp_path / "runs",
    )

    await runner.run_and_capture(
        "goal",
        run_id="run-order",
        session=session,
        store=store,
    )

    assert observed
    assert any(message.get("role") == "assistant" for message in observed[0])


# 功能：验证最终 thread persistence 失败时不会发布虚假的 run.finished 成功事件
# 设计：注入 append_messages fail-fast，保留事件观察器并断言持久化异常先于 terminal publication 传播
@pytest.mark.asyncio
async def test_persistence_failure_blocks_run_finished(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-persist-failure",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
    )
    store.write_meta(session)
    observed: list[object] = []

    async def collect(event: object) -> None:
        observed.append(event)

    def fail_append(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected message persistence failure")

    store.append_messages = fail_append  # type: ignore[method-assign]
    runner = AgentRunner(
        KamaConfig(),
        workspace_root=tmp_path.resolve(),
        provider=_FiniteProvider([LlmResponse(stop_reason="end_turn", text="done")]),  # type: ignore[arg-type]
        extra_handlers=[collect],
        runs_dir=tmp_path / "runs",
    )

    with pytest.raises(OSError, match="persistence"):
        await runner.run_and_capture("goal", run_id="run-persist-failure", session=session, store=store)
    assert not any(isinstance(event, RunFinishedEvent) for event in observed)


# 功能：验证两个真实 CoreApp.run 不能同时持有同一 sessions-root lock，取消后锁可恢复
# 设计：使用真实 daemon startup 路径，仅替换网络监听和 provider，覆盖 ownership wiring 与 cancellation cleanup
@pytest.mark.asyncio
async def test_coreapp_root_lock_competes_and_releases_on_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = KamaConfig()
    config.trace.enabled = False
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda _model: _FiniteProvider([]))
    started = asyncio.Event()
    stop = asyncio.Event()

    async def fake_start(self: object) -> tuple[str, int]:
        started.set()
        await stop.wait()
        return ("127.0.0.1", 0)

    async def fake_stop(self: object) -> None:
        del self

    monkeypatch.setattr(app_module.SocketServer, "start", fake_start)
    monkeypatch.setattr(app_module.SocketServer, "stop", fake_stop)

    first = app_module.CoreApp()
    first_task = asyncio.create_task(first.run())
    await asyncio.wait_for(started.wait(), timeout=2)
    assert first._daemon_root_lock is not None
    assert first._daemon_root_lock.held

    with pytest.raises(RuntimeError, match="already owned"):
        await app_module.CoreApp().run()

    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    recovered = app_module.DaemonRootLock(tmp_path / ".kama" / "sessions")
    recovered.acquire()
    recovered.release()


# 功能：验证 Bash 经 DirectToolInvoker 的真实 ingress 对超大 stdout 施加 bounded receipt
# 设计：执行真实 Python 子进程而非只调用 bound_tool_output，覆盖 producer 到 model-visible 两层预算
@pytest.mark.asyncio
async def test_oversized_bash_result_is_bounded_through_invocation(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(BashTool(HostExecutor(), workspace_root=tmp_path))
    invoker = DirectToolInvoker(registry, EventBus(), "run-bash-budget")
    command = f"{shlex.quote(sys.executable)} -c \"print('x' * 200000)\""

    result = await invoker.invoke(
        ToolCallBlock(id="bash-1", name="bash", input={"command": command})
    )

    assert len(result.content) <= 8_100
    assert result.content.endswith("[truncated]")


# 功能：验证 MCP 超大远端 payload 经真实 McpTool + invocation pipeline 被截断
# 设计：fake client 只替换远端边界，仍走 schema lookup、event 和 receipt 生成路径
@pytest.mark.asyncio
async def test_oversized_mcp_result_is_bounded_through_invocation() -> None:
    client = AsyncMock(spec=McpClient)
    client.call_tool = AsyncMock(return_value="m" * 400_000)
    tool = McpTool(
        client,
        "remote",
        McpToolDef(name="fetch", description="fetch", input_schema={"type": "object"}),
    )
    registry = ToolRegistry()
    registry.register(tool)
    result = await DirectToolInvoker(registry, EventBus(), "run-mcp-budget").invoke(
        ToolCallBlock(id="mcp-1", name="remote__fetch", input={})
    )

    assert len(result.content) <= 8_000
    assert result.raw_truncated is True
    assert result.original_size == 400_000


# 功能：验证 subagent terminal overflow 只向 parent 暴露 bounded SubagentOutcome receipt
# 设计：通过 AgentResultTool + DirectToolInvoker 查询真实后台注册表，不注入 child traceback/history
@pytest.mark.asyncio
async def test_oversized_subagent_result_is_bounded_parent_receipt() -> None:
    registry_state = BackgroundTaskRegistry()
    child_context = _context()
    child_context.result = "child traceback " * 10_000
    child_context.mark_failed("CONTEXT_WINDOW_EXCEEDED")

    async def finished() -> None:
        return None

    child_task = asyncio.create_task(finished())
    await child_task
    child_id = "child-overflow"
    registry_state.register(child_id, child_task, child_context)
    registry = ToolRegistry()
    registry.register(AgentResultTool(registry_state))

    result = await DirectToolInvoker(registry, EventBus(), "parent-run").invoke(
        ToolCallBlock(id="agent-result-1", name="agent_result", input={"run_id": child_id})
    )

    assert result.is_error
    assert result.error_type == "context_window_exceeded", (result.content, result.terminal_outcome)
    assert len(result.content) <= 4_000
    assert result.content.count("child traceback") < 300
    assert result.terminal_outcome is not None


# 功能：验证 SessionManager→AgentRunner→AgentLoop→Gateway→tool→Compactor→Store→replay 全链路
# 设计：有限 provider 依次返回 tool_use、end_turn、summary，随后用新 SessionStore 实例重放
@pytest.mark.asyncio
async def test_deterministic_session_lifecycle_replays_after_compaction(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    bus = EventBus()
    config = KamaConfig()
    config.agent.max_steps = 3
    config.compaction.auto_compact = False
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="bash-life", name="bash", input={"command": f"{shlex.quote(sys.executable)} -c \\\"print('x' * 12000)\\\""})],
            ),
            LlmResponse(stop_reason="end_turn", text="work complete"),
            LlmResponse(stop_reason="end_turn", text=_VALID_SUMMARY, usage=UsageStats(input_tokens=10, output_tokens=30)),
        ]
    )

    def runner_factory(workspace_root: Path) -> AgentRunner:
        return AgentRunner(
            config,
            workspace_root=workspace_root,
            provider=provider,  # type: ignore[arg-type]
            runs_dir=tmp_path / "runs",
        )

    manager = SessionManager(store, runner_factory, bus, provider=provider)  # type: ignore[arg-type]
    session = await manager.create("chat", workspace_root=tmp_path.resolve())
    await manager.send_message(session.id, "run the bounded tool")
    task = manager._running_runs[session.id]
    await task

    compacted = await manager.compact(session.id, focus="preserve tool evidence")
    assert compacted.summary_tokens == 30
    assert store.read_committed_checkpoints(session.id)

    restarted_store = SessionStore(tmp_path / "sessions")
    replayed = restarted_store.read_messages(session.id)
    assert any(message.get("role") == "user" for message in replayed)
    assert any(message.get("role") == "assistant" for message in replayed)
    assert not any("bash-life" in str(message) for message in replayed)


# 功能：验证同一 run 内自动压缩前产生的 assistant/tool raw rows 先进入 durable transcript
# 设计：开启真实 proactive compaction，检查 raw step_commit、checkpoint、surface_replace 的顺序与 unit 关联
@pytest.mark.asyncio
async def test_auto_compaction_durably_appends_current_run_raw_rows(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    bus = EventBus()
    config = KamaConfig()
    config.trace.enabled = False
    config.agent.max_steps = 2
    config.compaction.auto_compact = True
    config.compaction.soft_trigger_ratio = 0.25
    config.compaction.target_ratio = 0.15
    config.compaction.recent_tail_ratio = 0.0
    config.compaction.summary_max_tokens = 64
    config.compaction.tool_result_limit = 40_000
    config.compaction.tool_result_keep = 30_000
    provider = _FiniteProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="auto-tool-a",
                        name="bash",
                        input={
                            "command": (
                                f"{shlex.quote(sys.executable)} -c "
                                + shlex.quote("print('x' * 12000)")
                            )
                        },
                    )
                ],
            ),
            LlmResponse(
                stop_reason="end_turn",
                text=_VALID_SUMMARY,
                usage=UsageStats(input_tokens=1, output_tokens=30),
            ),
            LlmResponse(stop_reason="end_turn", text="done"),
        ]
    )
    provider.context_window = 50_000

    def runner_factory(workspace_root: Path) -> AgentRunner:
        return AgentRunner(
            config,
            workspace_root=workspace_root,
            provider=provider,  # type: ignore[arg-type]
            runs_dir=tmp_path / "runs",
        )

    manager = SessionManager(store, runner_factory, bus, provider=provider)  # type: ignore[arg-type]
    session = await manager.create("chat", workspace_root=tmp_path.resolve())
    await manager.send_message(session.id, "run the auto-compacted tool", run_id="run-auto-raw")
    run_task = manager._running_runs[session.id]
    await run_task

    rows = store._read_thread_rows(session.id)
    raw_rows = [
        row
        for row in rows
        if row.get("record_type") == "step_commit" and row.get("run_id") == "run-auto-raw"
    ]
    assistant_rows = [
        row
        for row in raw_rows
        if row.get("role") == "assistant"
        and "auto-tool-a" in str(row.get("content", ""))
    ]
    tool_rows = [
        row
        for row in raw_rows
        if row.get("role") == "user"
        and "auto-tool-a" in str(row.get("content", ""))
    ]
    assert assistant_rows, "auto-compacted assistant raw row was not durable"
    assert tool_rows, "auto-compacted tool-result raw row was not durable"

    checkpoints = store.read_committed_checkpoints(session.id)
    assert checkpoints
    checkpoint_index = next(
        index for index, row in enumerate(rows) if row.get("record_type") == "checkpoint"
    )
    replacement_index = next(
        index for index, row in enumerate(rows) if row.get("record_type") == "surface_replace"
    )
    raw_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("record_type") == "step_commit" and row.get("run_id") == "run-auto-raw"
    ]
    assert raw_indices and max(raw_indices) < checkpoint_index < replacement_index
    selected_ids = set(checkpoints[-1].get("selected_unit_ids", []))
    durable_unit_ids = {
        str(row.get("unit_id"))
        for row in raw_rows
        if row.get("unit_id")
    }
    assert selected_ids <= durable_unit_ids

    replacements = [row for row in rows if row.get("record_type") == "surface_replace"]
    assert replacements
    shadowed_ids = {
        str(message_id)
        for message_id in replacements[-1].get("shadowed_message_ids", [])
    }
    assert {
        str(assistant_rows[0]["message_id"]),
        str(tool_rows[0]["message_id"]),
    } <= shadowed_ids
    active = store.read_messages(session.id)
    assert "auto-tool-a" not in str(active)


# 功能：验证三代 checkpoint 在 contract correction、连续 compaction 与 restart 后保留语义事实
# 设计：真实 append-only SessionStore 写入 v3→v4→v5，断言 A/C 与 prohibition 保留且 B 的旧 action 不复活
def test_three_generation_semantic_retention_survives_restart(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-generations",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
    )
    store.write_meta(session)
    a_id = store.append_message(session.id, "user", "requirement A")
    b_id = store.append_message(session.id, "user", "must modify B")
    store.append_message(session.id, "assistant", "completed A and B")
    contract_v3 = TaskContractRecord.create(
        version=3,
        goal="deliver A and B",
        requirements=("A", "B"),
        source_message_ids=(a_id, b_id),
    )
    store.append_task_contract(session.id, contract_v3)

    envelope_v3 = CompactionCheckpointEnvelope(
        generation=3,
        base_checkpoint_id=None,
        contract_digest=contract_v3.digest,
        selected_unit_ids=("unit-a",),
        selected_span_digest="span-v3",
        route_identity="test",
        route_epoch="route-v1",
        prefix_epoch="prefix-v1",
        summary_route="isolated",
        transaction_id="checkpoint-v3",
        payload=CompactionCheckpointPayload(
            progress="completed A and B",
            current_work="modify B",
            files_or_code=("A.py", "B.py"),
            errors_or_evidence=("tests for A passed",),
            pending=("modify B",),
            next_step="modify B",
        ),
    )
    store.append_checkpoint(session.id, envelope_v3.to_record(), atomic_surface_replace=True)
    store.write_compacted(
        session.id,
        [
            {"role": "user", "content": "checkpoint v3", "_checkpoint_id": "checkpoint-v3"},
            {"role": "assistant", "content": "continue from v3", "_checkpoint_id": "checkpoint-v3"},
        ],
        contract_digest=contract_v3.digest,
        base_surface_revision=0,
        surface_revision=1,
        checkpoint_id="checkpoint-v3",
    )
    store.append_surface_state(
        session.id,
        surface_revision=1,
        active_head_id=store.surface_head_id(session.id),
        contract_version=3,
        contract_digest=contract_v3.digest,
        active_checkpoint_id="checkpoint-v3",
        active_checkpoint_status="ACTIVE_CHECKPOINT",
    )

    correction_id = store.append_message(session.id, "user", "do not modify B; add C")
    contract_v4 = contract_v3.with_semantic_state(
        requirements=("A", "C"),
        prohibitions=("do not modify B",),
        source_message_ids=(correction_id,),
    )
    store.append_task_contract(session.id, contract_v4)
    store.append_message(session.id, "assistant", "completed factual work for C")
    stale_replay = store.read_messages(session.id)
    stale_rendered = "\n".join(str(message.get("content", "")) for message in stale_replay)
    assert "Historical execution facts" in stale_rendered
    assert "B.py" in stale_rendered
    assert "next_step" not in stale_rendered

    envelope_v4 = CompactionCheckpointEnvelope(
        generation=4,
        base_checkpoint_id="checkpoint-v3",
        contract_digest=contract_v4.digest,
        selected_unit_ids=("unit-c",),
        selected_span_digest="span-v4",
        route_identity="test",
        route_epoch="route-v1",
        prefix_epoch="prefix-v1",
        summary_route="isolated",
        transaction_id="checkpoint-v4",
        payload=CompactionCheckpointPayload(
            progress="A retained; C completed",
            current_work="verify C",
            files_or_code=("A.py", "B.py", "C.py"),
            errors_or_evidence=("C tests passed",),
            next_step="verify C",
        ),
    )
    store.append_checkpoint(session.id, envelope_v4.to_record(), atomic_surface_replace=True)
    revision = store.read_surface_revision(session.id)
    store.write_compacted(
        session.id,
        [
            {"role": "user", "content": "checkpoint v4: A.py and B.py facts retained; C.py completed", "_checkpoint_id": "checkpoint-v4"},
            {"role": "assistant", "content": "continue from v4", "_checkpoint_id": "checkpoint-v4"},
        ],
        contract_digest=contract_v4.digest,
        base_surface_revision=revision,
        surface_revision=revision + 1,
        checkpoint_id="checkpoint-v4",
    )

    envelope_v5 = CompactionCheckpointEnvelope(
        generation=5,
        base_checkpoint_id="checkpoint-v4",
        contract_digest=contract_v4.digest,
        selected_unit_ids=("unit-final",),
        selected_span_digest="span-v5",
        route_identity="test",
        route_epoch="route-v1",
        prefix_epoch="prefix-v1",
        summary_route="isolated",
        transaction_id="checkpoint-v5",
        payload=CompactionCheckpointPayload(
            progress="A and C remain complete",
            current_work="run final tests",
            files_or_code=("A.py", "C.py"),
            errors_or_evidence=("final tests pending",),
            next_step="run final tests",
        ),
    )
    store.append_checkpoint(session.id, envelope_v5.to_record(), atomic_surface_replace=True)
    revision = store.read_surface_revision(session.id)
    store.write_compacted(
        session.id,
        [
            {"role": "user", "content": "checkpoint v5: A.py, B.py factual history, C.py remain available; do not modify B", "_checkpoint_id": "checkpoint-v5"},
            {"role": "assistant", "content": "continue from v5", "_checkpoint_id": "checkpoint-v5"},
        ],
        contract_digest=contract_v4.digest,
        base_surface_revision=revision,
        surface_revision=revision + 1,
        checkpoint_id="checkpoint-v5",
    )

    replayed = SessionStore(tmp_path / "sessions").read_messages(session.id)
    rendered = "\n".join(str(message.get("content", "")) for message in replayed)
    assert "A.py" in rendered and "C.py" in rendered
    assert "do not modify B" in rendered
    assert "must modify B" not in rendered
    assert rendered.count("modify B") == 1
