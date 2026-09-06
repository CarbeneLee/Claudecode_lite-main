from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.compact.protocol import (
    accept_semantic_payload,
    build_isolated_request,
    build_same_route_request,
)
from kama_claude.core.compact.state import (
    CompactionPolicy,
    CompactionState,
    classify_compaction_state,
    hard_admission_required,
    proactive_compaction_may_fail_soft,
)
from kama_claude.core.context import ExecutionContext
from kama_claude.core.llm.gateway import ProviderRequestGateway
from kama_claude.core.llm.types import (
    DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
    LlmResponse,
    ProviderContinuationState,
)
from kama_claude.core.llm.usage import (
    NormalizedUsage,
    ProviderUsageNormalizer,
    RawUsageEnvelope,
    ReplayAwareTokenMeter,
    estimate_input_tokens,
    normalize_output_reserve,
)
from kama_claude.core.session.surface import (
    CompactionCandidate,
    SurfaceState,
)
from kama_claude.core.subagent.protocol import SubagentOutcome, authorize_child_evidence
from kama_claude.core.task_contract import (
    DirectiveCoverageRecord,
    PendingDirectiveSet,
    TaskContractRecord,
    TaskContractState,
    project_stale_checkpoint,
)
from kama_claude.core.tools.evidence import EvidenceSpool, ToolEvidenceBudget, bound_tool_output


class _PrefixProvider:
    """仅暴露 gateway 构造 stable prefix 所需的 route capability。"""

    model = "deepseek-v4-flash"
    protocol = "anthropic"
    usage_schema = "deepseek_anthropic_messages_v1"
    route_identity = "deepseek-v4-flash:anthropic"
    context_window = 1_000_000
    max_output_tokens = 256
    output_budget_semantics = "inclusive_total"


# 功能：验证 DeepSeek thinking blocks 作为独立 continuation state 原样保存并参与身份摘要
# 设计：使用带签名的原始 block，分别检查 carrier、verbatim 内容和 digest 稳定性，避免把 reasoning 当普通文本重建
def test_deepseek_continuation_state_is_verbatim_and_hashed() -> None:
    blocks = ({"type": "thinking", "thinking": "R1", "signature": "sig-1"},)
    state = ProviderContinuationState(
        blocks=blocks,
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
        route_identity="deepseek-v4-flash:anthropic",
    )

    assert state.as_blocks() == list(blocks)
    assert state.policy.required_for_followup is True
    assert state.identity_digest() == state.identity_digest()
    response = LlmResponse(
        stop_reason="tool_use",
        text="visible",
        continuation_state=state,
    )
    assert response.thinking_blocks == list(blocks)
    legacy = ProviderContinuationState.from_dict(
        {
            "thinking_blocks": list(blocks),
            "policy": {"carrier": "none"},
            "route_identity": "deepseek-v4-flash:anthropic",
        }
    )
    assert legacy.policy == DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY


# 功能：验证按实际 wire schema 归一化 DeepSeek Anthropic usage，不因 vendor 名称猜字段
# 设计：fixture 同时给出 Anthropic input/cache 字段和 DeepSeek hit/miss 字段，断言 adapter 标注的 schema 决定解析结果
def test_usage_normalizer_dispatches_by_wire_schema() -> None:
    normalizer = ProviderUsageNormalizer()
    result = normalizer.normalize(
        RawUsageEnvelope(
            usage_schema="deepseek_anthropic_messages_v1",
            payload={
                "input_tokens": 100,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 5,
                "output_tokens": 7,
            },
            route="deepseek-v4-flash:anthropic",
        )
    )
    assert result.total_input_tokens == 125
    assert result.cache_hit_tokens == 20
    assert result.cache_miss_tokens == 105
    assert result.output_tokens == 7


# 功能：验证未知 wire usage schema fail closed，不依据 vendor 名称猜测 context occupancy
# 设计：注入未注册 schema，断言 normalizer 立即抛错而不是静默返回估算值
def test_usage_normalizer_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported usage schema"):
        ProviderUsageNormalizer().normalize(
            RawUsageEnvelope(
                usage_schema="unknown_wire_v9",
                payload={"prompt_tokens": 100},
                route="deepseek-v4-flash",
            )
        )


# 功能：验证 DeepSeek Chat usage 的 prompt_tokens 已包含 hit+miss，不会被二次相加
# 设计：提供官方三字段 fixture，断言 total 取 prompt_tokens 且 miss 只作 telemetry
def test_deepseek_chat_usage_does_not_double_count_cache_tokens() -> None:
    result = ProviderUsageNormalizer().normalize(
        RawUsageEnvelope(
            usage_schema="deepseek_chat_completions_v1",
            payload={
                "prompt_tokens": 125,
                "prompt_cache_hit_tokens": 20,
                "prompt_cache_miss_tokens": 105,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
            route="deepseek-v4-flash:chat",
        )
    )
    assert result.total_input_tokens == 125
    assert result.cache_hit_tokens == 20
    assert result.cache_miss_tokens == 105
    assert result.reasoning_output_tokens == 3


# 功能：验证 inclusive max output 语义不会把 reasoning tokens 再次加入 O_reserved
# 设计：用最小 request/context 对象测试 provider-normalized 输出预算的单一计数来源
def test_output_reserve_inclusive_max_tokens_is_not_double_counted() -> None:
    request = {"max_tokens": 4096}
    context = {"output_budget_semantics": "inclusive_total"}
    assert normalize_output_reserve(request, context) == 4096
    assert normalize_output_reserve(
        {"visible_output_tokens": 300, "reasoning_budget": 100},
        {"output_budget_semantics": "visible_plus_reasoning"},
    ) == 400


# 功能：验证保守 wire-byte 估算不会把 CJK、代码、JSON 和 continuation 按 chars/4 低估
# 设计：构造高 UTF-8 密度消息与工具 schema，比较旧启发式和生产估算，锁定 hard admission 的保守方向
def test_wire_byte_estimate_does_not_underestimate_multibyte_surface() -> None:
    messages = [
        {
            "role": "user",
            "content": "界" * 500 + "\n{" + '"key":"value"' * 80 + "}",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "R" * 500, "signature": "sig"},
                {"type": "text", "text": "def f(x):\n    return x * 2\n" * 40},
            ],
        },
    ]
    tools = [
        {
            "name": "write_file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]
    wire_estimate = estimate_input_tokens(messages, tools, "system")
    old_chars_div_four = len(
        str({"messages": messages, "tools": tools, "system": "system"})
    ) // 4
    assert wire_estimate > old_chars_div_four


# 功能：验证 ReplayAwareTokenMeter 在兼容 envelope 下可应用 committed signed delta，并在 route 变更时失效
# 设计：先写 provider baseline，再分别传入 surface delta 和 route mismatch，覆盖 derived 与 fail-closed 分支
def test_replay_aware_meter_uses_signed_delta_and_invalidates_incompatible_route() -> None:
    meter = ReplayAwareTokenMeter()
    meter.record(
        NormalizedUsage(total_input_tokens=100, confidence="provider"),
        surface_revision=1,
        route_epoch="route-1",
        prefix_epoch="prefix-1",
        envelope_id="env-1",
        continuation_policy_identity="policy-1",
    )

    assert meter.estimate(
        surface_revision=2,
        route_epoch="route-1",
        prefix_epoch="prefix-1",
        envelope_id="env-2",
        continuation_policy_identity="policy-1",
        signed_delta_tokens=50,
    ) == (150, "derived")
    assert meter.estimate(
        surface_revision=2,
        route_epoch="route-2",
        prefix_epoch="prefix-1",
        envelope_id="env-2",
        continuation_policy_identity="policy-1",
        signed_delta_tokens=50,
    ) is None


# 功能：验证 gateway 的 route/prefix epoch 变化会清除旧 usage baseline
# 设计：先完成一次带 usage 的 request，再分别切换 route 与 prompt prefix，锁定完整 reprice 边界
@pytest.mark.asyncio
async def test_gateway_route_and_prefix_changes_invalidate_meter() -> None:
    from kama_claude.core.llm.gateway import ProviderRequestGateway

    class _UsageProvider(_PrefixProvider):
        async def chat(
            self,
            _messages: list[dict[str, object]],
            _tool_schemas: list[dict[str, object]],
            _bus: object,
            _run_id: str,
            **_: object,
        ) -> LlmResponse:
            return LlmResponse(
                stop_reason="end_turn",
                text="ok",
                usage=NormalizedUsage(total_input_tokens=12).to_usage_stats(1_000_000),
            )

    gateway = ProviderRequestGateway(_UsageProvider(), safety_margin=0)
    await gateway.chat(
        messages=[{"role": "user", "content": "goal"}],
        tool_schemas=[],
        bus=object(),  # type: ignore[arg-type]
        run_id="meter-route",
    )
    assert gateway.meter_baseline is not None
    old_route = gateway.route_epoch
    gateway.change_route(
        "openai-chat:test",
        protocol="openai",
        model="gpt-test",
        context_window=100_000,
    )
    assert gateway.route_epoch != old_route
    assert gateway.meter_baseline is None

    old_prefix = gateway.prefix_epoch
    gateway.invalidate_prefix("repository-rules-v2")
    assert gateway.prefix_epoch != old_prefix
    assert gateway.meter_baseline is None


# 功能：验证语义 no-op 不增加 TaskContract 版本，但可以独立提交 message coverage
# 设计：以相同 canonical fields 构造两次记录，coverage 使用不同 message_id，锁定 provenance 与 coverage 分离
def test_task_contract_noop_keeps_version_and_coverage_is_independent() -> None:
    first = TaskContractRecord.create(
        version=1,
        goal="modify A",
        requirements=("run tests",),
        source_message_ids=("m1",),
    )
    second = first.with_semantic_state(source_message_ids=("m2",))
    coverage = DirectiveCoverageRecord.covered(
        message_id="m2",
        semantic_contract_digest=first.digest,
        classification="task_shaping",
        exact_text="continue",
    )

    assert second.version == first.version
    assert second.digest == first.digest
    assert second.source_message_ids == first.source_message_ids
    assert coverage.coverage_status == "covered"
    assert "m2" not in second.source_message_ids


# 功能：验证无法安全 additive merge 的 correction 保持 unresolved，而不是同时激活旧新动作
# 设计：输入明确撤销 B 并改为 C 的句子，断言 reducer 返回 None 以触发原文 pending overlay
def test_ambiguous_correction_remains_unresolved() -> None:
    from kama_claude.core.task_contract import apply_directive_to_contract

    previous = TaskContractRecord.create(
        version=1,
        goal="deliver A and B",
        requirements=("modify B",),
    )
    assert (
        apply_directive_to_contract(
            previous,
            message_id="m2",
            raw_text="Correction: do not modify B; modify C instead.",
            classification="task_shaping",
        )
        is None
    )
    state = TaskContractState(previous)
    state.evaluate_message(
        message_id="m2",
        raw_text="Correction: do not modify B; modify C instead.",
        updater=lambda _previous, _text: None,
        classification="task_shaping",
    )
    assert state.coverage["m2"].coverage_status == "unresolved"
    assert state.pending.protected_message_ids() == frozenset({"m2"})


# 功能：验证连续 updater 失败时 pending directives 按原始顺序全部保留
# 设计：先加入 prohibition 再加入 requirement，断言新 directive 不覆盖旧 directive 且渲染顺序稳定
def test_pending_directive_set_preserves_ordered_unresolved_overlays() -> None:
    pending = PendingDirectiveSet()
    pending.add("m1", "do not modify foo.py", "prohibition")
    pending.add("m2", "tests must use pytest", "requirement")

    assert [item.message_id for item in pending.unresolved()] == ["m1", "m2"]
    assert [item.raw_text for item in pending.render()] == [
        "do not modify foo.py",
        "tests must use pytest",
    ]


# 功能：验证 updater 拒绝非 steering directive 时 coverage 保持 unresolved 并保护旧 action
# 设计：注入返回 None 的有限 updater，区分明确 no-op 与无法 reconciliation 的 task-shaping directive
def test_declined_task_directive_remains_pending_until_reconciled() -> None:
    contract = TaskContractRecord.create(
        version=1,
        goal="modify A and B",
        requirements=("modify B",),
    )
    state = TaskContractState(contract)

    result = state.evaluate_message(
        message_id="m2",
        raw_text="do not modify B",
        updater=lambda _previous, _text: None,
        classification="prohibition",
    )

    assert result == contract
    assert state.coverage["m2"].coverage_status == "unresolved"
    assert [item.message_id for item in state.pending.unresolved()] == ["m2"]
    assert state.is_compactable("m2") is False


# 功能：验证 contract digest 变化只把 checkpoint 降级为 factual background，不丢失执行事实
# 设计：输入含 next_step/current_work 与 files/errors，断言 action fields 被抑制而事实字段保留
def test_stale_checkpoint_projection_suppresses_action_fields() -> None:
    projection = project_stale_checkpoint(
        {
            "progress": "implemented parser",
            "current_work": "modify foo.py",
            "next_step": "edit foo.py",
            "pending": ["run tests"],
            "files_or_code": ["foo.py"],
            "errors_or_evidence": ["test x passed"],
        },
        reason="contract digest changed",
    )

    assert projection.status == "HISTORICAL_BACKGROUND"
    assert projection.facts["progress"] == "implemented parser"
    assert projection.facts["files_or_code"] == ["foo.py"]
    assert "next_step" not in projection.facts
    assert "current_work" not in projection.facts
    assert "pending" not in projection.facts


# 功能：验证自由文本 summary 在 checkpoint 中按语义 section 分栏，stale projection 不泄漏旧 TODO
# 设计：使用生产 compaction prompt 的六段 markdown 形状，先解析再降级，避免把整段 summary 当作 progress 事实
def test_markdown_checkpoint_payload_drops_old_todo_on_stale_projection() -> None:
    payload = accept_semantic_payload(
        "## 1. Original Goal\nmodify A\n\n"
        "## 2. Completed Steps\n- edited parser.py\n\n"
        "## 4. Current File State\n- parser.py: updated\n\n"
        "## 5. Remaining TODOs\n1. edit foo.py\n"
    )
    projection = project_stale_checkpoint(payload.to_dict())
    assert "parser.py" in str(projection.facts)
    assert "foo.py" not in str(projection.facts)


# 功能：验证 SurfaceSnapshot 与事后选择的 CompactionCandidate 分离，并由 SurfaceState 执行 CAS
# 设计：先捕获快照再选择 span，随后修改 surface revision，断言旧 candidate 被冲突拒绝
def test_surface_candidate_cas_rejects_surface_mutation() -> None:
    state = SurfaceState()
    snapshot = state.snapshot()
    candidate = CompactionCandidate.from_snapshot(
        snapshot,
        selected_unit_ids=("u1",),
        selected_span_digest="span-a",
        route_epoch=state.route_epoch,
        prefix_epoch=state.prefix_epoch,
        request_envelope_id="env-a",
    )
    state.note_surface_mutation("new-user")

    assert state.commit_candidate(candidate) is False


# 功能：验证 route epoch 变化不增加 surface revision 但会拒绝旧 compaction candidate
# 设计：单独改变 route，检查两个身份的职责边界，防止把模型切换误当作会话内容 mutation
def test_route_change_is_separate_from_surface_revision() -> None:
    state = SurfaceState()
    snapshot = state.snapshot()
    revision = snapshot.surface_revision
    state.change_route("deepseek-v4-flash:anthropic")
    assert state.surface_revision == revision
    assert state.route_epoch != "route-0"
    candidate = CompactionCandidate.from_snapshot(
        snapshot,
        selected_unit_ids=(),
        selected_span_digest="span-a",
        route_epoch="route-0",
        prefix_epoch=state.prefix_epoch,
        request_envelope_id="env-a",
    )
    assert state.commit_candidate(candidate) is False


# 功能：验证 contract/pending surface mutation 会拒绝已经在异步 summarizer 中的旧 candidate
# 设计：candidate 选择后只改变语义 contract 字段，保持 active head 不变，锁定 CAS 对 contract identity 的检查
def test_contract_change_rejects_inflight_compaction_candidate() -> None:
    state = SurfaceState()
    snapshot = state.snapshot()
    candidate = CompactionCandidate.from_snapshot(
        snapshot,
        selected_unit_ids=("u1",),
        selected_span_digest="span-a",
        route_epoch=state.route_epoch,
        prefix_epoch=state.prefix_epoch,
        request_envelope_id="env-a",
    )
    state.note_surface_mutation(contract_version=1, contract_digest="contract-v2")

    assert state.surface_revision != candidate.base_surface_revision
    assert state.commit_candidate(candidate) is False


# 功能：验证 TaskContract、checkpoint 和新 raw message 变化不改变 Layer A/B stable prefix
# 设计：反复准备不同 Layer C 内容，断言 immutable/semi-stable/provider hashes 与 prefix epoch 保持不变
def test_dynamic_surface_changes_preserve_stable_prefix_identity() -> None:
    gateway = ProviderRequestGateway(_PrefixProvider())
    immutable = "immutable harness instructions"
    semi = "repository rules v1"
    tools = [
        {"name": "write_file", "input_schema": {"type": "object"}},
        {"name": "read_file", "input_schema": {"type": "object"}},
    ]
    first = gateway.prepare_request(
        messages=[{"role": "user", "content": "goal v1"}],
        tool_schemas=list(reversed(tools)),
        system=f"{immutable}\n\n{semi}\n\ncontract v1",
        immutable_system=immutable,
        semi_stable_context=semi,
        surface_revision=1,
    )
    second = gateway.prepare_request(
        messages=[
            {"role": "user", "content": "goal v2"},
            {"role": "assistant", "content": "checkpoint facts"},
        ],
        tool_schemas=tools,
        system=f"{immutable}\n\n{semi}\n\ncontract v2\ncheckpoint v2",
        immutable_system=immutable,
        semi_stable_context=semi,
        surface_revision=2,
    )

    assert first.stable_prefix.immutable_hash == second.stable_prefix.immutable_hash
    assert first.stable_prefix.semi_stable_hash == second.stable_prefix.semi_stable_hash
    assert first.stable_prefix.provider_serialized_hash == second.stable_prefix.provider_serialized_hash
    assert first.prefix_epoch == second.prefix_epoch
    assert first.tool_schemas == second.tool_schemas


# 功能：验证 repository rule/tool schema 变化会使 PrefixEpoch 失效且 canonical tool order 保持稳定
# 设计：先交换 schema 注册顺序，再改变 schema 内容，区分排序稳定性与真正的缓存前缀变化
def test_repository_or_tool_schema_change_invalidates_prefix_epoch() -> None:
    gateway = ProviderRequestGateway(_PrefixProvider())
    base_tools = [
        {"name": "b", "input_schema": {"type": "object"}},
        {"name": "a", "input_schema": {"type": "object"}},
    ]
    first = gateway.prepare_request(
        messages=[],
        tool_schemas=base_tools,
        system="immutable\n\nrepo-v1",
        immutable_system="immutable",
        semi_stable_context="repo-v1",
    )
    reordered = gateway.prepare_request(
        messages=[],
        tool_schemas=list(reversed(base_tools)),
        system="immutable\n\nrepo-v1",
        immutable_system="immutable",
        semi_stable_context="repo-v1",
    )
    changed = gateway.prepare_request(
        messages=[],
        tool_schemas=[
            {"name": "a", "description": "changed", "input_schema": {"type": "object"}},
            base_tools[0],
        ],
        system="immutable\n\nrepo-v2",
        immutable_system="immutable",
        semi_stable_context="repo-v2",
    )

    assert first.tool_schemas == reordered.tool_schemas
    assert first.prefix_epoch == reordered.prefix_epoch
    assert changed.prefix_epoch != first.prefix_epoch
    assert changed.stable_prefix.provider_serialized_hash != first.stable_prefix.provider_serialized_hash


# 功能：验证 isolated summarizer 明确放弃原始 prompt cache reuse 并移除 provider reasoning/tools
# 设计：传入带 thinking/tool blocks 的历史，断言 isolated request 只有 bounded history 和 compaction instruction
def test_isolated_summarizer_declares_cache_miss_and_strips_private_state() -> None:
    request = build_isolated_request(
        selected_messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}},
                    {"type": "text", "text": "visible"},
                ],
            }
        ]
    )

    assert request.cache_reuse is False
    assert request.tool_schemas == ()
    assert request.messages[0]["content"].startswith("COMPACTION_INSTRUCTION")
    assert '"thinking"' not in str(request.messages)


# 功能：验证 Layer C 不复制 pending directive，最新 user unit 在 provider request 中只出现一次
# 设计：让同一 message_id 同时存在 pending overlay 和 RecentSurfaceUnits，断言 system 不含原文且消息内容唯一
def test_pending_directive_metadata_does_not_duplicate_recent_user_unit() -> None:
    raw_text = "do not modify foo.py"
    context = ExecutionContext(
        run_id="duplicate-layout",
        goal=raw_text,
        max_steps=1,
        prefill_messages=[
            {"role": "user", "content": raw_text, "_message_id": "m1"}
        ],
    )
    context.pending_directives.add("m1", raw_text, "prohibition")

    system = context.system_prompt("base system")
    messages = context.provider_messages()

    assert raw_text not in system
    assert sum(message.get("content") == raw_text for message in messages) == 1


# 功能：验证 producer/durable/model 三层 evidence budget 共同限制输出并保留 head/tail truth metadata
# 设计：使用比 durable cap 更大的 bytes 和更小的 model cap，同时覆盖 EvidenceSpool 的分块写入路径
def test_evidence_budgets_preserve_truncation_truth_and_streaming_cap(tmp_path: Path) -> None:
    budget = ToolEvidenceBudget(
        producer_max_bytes=12,
        durable_max_bytes=8,
        model_max_chars=6,
        preview_head_chars=2,
        preview_tail_chars=2,
    )
    receipt = bound_tool_output(b"0123456789abcdef", budget=budget, evidence_ref="artifact-1")
    assert receipt.raw_truncated is True
    assert receipt.captured_size == 8
    assert receipt.original_size == 16
    assert receipt.model_truncated is True
    assert len(receipt.to_model_text()) > len(receipt.preview)

    spool_path = tmp_path / "evidence.bin"
    with EvidenceSpool(spool_path, max_bytes=5) as spool:
        spool.write(b"abc")
        spool.write(b"defgh")
    assert spool.sizes == (8, 5)
    assert spool_path.read_bytes() == b"abcde"


# 功能：验证 SubagentOutcome 在成功、overflow、极小 receipt 和 child evidence 授权边界都保持 bounded JSON
# 设计：直接构造超大 child preview，分别断言合法 JSON、错误码保留与跨 child/foreign reference 拒绝
def test_subagent_outcome_and_child_evidence_authorization() -> None:
    outcome = SubagentOutcome(
        status="failed",
        error_code="CONTEXT_WINDOW_EXCEEDED",
        child_run_id="child-1",
        provider="deepseek",
        model="deepseek-v4-flash",
        preview="trace " * 10_000,
        evidence_ref="child:child-1:artifact-1",
    )
    receipt = outcome.to_parent_receipt(max_chars=240)
    assert len(receipt) <= 240
    assert receipt.startswith("{") and receipt.endswith("}")
    assert "CONTEXT_WINDOW_EXCEEDED" in receipt or receipt == "{}"
    assert authorize_child_evidence(
        "child:child-1:artifact-1",
        parent_run_id="parent-1",
        child_run_id="child-1",
    ) == "parent:parent-1:child:child-1:artifact-1"
    with pytest.raises(ValueError):
        authorize_child_evidence(
            "child:child-2:artifact-1",
            parent_run_id="parent-1",
            child_run_id="child-1",
        )
    with pytest.raises(ValueError):
        authorize_child_evidence(
            "session:foreign:artifact-1",
            parent_run_id="parent-1",
            child_run_id="child-1",
        )

    class _ChildContext:
        status = "success"
        result = "completed"
        reason = None
        step = 2
        provider_name = "deepseek"
        provider_model = "deepseek-v4-flash"
        provider_attempt_count = 1

    success = SubagentOutcome.from_context(_ChildContext(), child_run_id="child-2")
    assert success.error_code is None
    assert success.status == "success"


# 功能：验证 same-route summarizer 在原始 conversation prefix 后追加 compaction instruction
# 设计：保留 tool schema 与 thinking block 的 exact bytes，断言 instruction 只出现在尾部且不替换原 system
def test_same_route_summarizer_replays_prefix_and_appends_instruction() -> None:
    state = ProviderContinuationState.from_thinking_blocks(
        [{"type": "thinking", "thinking": "R1", "signature": "sig"}],
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
        route_identity="deepseek-v4-flash:anthropic",
    )
    request = build_same_route_request(
        prefix_messages=[
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ],
        tool_schemas=[{"name": "read_file"}],
        system="immutable system",
        continuation_states=[state],
    )
    assert request.cache_reuse is True
    assert request.tool_schemas == ({"name": "read_file"},)
    assert request.messages[-1]["content"].startswith("COMPACTION_INSTRUCTION")
    assert request.messages[0]["content"] == "goal"


# 功能：验证 DaemonRootLock 能在同一 sessions root 内阻止第二个持有者
# 设计：同一进程打开两个 lock，第二次必须失败，避免多端口 daemon 并发写同一 session 根目录
def test_daemon_root_lock_is_non_reentrant(tmp_path: Path) -> None:
    from kama_claude.core.session.lock import DaemonRootLock

    first = DaemonRootLock(tmp_path)
    first.acquire()
    second = DaemonRootLock(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release()
        second.release()


# 功能：验证 async mutation lane 可串行化同一 session 的提交
# 设计：两个 coroutine 共享 SurfaceState 的 lane，第二个只能在第一个释放后进入
async def test_surface_mutation_lane_serializes() -> None:
    state = SurfaceState()
    order: list[str] = []

    # 功能：验证单个 coroutine 在 mutation lane 中按进入和退出顺序记录
    # 设计：让 worker 主动让出控制权，确保并发 gather 仍只能串行进入临界区
    async def worker(name: str) -> None:
        async with state.mutation_lane:
            order.append(f"{name}:enter")
            await asyncio.sleep(0)
            order.append(f"{name}:exit")

    await asyncio.gather(worker("a"), worker("b"))
    assert order in (["a:enter", "a:exit", "b:enter", "b:exit"], ["b:enter", "b:exit", "a:enter", "a:exit"])


# 功能：验证 active inference unit 的 provider continuation 在下一次 request 中原样 replay
# 设计：先注入可见文本但缺失 thinking block，再由 provider_messages 补回原始签名 block
def test_execution_context_replays_active_continuation_state() -> None:
    state = ProviderContinuationState.from_thinking_blocks(
        [{"type": "thinking", "thinking": "R1", "signature": "sig"}],
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
    )
    context = ExecutionContext(run_id="r1", goal="goal", max_steps=3)
    context.add_assistant_message(
        [{"type": "text", "text": "visible"}],
        continuation_state=state,
    )

    assistant = context.provider_messages()[1]
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "R1",
        "signature": "sig",
    }
    context.shadow_inference_units({context.inference_units[0].unit_id})
    assert all(
        block.get("type") != "thinking"
        for block in context.provider_messages()[1]["content"]
    )


# 功能：验证 step_commit 持久化 provider continuation metadata 且 TUI projection 默认隐藏 reasoning
# 设计：写入含 thinking block 的 assistant message，再分别读取 provider history 与 UI history，锁定双投影边界
def test_store_persists_continuation_but_hides_it_from_history_projection(tmp_path: Path) -> None:
    from kama_claude.core.session.store import SessionStore

    store = SessionStore(tmp_path)
    store.append_messages(
        "sess-1",
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "R1", "signature": "s"},
                    {"type": "text", "text": "visible"},
                ],
            }
        ],
        run_id="r1",
    )
    assert store.read_messages("sess-1")[0]["content"][0]["type"] == "thinking"
    assert all(
        block.get("type") != "thinking"
        for block in store.read_history_projection("sess-1")[0]["content"]
    )


# 功能：验证 daemon restart 从 durable continuation metadata 恢复 provider replay contract
# 设计：故意把 visible text 与 thinking state 分开写入，模拟 provider adapter 只持久化 canonical state 的路径
def test_restart_rehydrates_durable_continuation_metadata(tmp_path: Path) -> None:
    from kama_claude.core.session.store import SessionStore

    store = SessionStore(tmp_path)
    state = ProviderContinuationState.from_thinking_blocks(
        [{"type": "thinking", "thinking": "R1", "signature": "sig"}],
        policy=DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
        route_identity="deepseek-v4-flash:anthropic",
    )
    store.append_continuation_message(
        "sess-1",
        role="assistant",
        content="visible",
        continuation_state=state.to_dict(),
        unit_id="unit-1",
    )

    context = ExecutionContext(
        run_id="r1",
        goal="goal",
        max_steps=3,
        prefill_messages=store.read_messages_with_metadata("sess-1"),
    )
    assert context.inference_units[0].unit_id == "unit-1"
    assert context.provider_messages()[0]["content"][0] == {
        "type": "thinking",
        "thinking": "R1",
        "signature": "sig",
    }


# 功能：验证 contract digest 变化后 shadowed checkpoint 仅以事实背景回放而不复活旧 next_step
# 设计：先写 raw history 与 v3 replacement，再追加 v4 contract/checkpoint，读取 provider continuation 断言事实保留且旧动作消失
def test_stale_checkpoint_keeps_facts_without_resurrecting_raw_history(tmp_path: Path) -> None:
    from kama_claude.core.session.store import SessionStore

    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "500K old history")
    store.write_compacted(
        "sess-1",
        [{"role": "user", "content": "old next_step: edit foo.py"}],
    )
    store.append_task_contract(
        "sess-1",
        TaskContractRecord.create(version=4, goal="do not modify foo.py"),
    )
    store.append_checkpoint(
        "sess-1",
        {
            "generation": 3,
            "contract_digest": "contract-v3",
            "payload": {
                "progress": "implemented parser",
                "next_step": "edit foo.py",
                "files_or_code": ["parser.py"],
                "errors_or_evidence": ["tests passed"],
            },
        },
    )

    rendered = store.read_messages("sess-1")
    text = "\n".join(str(message["content"]) for message in rendered)
    assert "500K old history" not in text
    assert "old next_step: edit foo.py" not in text
    assert '"next_step"' not in text
    assert "parser.py" in text
    assert "tests passed" in text


# 功能：验证压缩后处于 safe-above-target 时请求仍安全且不被判定为 fatal convergence failure
# 设计：使用 65% occupancy、60% target、80% trigger 的明确比例，区分 hard safety 与优化目标
def test_safe_above_target_is_not_hard_failure() -> None:
    assert classify_compaction_state(65, 100) is CompactionState.SAFE_ABOVE_TARGET


# 功能：验证 hard/soft/target 四态边界分别映射到正确的 fail-open/fail-closed 语义
# 设计：使用同一容量的四个精确 occupancy，断言 hard 只有超限、soft 允许 proactive fail-soft
def test_compaction_state_machine_boundaries() -> None:
    policy = CompactionPolicy(soft_trigger_ratio=0.80, target_ratio=0.60)
    states = [
        classify_compaction_state(101, 100, policy=policy),
        classify_compaction_state(80, 100, policy=policy),
        classify_compaction_state(61, 100, policy=policy),
        classify_compaction_state(60, 100, policy=policy),
    ]

    assert states == [
        CompactionState.HARD_UNSAFE,
        CompactionState.SOFT_PRESSURED,
        CompactionState.SAFE_ABOVE_TARGET,
        CompactionState.AT_TARGET,
    ]
    assert hard_admission_required(states[0]) is True
    assert hard_admission_required(states[2]) is False
    assert proactive_compaction_may_fail_soft(states[0]) is False
    assert proactive_compaction_may_fail_soft(states[1]) is True
