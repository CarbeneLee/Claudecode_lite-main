from __future__ import annotations

import json
from pathlib import Path

import pytest

from kama_claude.core.bus.events import PlannerDecisionReadyEvent, RunFinishedEvent
from kama_claude.core.events.journal import (
    EventJournalCoordinator,
    JournalCorruptionError,
    _validate_event_payload,
)
from kama_claude.core.plan_view import (
    MAX_LIST_ITEM_CHARS,
    PLAN_VIEW_MAX_BYTES,
    LegacyPlanViewV0,
    PlanReadyCommitReducer,
    decode_plan_view_record,
)
from kama_claude.core.planning import (
    ExactPlannerDecisionV2,
    IntendedChange,
    Requirement,
    VerificationStrategy,
    build_plan_view,
    render_planner_decision_execution_summary,
)


# 构造完整的 exact domain decision，供 PlanView projection tests 复用
def _decision(**updates: object) -> ExactPlannerDecisionV2:
    values: dict[str, object] = {
        "decision_id": "decision-1",
        "version": 1,
        "goal": "Change target behavior",
        "requirements": (
            Requirement(
                requirement_id="R1",
                statement="Change target behavior.",
                required=True,
            ),
        ),
        "architecture_slice_id": "slice-1",
        "architecture_slice_version": 1,
        "architecture_slice_content_digest": "slice-digest",
        "snapshot_digest": "snapshot-digest",
        "architecture_mode": "preserve",
        "selected_approach": "Edit the existing module.",
        "existing_patterns_reused": ("existing pattern",),
        "intended_changes": (
            IntendedChange(
                change_id="C1",
                description="Update the module.",
                requirement_ids=("R1",),
                target_paths=("src/target.py",),
                evidence_refs=("read-target",),
            ),
        ),
        "files_to_modify": ("src/target.py",),
        "files_to_create": (),
        "allowed_capabilities": ("read_file", "write_file"),
        "dependency_changes": (),
        "protocol_or_schema_changes": (),
        "verification_plan": (
            VerificationStrategy(
                requirement_ids=("R1",),
                strategy="Run focused tests.",
            ),
        ),
        "non_goals": ("No cleanup.",),
        "assumptions": ("The existing test suite remains available.",),
        "unresolved_questions": (),
        "requires_user_approval": True,
        "content_digest": "decision-digest",
    }
    values.update(updates)
    return ExactPlannerDecisionV2.model_validate(values)


# 构造新的 PlanReady event，集中验证 projection identity 和 commit reducer
def _plan_ready(*, run_id: str = "run-1", approach: str = "Edit the existing module.") -> PlannerDecisionReadyEvent:
    plan = build_plan_view(
        _decision(selected_approach=approach),
        top_level_run_id=run_id,
    )
    return PlannerDecisionReadyEvent(
        event_id=f"plan-ready:{plan.projection_key}",
        run_id=run_id,
        planner_run_id="planner-1",
        session_id="sess-1",
        plan=plan,
        ts="2026-08-15T00:00:00Z",
    )


# 功能：验证 PlanView 使用 decision/projection 双 identity 且 timestamp 不参与 digest
# 设计：同一 exact decision 在两个 run 中生成 projection，比较 identity 和稳定 digest
def test_plan_view_identity_and_digest_are_deterministic() -> None:
    first = build_plan_view(_decision(), top_level_run_id="run-1")
    repeated = build_plan_view(_decision(), top_level_run_id="run-1")
    second_run = build_plan_view(_decision(), top_level_run_id="run-2")

    assert first.decision_key == "decision-1:v1"
    assert first.projection_key == "pv1:run-1:decision-1:v1"
    assert first.projection_digest == repeated.projection_digest
    assert first.projection_key != second_run.projection_key
    assert first.projection_digest == second_run.projection_digest
    assert "ts" not in first.model_dump(mode="json")


# 功能：验证 bounded PlanView 遵守总字节和列表预算并记录 omission
# 设计：构造超过所有 section 上限的合法 domain decision，断言 projection 在边界内而非静默膨胀
def test_plan_view_applies_fixed_bounds_and_omission_counts() -> None:
    requirements = tuple(
        Requirement(
            requirement_id=f"R{index}",
            statement="x" * 4096,
            required=True,
        )
        for index in range(100)
    )
    decision = _decision(requirements=requirements)
    plan = build_plan_view(decision, top_level_run_id="run-1")
    encoded = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False).encode()

    assert len(encoded) <= PLAN_VIEW_MAX_BYTES
    assert len(plan.requirements) <= 64
    assert plan.omitted_counts["requirements"] > 0
    assert "requirements" in plan.section_limits
    assert plan.execution_available is False
    assert plan.requires_user_approval is True


# 功能：验证 identifier/path 超限时不会生成虚假的 prefix 路径
# 设计：使用超过 item budget 的真实路径，要求整项省略并记录 omitted count
def test_plan_view_does_not_prefix_truncate_identifier_paths() -> None:
    long_path = "src/" + ("authorization_policy_" * MAX_LIST_ITEM_CHARS) + ".py"
    decision = _decision(files_to_modify=(long_path,))
    plan = build_plan_view(decision, top_level_run_id="run-1")

    assert long_path not in plan.files_to_modify
    assert not any(long_path[:32] in path for path in plan.files_to_modify)
    assert plan.omitted_counts["files_to_modify"] == 1


# 功能：验证 legacy PlanView 可以在 Event adapter 之前被显式解码
# 设计：保留旧 plan_key/content_digest/ts shape，未知 schema marker 则必须 fail closed
def test_legacy_plan_view_decode_and_unknown_schema() -> None:
    legacy = {
        "schema_version": 1,
        "plan_key": "decision-1:v1",
        "goal": "Legacy goal",
        "selected_approach": "Legacy approach",
        "content_digest": "legacy-content",
        "ts": "2026-08-15T00:00:00Z",
    }
    record = decode_plan_view_record(legacy)
    assert isinstance(record, LegacyPlanViewV0)
    assert record.plan_key == "decision-1:v1"

    with pytest.raises(ValueError, match="schema"):
        decode_plan_view_record({"schema_version": 99, "goal": "unknown"})


# 功能：验证 Event adapter 能重放旧 PlanReady outer event 而不是报告 journal corruption
# 设计：直接走 journal 的 strict event validation seam，覆盖 replay 实际使用的 decoder
def test_legacy_plan_ready_event_replays_through_event_adapter() -> None:
    raw = {
        "type": "planner.decision_ready",
        "event_id": "legacy-plan-ready",
        "run_id": "run-1",
        "planner_run_id": "planner-1",
        "session_id": "sess-1",
        "plan": {
            "schema_version": 1,
            "plan_key": "decision-1:v1",
            "goal": "Legacy goal",
            "selected_approach": "Legacy approach",
            "content_digest": "legacy-content",
            "ts": "2026-08-15T00:00:00Z",
        },
        "plan_key": "decision-1:v1",
        "decision_id": "decision-1",
        "decision_version": 1,
        "ts": "2026-08-15T00:00:00Z",
        "snapshot_digest": "snapshot",
        "content_digest": "legacy-content",
    }

    event = _validate_event_payload(raw, role="v2")

    assert event["type"] == "planner.decision_ready"
    assert event["plan"]["plan_key"] == "decision-1:v1"


# 功能：验证旧 PlanReady JSONL 在 journal reopen 后仍通过 Event adapter 解码
# 设计：真实写入并关闭旧事件，再由新 coordinator 注册同一 stream 读取 replay，覆盖磁盘边界
async def test_legacy_plan_ready_journal_reopen_replays(tmp_path: Path) -> None:
    session_path = tmp_path / "sessions" / "sess-1"
    run_path = session_path / "runs" / "run-1"
    coordinator = EventJournalCoordinator()
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run("run-1", run_path, session_id="sess-1")
    plan = LegacyPlanViewV0(
        plan_key="decision-1:v1",
        goal="Legacy goal",
        selected_approach="Legacy approach",
        content_digest="legacy-content",
        ts="2026-08-15T00:00:00Z",
    )
    await coordinator.publish_required_durable(
        PlannerDecisionReadyEvent(
            event_id="legacy-plan-ready",
            run_id="run-1",
            planner_run_id="planner-1",
            session_id="sess-1",
            plan=plan,
            plan_key=plan.plan_key,
            decision_id="decision-1",
            decision_version=1,
            ts=plan.ts,
            snapshot_digest="snapshot",
            content_digest=plan.content_digest,
        )
    )
    await coordinator.flush_all()
    await coordinator.close()

    reopened = EventJournalCoordinator()
    await reopened.register_run("run-1", run_path, session_id=None)
    replay = await reopened.read_replay("run:run-1", after_seq=0, high_watermark=1)
    assert replay.records[0].event["plan"]["plan_key"] == "decision-1:v1"
    await reopened.close()


# 功能：验证 session stream 中的 PlanReady candidate 只有紧随其后的成功终态才提交一张计划卡
# 设计：真实写入 session durable stream 后再用客户端 reducer replay，覆盖 reconnect 主消费路径而非只测内存顺序
async def test_session_stream_plan_ready_commit_barrier(tmp_path: Path) -> None:
    session_path = tmp_path / "sessions" / "sess-1"
    run_path = session_path / "runs" / "run-v1"
    coordinator = EventJournalCoordinator()
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run("run-v1", run_path, session_id="sess-1")
    plan = build_plan_view(_decision(), top_level_run_id="run-v1")
    await coordinator.publish_required_durable(
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-v1",
            planner_run_id="planner-v1",
            session_id="sess-1",
            plan=plan,
            ts="2026-08-15T00:00:00Z",
        )
    )
    await coordinator.handle(
        RunFinishedEvent(
            run_id="run-v1",
            status="success",
            reason=None,
            steps=1,
            ts="2026-08-15T00:00:01Z",
        )
    )
    await coordinator.flush_all()
    await coordinator.register_run("run-v2", session_path / "runs" / "run-v2", session_id="sess-1")
    failed_plan = build_plan_view(_decision(selected_approach="failed"), top_level_run_id="run-v2")
    await coordinator.publish_required_durable(
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{failed_plan.projection_key}",
            run_id="run-v2",
            planner_run_id="planner-v2",
            session_id="sess-1",
            plan=failed_plan,
            ts="2026-08-15T00:00:02Z",
        )
    )
    await coordinator.handle(
        RunFinishedEvent(
            run_id="run-v2",
            status="failed",
            reason="planner failure",
            steps=1,
            ts="2026-08-15T00:00:03Z",
        )
    )
    await coordinator.flush_all()
    replay = await coordinator.read_replay(
        "session:sess-1",
        after_seq=0,
        high_watermark=coordinator.high_watermark("session:sess-1"),
    )
    assert [record.event["type"] for record in replay.records] == [
        "planner.decision_ready",
        "run.finished",
        "planner.decision_ready",
        "run.finished",
    ]
    reducer = PlanReadyCommitReducer()
    committed = [
        plan
        for record in replay.records
        for plan in reducer.ingest(record.event)
    ]
    assert len(committed) == 1
    assert committed[0].projection_key == plan.projection_key
    await coordinator.close()


# 功能：验证 PlanReady outer identity 只能与 PlanViewV1 派生值一致
# 设计：构造一个故意冲突的 projection digest，要求 event model 在 wire boundary 拒绝第二事实源
def test_plan_ready_outer_identity_must_match_plan() -> None:
    plan = build_plan_view(_decision(), top_level_run_id="run-1")

    with pytest.raises(ValueError, match="projection"):
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-1",
            planner_run_id="planner-1",
            session_id="sess-1",
            plan=plan,
            projection_digest="forged-projection",
            ts="2026-08-15T00:00:00Z",
        )


# 功能：验证直接传入的 V1 PlanView 也必须通过 projection digest 校验
# 设计：model_copy 模拟内部调用方伪造 digest，防止 Event adapter 只校验 outer alias 而接受不完整 projection
def test_plan_ready_rejects_forged_plan_digest() -> None:
    plan = build_plan_view(_decision(), top_level_run_id="run-1")
    forged = plan.model_copy(update={"projection_digest": "forged-projection"})

    with pytest.raises(ValueError, match="digest"):
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-1",
            planner_run_id="planner-1",
            session_id="sess-1",
            plan=forged,
            ts="2026-08-15T00:00:00Z",
        )


# 功能：验证 replay payload 不能伪造 approval 或 execution capability
# 设计：在真实 V1 JSON projection 上篡改固定字面量，再走 persistence decoder 的 strict boundary
def test_plan_view_replay_rejects_forged_fixed_flags() -> None:
    event = _plan_ready()
    forged = event.model_dump(mode="json")
    forged_plan = forged["plan"]
    assert isinstance(forged_plan, dict)
    forged_plan["requires_user_approval"] = False
    forged_plan["execution_available"] = True

    with pytest.raises(JournalCorruptionError):
        _validate_event_payload(forged, role="v2")


# 功能：验证 PlanReady 的 projection identity 必须绑定 outer top-level run
# 设计：复用合法 projection 但替换 event run_id，锁定 run/projection mismatch 的 wire 拒绝边界
def test_plan_ready_rejects_run_projection_mismatch() -> None:
    plan = build_plan_view(_decision(), top_level_run_id="run-1")

    with pytest.raises(ValueError, match="run"):
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-2",
            planner_run_id="planner-1",
            session_id="sess-1",
            plan=plan,
            ts="2026-08-15T00:00:00Z",
        )


# 功能：验证 PlanReady candidate 只有在 run.finished(success) 后才提交一次
# 设计：覆盖 candidate-first、success commit、failed discard、重复 event id 和 projection key 去重
def test_plan_ready_commit_reducer_waits_for_success_and_deduplicates() -> None:
    reducer = PlanReadyCommitReducer()
    ready = _plan_ready()
    ready_payload = ready.model_dump(mode="json")

    assert reducer.ingest(ready_payload) == []
    assert reducer.ingest(ready_payload) == []
    committed = reducer.ingest(
        {
            "type": "run.finished",
            "run_id": "run-1",
            "status": "success",
        }
    )
    assert len(committed) == 1
    assert committed[0].projection_key == ready.plan.projection_key
    assert reducer.ingest(ready_payload) == []

    failed = _plan_ready(run_id="run-2")
    assert reducer.ingest(failed.model_dump(mode="json")) == []
    assert reducer.ingest(
        {"type": "run.finished", "run_id": "run-2", "status": "failed"}
    ) == []


# 功能：验证相同 event_id 携带不同 projection digest 时不覆盖既有 candidate
# 设计：模拟重放冲突 payload，要求只记录 integrity warning 而不产生第二张计划卡
def test_plan_ready_conflicting_duplicate_is_warning_only() -> None:
    reducer = PlanReadyCommitReducer()
    ready = _plan_ready()
    payload = ready.model_dump(mode="json")
    assert reducer.ingest(payload) == []
    conflict = dict(payload)
    conflict_plan = build_plan_view(
        _decision(selected_approach="conflicting"), top_level_run_id="run-1"
    )
    conflict["plan"] = conflict_plan.model_dump(mode="json")
    assert reducer.ingest(conflict) == []
    assert "plan-projection-integrity-conflict" in reducer.warnings


# 功能：验证 one-shot 在 terminal 之后订阅仍能从 candidate replay 出唯一 committed plan
# 设计：先摄取 run.finished 再摄取 PlanReady，模拟 subscribe-after-terminal 的 durable replay 顺序
def test_plan_ready_replay_after_terminal_commits_once() -> None:
    reducer = PlanReadyCommitReducer()
    ready = _plan_ready()
    assert reducer.ingest(
        {"type": "run.finished", "run_id": "run-1", "status": "success"}
    ) == []
    committed = reducer.ingest(ready.model_dump(mode="json"))
    assert len(committed) == 1
    assert reducer.ingest(ready.model_dump(mode="json")) == []


# 功能：验证完整 orchestrate renderer 不复用 bounded PlanView 且超出 transport budget 时 fail closed
# 设计：用完整 decision renderer 生成 agent-facing payload，并以超长 approach 覆盖静默 tool truncation 风险
def test_execution_renderer_is_full_and_budget_checked() -> None:
    decision = _decision(selected_approach="x" * 20_000)
    with pytest.raises(ValueError, match="too-large"):
        render_planner_decision_execution_summary(decision)

    rendered = render_planner_decision_execution_summary(_decision())
    assert "requirements" in rendered
    assert "intended_changes" in rendered
    assert "verification_plan" in rendered
