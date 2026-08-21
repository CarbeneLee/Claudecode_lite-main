from __future__ import annotations

import json
from pathlib import Path

import pytest

from kama_claude.core.approval import (
    ApprovalError,
    ApprovalRecordCorrupt,
    ApprovalRequestOwner,
    ApprovalService,
    ApprovalSnapshot,
    ApprovalSnapshotState,
    materialize_committed_plan_receipt,
)
from kama_claude.core.bus.events import PlannerDecisionReadyEvent, RunFinishedEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.grounding import canonical_digest
from kama_claude.core.planning import ExactPlannerDecisionV2, build_plan_view
from kama_claude.core.session.manager import SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore


# 构造可写入现有 SessionStore 双层 digest 校验的 exact decision
def _decision() -> ExactPlannerDecisionV2:
    payload = {
        "schema_version": 2,
        "decision_id": "decision-1",
        "version": 1,
        "goal": "change behavior",
        "requirements": [
            {"requirement_id": "R1", "statement": "change behavior", "required": True}
        ],
        "architecture_slice_id": "slice-1",
        "architecture_slice_version": 1,
        "architecture_slice_content_digest": "slice-digest",
        "snapshot_digest": "snapshot-digest",
        "architecture_mode": "preserve",
        "selected_approach": "edit existing module",
        "existing_patterns_reused": ["existing pattern"],
        "intended_changes": [
            {
                "change_id": "C1",
                "description": "edit module",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["read-target"],
            }
        ],
        "files_to_modify": ["src/target.py"],
        "files_to_create": [],
        "allowed_capabilities": ["read_file", "write_file"],
        "dependency_changes": [],
        "protocol_or_schema_changes": [],
        "verification_plan": [{"requirement_ids": ["R1"], "strategy": "run tests"}],
        "non_goals": [],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
    }
    payload["content_digest"] = canonical_digest(payload)
    return ExactPlannerDecisionV2.model_validate(payload)


# 创建真实 journal 证据并返回 store、coordinator、projection identity
async def _committed_fixture(tmp_path: Path) -> tuple[SessionStore, EventJournalCoordinator, str]:
    store = SessionStore(tmp_path / "sessions")
    decision = _decision()
    store.write_decision("sess-1", decision.decision_id, decision.version, decision.model_dump(mode="json"))
    plan = build_plan_view(decision, top_level_run_id="run-1")
    session_path = store.session_dir("sess-1")
    run_path = store.runs_dir("sess-1") / "run-1"
    coordinator = EventJournalCoordinator()
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run("run-1", run_path, session_id="sess-1")
    await coordinator.publish_required_durable(
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-1",
            planner_run_id="planner-1",
            session_id="sess-1",
            plan=plan,
            ts="t1",
        )
    )
    await coordinator.handle(
        RunFinishedEvent(
            run_id="run-1",
            status="success",
            reason=None,
            steps=1,
            ts="t2",
        )
    )
    await coordinator.flush_all()
    return store, coordinator, plan.projection_key


# 功能：验证同一 projection 的冲突 terminal snapshot 进入 conflicted/unknown
# 设计：直接驱动纯状态机，不依赖 socket 或事件循环，锁定冲突不会伪装成 approved/rejected
def test_conflicting_terminal_snapshots_enter_conflicted_state() -> None:
    owner = ApprovalRequestOwner(
        client_identity="cli-1",
        session_id="sess-1",
        daemon_instance_id="daemon-1",
        projection_key="pv1:run-1:decision:v1",
    )
    state = ApprovalSnapshotState(owner)
    state.merge(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            record_digest="record-a",
            action="approve",
        )
    )

    relation = state.merge(
        ApprovalSnapshot(
            status="rejected",
            projection_key=owner.projection_key,
            record_digest="record-b",
            action="reject",
        )
    )

    assert relation == "conflict"
    assert state.snapshot.status == "conflicted/unknown"
    assert state.refresh_required is True
    assert state.conflict_epoch == 1


# 功能：验证旧 pending replay 不能把已经 resolved 的 approval 降级
# 设计：先应用 approved，再注入 pending，锁定 snapshot 偏序中的 stale 分支
def test_stale_pending_snapshot_cannot_downgrade_resolved() -> None:
    owner = ApprovalRequestOwner("cli-1", "sess-1", "daemon-1", "pv1:run:decision:v1")
    state = ApprovalSnapshotState(owner)
    approved = ApprovalSnapshot(
        status="approved",
        projection_key=owner.projection_key,
        record_digest="record-a",
        action="approve",
    )
    assert state.merge(approved) == "apply"

    assert (
        state.merge(
            ApprovalSnapshot(status="pending", projection_key=owner.projection_key)
        )
        == "stale"
    )
    assert state.snapshot == approved


# 功能：验证完全相同的 resolved snapshot 重复到达时幂等
# 设计：重放同一 record/action/digest，确认不产生冲突 epoch 或状态变化
def test_same_resolved_snapshot_is_idempotent() -> None:
    owner = ApprovalRequestOwner("cli-1", "sess-1", "daemon-1", "pv1:run:decision:v1")
    state = ApprovalSnapshotState(owner)
    snapshot = ApprovalSnapshot(
        status="rejected",
        projection_key=owner.projection_key,
        record_digest="record-a",
        action="reject",
        commit_receipt_digest="receipt-a",
    )
    assert state.merge(snapshot) == "apply"
    assert state.merge(snapshot.model_copy()) == "noop"
    assert state.conflict_epoch is None


# 功能：验证延迟的 pending authority GET 不能覆盖已经应用的 terminal approval
# 设计：先用 live terminal snapshot 建立 resolved，再走独立 authority seed，锁定 response reordering 防护
def test_stale_pending_authority_cannot_downgrade_terminal() -> None:
    owner = ApprovalRequestOwner("cli-1", "sess-1", "daemon-1", "pv1:run:decision:v1")
    state = ApprovalSnapshotState(owner)
    approved = ApprovalSnapshot(
        status="approved",
        projection_key=owner.projection_key,
        record_digest="record-a",
        action="approve",
    )
    assert state.merge(approved) == "apply"
    assert state.seed_authoritative_snapshot(
        ApprovalSnapshot(status="pending", projection_key=owner.projection_key),
        owner=owner,
    ) is False
    assert state.snapshot == approved


# 功能：验证重复冲突只创建一个 refresh epoch 并支持权威结果绕过普通冲突分类器
# 设计：重复提交不同 terminal record 后只允许一次 refresh，随后独立 authority apply 清除冲突
def test_conflict_refresh_is_coalesced_and_authority_replaces_state() -> None:
    owner = ApprovalRequestOwner(
        client_identity="cli-1",
        session_id="sess-1",
        daemon_instance_id="daemon-1",
        projection_key="pv1:run-1:decision:v1",
    )
    state = ApprovalSnapshotState(owner)
    state.merge(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            record_digest="record-a",
            action="approve",
        )
    )
    state.merge(
        ApprovalSnapshot(
            status="rejected",
            projection_key=owner.projection_key,
            record_digest="record-b",
            action="reject",
        )
    )
    state.merge(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            record_digest="record-c",
            action="approve",
        )
    )

    epoch = state.begin_refresh()
    assert epoch == 1
    assert state.begin_refresh() is None

    assert state.apply_authoritative_snapshot(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            record_digest="record-authority",
            action="approve",
        ),
        epoch=epoch,
    ) is True
    assert state.snapshot.status == "approved"
    assert state.refresh_required is False
    assert state.conflict_epoch is None


# 功能：验证 authority refresh 失败不会自动重试或把未知状态伪装成 terminal
# 设计：启动一个 epoch 后显式失败，再重复 begin_refresh，锁定同一 epoch 不会形成 refresh loop
def test_conflict_refresh_failure_keeps_unknown_without_loop() -> None:
    owner = ApprovalRequestOwner("cli-1", "sess-1", "daemon-1", "pv1:run:decision:v1")
    state = ApprovalSnapshotState(owner)
    state.merge(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            action="approve",
            record_digest="a",
        )
    )
    state.merge(
        ApprovalSnapshot(
            status="rejected",
            projection_key=owner.projection_key,
            action="reject",
            record_digest="b",
        )
    )
    epoch = state.begin_refresh()
    assert epoch == 1
    assert state.fail_refresh(epoch=epoch) is True
    assert state.snapshot.status == "conflicted/unknown"
    assert state.begin_refresh() is None


# 功能：验证旧 conflict epoch 的 authority response 不能覆盖新一轮冲突
# 设计：完成第一轮失败后创建新冲突 epoch，再提交旧 epoch response，锁定 owner/epoch 双重防护
def test_stale_authority_response_is_discarded() -> None:
    owner = ApprovalRequestOwner("cli-1", "sess-1", "daemon-1", "pv1:run:decision:v1")
    state = ApprovalSnapshotState(owner)
    state.merge(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            action="approve",
            record_digest="a",
        )
    )
    state.merge(
        ApprovalSnapshot(
            status="rejected",
            projection_key=owner.projection_key,
            action="reject",
            record_digest="b",
        )
    )
    first_epoch = state.begin_refresh()
    assert first_epoch == 1
    assert state.fail_refresh(epoch=first_epoch) is True
    # 新的 externally observed terminal conflict 允许开启下一 epoch
    state.force_new_conflict_epoch()
    second_epoch = state.begin_refresh()
    assert second_epoch == 2
    assert state.apply_authoritative_snapshot(
        ApprovalSnapshot(
            status="approved",
            projection_key=owner.projection_key,
            action="approve",
            record_digest="authority",
        ),
        epoch=first_epoch,
    ) is False
    assert state.snapshot.status == "conflicted/unknown"


# 功能：验证损坏 ApprovalRecord 永不被普通 approve/reject 覆盖或自动重建
# 设计：先制造损坏字节，再调用严格读取入口，断言稳定 corruption 错误且原始 bytes 不变
def test_corrupted_approval_record_fails_closed_without_reconstruction(tmp_path) -> None:
    from kama_claude.core.session.store import SessionStore

    store = SessionStore(tmp_path)
    path = store.approval_record_path("sess-1", "pv1:run-1:decision:v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="approval record corrupt"):
        store.read_approval_record("sess-1", "pv1:run-1:decision:v1")

    assert path.read_bytes() == before


# 功能：验证 envelope 有效但 schema 损坏的 ApprovalRecord 也先 fail closed 而不触发 receipt 修复
# 设计：写入可解析 envelope 后用 materializer 哨兵，区分 record schema corruption 与 committed evidence failure
@pytest.mark.asyncio
async def test_invalid_approval_record_schema_precedes_receipt_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path)
    projection_key = "pv1:run-1:decision:v1"
    store.write_approval_record(
        "sess-1",
        projection_key,
        {
            "schema_version": 1,
            "session_id": "sess-1",
            "projection_key": projection_key,
            "decision_id": "decision",
            "decision_version": 1,
            "content_digest": "content",
            "commit_receipt_digest": "receipt",
            "action": "not-an-action",
            "actor": "user",
            "resolved_at": "t",
            "record_digest": "not-recomputed",
        },
    )
    materialize_called = False

    async def unexpected_materialize(**_kwargs: object) -> object:
        nonlocal materialize_called
        materialize_called = True
        raise AssertionError("receipt materialization must follow record validation")

    monkeypatch.setattr(
        "kama_claude.core.approval.materialize_committed_plan_receipt",
        unexpected_materialize,
    )
    service = ApprovalService(store, None)

    with pytest.raises(ApprovalRecordCorrupt, match="approval-record-corrupt"):
        await service.resolve(
            session_id="sess-1",
            projection_key=projection_key,
            action="approve",
            decision_id="decision",
            decision_version=1,
            content_digest="content",
            commit_receipt_digest="receipt",
        )

    assert materialize_called is False


# 功能：验证 committed receipt 的 namespace 使用 projection key 的稳定 hash
# 设计：不依赖具体 decision 内容，确保 receipt 文件名不直接暴露带冒号的 projection key
def test_committed_receipt_path_is_projection_hashed(tmp_path) -> None:
    store = SessionStore(tmp_path)
    path = store.committed_plan_receipt_path("sess-1", "pv1:run-1:decision:v1")
    assert path.parent.name == "committed"
    assert path.suffix == ".json"
    assert ":" not in path.name


# 功能：验证只有 run/session 两条 durable stream 都有成功终态时才能 materialize receipt
# 设计：使用真实 EventJournal 写入 PlanReady 与 RunFinished，再通过 verifier 读取双 stream 证据
@pytest.mark.asyncio
async def test_materialize_committed_receipt_requires_cross_stream_evidence(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)

    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=journal,
        session_id="sess-1",
        projection_key=projection_key,
    )

    assert receipt.projection_key == projection_key
    assert receipt.top_level_run_id == "run-1"
    assert receipt.run_finished_journal_event_id.startswith("evt-")
    receipt.verify_digest()
    assert store.read_committed_plan_receipt("sess-1", projection_key) is not None
    await journal.close()


# 功能：验证只有 run stream 的成功终态不能产生 committed receipt
# 设计：真实 journal 只注册无 session 映射的 run stream，排除 session evidence 后断言 fail closed
@pytest.mark.asyncio
async def test_run_stream_only_success_cannot_materialize_receipt(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    decision = _decision()
    store.write_decision(
        "sess-1",
        decision.decision_id,
        decision.version,
        decision.model_dump(mode="json"),
    )
    plan = build_plan_view(decision, top_level_run_id="run-only")
    journal = EventJournalCoordinator()
    await journal.register_run(
        "run-only",
        store.runs_dir("sess-1") / "run-only",
        session_id=None,
    )
    await journal.publish_required_durable(
        PlannerDecisionReadyEvent(
            event_id=f"plan-ready:{plan.projection_key}",
            run_id="run-only",
            planner_run_id="planner-only",
            session_id="",
            plan=plan,
            ts="t1",
        )
    )
    await journal.handle(
        RunFinishedEvent(
            run_id="run-only",
            status="success",
            reason=None,
            steps=1,
            ts="t2",
        )
    )
    await journal.flush_all()

    with pytest.raises(ApprovalError, match="evidence unavailable"):
        await materialize_committed_plan_receipt(
            store=store,
            journal=journal,
            session_id="sess-1",
            projection_key=plan.projection_key,
        )

    await journal.close()


# 功能：验证缺少已注册 journal 时 committed-plan eligibility 明确 fail closed
# 设计：直接调用 materializer，避免把普通 session/unit wiring 的缺 journal 误报成 AttributeError
@pytest.mark.asyncio
async def test_missing_journal_fails_closed_for_committed_receipt(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ApprovalError, match="evidence unavailable"):
        await materialize_committed_plan_receipt(
            store=store,
            journal=None,
            session_id="sess-1",
            projection_key="pv1:run-1:decision:v1",
        )


# 功能：验证 daemon/journal 重启后可从 run+session replay 重新 materialize receipt
# 设计：关闭首个 coordinator 后用同一磁盘 stream 重新注册，覆盖 restart recovery 而非只测内存对象
@pytest.mark.asyncio
async def test_committed_receipt_recovery_after_journal_restart(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    await journal.close()
    reopened = EventJournalCoordinator()
    await reopened.register_session("sess-1", store.session_dir("sess-1"))
    await reopened.register_run(
        "run-1",
        store.runs_dir("sess-1") / "run-1",
        session_id="sess-1",
    )

    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=reopened,
        session_id="sess-1",
        projection_key=projection_key,
    )

    assert receipt.projection_key == projection_key
    await reopened.close()


# 功能：验证 SessionManager 重启 reconciliation 后仍能通过真实 authority 查询 committed plan
# 设计：关闭首个 journal 后由新 manager 重新注册 session/run streams，再调用 get_approval 而非直接读文件
@pytest.mark.asyncio
async def test_session_manager_recovery_replays_committed_approval_authority(
    tmp_path: Path,
) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    session = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="plan",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
        run_ids=["run-1"],
    )
    store.write_meta(session)
    await journal.close()

    reopened = EventJournalCoordinator()
    manager = SessionManager(
        store,
        lambda _root: object(),  # type: ignore[arg-type]
        EventBus(),
        journal=reopened,
    )
    await manager.reconcile_persisted_sessions()
    result = await manager.get_approval("sess-1", projection_key)

    assert result.status == "pending"
    assert result.projection_key == projection_key
    assert result.commit_receipt_digest
    assert store.read_committed_plan_receipt("sess-1", projection_key) is not None
    await reopened.close()


# 功能：验证相同 event_id 但跨 stream PlanReady payload 不一致时不能形成 committed receipt
# 设计：先写入真实双流证据，再仅篡改 run stream 的 planner identity，覆盖 journal 之外的 cross-stream integrity 检查
@pytest.mark.asyncio
async def test_cross_stream_plan_ready_payload_conflict_fails_closed(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    await journal.close()
    run_events = store.runs_dir("sess-1") / "run-1" / "events.v2.jsonl"
    rows = [json.loads(line) for line in run_events.read_text(encoding="utf-8").splitlines()]
    rows[0]["event"]["planner_run_id"] = "forged-planner"
    run_events.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    reopened = EventJournalCoordinator()
    await reopened.register_session("sess-1", store.session_dir("sess-1"))
    await reopened.register_run(
        "run-1",
        store.runs_dir("sess-1") / "run-1",
        session_id="sess-1",
    )
    with pytest.raises(ApprovalError, match="evidence conflict"):
        await materialize_committed_plan_receipt(
            store=store,
            journal=reopened,
            session_id="sess-1",
            projection_key=projection_key,
        )
    await reopened.close()


# 功能：验证损坏的 derived receipt 可由独立 decision/journal 证据修复
# 设计：先写入不可解析 receipt，再调用 materializer，断言修复后的 canonical receipt 不采信损坏字段
@pytest.mark.asyncio
async def test_corrupt_committed_receipt_is_repaired_from_authoritative_evidence(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    path = store.committed_plan_receipt_path("sess-1", projection_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"payload": {"projection_key": projection_key, "top_level_run_id": "forged"}, "digest": "bad"}),
        encoding="utf-8",
    )

    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=journal,
        session_id="sess-1",
        projection_key=projection_key,
    )

    assert receipt.top_level_run_id == "run-1"
    restored = store.read_committed_plan_receipt("sess-1", projection_key)
    assert restored is not None
    assert restored["top_level_run_id"] == "run-1"
    await journal.close()


# 功能：验证缺少 cross-stream authority evidence 时损坏 receipt 不会被重建
# 设计：只创建损坏 receipt 而不创建 journal 证据，断言失败后原始 bytes 完整保留
@pytest.mark.asyncio
async def test_corrupt_receipt_without_evidence_fails_closed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    journal = EventJournalCoordinator()
    projection_key = "pv1:run-missing:decision-1:v1"
    path = store.committed_plan_receipt_path("sess-1", projection_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("corrupt receipt", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ApprovalError, match="evidence"):
        await materialize_committed_plan_receipt(
            store=store,
            journal=journal,
            session_id="sess-1",
            projection_key=projection_key,
        )

    assert path.read_bytes() == before


# 功能：验证损坏的 ApprovalRecord 即使有有效 receipt 也不能被普通 approve/reject 重建
# 设计：先建立 committed evidence 与损坏 user authority，再覆盖两种普通 action，断言 bytes 和 corruption 状态保持
@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_corrupt_approval_record_cannot_be_reconstructed_by_approve_or_reject(
    tmp_path: Path,
    action: str,
) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=journal,
        session_id="sess-1",
        projection_key=projection_key,
    )
    path = store.approval_record_path("sess-1", projection_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("corrupt approval bytes", encoding="utf-8")
    before = path.read_bytes()
    service = ApprovalService(store, journal)

    with pytest.raises(ApprovalRecordCorrupt, match="approval-record-corrupt"):
        await service.resolve(
            session_id="sess-1",
            projection_key=projection_key,
            action=action,  # type: ignore[arg-type]
            decision_id=receipt.decision_id,
            decision_version=receipt.decision_version,
            content_digest=receipt.decision_content_digest,
            commit_receipt_digest=receipt.receipt_digest,
        )

    assert path.read_bytes() == before
    records = await journal.read_replay(
        "session:sess-1",
        after_seq=0,
        high_watermark=journal.high_watermark("session:sess-1"),
    )
    assert not any(
        record.event.get("type") == "plan.approval_changed"
        for record in records.records
    )
    await journal.close()


# 功能：验证客户端伪造 commit receipt digest 时 approval 不会写入 authority record
# 设计：使用真实 committed evidence 但替换 receipt precondition，断言稳定拒绝且 session stream 不新增 approval event
@pytest.mark.asyncio
async def test_forged_receipt_digest_is_rejected_before_approval_write(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=journal,
        session_id="sess-1",
        projection_key=projection_key,
    )
    service = ApprovalService(store, journal)

    with pytest.raises(ApprovalError, match="approval-target-mismatch"):
        await service.resolve(
            session_id="sess-1",
            projection_key=projection_key,
            action="approve",
            decision_id=receipt.decision_id,
            decision_version=receipt.decision_version,
            content_digest=receipt.decision_content_digest,
            commit_receipt_digest="forged-receipt-digest",
        )

    assert store.read_approval_record("sess-1", projection_key) is None
    records = await journal.read_replay(
        "session:sess-1",
        after_seq=0,
        high_watermark=journal.high_watermark("session:sess-1"),
    )
    assert not any(
        record.event.get("type") == "plan.approval_changed"
        for record in records.records
    )
    await journal.close()


# 功能：验证首次 approval durable event、重复相同响应幂等以及冲突响应拒绝
# 设计：在真实 session stream 上检查 event_id 数量，证明 user authority 与通知去重分别成立
@pytest.mark.asyncio
async def test_approval_resolution_is_idempotent_and_durable(tmp_path: Path) -> None:
    store, journal, projection_key = await _committed_fixture(tmp_path)
    receipt = await materialize_committed_plan_receipt(
        store=store,
        journal=journal,
        session_id="sess-1",
        projection_key=projection_key,
    )
    service = ApprovalService(store, journal)
    first = await service.resolve(
        session_id="sess-1",
        projection_key=projection_key,
        action="approve",
        decision_id=receipt.decision_id,
        decision_version=receipt.decision_version,
        content_digest=receipt.decision_content_digest,
        commit_receipt_digest=receipt.receipt_digest,
    )
    second = await service.resolve(
        session_id="sess-1",
        projection_key=projection_key,
        action="approve",
        decision_id=receipt.decision_id,
        decision_version=receipt.decision_version,
        content_digest=receipt.decision_content_digest,
        commit_receipt_digest=receipt.receipt_digest,
    )
    assert first == second
    with pytest.raises(Exception, match="already-resolved-conflict"):
        await service.resolve(
            session_id="sess-1",
            projection_key=projection_key,
            action="reject",
            decision_id=receipt.decision_id,
            decision_version=receipt.decision_version,
            content_digest=receipt.decision_content_digest,
            commit_receipt_digest=receipt.receipt_digest,
        )
    records = await journal.read_replay(
        "session:sess-1",
        after_seq=0,
        high_watermark=journal.high_watermark("session:sess-1"),
    )
    approval_events = [
        record for record in records.records if record.event.get("type") == "plan.approval_changed"
    ]
    assert len(approval_events) == 1
    await journal.close()


# 功能：验证 approval command/event 的 discriminated wire models 能往返并保留 exact identity
# 设计：直接走 Command/Event TypeAdapter，覆盖新增协议模型确实接入 union 而非只存在 Python 类
def test_approval_protocol_models_roundtrip() -> None:
    from pydantic import TypeAdapter

    from kama_claude.core.bus.commands import Command, PlanApproveCommand
    from kama_claude.core.bus.events import Event, PlanApprovalChangedEvent

    command = PlanApproveCommand(
        session_id="sess-1",
        projection_key="pv1:run-1:decision-1:v1",
        decision_id="decision-1",
        decision_version=1,
        content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
    )
    restored_command = TypeAdapter(Command).validate_json(command.model_dump_json())
    assert isinstance(restored_command, PlanApproveCommand)
    event = PlanApprovalChangedEvent(
        event_id="plan-approval:record-digest",
        session_id="sess-1",
        projection_key=command.projection_key,
        status="approved",
        action="approve",
        record_digest="record-digest",
        commit_receipt_digest=command.commit_receipt_digest,
        ts="t",
    )
    restored_event = TypeAdapter(Event).validate_json(event.model_dump_json())
    assert isinstance(restored_event, PlanApprovalChangedEvent)
