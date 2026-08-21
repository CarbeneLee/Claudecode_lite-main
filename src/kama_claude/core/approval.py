from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.plan_view import PlanViewV1, decode_plan_view_record

if TYPE_CHECKING:
    from kama_claude.core.session.store import SessionStore

logger = logging.getLogger(__name__)


# 使用稳定 JSON 编码计算 approval artifact digest，避免导入 session package 形成循环
def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

ApprovalStatus = Literal["pending", "approved", "rejected", "conflicted/unknown"]
ApprovalAction = Literal["approve", "reject"]


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class ApprovalError(RuntimeError):
    # 创建 approval domain 的稳定错误
    pass


class ApprovalRecordCorrupt(ApprovalError):
    # 表示用户 authority artifact 已损坏且不能自动修复
    pass


class ApprovalConflict(ApprovalError):
    # 表示同一 projection 已存在冲突的 terminal approval
    pass


class ApprovalSnapshotRelation(StrEnum):
    # 描述 approval snapshot 的单调合并关系
    NEW = "new"
    NOOP = "noop"
    APPLY = "apply"
    STALE = "stale"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class ApprovalRequestOwner:
    client_identity: str
    session_id: str
    daemon_instance_id: str
    projection_key: str


@dataclass(frozen=True, slots=True)
class CommittedApprovalTarget:
    # 保存一个 committed PlanView 对应的不可猜测 approval identity
    session_id: str
    projection_key: str
    decision_id: str
    decision_version: int
    decision_content_digest: str

    # 从已经 committed 的 PlanView 建立 approval target
    @classmethod
    def from_plan(cls, session_id: str, plan: PlanViewV1) -> CommittedApprovalTarget:
        return cls(
            session_id=session_id,
            projection_key=plan.projection_key,
            decision_id=plan.decision_id,
            decision_version=plan.decision_version,
            decision_content_digest=plan.decision_content_digest,
        )

    # 验证 RPC 或 notification 仍指向同一个 committed decision
    def matches_payload(self, payload: Mapping[str, Any]) -> bool:
        return (
            payload.get("session_id") == self.session_id
            and payload.get("projection_key") == self.projection_key
            and payload.get("decision_id") == self.decision_id
            and payload.get("decision_version") == self.decision_version
            and payload.get("content_digest") == self.decision_content_digest
        )


class ApprovalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ApprovalStatus
    projection_key: str = Field(min_length=1)
    record_digest: str | None = None
    action: ApprovalAction | None = None
    commit_receipt_digest: str | None = None


# 将 approval RPC/event 的受限字段解析为统一 snapshot
def approval_snapshot_from_payload(payload: Mapping[str, Any]) -> ApprovalSnapshot:
    return ApprovalSnapshot.model_validate(
        {
            "status": payload["status"],
            "projection_key": payload["projection_key"],
            "record_digest": payload.get("record_digest"),
            "action": payload.get("action"),
            "commit_receipt_digest": payload.get("commit_receipt_digest"),
        }
    )


class ApprovalSnapshotState:
    # 初始化一个 projection 绑定的客户端 approval 状态机
    def __init__(self, owner: ApprovalRequestOwner) -> None:
        self.owner = owner
        self.snapshot = ApprovalSnapshot(
            status="pending",
            projection_key=owner.projection_key,
        )
        self.refresh_required = False
        self.refresh_in_flight = False
        self.conflict_epoch: int | None = None
        self._next_conflict_epoch = 0

    # 合并普通 live/replay snapshot，冲突时只登记一个 refresh epoch
    def merge(
        self,
        incoming: ApprovalSnapshot,
        *,
        owner: ApprovalRequestOwner | None = None,
    ) -> ApprovalSnapshotRelation:
        if owner is not None and owner != self.owner:
            return ApprovalSnapshotRelation.STALE
        if incoming.projection_key != self.owner.projection_key:
            return ApprovalSnapshotRelation.STALE
        current = self.snapshot
        if current.status == "conflicted/unknown":
            if incoming.status == "conflicted/unknown":
                return ApprovalSnapshotRelation.NOOP
            return ApprovalSnapshotRelation.STALE
        if current.status == "pending":
            if incoming.status == "pending":
                return ApprovalSnapshotRelation.NOOP
            self.snapshot = incoming
            return ApprovalSnapshotRelation.APPLY
        if incoming.status == "pending":
            return ApprovalSnapshotRelation.STALE
        if (
            current.status == incoming.status
            and current.action == incoming.action
            and current.record_digest == incoming.record_digest
            and current.commit_receipt_digest == incoming.commit_receipt_digest
        ):
            return ApprovalSnapshotRelation.NOOP
        self.snapshot = ApprovalSnapshot(
            status="conflicted/unknown",
            projection_key=self.owner.projection_key,
        )
        if self.conflict_epoch is None:
            self._next_conflict_epoch += 1
            self.conflict_epoch = self._next_conflict_epoch
            self.refresh_required = True
        return ApprovalSnapshotRelation.CONFLICT

    # 用首次 authority GET 建立状态，避免把 authority 响应误判成普通冲突
    def seed_authoritative_snapshot(
        self,
        incoming: ApprovalSnapshot,
        *,
        owner: ApprovalRequestOwner | None = None,
    ) -> bool:
        if owner is not None and owner != self.owner:
            return False
        if incoming.projection_key != self.owner.projection_key:
            return False
        if self.refresh_in_flight:
            return False
        if self.snapshot.status in ("approved", "rejected") and incoming.status == "pending":
            return False
        self.snapshot = incoming
        self.refresh_required = False
        self.conflict_epoch = None
        return True

    # 为当前 conflict epoch 启动最多一个 authority refresh
    def begin_refresh(self) -> int | None:
        if (
            self.conflict_epoch is None
            or self.refresh_in_flight
            or not self.refresh_required
        ):
            return None
        self.refresh_in_flight = True
        self.refresh_required = False
        return self.conflict_epoch

    # 使用 authority GET 的结果直接替换状态，绕过普通 conflict classifier
    def apply_authoritative_snapshot(
        self,
        incoming: ApprovalSnapshot,
        *,
        epoch: int,
        owner: ApprovalRequestOwner | None = None,
    ) -> bool:
        if owner is not None and owner != self.owner:
            return False
        if incoming.projection_key != self.owner.projection_key:
            return False
        if self.conflict_epoch != epoch:
            return False
        if not self.refresh_in_flight:
            return False
        self.snapshot = incoming
        self.refresh_in_flight = False
        self.refresh_required = False
        self.conflict_epoch = None
        return True

    # 结束失败的 authority refresh 但保留 conflicted/unknown，不自动循环重试
    def fail_refresh(self, *, epoch: int) -> bool:
        if self.conflict_epoch != epoch:
            return False
        self.refresh_in_flight = False
        self.refresh_required = False
        return True

    # 记录用户或新 projection event 明确开启的新冲突 epoch
    def force_new_conflict_epoch(self) -> int:
        self._next_conflict_epoch += 1
        self.conflict_epoch = self._next_conflict_epoch
        self.refresh_required = True
        self.refresh_in_flight = False
        self.snapshot = ApprovalSnapshot(
            status="conflicted/unknown",
            projection_key=self.owner.projection_key,
        )
        return self._next_conflict_epoch


class CommittedPlanEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    top_level_run_id: str
    planner_run_id: str
    decision_id: str
    decision_version: int = Field(ge=1)
    decision_content_digest: str
    projection_key: str
    projection_digest: str
    plan_ready_journal_event_id: str
    run_finished_journal_event_id: str


class CommittedPlanReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    top_level_run_id: str
    planner_run_id: str
    decision_id: str
    decision_version: int = Field(ge=1)
    decision_content_digest: str
    projection_key: str
    projection_digest: str
    plan_ready_journal_event_id: str
    run_finished_journal_event_id: str
    receipt_digest: str

    # 从已验证 cross-stream evidence 计算 derived receipt digest
    @classmethod
    def from_evidence(cls, evidence: CommittedPlanEvidence) -> CommittedPlanReceipt:
        payload = evidence.model_dump(mode="json")
        return cls(**payload, receipt_digest=_digest(payload))

    # 验证 receipt 自身 canonical digest，不把 receipt 当作 authority 来源
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"receipt_digest"})
        if _digest(payload) != self.receipt_digest:
            raise ValueError("committed plan receipt digest mismatch")


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    projection_key: str
    decision_id: str
    decision_version: int = Field(ge=1)
    content_digest: str
    commit_receipt_digest: str
    action: ApprovalAction
    actor: Literal["user"] = "user"
    resolved_at: str
    record_digest: str

    # 从完整用户响应字段创建 immutable authority record
    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        projection_key: str,
        decision_id: str,
        decision_version: int,
        content_digest: str,
        commit_receipt_digest: str,
        action: ApprovalAction,
        resolved_at: str | None = None,
    ) -> ApprovalRecord:
        payload = {
            "schema_version": 1,
            "session_id": session_id,
            "projection_key": projection_key,
            "decision_id": decision_id,
            "decision_version": decision_version,
            "content_digest": content_digest,
            "commit_receipt_digest": commit_receipt_digest,
            "action": action,
            "actor": "user",
            "resolved_at": resolved_at or _now(),
        }
        return cls.model_validate({**payload, "record_digest": _digest(payload)})

    # 验证 user authority record 的 canonical digest
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"record_digest"})
        if _digest(payload) != self.record_digest:
            raise ValueError("approval record digest mismatch")


# 验证两条 durable stream 是否共同提交了同一 committed Plan
async def verify_committed_plan(
    *,
    store: SessionStore,
    journal: Any,
    session_id: str,
    projection_key: str,
) -> CommittedPlanEvidence:
    from kama_claude.core.planning import decode_exact_planner_decision

    if journal is None:
        raise ApprovalError("committed plan evidence unavailable")
    session_stream = f"session:{session_id}"
    if not journal.has_stream(session_stream):
        raise ApprovalError("committed plan evidence unavailable")
    session_records = await _read_all_records(journal, session_stream)
    ready_candidates = [
        record
        for record in session_records
        if record.event.get("type") == "planner.decision_ready"
    ]
    matching_ready = []
    for record in ready_candidates:
        try:
            plan = decode_plan_view_record(record.event.get("plan"))
        except (TypeError, ValueError):
            continue
        if isinstance(plan, PlanViewV1) and plan.projection_key == projection_key:
            matching_ready.append((record, plan))
    if not matching_ready:
        raise ApprovalError("committed plan evidence unavailable")
    ready_record, plan = matching_ready[-1]
    run_id = str(ready_record.event.get("run_id", ""))
    planner_run_id = str(ready_record.event.get("planner_run_id", ""))
    if not run_id or not planner_run_id:
        raise ApprovalError("committed plan evidence unavailable")
    if not journal.has_stream(f"run:{run_id}"):
        raise ApprovalError("committed plan evidence unavailable")
    run_records = await _read_all_records(journal, f"run:{run_id}")
    run_ready = next(
        (
            record
            for record in run_records
            if record.event.get("type") == "planner.decision_ready"
            and record.event.get("event_id") == ready_record.event.get("event_id")
        ),
        None,
    )
    session_finished = _successful_run_finished(session_records, run_id)
    run_finished = _successful_run_finished(run_records, run_id)
    if run_ready is None or session_finished is None or run_finished is None:
        raise ApprovalError("committed plan evidence unavailable")
    if run_ready.event != ready_record.event:
        raise ApprovalError("committed plan evidence conflict")
    if session_finished.event != run_finished.event:
        raise ApprovalError("committed plan evidence conflict")
    if session_finished.event_id != run_finished.event_id:
        raise ApprovalError("committed plan evidence conflict")
    if session_finished.seq <= ready_record.seq or run_finished.seq <= run_ready.seq:
        raise ApprovalError("committed plan evidence ordering invalid")
    try:
        payload = store.read_decision(session_id, plan.decision_id, plan.decision_version)
        decision = decode_exact_planner_decision(payload)
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
        raise ApprovalError("committed plan decision unavailable") from exc
    if (
        decision.content_digest != plan.decision_content_digest
        or decision.decision_id != plan.decision_id
        or decision.version != plan.decision_version
    ):
        raise ApprovalError("committed plan identity conflict")
    return CommittedPlanEvidence(
        session_id=session_id,
        top_level_run_id=run_id,
        planner_run_id=planner_run_id,
        decision_id=decision.decision_id,
        decision_version=decision.version,
        decision_content_digest=decision.content_digest,
        projection_key=plan.projection_key,
        projection_digest=plan.projection_digest,
        plan_ready_journal_event_id=ready_record.event_id,
        run_finished_journal_event_id=run_finished.event_id,
    )


class PlanCommitVerifier:
    # 绑定 store 与 generic journal，验证一个 projection 的 committed evidence
    def __init__(self, store: SessionStore, journal: Any) -> None:
        self._store = store
        self._journal = journal

    # 读取两条 durable stream 并返回 canonical cross-stream evidence
    async def verify(
        self,
        session_id: str,
        projection_key: str,
    ) -> CommittedPlanEvidence:
        return await verify_committed_plan(
            store=self._store,
            journal=self._journal,
            session_id=session_id,
            projection_key=projection_key,
        )


# 读取一个 stream 的全部 durable records，复用 EventJournal 的通用 replay primitive
async def _read_all_records(journal: Any, stream_id: str) -> tuple[Any, ...]:
    high_watermark = journal.high_watermark(stream_id)
    if high_watermark <= 0:
        return ()
    batch = await journal.read_replay(
        stream_id,
        after_seq=0,
        high_watermark=high_watermark,
    )
    return cast(tuple[Any, ...], batch.records)


# 从 stream 中取出指定 run 的 successful terminal record
def _successful_run_finished(records: tuple[Any, ...], run_id: str) -> Any | None:
    return next(
        (
            record
            for record in records
            if record.event.get("type") == "run.finished"
            and record.event.get("run_id") == run_id
            and record.event.get("status") == "success"
        ),
        None,
    )


# 校验或独立重建 derived committed receipt；绝不信任损坏 receipt 内容
async def materialize_committed_plan_receipt(
    *,
    store: SessionStore,
    journal: Any,
    session_id: str,
    projection_key: str,
) -> CommittedPlanReceipt:
    existing: CommittedPlanReceipt | None = None
    corrupt = False
    try:
        raw = store.read_committed_plan_receipt(session_id, projection_key)
        if raw is not None:
            existing = CommittedPlanReceipt.model_validate(raw)
            existing.verify_digest()
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
        corrupt = True
    evidence = await verify_committed_plan(
        store=store,
        journal=journal,
        session_id=session_id,
        projection_key=projection_key,
    )
    canonical = CommittedPlanReceipt.from_evidence(evidence)
    if existing is not None:
        if existing != canonical:
            corrupt = True
        else:
            return existing
    store.write_committed_plan_receipt(
        session_id,
        projection_key,
        canonical.model_dump(mode="json"),
        replace=corrupt,
    )
    if corrupt:
        logger.warning(
            "repaired derived committed plan receipt sid=%s projection_key=%s",
            session_id,
            projection_key,
        )
    return canonical


class ApprovalService:
    # 初始化 session-scoped approval authority service
    def __init__(self, store: SessionStore, journal: Any, bus: Any = None) -> None:
        self._store = store
        self._journal = journal
        self._bus = bus

    # 获取 committed projection 的 pending 或已解析 approval snapshot
    async def get_snapshot(self, session_id: str, projection_key: str) -> ApprovalSnapshot:
        try:
            raw = self._store.read_approval_record(session_id, projection_key)
        except ValueError as exc:
            raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
        existing: ApprovalRecord | None = None
        if raw is not None:
            try:
                existing = ApprovalRecord.model_validate(raw)
                existing.verify_digest()
            except (TypeError, ValueError) as exc:
                raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
        receipt = await materialize_committed_plan_receipt(
            store=self._store,
            journal=self._journal,
            session_id=session_id,
            projection_key=projection_key,
        )
        if existing is None:
            return ApprovalSnapshot(status="pending", projection_key=projection_key)
        try:
            _validate_record_binding(existing, receipt, session_id, projection_key)
        except ValueError as exc:
            raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
        return _snapshot_from_record(existing, receipt)

    # 在严格读取 corrupted record 后创建一次 immutable user approval
    async def resolve(
        self,
        *,
        session_id: str,
        projection_key: str,
        action: ApprovalAction,
        decision_id: str,
        decision_version: int,
        content_digest: str,
        commit_receipt_digest: str,
    ) -> ApprovalSnapshot:
        # 先读取 user authority；corrupt record 不能被 receipt 或普通响应绕过
        try:
            raw = self._store.read_approval_record(session_id, projection_key)
        except ValueError as exc:
            raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
        existing: ApprovalRecord | None = None
        if raw is not None:
            try:
                existing = ApprovalRecord.model_validate(raw)
                existing.verify_digest()
            except (TypeError, ValueError) as exc:
                raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
        receipt = await materialize_committed_plan_receipt(
            store=self._store,
            journal=self._journal,
            session_id=session_id,
            projection_key=projection_key,
        )
        if (
            receipt.decision_id != decision_id
            or receipt.decision_version != decision_version
            or receipt.decision_content_digest != content_digest
            or receipt.receipt_digest != commit_receipt_digest
        ):
            raise ApprovalError("approval-target-mismatch")
        if existing is not None:
            try:
                _validate_record_binding(existing, receipt, session_id, projection_key)
            except ValueError as exc:
                raise ApprovalRecordCorrupt("approval-record-corrupt") from exc
            if (
                existing.action == action
                and existing.content_digest == content_digest
                and existing.commit_receipt_digest == commit_receipt_digest
            ):
                return _snapshot_from_record(existing, receipt)
            raise ApprovalConflict("approval-already-resolved-conflict")
        record = ApprovalRecord.create(
            session_id=session_id,
            projection_key=projection_key,
            decision_id=decision_id,
            decision_version=decision_version,
            content_digest=content_digest,
            commit_receipt_digest=commit_receipt_digest,
            action=action,
        )
        self._store.write_approval_record(
            session_id,
            projection_key,
            record.model_dump(mode="json"),
        )
        await self._publish_change(record, receipt)
        return _snapshot_from_record(record, receipt)

    # 持久化成功后发布 approval 事件；通知失败不回滚 user authority
    async def _publish_change(
        self,
        record: ApprovalRecord,
        receipt: CommittedPlanReceipt,
    ) -> None:
        from kama_claude.core.bus.events import PlanApprovalChangedEvent

        event = PlanApprovalChangedEvent(
            event_id=f"plan-approval:{record.record_digest}",
            session_id=record.session_id,
            projection_key=record.projection_key,
            status="approved" if record.action == "approve" else "rejected",
            action=record.action,
            record_digest=record.record_digest,
            commit_receipt_digest=receipt.receipt_digest,
            ts=record.resolved_at,
        )
        try:
            if self._journal is not None:
                await self._journal.publish_required_durable(event)
            elif self._bus is not None:
                await self._bus.publish(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "approval notification failed sid=%s projection_key=%s",
                record.session_id,
                record.projection_key,
            )


# 从 immutable user record 构造客户端 snapshot
def _snapshot_from_record(
    record: ApprovalRecord,
    receipt: CommittedPlanReceipt,
) -> ApprovalSnapshot:
    return ApprovalSnapshot(
        status="approved" if record.action == "approve" else "rejected",
        projection_key=record.projection_key,
        record_digest=record.record_digest,
        action=record.action,
        commit_receipt_digest=receipt.receipt_digest,
    )


# 验证 user authority record 仍绑定当前 session、projection 与 committed receipt
def _validate_record_binding(
    record: ApprovalRecord,
    receipt: CommittedPlanReceipt,
    session_id: str,
    projection_key: str,
) -> None:
    if (
        record.session_id != session_id
        or record.projection_key != projection_key
        or record.decision_id != receipt.decision_id
        or record.decision_version != receipt.decision_version
        or record.content_digest != receipt.decision_content_digest
        or record.commit_receipt_digest != receipt.receipt_digest
    ):
        raise ValueError("approval record binding mismatch")
