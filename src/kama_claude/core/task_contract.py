from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

CoverageStatus = Literal["unresolved", "covered", "superseded"]
DirectiveClassification = Literal[
    "task_shaping",
    "prohibition",
    "requirement",
    "authorization",
    "steering",
    "unknown",
]

_SEMANTIC_NOOP_DIRECTIVES = frozenset(
    {
        "continue",
        "ok",
        "okay",
        "yes",
        "y",
        "yep",
        "go ahead",
        "proceed",
        "do it",
        "run tests",
        "run the tests",
        "please run tests",
        "please run the tests",
        "继续",
        "好的",
        "好",
        "执行测试",
        "运行测试",
    }
)


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 对语义 contract 字段做稳定编码并计算 digest
def contract_digest(
    goal: str,
    requirements: tuple[str, ...],
    constraints: tuple[str, ...],
    prohibitions: tuple[str, ...],
    acceptance_criteria: tuple[str, ...],
    authorization: tuple[str, ...],
) -> str:
    payload = {
        "goal": goal,
        "requirements": requirements,
        "constraints": constraints,
        "prohibitions": prohibitions,
        "acceptance_criteria": acceptance_criteria,
        "authorization": authorization,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 对用户 directive 原文计算 durable coverage 使用的精确摘要
def directive_text_digest(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


# 判断只推进当前工作而不改变 semantic TaskContract 的常见用户 steering
def is_semantic_noop_directive(raw_text: str) -> bool:
    return raw_text.strip().lower() in _SEMANTIC_NOOP_DIRECTIVES


@dataclass(frozen=True, slots=True)
class TaskContractRecord:
    """持久化的语义 contract；版本只随语义字段变化而增长。"""

    version: int
    digest: str
    goal: str = ""
    requirements: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    prohibitions: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    authorization: tuple[str, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    created_at: str = ""
    transaction_id: str = ""

    # 用语义字段构造 contract record，source IDs 仅表示字段 provenance
    @classmethod
    def create(
        cls,
        *,
        version: int,
        goal: str = "",
        requirements: tuple[str, ...] = (),
        constraints: tuple[str, ...] = (),
        prohibitions: tuple[str, ...] = (),
        acceptance_criteria: tuple[str, ...] = (),
        authorization: tuple[str, ...] = (),
        source_message_ids: tuple[str, ...] = (),
        transaction_id: str = "",
    ) -> TaskContractRecord:
        digest = contract_digest(
            goal,
            tuple(requirements),
            tuple(constraints),
            tuple(prohibitions),
            tuple(acceptance_criteria),
            tuple(authorization),
        )
        return cls(
            version=version,
            digest=digest,
            goal=goal,
            requirements=tuple(requirements),
            constraints=tuple(constraints),
            prohibitions=tuple(prohibitions),
            acceptance_criteria=tuple(acceptance_criteria),
            authorization=tuple(authorization),
            source_message_ids=tuple(source_message_ids),
            created_at=_now(),
            transaction_id=transaction_id,
        )

    # 从 durable task_contract record 恢复原始 digest/version 与 metadata
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskContractRecord:
        return cls(
            version=int(payload.get("version", 0)),
            digest=str(payload.get("digest", "")),
            goal=str(payload.get("goal", "")),
            requirements=tuple(str(item) for item in payload.get("requirements", ())),
            constraints=tuple(str(item) for item in payload.get("constraints", ())),
            prohibitions=tuple(str(item) for item in payload.get("prohibitions", ())),
            acceptance_criteria=tuple(
                str(item) for item in payload.get("acceptance_criteria", ())
            ),
            authorization=tuple(str(item) for item in payload.get("authorization", ())),
            source_message_ids=tuple(
                str(item) for item in payload.get("source_message_ids", ())
            ),
            created_at=str(payload.get("created_at", "")),
            transaction_id=str(payload.get("transaction_id", "")),
        )

    # 从候选字段生成新 record；语义不变时保留版本和字段 provenance
    def with_semantic_state(self, **changes: Any) -> TaskContractRecord:
        values: dict[str, Any] = {
            "goal": self.goal,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "prohibitions": self.prohibitions,
            "acceptance_criteria": self.acceptance_criteria,
            "authorization": self.authorization,
        }
        values.update(
            {
                key: tuple(value) if key != "goal" else value
                for key, value in changes.items()
                if key in values
            }
        )
        next_digest = contract_digest(
            values["goal"],
            values["requirements"],
            values["constraints"],
            values["prohibitions"],
            values["acceptance_criteria"],
            values["authorization"],
        )
        semantic_changed = next_digest != self.digest
        return TaskContractRecord(
            version=self.version + 1 if semantic_changed else self.version,
            digest=next_digest,
            goal=values["goal"],
            requirements=values["requirements"],
            constraints=values["constraints"],
            prohibitions=values["prohibitions"],
            acceptance_criteria=values["acceptance_criteria"],
            authorization=values["authorization"],
            source_message_ids=(
                tuple(changes["source_message_ids"])
                if semantic_changed and "source_message_ids" in changes
                else self.source_message_ids
            ),
            created_at=_now() if semantic_changed else self.created_at,
            transaction_id=(
                str(changes.get("transaction_id", ""))
                if semantic_changed
                else self.transaction_id
            ),
        )

    # 生成不包含 persistence metadata 的 canonical model-visible contract
    def model_view(self) -> TaskContractModelView:
        return TaskContractModelView.from_record(self)


@dataclass(frozen=True, slots=True)
class TaskContractModelView:
    """给模型看的简洁 contract，不暴露版本、digest 和 transaction metadata。"""

    goal: str
    requirements: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    prohibitions: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    authorization: tuple[str, ...] = ()

    # 从持久化 record 提取纯语义字段
    @classmethod
    def from_record(cls, record: TaskContractRecord) -> TaskContractModelView:
        return cls(
            goal=record.goal,
            requirements=record.requirements,
            constraints=record.constraints,
            prohibitions=record.prohibitions,
            acceptance_criteria=record.acceptance_criteria,
            authorization=record.authorization,
        )

    # 将 contract 以稳定且简洁的文本渲染给模型
    def render(self) -> str:
        lines = ["## Current Task Contract", f"Goal: {self.goal}"]
        sections = (
            ("Requirements", self.requirements),
            ("Constraints", self.constraints),
            ("Prohibitions", self.prohibitions),
            ("Acceptance Criteria", self.acceptance_criteria),
            ("Authorization", self.authorization),
        )
        for title, values in sections:
            if values:
                lines.append(f"{title}:")
                lines.extend(f"- {value}" for value in values)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DirectiveCoverageRecord:
    """单条用户消息的 durable semantic coverage 状态。"""

    message_id: str
    semantic_contract_digest: str
    classification: DirectiveClassification
    exact_text_digest: str
    coverage_status: CoverageStatus
    covered_at: str | None = None

    # 创建已覆盖消息的独立 coverage record，不增加 contract semantic version
    @classmethod
    def covered(
        cls,
        *,
        message_id: str,
        semantic_contract_digest: str,
        classification: DirectiveClassification,
        exact_text: str,
    ) -> DirectiveCoverageRecord:
        return cls(
            message_id=message_id,
            semantic_contract_digest=semantic_contract_digest,
            classification=classification,
            exact_text_digest=directive_text_digest(exact_text),
            coverage_status="covered",
            covered_at=_now(),
        )

    # 创建尚未被 updater 覆盖的消息记录
    @classmethod
    def unresolved(
        cls,
        *,
        message_id: str,
        semantic_contract_digest: str,
        classification: DirectiveClassification,
        exact_text: str,
    ) -> DirectiveCoverageRecord:
        return cls(
            message_id=message_id,
            semantic_contract_digest=semantic_contract_digest,
            classification=classification,
            exact_text_digest=directive_text_digest(exact_text),
            coverage_status="unresolved",
        )

    # 将 durable coverage record 序列化，避免通过 TaskContract provenance 隐式表示覆盖关系
    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "semantic_contract_digest": self.semantic_contract_digest,
            "classification": self.classification,
            "exact_text_digest": self.exact_text_digest,
            "coverage_status": self.coverage_status,
            "covered_at": self.covered_at,
        }

    # 从 append-only coverage record 恢复状态，未知字段不进入模型视图
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DirectiveCoverageRecord:
        return cls(
            message_id=str(payload.get("message_id", "")),
            semantic_contract_digest=str(payload.get("semantic_contract_digest", "")),
            classification=cast(
                DirectiveClassification, payload.get("classification", "unknown")
            ),
            exact_text_digest=str(payload.get("exact_text_digest", "")),
            coverage_status=cast(
                CoverageStatus, payload.get("coverage_status", "unresolved")
            ),
            covered_at=(str(payload["covered_at"]) if payload.get("covered_at") else None),
        )


@dataclass(frozen=True, slots=True)
class PendingDirectiveOverlay:
    """有序、原文保留且不可压缩的用户 directive overlay。"""

    message_id: str
    original_order: int
    raw_text: str
    classification: DirectiveClassification
    status: CoverageStatus = "unresolved"


@dataclass
class PendingDirectiveSet:
    """自上次 reconciliation watermark 以来所有未解决 directive 的集合。"""

    reconciliation_watermark: str = ""
    overlays: list[PendingDirectiveOverlay] = field(default_factory=list)

    # 按用户原始顺序追加 directive，不覆盖此前 unresolved overlay
    def add(
        self,
        message_id: str,
        raw_text: str,
        classification: DirectiveClassification = "unknown",
        *,
        original_order: int | None = None,
    ) -> PendingDirectiveOverlay:
        existing = next((item for item in self.overlays if item.message_id == message_id), None)
        if existing is not None:
            return existing
        overlay = PendingDirectiveOverlay(
            message_id=message_id,
            original_order=(
                max((item.original_order for item in self.overlays), default=-1) + 1
                if original_order is None
                else original_order
            ),
            raw_text=raw_text,
            classification=classification,
        )
        self.overlays.append(overlay)
        return overlay

    # 返回仍需保护的 unresolved overlays，保持原始用户顺序
    def unresolved(self) -> list[PendingDirectiveOverlay]:
        return sorted(
            (item for item in self.overlays if item.status == "unresolved"),
            key=lambda item: item.original_order,
        )

    # 返回 ordered overlays 的模型可见 metadata 与原始 directive 文本
    def render(self) -> list[PendingDirectiveOverlay]:
        return sorted(self.unresolved(), key=lambda item: item.original_order)

    # 通过显式成功的单条 reconciliation 移除 directive
    def reconcile(self, message_id: str, *, watermark: str | None = None) -> bool:
        for index, item in enumerate(self.overlays):
            if item.message_id == message_id and item.status == "unresolved":
                self.overlays[index] = PendingDirectiveOverlay(
                    message_id=item.message_id,
                    original_order=item.original_order,
                    raw_text=item.raw_text,
                    classification=item.classification,
                    status="covered",
                )
                if watermark is not None:
                    self.reconciliation_watermark = watermark
                return True
        return False

    # 返回待保护的用户消息 ID 集合，供 compaction range 过滤
    def protected_message_ids(self) -> frozenset[str]:
        return frozenset(item.message_id for item in self.unresolved())

    # 计算 ordered pending state 的稳定 digest，供 surface CAS 捕获无文本变更
    def digest(self) -> str:
        payload = {
            "watermark": self.reconciliation_watermark,
            "overlays": [
                {
                    "message_id": item.message_id,
                    "original_order": item.original_order,
                    "raw_text": item.raw_text,
                    "classification": item.classification,
                    "status": item.status,
                }
                for item in sorted(self.overlays, key=lambda value: value.original_order)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


class TaskContractState:
    """运行时 contract reducer，处理语义 no-op、coverage 与 updater failure。"""

    # 初始化当前 contract、pending overlays 与 coverage 索引
    def __init__(self, contract: TaskContractRecord | None = None) -> None:
        self.contract = contract
        self.pending = PendingDirectiveSet()
        self.coverage: dict[str, DirectiveCoverageRecord] = {}

    # 评估一条用户消息；updater failure 时保留 ordered verbatim overlay 并遮蔽旧 action
    def evaluate_message(
        self,
        *,
        message_id: str,
        raw_text: str,
        updater: Any,
        classification: DirectiveClassification = "task_shaping",
    ) -> TaskContractRecord | None:
        previous = self.contract
        try:
            candidate = updater(previous, raw_text)
        except Exception:
            digest = previous.digest if previous is not None else ""
            coverage = DirectiveCoverageRecord.unresolved(
                message_id=message_id,
                semantic_contract_digest=digest,
                classification=classification,
                exact_text=raw_text,
            )
            self.coverage[message_id] = coverage
            self.pending.add(message_id, raw_text, classification)
            return previous

        if candidate is None:
            digest = previous.digest if previous is not None else ""
            # A declined candidate is only a durable no-op for explicit
            # steering/acknowledgement text.  For task-shaping or unknown
            # directives it means reconciliation is unresolved, so preserve
            # the raw directive and keep older action fields masked.
            if classification == "steering" or is_semantic_noop_directive(raw_text):
                coverage = DirectiveCoverageRecord.covered(
                    message_id=message_id,
                    semantic_contract_digest=digest,
                    classification=classification,
                    exact_text=raw_text,
                )
                self.pending.reconcile(message_id)
            else:
                coverage = DirectiveCoverageRecord.unresolved(
                    message_id=message_id,
                    semantic_contract_digest=digest,
                    classification=classification,
                    exact_text=raw_text,
                )
                self.pending.add(message_id, raw_text, classification)
            self.coverage[message_id] = coverage
            return previous

        if not isinstance(candidate, TaskContractRecord):
            raise TypeError("contract updater must return TaskContractRecord or None")
        if previous is not None and candidate.digest == previous.digest:
            self.coverage[message_id] = DirectiveCoverageRecord.covered(
                message_id=message_id,
                semantic_contract_digest=previous.digest,
                classification=classification,
                exact_text=raw_text,
            )
            self.pending.reconcile(message_id)
            return previous

        self.contract = candidate
        self.coverage[message_id] = DirectiveCoverageRecord.covered(
            message_id=message_id,
            semantic_contract_digest=candidate.digest,
            classification=classification,
            exact_text=raw_text,
        )
        self.pending.reconcile(message_id)
        return candidate

    # 判断一条消息是否已被 durable coverage 覆盖且不在 unresolved watermark 之前
    def is_compactable(self, message_id: str) -> bool:
        coverage = self.coverage.get(message_id)
        return coverage is not None and coverage.coverage_status in {"covered", "superseded"} and (
            message_id not in self.pending.protected_message_ids()
        )

    # 返回当前模型可见 authority framing；pending raw directives 永远优先旧 contract
    def render_model_view(self) -> str:
        checkpoint: CheckpointProjection | None = None
        return render_authority_framing(self.contract, self.pending, checkpoint)


# 用确定性字段映射完成最小 contract evaluation，避免普通消息永久阻塞 compaction
def apply_directive_to_contract(
    previous: TaskContractRecord | None,
    *,
    message_id: str,
    raw_text: str,
    classification: DirectiveClassification,
) -> TaskContractRecord | None:
    text = raw_text.strip()
    if not text:
        return previous
    if classification == "steering" or is_semantic_noop_directive(text):
        return previous
    if _is_ambiguous_correction(text):
        # A sentence that both retracts/restricts an old action and introduces
        # a replacement cannot be safely represented by additive fields alone.
        # Keep it unresolved so the raw directive remains higher authority.
        return None
    if previous is None:
        if classification == "prohibition":
            return TaskContractRecord.create(
                version=1,
                goal=text,
                prohibitions=(text,),
                source_message_ids=(message_id,),
                transaction_id=f"contract-{message_id}",
            )
        if classification in {"requirement", "authorization", "task_shaping"}:
            values: dict[str, Any] = {
                "version": 1,
                "goal": text,
                "source_message_ids": (message_id,),
                "transaction_id": f"contract-{message_id}",
            }
            if classification == "authorization":
                values["authorization"] = (text,)
            else:
                values["requirements"] = (text,)
            return TaskContractRecord.create(**values)
        return None

    if classification == "prohibition":
        if text in previous.prohibitions:
            return previous
        return previous.with_semantic_state(
            prohibitions=(*previous.prohibitions, text),
            source_message_ids=(*previous.source_message_ids, message_id),
            transaction_id=f"contract-{message_id}",
        )
    if classification == "authorization":
        if text in previous.authorization:
            return previous
        return previous.with_semantic_state(
            authorization=(*previous.authorization, text),
            source_message_ids=(*previous.source_message_ids, message_id),
            transaction_id=f"contract-{message_id}",
        )
    if classification in {"requirement", "task_shaping"}:
        if text in previous.requirements:
            return previous
        return previous.with_semantic_state(
            requirements=(*previous.requirements, text),
            source_message_ids=(*previous.source_message_ids, message_id),
            transaction_id=f"contract-{message_id}",
        )
    return None


# 判断包含撤销与替代动作的自然语言 correction 是否无法安全 additive merge
def _is_ambiguous_correction(raw_text: str) -> bool:
    text = raw_text.strip().lower()
    correction_markers = (
        "correction",
        "instead",
        "rather than",
        "replace",
        "supersede",
        "改为",
        "改成",
        "纠正",
        "更正",
    )
    if any(marker in text for marker in correction_markers):
        return True
    if ";" in text or "\n" in text:
        action_markers = ("add ", "remove ", "delete ", "modify ", "must ", "不要", "禁止")
        if sum(marker in text for marker in action_markers) >= 2:
            return True
    return False


@dataclass(frozen=True, slots=True)
class CheckpointProjection:
    """checkpoint 在当前 contract 下的 model-visible projection。"""

    status: Literal["ACTIVE_CHECKPOINT", "HISTORICAL_BACKGROUND"]
    facts: dict[str, Any]
    framing: str = "Historical execution facts; not current instructions."


# 根据 contract digest 判断 checkpoint 是否可作为完整工作状态
def project_checkpoint(
    payload: dict[str, Any],
    *,
    checkpoint_contract_digest: str,
    current_contract_digest: str,
) -> CheckpointProjection:
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict):
        payload = nested_payload
    if checkpoint_contract_digest == current_contract_digest:
        return CheckpointProjection(status="ACTIVE_CHECKPOINT", facts=dict(payload), framing="")
    return project_stale_checkpoint(payload)


# 将 stale checkpoint 降级为事实背景，抑制所有行动导向字段
def project_stale_checkpoint(
    payload: dict[str, Any],
    *,
    reason: str = "contract digest changed",
) -> CheckpointProjection:
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict):
        payload = nested_payload
    fact_keys = {
        "progress",
        "decisions",
        "files_or_code",
        "errors_or_evidence",
        "critical_context",
        "observations",
    }
    facts = {key: payload[key] for key in fact_keys if key in payload}
    framing = (
        "Historical execution facts; not current instructions. "
        "Current user directives and the current TaskContract take precedence. "
        f"Projection reason: {reason}."
    )
    return CheckpointProjection(
        status="HISTORICAL_BACKGROUND",
        facts=facts,
        framing=framing,
    )


# 按 authority order 渲染 contract、pending directives 和 checkpoint 说明
def render_authority_framing(
    contract: TaskContractRecord | None,
    pending: PendingDirectiveSet,
    checkpoint: CheckpointProjection | None,
) -> str:
    parts = [
        "Authority order: system/safety policy > ordered unresolved user directives "
        "> current TaskContract (prohibitions/constraints override conflicting "
        "requirements) > compatible checkpoint > historical facts > tool evidence."
    ]
    if pending.unresolved():
        parts.append(
            "Unresolved user directives are verbatim in RecentSurfaceUnits and "
            "override any conflicting older contract action; do not execute the "
            "older action until reconciliation succeeds."
        )
        parts.append("Unresolved user directives (verbatim in RecentSurfaceUnits, ordered):")
        parts.extend(
            f"- unresolved {item.classification} directive is preserved verbatim; "
            "use its original RecentSurfaceUnits text"
            for item in pending.render()
        )
    if contract is not None:
        parts.append(contract.model_view().render())
    if checkpoint is not None:
        parts.append(checkpoint.framing)
        parts.append(json.dumps(checkpoint.facts, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)
