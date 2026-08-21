from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

PLAN_VIEW_MAX_BYTES = 256 * 1024
MAX_SECTION_BYTES = 32 * 1024
MAX_LIST_ITEMS = 64
MAX_EXPLANATORY_TEXT_CHARS = 4096
MAX_LIST_ITEM_CHARS = 1024

_IDENTIFIER_LIST_SECTIONS = {
    "files_to_modify",
    "files_to_create",
    "allowed_capabilities",
    "dependency_changes",
    "protocol_or_schema_changes",
}
_TEXT_LIST_SECTIONS = {
    "existing_patterns_reused",
    "non_goals",
    "assumptions",
    "unresolved_questions",
}
_STRUCTURED_LIST_SECTIONS = {
    "requirements",
    "intended_changes",
    "verification_plan",
}


class LegacyPlanViewV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    plan_key: str
    goal: str
    architecture_mode: str = "preserve"
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    selected_approach: str
    intended_changes: list[dict[str, Any]] = Field(default_factory=list)
    existing_patterns_reused: list[str] = Field(default_factory=list)
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    dependency_changes: list[str] = Field(default_factory=list)
    protocol_or_schema_changes: list[str] = Field(default_factory=list)
    verification_plan: list[dict[str, Any]] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    section_budgets: dict[str, int] = Field(default_factory=dict)
    content_digest: str = ""
    ts: str = ""


class PlanViewV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    decision_key: str = Field(min_length=1)
    projection_key: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    decision_version: int = Field(ge=1)
    decision_content_digest: str = Field(min_length=1)
    architecture_slice_id: str = Field(min_length=1)
    architecture_slice_version: int = Field(ge=1)
    architecture_slice_content_digest: str = Field(min_length=1)
    projection_digest: str = Field(min_length=1)
    snapshot_digest: str = Field(min_length=1)
    goal: str
    architecture_mode: str
    selected_approach: str
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    existing_patterns_reused: list[str] = Field(default_factory=list)
    intended_changes: list[dict[str, Any]] = Field(default_factory=list)
    files_to_modify: list[str] = Field(default_factory=list)
    files_to_create: list[str] = Field(default_factory=list)
    allowed_capabilities: list[str] = Field(default_factory=list)
    dependency_changes: list[str] = Field(default_factory=list)
    protocol_or_schema_changes: list[str] = Field(default_factory=list)
    verification_plan: list[dict[str, Any]] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    requires_user_approval: Literal[True] = True
    execution_available: Literal[False] = False
    section_limits: dict[str, int] = Field(default_factory=dict)
    omitted_counts: dict[str, int] = Field(default_factory=dict)
    truncated_sections: list[str] = Field(default_factory=list)

    # 保留旧测试与内部调用读取 section_budgets 的兼容只读视图
    @property
    def section_budgets(self) -> dict[str, int]:
        return {
            "requirements": len(self.requirements),
            "intended_changes": len(self.intended_changes),
            "verification_plan": len(self.verification_plan),
        }


# 保留旧导入名；PlanView 现在代表 active V1 projection
PlanView = PlanViewV1
PlanViewRecord = LegacyPlanViewV0 | PlanViewV1


# 使用稳定 JSON 编码计算 projection digest，明确排除 digest 自身
def projection_digest(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("projection_digest", None)
    # projection key 只标识展示实例，不参与内容摘要
    content.pop("projection_key", None)
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 将旧/新 PlanView payload 在 persistence/history boundary 做显式版本分发
def decode_plan_view_record(payload: object) -> PlanViewRecord:
    if isinstance(payload, LegacyPlanViewV0):
        return payload
    if isinstance(payload, PlanViewV1):
        if projection_digest(payload.model_dump(mode="json")) != payload.projection_digest:
            raise ValueError("PlanView projection digest mismatch")
        return payload
    if not isinstance(payload, dict):
        raise ValueError("PlanView payload must be an object")
    schema_version = payload.get("schema_version")
    markers = {
        key
        for key in (
            "decision_key",
            "projection_key",
            "decision_content_digest",
            "projection_digest",
        )
        if key in payload
    }
    if markers:
        if schema_version != 1 or markers != {
            "decision_key",
            "projection_key",
            "decision_content_digest",
            "projection_digest",
        }:
            raise ValueError("invalid PlanView schema marker")
        record = PlanViewV1.model_validate(payload)
        if projection_digest(payload) != record.projection_digest:
            raise ValueError("PlanView projection digest mismatch")
        return record
    if schema_version not in (None, 1):
        raise ValueError("unknown PlanView schema version")
    return LegacyPlanViewV0.model_validate(payload)


# 让 Pydantic Event adapter 在 strict replay 中复用显式 legacy/V1 decoder
def _decode_plan_view_before_validation(value: object) -> PlanViewRecord:
    return decode_plan_view_record(value)


PlanViewEventValue = Annotated[
    PlanViewRecord,
    BeforeValidator(_decode_plan_view_before_validation),
]


# 对解释性文本做确定性截断，并记录发生截断的 section
def _bound_text(value: str, section: str, truncated: set[str]) -> str:
    if len(value) <= MAX_EXPLANATORY_TEXT_CHARS:
        return value
    truncated.add(section)
    return value[:MAX_EXPLANATORY_TEXT_CHARS]


# 复制并约束结构化 list item，identifier 超限时整项省略
def _bound_structured_item(
    value: dict[str, Any],
    section: str,
    truncated: set[str],
) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    identifier_keys = {
        "requirement_id",
        "change_id",
        "requirement_ids",
        "target_paths",
        "evidence_refs",
    }
    for key, item in value.items():
        if isinstance(item, str):
            if key in identifier_keys:
                if len(item) > MAX_LIST_ITEM_CHARS:
                    return None
                result[key] = item
            else:
                result[key] = _bound_text(item, section, truncated)
        elif isinstance(item, list):
            normalized: list[Any] = []
            for nested in item:
                if isinstance(nested, str) and key in identifier_keys:
                    if len(nested) > MAX_LIST_ITEM_CHARS:
                        return None
                    normalized.append(nested)
                elif isinstance(nested, str):
                    normalized.append(_bound_text(nested, section, truncated))
                else:
                    normalized.append(nested)
            result[key] = normalized
        else:
            result[key] = item
    return result


# 约束单个 projection section 的 item 数量、item bytes 和 identifier 语义
def _bound_section(
    section: str,
    values: list[Any],
    truncated: set[str],
) -> tuple[list[Any], int]:
    omitted = 0
    bounded: list[Any] = []
    for value in values[:MAX_LIST_ITEMS]:
        if section in _IDENTIFIER_LIST_SECTIONS:
            if not isinstance(value, str) or len(value) > MAX_LIST_ITEM_CHARS:
                omitted += 1
                continue
            bounded.append(value)
        elif section in _TEXT_LIST_SECTIONS:
            if not isinstance(value, str):
                omitted += 1
                continue
            bounded.append(_bound_text(value, section, truncated))
        else:
            if not isinstance(value, dict):
                omitted += 1
                continue
            item = _bound_structured_item(value, section, truncated)
            if item is None:
                omitted += 1
                continue
            bounded.append(item)
    omitted += max(0, len(values) - MAX_LIST_ITEMS)
    while len(json.dumps(bounded, ensure_ascii=False).encode("utf-8")) > MAX_SECTION_BYTES:
        if not bounded:
            raise ValueError("plan-view-too-large")
        bounded.pop()
        omitted += 1
    return bounded, omitted


# 从未带 digest 的 payload 生成固定 bounds、omission metadata 和 V1 digest
def finalize_plan_view_payload(payload: dict[str, Any]) -> PlanViewV1:
    bounded = dict(payload)
    truncated: set[str] = set()
    omitted_counts: dict[str, int] = {}
    for section in (
        *_IDENTIFIER_LIST_SECTIONS,
        *_TEXT_LIST_SECTIONS,
        *_STRUCTURED_LIST_SECTIONS,
    ):
        values = bounded.get(section, [])
        if not isinstance(values, list):
            raise ValueError(f"plan-view-invalid-section:{section}")
        bounded[section], omitted = _bound_section(section, values, truncated)
        if omitted:
            omitted_counts[section] = omitted

    for field in ("goal", "selected_approach"):
        value = bounded.get(field)
        if not isinstance(value, str):
            raise ValueError(f"plan-view-invalid-field:{field}")
        bounded[field] = _bound_text(value, field, truncated)

    identity_fields = (
        "decision_key",
        "projection_key",
        "decision_id",
        "decision_content_digest",
        "architecture_slice_id",
        "architecture_slice_content_digest",
        "snapshot_digest",
    )
    for field in identity_fields:
        value = bounded.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"plan-view-invalid-identity:{field}")
        if len(value) > MAX_LIST_ITEM_CHARS:
            raise ValueError("plan-view-too-large")
    if not isinstance(bounded.get("architecture_slice_version"), int):
        raise ValueError("plan-view-invalid-identity:architecture_slice_version")

    bounded["requires_user_approval"] = True
    bounded["execution_available"] = False
    bounded["section_limits"] = {
        "total_bytes": PLAN_VIEW_MAX_BYTES,
        "section_bytes": MAX_SECTION_BYTES,
        "list_items": MAX_LIST_ITEMS,
        "explanatory_text_chars": MAX_EXPLANATORY_TEXT_CHARS,
        "list_item_chars": MAX_LIST_ITEM_CHARS,
    }
    bounded["section_limits"].update(
        {
            section: MAX_SECTION_BYTES
            for section in (
                *_IDENTIFIER_LIST_SECTIONS,
                *_TEXT_LIST_SECTIONS,
                *_STRUCTURED_LIST_SECTIONS,
            )
        }
    )
    bounded["omitted_counts"] = omitted_counts
    bounded["truncated_sections"] = sorted(truncated)
    bounded["projection_digest"] = projection_digest(bounded)
    encoded = json.dumps(bounded, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(encoded) > PLAN_VIEW_MAX_BYTES:
        raise ValueError("plan-view-too-large")
    return PlanViewV1.model_validate(bounded)


# 将 PlanReady candidate 在对应 run 成功后提交为一次用户可见 projection
class PlanReadyCommitReducer:
    # 初始化 event/projection 去重与 candidate 状态
    def __init__(self) -> None:
        self._seen_event_ids: set[str] = set()
        self._event_digests: dict[str, str] = {}
        self._pending: dict[str, tuple[str, str, PlanViewRecord]] = {}
        self._committed: dict[str, str] = {}
        self._run_status: dict[str, str] = {}
        self.warnings: list[str] = []

    # 摄取 PlanReady 或 run.finished，并只返回新 committed PlanView
    def ingest(self, event: dict[str, Any]) -> list[PlanViewRecord]:
        event_type = event.get("type")
        if event_type == "planner.decision_ready":
            event_id = str(event.get("event_id", ""))
            plan = decode_plan_view_record(event.get("plan"))
            key = (
                plan.projection_key
                if isinstance(plan, PlanViewV1)
                else f"legacy:{event_id}"
            )
            digest = (
                plan.projection_digest
                if isinstance(plan, PlanViewV1)
                else plan.content_digest
            )
            if event_id and event_id in self._seen_event_ids:
                if self._event_digests.get(event_id) != digest:
                    self.warnings.append("plan-projection-integrity-conflict")
                return []
            if event_id:
                self._seen_event_ids.add(event_id)
                self._event_digests[event_id] = digest
            existing = self._committed.get(key)
            if existing is None and key in self._pending:
                existing = self._pending[key][1]
            if existing is not None:
                if existing != digest:
                    self.warnings.append("plan-projection-integrity-conflict")
                return []
            run_id = str(event.get("run_id", ""))
            status = self._run_status.get(run_id)
            if status in {"failed", "cancelled"}:
                return []
            if status == "success":
                self._committed[key] = digest
                return [plan]
            self._pending[key] = (run_id, digest, plan)
            return []

        if event_type != "run.finished":
            return []
        run_id = str(event.get("run_id", ""))
        status = str(event.get("status", ""))
        self._run_status[run_id] = status
        committed: list[PlanViewRecord] = []
        for key, (candidate_run, digest, plan) in list(self._pending.items()):
            if candidate_run != run_id:
                continue
            del self._pending[key]
            if status == "success":
                existing = self._committed.get(key)
                if existing is not None and existing != digest:
                    self.warnings.append("plan-projection-integrity-conflict")
                    continue
                if existing is None:
                    self._committed[key] = digest
                    committed.append(plan)
        return committed
