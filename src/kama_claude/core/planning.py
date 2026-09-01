from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from kama_claude.core.compact.budget import TOOL_RESULT_LIMIT
from kama_claude.core.grounding import (
    ArchitectureSlice,
    RepositorySnapshot,
    SnapshotBuilder,
    canonical_digest,
)
from kama_claude.core.plan_view import PlanViewV1, finalize_plan_view_payload
from kama_claude.core.session.store import SessionStore
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_LOGGER = logging.getLogger(__name__)

ArchitectureMode = Literal["preserve", "extend", "refactor"]

_MANIFEST_NAMES = {
    "Cargo.lock",
    "Cargo.toml",
    "Gemfile",
    "Gemfile.lock",
    "go.mod",
    "go.sum",
    "package-lock.json",
    "package.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pyproject.toml",
    "uv.lock",
    "yarn.lock",
}

_PLANNER_FAILURE_REASONS = frozenset(
    {
        "missing-terminal-decision",
        "planning-grounding-missing",
        "planning-input-incomplete",
        "artifact-corrupt",
        "snapshot-stale",
        "goal-mismatch",
        "terminal-contract-violation",
        "session/workspace-mismatch",
        "plan-event-append-failed",
        "planner-result-too-large",
        "projection-incomplete",
    }
)


# 将 Planner contract failure 归一化为不泄漏模型文本或路径的稳定摘要
def planner_failure_message(reason: str) -> str:
    aliases = {
        "session/workspace mismatch": "session/workspace-mismatch",
        "session-workspace-mismatch": "session/workspace-mismatch",
        "session_workspace_mismatch": "session/workspace-mismatch",
    }
    stable_reason = aliases.get(reason, reason)
    if stable_reason not in _PLANNER_FAILURE_REASONS:
        stable_reason = "planner-contract-failure"
    return (
        "Planner failed to produce a valid terminal PlannerDecision:\n"
        + stable_reason
    )


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    statement: str
    required: bool = True


class IntendedChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_id: str
    description: str
    requirement_ids: tuple[str, ...]
    target_paths: tuple[str, ...]
    evidence_refs: tuple[str, ...]


class VerificationStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_ids: tuple[str, ...]
    strategy: str


class PlannerDecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str | None = None
    requirements: tuple[Requirement, ...]
    architecture_slice_id: str
    architecture_slice_version: int
    architecture_mode: ArchitectureMode
    selected_approach: str
    existing_patterns_reused: tuple[str, ...] = ()
    intended_changes: tuple[IntendedChange, ...]
    files_to_modify: tuple[str, ...]
    files_to_create: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...]
    dependency_changes: tuple[str, ...] = ()
    protocol_or_schema_changes: tuple[str, ...] = ()
    verification_plan: tuple[VerificationStrategy, ...]
    non_goals: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    requires_user_approval: bool = True


# 保留旧导入名；PlannerDecisionDraftV2 是当前 active submit contract 的语义别名
PlannerDecisionDraftV2 = PlannerDecisionDraft


class LegacyPlannerDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    version: int
    goal: str
    requirements: tuple[Requirement, ...]
    architecture_slice_id: str
    snapshot_digest: str
    architecture_mode: ArchitectureMode
    selected_approach: str
    existing_patterns_reused: tuple[str, ...]
    intended_changes: tuple[IntendedChange, ...]
    files_to_modify: tuple[str, ...]
    files_to_create: tuple[str, ...]
    allowed_capabilities: tuple[str, ...]
    dependency_changes: tuple[str, ...]
    protocol_or_schema_changes: tuple[str, ...]
    verification_plan: tuple[VerificationStrategy, ...]
    non_goals: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    requires_user_approval: bool
    content_digest: str


class ExactPlannerDecisionV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    decision_id: str
    version: int
    goal: str
    requirements: tuple[Requirement, ...]
    architecture_slice_id: str
    architecture_slice_version: int
    architecture_slice_content_digest: str
    snapshot_digest: str
    architecture_mode: ArchitectureMode
    selected_approach: str
    existing_patterns_reused: tuple[str, ...]
    intended_changes: tuple[IntendedChange, ...]
    files_to_modify: tuple[str, ...]
    files_to_create: tuple[str, ...]
    allowed_capabilities: tuple[str, ...]
    dependency_changes: tuple[str, ...]
    protocol_or_schema_changes: tuple[str, ...]
    verification_plan: tuple[VerificationStrategy, ...]
    non_goals: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    requires_user_approval: bool
    content_digest: str


PlannerDecisionRecord = LegacyPlannerDecisionV1 | ExactPlannerDecisionV2
# 新 Planner runtime 只处理 exact-bound V2；legacy union 留在 persistence/history 边界
PlannerDecision = ExactPlannerDecisionV2
# 保留旧导入名；active runtime 的 PlanView 仅指向当前 V1 projection
PlanView = PlanViewV1


# 依据 schema marker 在 persistence/history boundary 解码一个 immutable record
def decode_planner_decision_record(payload: dict[str, Any]) -> PlannerDecisionRecord:
    if payload.get("schema_version") == 2:
        record = ExactPlannerDecisionV2.model_validate(payload)
        if _exact_content_digest(payload) != record.content_digest:
            raise ValueError("exact PlannerDecision content digest mismatch")
        return record
    legacy_record = LegacyPlannerDecisionV1.model_validate(payload)
    if _legacy_content_digest(payload) != legacy_record.content_digest:
        raise ValueError("legacy PlannerDecision content digest mismatch")
    return legacy_record


# 在 persistence boundary 解码并收窄 active runtime 允许的 exact V2 record
def decode_exact_planner_decision(payload: dict[str, Any]) -> ExactPlannerDecisionV2:
    record = decode_planner_decision_record(payload)
    if not isinstance(record, ExactPlannerDecisionV2):
        raise ValueError("legacy decision is not terminal-runtime eligible")
    return record


# 使用 V1 原始 payload contract 计算 legacy domain digest
def _legacy_content_digest(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("content_digest", None)
    return canonical_digest(content)


# 使用 V2 exact payload contract 计算当前 domain digest
def _exact_content_digest(payload: dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("content_digest", None)
    return canonical_digest(content)


class PlannerValidationError(ValueError):
    # 构造带稳定 validation category 的拒绝异常
    def __init__(
        self,
        category: str,
        message: str,
        *,
        incomplete: bool = False,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.incomplete = incomplete
        self.code = code or category


@dataclass(frozen=True)
class SubmittedDecisionIdentity:
    decision_id: str
    version: int
    snapshot_digest: str
    content_digest: str


class PlannerDecisionService:
    # 绑定一个 session 的 workspace、goal 与 immutable planning store
    def __init__(
        self,
        *,
        workspace_root: Path,
        session_id: str,
        store: SessionStore,
        goal: str,
        run_id: str = "",
    ) -> None:
        self._resolver = WorkspacePathResolver(workspace_root)
        self._root = self._resolver.root
        self._access_policy = WorkspaceAccessPolicy(self._root)
        self._session_id = session_id
        self._store = store
        self._goal = goal
        self._run_id = run_id
        self.last_validation: dict[str, bool] | None = None
        self._last_submit_outcome = "none"
        self._last_incomplete_reason: str | None = None
        self._last_submit_category: str | None = None
        self._terminal_decision: SubmittedDecisionIdentity | None = None
        self._terminal_draft_digest: str | None = None
        self._missing_grounding_submit_count = 0
        self._invocation_terminal_reason: str | None = None

    # 返回本次 Planner run 是否已经提交 terminal decision
    @property
    def is_terminal_committed(self) -> bool:
        return self._terminal_decision is not None

    # 返回本次 Planner run 的 immutable terminal identity
    @property
    def terminal_decision(self) -> SubmittedDecisionIdentity | None:
        return self._terminal_decision

    # 读取当前 run 已提交的 immutable decision，供 PlanReady projection 使用
    def read_terminal_decision(self) -> PlannerDecision:
        return self._read_terminal_decision()

    # 将已验证 decision 转成受预算约束的结构化 PlanView
    def build_plan_view(self, *, top_level_run_id: str | None = None) -> PlanViewV1:
        return build_plan_view(
            self._read_terminal_decision(),
            top_level_run_id=top_level_run_id,
        )

    # 返回本次 Planner service 最后一次 submit 的内部结果分类
    @property
    def last_submit_outcome(self) -> str:
        return self._last_submit_outcome

    # 返回最后一次显式 incomplete validation 的稳定原因
    @property
    def last_incomplete_reason(self) -> str | None:
        return self._last_incomplete_reason

    # 返回 Planner tool invoker 应锁存的硬终止原因
    @property
    def invocation_terminal_reason(self) -> str | None:
        return self._invocation_terminal_reason

    # 记录未能解析为 draft 的最后一次 submit，避免复用旧 incomplete 状态
    def note_invalid_submit(self, category: str = "structure-invalid") -> None:
        self.last_validation = None
        self._last_submit_outcome = "invalid"
        self._last_incomplete_reason = None
        self._last_submit_category = (
            "terminal-contract-violation"
            if self._terminal_decision is not None
            else category
        )

    # 校验 draft 的结构、引用、范围、来源与 snapshot 后持久化新版本
    def submit(self, draft: PlannerDecisionDraft) -> PlannerDecision:
        self.last_validation = None
        self._last_submit_outcome = "invalid"
        self._last_incomplete_reason = None
        self._last_submit_category = None
        try:
            decision = self._submit_validated(draft)
        except PlannerValidationError as exc:
            self._last_submit_category = exc.category
            if exc.code == "grounding-missing":
                self._missing_grounding_submit_count += 1
                if self._missing_grounding_submit_count >= 2:
                    self._invocation_terminal_reason = "planning-grounding-missing"
            else:
                self._missing_grounding_submit_count = 0
            if exc.incomplete:
                self._last_submit_outcome = "incomplete"
                self._last_incomplete_reason = exc.category
            raise
        self._missing_grounding_submit_count = 0
        self._invocation_terminal_reason = None
        self._last_submit_outcome = "accepted"
        self._last_submit_category = None
        return decision

    # 执行一次完整校验与 immutable decision 写入，供 submit 统一记录最后结果
    def _submit_validated(self, draft: PlannerDecisionDraft) -> PlannerDecision:
        draft_digest = canonical_digest(draft.model_dump(mode="json"))
        if self._terminal_decision is not None:
            if draft_digest != self._terminal_draft_digest:
                raise PlannerValidationError(
                    "terminal-contract-violation",
                    "a different draft cannot follow a terminal PlannerDecision",
                )
            stale_reason = self.terminal_failure_reason()
            if stale_reason is not None:
                raise PlannerValidationError(
                    stale_reason,
                    "terminal PlannerDecision is no longer current",
                )
            decision = self._read_terminal_decision()
            self.last_validation = {
                "structure-valid": True,
                "reference-valid": True,
                "scope-valid": True,
                "provenance-valid": True,
                "snapshot-current": True,
            }
            return decision
        self._validate_structure(draft)
        architecture_slice, snapshot = self._load_grounding(draft)
        self._validate_provenance(draft, architecture_slice)
        self._validate_references(draft, architecture_slice)
        if not SnapshotBuilder(self._root).is_current(snapshot):
            raise PlannerValidationError(
                "snapshot-stale",
                "relevant repository snapshot is stale",
            )
        normalized = self._validate_scope(draft, architecture_slice, snapshot)
        decision_id, version = self._next_identity(draft.decision_id)
        payload = self._decision_payload(
            draft,
            decision_id=decision_id,
            version=version,
            normalized=normalized,
            architecture_slice=architecture_slice,
            snapshot=snapshot,
        )
        decision = PlannerDecision(
            **payload,
            content_digest=_exact_content_digest(payload),
        )
        self._store.write_decision(
            self._session_id,
            decision.decision_id,
            decision.version,
            decision.model_dump(mode="json"),
        )
        self._terminal_decision = SubmittedDecisionIdentity(
            decision_id=decision.decision_id,
            version=decision.version,
            snapshot_digest=decision.snapshot_digest,
            content_digest=decision.content_digest,
        )
        self._terminal_draft_digest = draft_digest
        self.last_validation = {
            "structure-valid": True,
            "reference-valid": True,
            "scope-valid": True,
            "provenance-valid": True,
            "snapshot-current": True,
        }
        return decision

    # 重新读取本次 terminal decision，发现任何持久化异常都安全归类为 artifact-corrupt
    def _read_terminal_decision(self) -> PlannerDecision:
        identity = self._terminal_decision
        if identity is None:
            raise PlannerValidationError(
                "missing-terminal-decision",
                "planner has not submitted a terminal decision",
            )
        try:
            payload = self._store.read_decision(
                self._session_id,
                identity.decision_id,
                identity.version,
            )
            decision = decode_exact_planner_decision(payload)
            if _exact_content_digest(payload) != decision.content_digest:
                raise ValueError("content digest mismatch")
            if (
                decision.decision_id != identity.decision_id
                or decision.version != identity.version
                or decision.snapshot_digest != identity.snapshot_digest
                or decision.content_digest != identity.content_digest
            ):
                raise ValueError("terminal identity mismatch")
            return decision
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            raise PlannerValidationError(
                "artifact-corrupt",
                "terminal PlannerDecision artifact is not readable",
            ) from exc

    # 在 child finished event 前验证本次 run 的 exact decision、grounding 和相关 snapshot
    def terminal_failure_reason(self) -> str | None:
        if self._invocation_terminal_reason is not None:
            return self._invocation_terminal_reason
        identity = self._terminal_decision
        if identity is None:
            if self._last_submit_outcome == "incomplete":
                return "planning-input-incomplete"
            return "missing-terminal-decision"
        if self._last_submit_category == "terminal-contract-violation":
            return "terminal-contract-violation"
        try:
            decision = self._read_terminal_decision()
            if decision.goal != self._goal:
                return "goal-mismatch"
            if not decision.requires_user_approval:
                return "missing-terminal-decision"
            grounding = self._store.read_grounding(self._session_id)
            if grounding is None:
                return "artifact-corrupt"
            raw_slices = grounding.get("architecture_slices")
            raw_snapshots = grounding.get("snapshots")
            if not isinstance(raw_slices, list) or not isinstance(raw_snapshots, list):
                return "artifact-corrupt"
            slices = [
                ArchitectureSlice.model_validate(item)
                for item in raw_slices
                if isinstance(item, dict)
                and item.get("slice_id") == decision.architecture_slice_id
                and item.get("version") == decision.architecture_slice_version
            ]
            snapshots = [
                RepositorySnapshot.model_validate(item)
                for item in raw_snapshots
                if isinstance(item, dict)
                and item.get("snapshot_digest") == decision.snapshot_digest
            ]
            if len(slices) != 1 or not snapshots:
                return "artifact-corrupt"
            architecture_slice = slices[0]
            if architecture_slice.completeness != "complete_for_task":
                return "artifact-corrupt"
            if architecture_slice.content_digest != decision.architecture_slice_content_digest:
                return "artifact-corrupt"
            if architecture_slice.snapshot_digest != decision.snapshot_digest:
                return "artifact-corrupt"
            slice_payload = architecture_slice.model_dump(
                mode="json", exclude={"content_digest"}
            )
            if canonical_digest(slice_payload) != architecture_slice.content_digest:
                return "artifact-corrupt"
            snapshot = snapshots[0]
            if not SnapshotBuilder(self._root).is_current(snapshot):
                return "snapshot-stale"
            return None
        except PlannerValidationError as exc:
            _LOGGER.warning(
                "planner terminal validation failed category=%s run_id=%s",
                exc.category,
                self._run_id,
            )
            return exc.category
        except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
            _LOGGER.warning(
                "planner terminal artifact validation failed run_id=%s",
                self._run_id,
            )
            return "artifact-corrupt"

    # 检查 required requirement mapping、重复 identity 与 blocking questions
    def _validate_structure(self, draft: PlannerDecisionDraft) -> None:
        if not draft.requirements or not draft.intended_changes:
            raise PlannerValidationError(
                "structure-invalid",
                "requirements and intended_changes must not be empty",
                incomplete=True,
            )
        if not draft.allowed_capabilities:
            raise PlannerValidationError(
                "structure-invalid",
                "allowed_capabilities must not be empty",
            )
        if len(set(draft.allowed_capabilities)) != len(draft.allowed_capabilities):
            raise PlannerValidationError(
                "structure-invalid",
                "allowed_capabilities must be unique",
            )
        if not draft.requires_user_approval:
            raise PlannerValidationError(
                "structure-invalid",
                "PlannerDecision must require user approval",
            )
        if not draft.selected_approach.strip():
            raise PlannerValidationError(
                "structure-invalid",
                "selected_approach must not be empty",
            )
        if draft.unresolved_questions:
            raise PlannerValidationError(
                "structure-invalid",
                "unresolved questions block decision validation",
            )
        requirement_ids = [item.requirement_id for item in draft.requirements]
        change_ids = [item.change_id for item in draft.intended_changes]
        if len(set(requirement_ids)) != len(requirement_ids) or len(set(change_ids)) != len(
            change_ids
        ):
            raise PlannerValidationError(
                "structure-invalid",
                "requirement_id and change_id values must be unique",
            )
        known_requirements = set(requirement_ids)
        for change in draft.intended_changes:
            if not change.description.strip() or not change.target_paths:
                raise PlannerValidationError(
                    "structure-invalid",
                    f"intended change {change.change_id} is incomplete",
                    incomplete=True,
                )
            if not set(change.requirement_ids).issubset(known_requirements):
                raise PlannerValidationError(
                    "structure-invalid",
                    f"intended change {change.change_id} references an unknown requirement",
                )
        for strategy in draft.verification_plan:
            if not strategy.strategy.strip() or not set(strategy.requirement_ids).issubset(
                known_requirements
            ):
                raise PlannerValidationError(
                    "structure-invalid",
                    "verification strategy is empty or references an unknown requirement",
                )
        required = {
            item.requirement_id for item in draft.requirements if item.required
        }
        changed = {
            requirement_id
            for change in draft.intended_changes
            for requirement_id in change.requirement_ids
        }
        verified = {
            requirement_id
            for strategy in draft.verification_plan
            for requirement_id in strategy.requirement_ids
        }
        if not required.issubset(changed) or not required.issubset(verified):
            raise PlannerValidationError(
                "structure-invalid",
                "every required requirement needs an intended change and verification strategy",
            )

    # 从 grounding artifact 读取并校验目标 slice 与 snapshot 的存在性
    def _load_grounding(
        self,
        draft: PlannerDecisionDraft,
    ) -> tuple[ArchitectureSlice, RepositorySnapshot]:
        try:
            grounding = self._store.read_grounding(self._session_id)
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            raise PlannerValidationError(
                "artifact-corrupt",
                "grounding artifact is not readable",
            ) from exc
        if grounding is None:
            raise PlannerValidationError(
                "reference-invalid",
                "grounding artifact does not exist",
                code="grounding-missing",
            )
        raw_slices = grounding.get("architecture_slices")
        raw_snapshots = grounding.get("snapshots")
        if not isinstance(raw_slices, list) or not isinstance(raw_snapshots, list):
            raise PlannerValidationError(
                "reference-invalid",
                "grounding artifact does not contain slice and snapshot records",
            )
        try:
            slices = [
                ArchitectureSlice.model_validate(item)
                for item in raw_slices
                if isinstance(item, dict)
                and item.get("slice_id") == draft.architecture_slice_id
                and item.get("version") == draft.architecture_slice_version
            ]
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            raise PlannerValidationError(
                "reference-invalid",
                "grounding artifact contains an invalid record",
            ) from exc
        if len(slices) != 1:
            raise PlannerValidationError(
                "reference-invalid",
                "referenced exact ArchitectureSlice does not exist",
        )
        architecture_slice = slices[0]
        try:
            snapshots = [
                RepositorySnapshot.model_validate(item)
                for item in raw_snapshots
                if isinstance(item, dict)
                and item.get("snapshot_digest") == architecture_slice.snapshot_digest
            ]
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            raise PlannerValidationError(
                "reference-invalid",
                "grounding artifact contains an invalid snapshot record",
            ) from exc
        if not snapshots:
            raise PlannerValidationError(
                "reference-invalid",
                "exact ArchitectureSlice snapshot does not exist",
            )
        return architecture_slice, snapshots[0]

    # 校验 slice content digest、complete_for_task 声明与 unresolved questions
    def _validate_provenance(
        self,
        draft: PlannerDecisionDraft,
        architecture_slice: ArchitectureSlice,
    ) -> None:
        payload = architecture_slice.model_dump(mode="json", exclude={"content_digest"})
        if canonical_digest(payload) != architecture_slice.content_digest:
            raise PlannerValidationError(
                "provenance-invalid",
                "ArchitectureSlice content digest is invalid",
            )
        if architecture_slice.completeness != "complete_for_task":
            raise PlannerValidationError(
                "provenance-invalid",
                "PlannerDecision requires a complete_for_task ArchitectureSlice",
            )
        if architecture_slice.unresolved_questions:
            raise PlannerValidationError(
                "provenance-invalid",
                "ArchitectureSlice contains unresolved questions",
            )

    # 校验 intended changes 只引用 slice 中已记录的 evidence IDs
    def _validate_references(
        self,
        draft: PlannerDecisionDraft,
        architecture_slice: ArchitectureSlice,
    ) -> None:
        evidence_ids = {
            evidence.tool_call_id for evidence in architecture_slice.evidence_refs
        }
        for change in draft.intended_changes:
            if not change.evidence_refs:
                raise PlannerValidationError(
                    "reference-invalid",
                    f"intended change {change.change_id} has no evidence",
                )
            if not set(change.evidence_refs).issubset(evidence_ids):
                raise PlannerValidationError(
                    "reference-invalid",
                    f"intended change {change.change_id} references unknown evidence",
                )

    # 校验 declared targets 的存在状态、grounding/read 关系与显式边界声明
    def _validate_scope(
        self,
        draft: PlannerDecisionDraft,
        architecture_slice: ArchitectureSlice,
        snapshot: RepositorySnapshot,
    ) -> dict[str, tuple[str, ...]]:
        modify = self._normalize_many(draft.files_to_modify)
        create = self._normalize_many(draft.files_to_create)
        if set(modify) & set(create):
            raise PlannerValidationError(
                "scope-invalid",
                "files_to_modify and files_to_create must be disjoint",
            )
        likely = set(self._normalize_many(architecture_slice.likely_change_targets))
        read_paths = {
            self._normalize(evidence.logical_path)
            for evidence in architecture_slice.evidence_refs
            if evidence.tool_name == "read_file" and evidence.logical_path is not None
        }
        existing_snapshot = set(snapshot.planned_existing_target_digests)
        new_snapshot = set(snapshot.planned_new_target_paths)
        if likely != existing_snapshot | new_snapshot:
            raise PlannerValidationError(
                "provenance-invalid",
                "ArchitectureSlice likely targets do not match its snapshot target scope",
            )
        for path in modify:
            candidate = self._root / path
            if (
                path not in likely
                or path not in read_paths
                or path not in existing_snapshot
                or not candidate.is_file()
            ):
                raise PlannerValidationError(
                    "scope-invalid",
                    f"files_to_modify target is not read, grounded, and existing: {path}",
                )
        for path in create:
            candidate = self._resolver.resolve_for_write(path)
            self._access_policy.ensure_allowed(path, candidate)
            if (
                path not in likely
                or path not in new_snapshot
                or candidate.exists()
                or candidate.is_symlink()
            ):
                raise PlannerValidationError(
                    "scope-invalid",
                    f"files_to_create target is not grounded and absent: {path}",
                )
        declared_targets = set(modify) | set(create)
        change_targets = {
            self._normalize(path)
            for change in draft.intended_changes
            for path in change.target_paths
        }
        if change_targets != declared_targets:
            raise PlannerValidationError(
                "scope-invalid",
                "intended change targets must equal the declared file scope",
            )
        dependency = set(self._normalize_many(draft.dependency_changes))
        protocol = set(self._normalize_many(draft.protocol_or_schema_changes))
        if not dependency.issubset(declared_targets) or not protocol.issubset(
            declared_targets
        ):
            raise PlannerValidationError(
                "scope-invalid",
                "dependency/protocol declarations must be inside the declared file scope",
            )
        missing_dependency = {
            path for path in declared_targets if self._is_manifest(path)
        } - dependency
        if missing_dependency:
            raise PlannerValidationError(
                "scope-invalid",
                "dependency_changes must explicitly include manifest or lockfile targets",
            )
        missing_protocol = {
            path for path in declared_targets if self._is_protocol_or_schema(path)
        } - protocol
        if missing_protocol:
            raise PlannerValidationError(
                "scope-invalid",
                "protocol_or_schema_changes must explicitly include protocol/schema targets",
            )
        return {
            "files_to_modify": modify,
            "files_to_create": create,
            "dependency_changes": tuple(sorted(dependency)),
            "protocol_or_schema_changes": tuple(sorted(protocol)),
        }

    # 将多个 logical paths 归一化并拒绝重复别名
    def _normalize_many(self, paths: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(self._normalize(path) for path in paths)
        if len(set(normalized)) != len(normalized):
            raise PlannerValidationError(
                "scope-invalid",
                "duplicate or aliased paths are not allowed",
            )
        return normalized

    # 将 logical path 解析为 workspace 内 canonical relative path
    def _normalize(self, logical_path: str) -> str:
        path = Path(logical_path)
        if not logical_path or path.is_absolute() or ".." in path.parts:
            raise PlannerValidationError(
                "scope-invalid",
                "decision paths must be workspace-relative",
            )
        try:
            resolved = self._resolver.resolve_for_write(logical_path)
            self._access_policy.ensure_allowed(logical_path, resolved)
            normalized = resolved.relative_to(self._root).as_posix()
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            raise PlannerValidationError(
                "scope-invalid",
                "decision path is outside the workspace",
            ) from exc
        if normalized == ".":
            raise PlannerValidationError(
                "scope-invalid",
                "decision path cannot be the workspace root",
            )
        return normalized

    # 判断 path 是否是可机械识别的 dependency manifest 或 lockfile
    def _is_manifest(self, path: str) -> bool:
        name = Path(path).name
        return name in _MANIFEST_NAMES or name.startswith("requirements")

    # 判断 path 是否是可机械识别的 bus protocol 或 schema 文件
    def _is_protocol_or_schema(self, path: str) -> bool:
        logical = Path(path)
        return (
            path == "WIRE_PROTOCOL.md"
            or path.startswith("src/kama_claude/core/bus/")
            or "schemas" in logical.parts
            or logical.name.startswith("schema.")
            or logical.name.endswith(".schema.json")
        )

    # 为新 lineage 分配 runtime ID，或为既有 lineage 分配下一 immutable version
    def _next_identity(self, requested_id: str | None) -> tuple[str, int]:
        decisions = self._store.list_decisions(self._session_id)
        if requested_id is None:
            return f"decision_{uuid.uuid4().hex}", 1
        lineage = [
            item for item in decisions if item.get("decision_id") == requested_id
        ]
        if not lineage:
            raise PlannerValidationError(
                "reference-invalid",
                "revision decision_id does not identify an existing lineage",
            )
        return requested_id, max(int(item["version"]) for item in lineage) + 1

    # 组装排除 content_digest 的 canonical PlannerDecision payload
    def _decision_payload(
        self,
        draft: PlannerDecisionDraft,
        *,
        decision_id: str,
        version: int,
        normalized: dict[str, tuple[str, ...]],
        architecture_slice: ArchitectureSlice,
        snapshot: RepositorySnapshot,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "decision_id": decision_id,
            "version": version,
            "goal": self._goal,
            "requirements": [item.model_dump(mode="json") for item in draft.requirements],
            "architecture_slice_id": draft.architecture_slice_id,
            "architecture_slice_version": architecture_slice.version,
            "architecture_slice_content_digest": architecture_slice.content_digest,
            "snapshot_digest": snapshot.snapshot_digest,
            "architecture_mode": draft.architecture_mode,
            "selected_approach": draft.selected_approach,
            "existing_patterns_reused": list(draft.existing_patterns_reused),
            "intended_changes": [
                item.model_dump(mode="json") for item in draft.intended_changes
            ],
            "files_to_modify": list(normalized["files_to_modify"]),
            "files_to_create": list(normalized["files_to_create"]),
            "allowed_capabilities": list(draft.allowed_capabilities),
            "dependency_changes": list(normalized["dependency_changes"]),
            "protocol_or_schema_changes": list(
                normalized["protocol_or_schema_changes"]
            ),
            "verification_plan": [
                item.model_dump(mode="json") for item in draft.verification_plan
            ],
            "non_goals": list(draft.non_goals),
            "assumptions": list(draft.assumptions),
            "unresolved_questions": list(draft.unresolved_questions),
            "requires_user_approval": draft.requires_user_approval,
        }


class PlannerDecisionSubmitTool(BaseTool):
    name = "planner_decision_submit"
    description = (
        "Submit the final snapshot-bound PlannerDecision as a terminal commit. "
        "Call this only when the plan is final: after a successful submission, no further "
        "repository exploration or plan revision is allowed. The runtime validates evidence, "
        "scope, provenance, requirement mappings, and current relevant content; an exact "
        "duplicate retry is idempotent."
    )
    params_model = PlannerDecisionDraftV2
    input_schema: dict[str, object] = PlannerDecisionDraftV2.model_json_schema()

    # 绑定当前 session 的 PlannerDecision service
    def __init__(self, service: PlannerDecisionService) -> None:
        self._service = service

    # 校验 draft 并返回 runtime 分配的 immutable decision identity
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        try:
            draft = PlannerDecisionDraft.model_validate(params)
            decision = self._service.submit(draft)
        except PlannerValidationError as exc:
            _LOGGER.warning(
                "planner decision submit rejected category=%s run_id=%s",
                exc.category,
                self._service._run_id,
            )
            if exc.code == "grounding-missing":
                guidance = (
                    "grounding artifact does not exist; call spawn_agent in foreground "
                    "with subagent_type='explorer', then retry with the returned "
                    "slice_id and version; do not guess identifiers"
                )
            else:
                guidance = "revise the draft or grounding and retry"
            return ToolResult(
                content=(
                    "planner_decision_submit rejected: "
                    f"{exc.category}; validation detail: {exc}; {guidance}"
                ),
                is_error=True,
                error_type="invalid_input",
            )
        except (TypeError, ValueError) as exc:
            self._service.note_invalid_submit("structure-invalid")
            _LOGGER.warning(
                "planner decision submit schema rejected run_id=%s error=%s",
                self._service._run_id,
                type(exc).__name__,
            )
            return ToolResult(
                content=(
                    "planner_decision_submit rejected: structure-invalid; "
                    "revise the draft and retry"
                ),
                is_error=True,
                error_type="invalid_input",
            )
        return ToolResult(
            content=json.dumps(
                {
                    "decision_id": decision.decision_id,
                    "version": decision.version,
                    "architecture_slice_id": decision.architecture_slice_id,
                    "architecture_slice_version": decision.architecture_slice_version,
                    "architecture_slice_content_digest": (
                        decision.architecture_slice_content_digest
                    ),
                    "snapshot_digest": decision.snapshot_digest,
                    "content_digest": decision.content_digest,
                    "validation": self._service.last_validation,
                },
                sort_keys=True,
            )
        )


# 将 bounded PlanView 渲染为用户可读但非权威的短摘要
def render_planner_decision(plan: PlanViewV1) -> str:
    lines = [
        "Plan generated.",
        f"Goal: {plan.goal}",
        f"Approach: {plan.selected_approach}",
        f"Architecture mode: {plan.architecture_mode}",
    ]
    if plan.files_to_modify:
        lines.append("Files to modify: " + ", ".join(plan.files_to_modify))
    if plan.files_to_create:
        lines.append("Files to create: " + ", ".join(plan.files_to_create))
    if plan.dependency_changes:
        lines.append("Dependency changes: " + ", ".join(plan.dependency_changes))
    if plan.protocol_or_schema_changes:
        lines.append(
            "Protocol/schema changes: " + ", ".join(plan.protocol_or_schema_changes)
        )
    if plan.unresolved_questions:
        lines.append("Unresolved: " + "; ".join(plan.unresolved_questions))
    if plan.assumptions:
        lines.append("Assumptions: " + "; ".join(plan.assumptions))
    if plan.non_goals:
        lines.append("Non-goals: " + "; ".join(plan.non_goals))
    lines.append(f"Decision: {plan.decision_key} ({plan.decision_content_digest})")
    lines.append("Approval and execution are not available in this build.")
    return "\n".join(lines)


# 将 exact PlannerDecision 投影为 bounded V1 PlanView，并绑定 top-level run projection identity
def build_plan_view(
    decision: PlannerDecision,
    *,
    top_level_run_id: str | None = None,
) -> PlanViewV1:
    decision_key = f"{decision.decision_id}:v{decision.version}"
    projection_owner = top_level_run_id or "standalone"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision_key": decision_key,
        "projection_key": f"pv1:{projection_owner}:{decision_key}",
        "decision_id": decision.decision_id,
        "decision_version": decision.version,
        "decision_content_digest": decision.content_digest,
        "architecture_slice_id": decision.architecture_slice_id,
        "architecture_slice_version": decision.architecture_slice_version,
        "architecture_slice_content_digest": decision.architecture_slice_content_digest,
        "snapshot_digest": decision.snapshot_digest,
        "goal": decision.goal,
        "architecture_mode": decision.architecture_mode,
        "requirements": [
            item.model_dump(mode="json") for item in decision.requirements
        ],
        "selected_approach": decision.selected_approach,
        "intended_changes": [
            item.model_dump(mode="json") for item in decision.intended_changes
        ],
        "existing_patterns_reused": list(decision.existing_patterns_reused),
        "files_to_modify": list(decision.files_to_modify),
        "files_to_create": list(decision.files_to_create),
        "allowed_capabilities": list(decision.allowed_capabilities),
        "unresolved_questions": list(decision.unresolved_questions),
        "assumptions": list(decision.assumptions),
        "dependency_changes": list(decision.dependency_changes),
        "protocol_or_schema_changes": list(decision.protocol_or_schema_changes),
        "verification_plan": [
            item.model_dump(mode="json") for item in decision.verification_plan
        ],
        "non_goals": list(decision.non_goals),
        "requires_user_approval": True,
        "execution_available": False,
    }
    return finalize_plan_view_payload(payload)


# 将完整 exact decision 渲染给 /orchestrate，超过 tool transport budget 时 fail closed
def render_planner_decision_execution_summary(decision: PlannerDecision) -> str:
    rendered = json.dumps(
        {"planner_decision": decision.model_dump(mode="json")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    # 以当前 tool_result transport cap 的字节保守边界拒绝静默截断
    if len(rendered.encode("utf-8")) > TOOL_RESULT_LIMIT:
        raise ValueError("planner-result-too-large")
    return rendered
