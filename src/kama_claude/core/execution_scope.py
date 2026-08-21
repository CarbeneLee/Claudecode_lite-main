from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

if TYPE_CHECKING:
    from kama_claude.core.approval import ApprovalRecord, CommittedPlanReceipt
    from kama_claude.core.grounding import RepositorySnapshot
    from kama_claude.core.planning import ExactPlannerDecisionV2

SCOPED_CAPABILITY_CEILING = frozenset(
    {"read_file", "list_dir", "search_code", "write_file"}
)
_POST_STATE_AUDIT_TIMEOUT_S = 5.0


# 计算 scope domain payload 的稳定 digest
def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 计算 workspace 文件的原始字节 digest
def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExecutionScopeError(RuntimeError):
    # 表示 scope derivation 或 invocation authorization 失败
    error_type = "scope_denied"


class ScopeDeniedError(ExecutionScopeError):
    # 表示调用不属于当前 immutable scope
    error_type = "scope_denied"


class ScopeRequiredError(ExecutionScopeError):
    # 表示 scoped invoker 缺少 mandatory execution context
    error_type = "scope_required"


class ExternalWorkspaceDriftError(ExecutionScopeError):
    # 表示当前 workspace 不再匹配本 execution 的 expected state
    error_type = "external_workspace_drift"


class ScopeMutationInconclusiveError(ExecutionScopeError):
    # 表示 observable mutation 无法证明等于授权 intended state
    error_type = "scope_mutation_inconclusive"


class ScopeAuditError(ExecutionScopeError):
    # 表示 post-state audit 失败且 execution 必须停止
    error_type = "scope_audit_failed"


@dataclass(frozen=True, slots=True)
class PathState:
    # 保存一个 logical target 的可比较存在性和内容状态
    exists: bool
    digest: str | None
    is_file: bool = False
    is_symlink: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MutationLedgerEntry:
    # 保存一次已观察 mutation 的 before/after 和 provenance
    path: str
    operation: str
    before: PathState
    after: PathState
    tool_call_id: str
    sequence: int
    classification: Literal[
        "authorized",
        "abnormal-but-authorized",
        "inconclusive",
    ]


@dataclass(frozen=True, slots=True)
class AuthorizationAttempt:
    # 返回 post-state audit 对当前 retry decision 的约束
    retry_allowed: bool = True
    mutation_observed: bool = False


@dataclass(frozen=True, slots=True, init=False)
class ExecutionScope:
    # 描述一次 approved committed plan 允许访问的 immutable capability 与 targets
    scope_version: int
    session_id: str
    workspace_id: str
    projection_key: str
    decision_id: str
    decision_version: int
    decision_content_digest: str
    commit_receipt_digest: str
    snapshot_digest: str
    files_to_modify: tuple[str, ...]
    files_to_create: tuple[str, ...]
    capabilities: tuple[str, ...]
    dependency_changes: tuple[str, ...]
    protocol_or_schema_changes: tuple[str, ...]
    scope_digest: str

    # 拒绝绕过 authoritative derivation 的任意 scope 构造
    def __init__(self, *_: object, **__: object) -> None:
        raise ScopeDeniedError("execution scope must come from approved artifacts")

    # 仅供 authoritative derivation 和本模块测试构造已校验 scope
    @classmethod
    def _create(
        cls,
        *,
        session_id: str,
        workspace_id: str,
        projection_key: str,
        decision_id: str,
        decision_version: int,
        decision_content_digest: str,
        commit_receipt_digest: str,
        snapshot_digest: str,
        files_to_modify: tuple[str, ...] = (),
        files_to_create: tuple[str, ...] = (),
        capabilities: tuple[str, ...] = (),
        dependency_changes: tuple[str, ...] = (),
        protocol_or_schema_changes: tuple[str, ...] = (),
        scope_version: int = 1,
    ) -> ExecutionScope:
        if scope_version != 1 or decision_version < 1:
            raise ScopeDeniedError("invalid execution scope version or decision version")
        normalized_caps = tuple(sorted(set(capabilities)))
        if not set(normalized_caps).issubset(SCOPED_CAPABILITY_CEILING):
            raise ScopeDeniedError("capability is outside scoped execution ceiling")
        normalized_modify = _normalize_declared_paths(files_to_modify)
        normalized_create = _normalize_declared_paths(files_to_create)
        if set(normalized_modify) & set(normalized_create):
            raise ScopeDeniedError("a target cannot be both modified and created")
        payload: dict[str, object] = {
            "scope_version": scope_version,
            "session_id": session_id,
            "workspace_id": workspace_id,
            "projection_key": projection_key,
            "decision_id": decision_id,
            "decision_version": decision_version,
            "decision_content_digest": decision_content_digest,
            "commit_receipt_digest": commit_receipt_digest,
            "snapshot_digest": snapshot_digest,
            "files_to_modify": list(normalized_modify),
            "files_to_create": list(normalized_create),
            "capabilities": list(normalized_caps),
            "dependency_changes": sorted(set(dependency_changes)),
            "protocol_or_schema_changes": sorted(set(protocol_or_schema_changes)),
        }
        return cls._materialize(
            scope_version=scope_version,
            session_id=session_id,
            workspace_id=workspace_id,
            projection_key=projection_key,
            decision_id=decision_id,
            decision_version=decision_version,
            decision_content_digest=decision_content_digest,
            commit_receipt_digest=commit_receipt_digest,
            snapshot_digest=snapshot_digest,
            files_to_modify=normalized_modify,
            files_to_create=normalized_create,
            capabilities=normalized_caps,
            dependency_changes=tuple(sorted(set(dependency_changes))),
            protocol_or_schema_changes=tuple(sorted(set(protocol_or_schema_changes))),
            scope_digest=_digest(payload),
        )

    # 通过 frozen slots dataclass 的内部入口物化不可变 scope
    @classmethod
    def _materialize(cls, **values: object) -> ExecutionScope:
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance

    # 从 exact V2 decision、approved record、receipt 和 current snapshot 派生 scope
    @classmethod
    def from_approved(
        cls,
        *,
        decision: ExactPlannerDecisionV2,
        approval_record: ApprovalRecord,
        receipt: CommittedPlanReceipt,
        snapshot: RepositorySnapshot,
        workspace_root: Path,
    ) -> ExecutionScope:
        from kama_claude.core.approval import ApprovalRecord, CommittedPlanReceipt
        from kama_claude.core.grounding import (
            RepositorySnapshot,
            SnapshotBuilder,
            canonical_digest,
            workspace_identity,
        )
        from kama_claude.core.planning import ExactPlannerDecisionV2

        if not isinstance(decision, ExactPlannerDecisionV2):
            raise ScopeDeniedError("exact V2 decision is required")
        if (
            not isinstance(approval_record, ApprovalRecord)
            or approval_record.action != "approve"
        ):
            raise ScopeDeniedError("approved ApprovalRecord is required")
        if not isinstance(receipt, CommittedPlanReceipt):
            raise ScopeDeniedError("committed receipt is required")
        if not isinstance(snapshot, RepositorySnapshot):
            raise ScopeDeniedError("current repository snapshot is required")
        try:
            approval_record.verify_digest()
            receipt.verify_digest()
            decision_payload = decision.model_dump(
                mode="json",
                exclude={"content_digest"},
            )
            if canonical_digest(decision_payload) != decision.content_digest:
                raise ValueError("decision content digest mismatch")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScopeDeniedError("approval or receipt integrity check failed") from exc
        if (
            approval_record.session_id != receipt.session_id
            or approval_record.projection_key != receipt.projection_key
            or approval_record.decision_id != decision.decision_id
            or approval_record.decision_version != decision.version
            or approval_record.content_digest != decision.content_digest
            or approval_record.commit_receipt_digest != receipt.receipt_digest
            or receipt.decision_id != decision.decision_id
            or receipt.decision_version != decision.version
            or receipt.decision_content_digest != decision.content_digest
            or receipt.projection_key
            != f"pv1:{receipt.top_level_run_id}:"
            f"{decision.decision_id}:v{decision.version}"
        ):
            raise ScopeDeniedError("approved artifacts do not bind to the exact decision")
        canonical_root, workspace_id = workspace_identity(workspace_root)
        if workspace_id != snapshot.workspace_id:
            raise ScopeDeniedError("workspace identity does not match snapshot")
        builder = SnapshotBuilder(canonical_root)
        if snapshot.snapshot_digest != decision.snapshot_digest or not builder.is_current(snapshot):
            raise ScopeDeniedError("relevant workspace snapshot is stale")
        modify = _normalize_declared_paths(decision.files_to_modify)
        create = _normalize_declared_paths(decision.files_to_create)
        _validate_current_targets(
            canonical_root,
            snapshot,
            modify,
            create,
        )
        return cls._create(
            session_id=approval_record.session_id,
            workspace_id=snapshot.workspace_id,
            projection_key=approval_record.projection_key,
            decision_id=decision.decision_id,
            decision_version=decision.version,
            decision_content_digest=decision.content_digest,
            commit_receipt_digest=receipt.receipt_digest,
            snapshot_digest=decision.snapshot_digest,
            files_to_modify=modify,
            files_to_create=create,
            capabilities=tuple(decision.allowed_capabilities),
            dependency_changes=tuple(decision.dependency_changes),
            protocol_or_schema_changes=tuple(decision.protocol_or_schema_changes),
        )

    # 提供更明确的 domain 命名，保持 from_approved 的兼容入口
    @classmethod
    def derive(cls, **kwargs: Any) -> ExecutionScope:
        return cls.from_approved(**kwargs)


# 规范化声明的 workspace-relative logical paths，拒绝 absolute 和 parent traversal
def _normalize_declared_paths(paths: Any) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in paths:
        if not isinstance(raw, str):
            raise ScopeDeniedError("scope target must be a string")
        path = Path(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            raise ScopeDeniedError("scope target must be workspace-relative")
        value = path.as_posix().removeprefix("./")
        if value in ("", ".") or value in normalized:
            raise ScopeDeniedError("scope target is empty or duplicated")
        normalized.append(value)
    return tuple(sorted(normalized))


# 在 scope materialization 时验证 target baseline 和 parent existence
def _validate_current_targets(
    workspace_root: Path,
    snapshot: Any,
    files_to_modify: tuple[str, ...],
    files_to_create: tuple[str, ...],
) -> None:
    resolver = WorkspacePathResolver(workspace_root)
    policy = WorkspaceAccessPolicy(resolver.root)
    existing = snapshot.planned_existing_target_digests
    new = set(snapshot.planned_new_target_paths)
    for logical in files_to_modify:
        if logical not in existing:
            raise ScopeDeniedError("existing target is not in current snapshot")
        candidate = resolver.resolve_for_write(logical)
        policy.ensure_allowed(logical, candidate)
        if candidate.relative_to(resolver.root).as_posix() != logical:
            raise ScopeDeniedError("symlink alias is not an approved logical target")
        if candidate.is_symlink() or not candidate.is_file():
            raise ScopeDeniedError("existing mutation target is not a regular file")
        if _file_digest(candidate) != existing[logical]:
            raise ScopeDeniedError("existing mutation target changed")
    for logical in files_to_create:
        if logical not in new:
            raise ScopeDeniedError("new target is not in current snapshot")
        candidate = resolver.resolve_for_write(logical)
        policy.ensure_allowed(logical, candidate)
        if candidate.relative_to(resolver.root).as_posix() != logical:
            raise ScopeDeniedError("symlink alias is not an approved logical target")
        if candidate.exists() or candidate.is_symlink():
            raise ScopeDeniedError("new target already exists")
        if not candidate.parent.exists() or not candidate.parent.is_dir():
            raise ScopeDeniedError("new target parent must already exist")
        if candidate.parent.is_symlink():
            raise ScopeDeniedError("new target parent cannot be a symlink alias")


@dataclass
class ExecutionMutationState:
    # 保存一个 scoped execution 的 baseline、expected state 和 mutation ledger
    execution_id: str
    scope_digest: str
    workspace_root: Path
    baseline_path_states: dict[str, PathState]
    expected_path_states: dict[str, PathState]
    mutation_ledger: list[MutationLedgerEntry] = field(default_factory=list)
    blocked: bool = False
    status: Literal["active", "blocked", "inconclusive"] = "active"
    _sequence: int = 0

    # 从 scope 当前捕获 baseline，允许 authorization 在调用前报告缺失 parent/target
    @classmethod
    def _capture(
        cls,
        scope: ExecutionScope,
        workspace_root: Path,
        execution_id: str,
    ) -> ExecutionMutationState:
        resolver = WorkspacePathResolver(workspace_root)
        states = {
            path: _read_path_state(resolver, path)
            for path in (*scope.files_to_modify, *scope.files_to_create)
        }
        return cls(
            execution_id=execution_id,
            scope_digest=scope.scope_digest,
            workspace_root=resolver.root,
            baseline_path_states=states.copy(),
            expected_path_states=states.copy(),
        )

    # 校验 attempt 开始前 target 仍等于本 execution 的 expected state
    def before_mutation(self, path: str, *, must_exist: bool) -> PathState:
        if self.blocked:
            raise ScopeMutationInconclusiveError("scoped execution is blocked")
        resolver = WorkspacePathResolver(self.workspace_root)
        current = _read_path_state(resolver, path)
        expected = self.expected_path_states[path]
        if current != expected:
            self._block_inconclusive()
            raise ExternalWorkspaceDriftError("workspace target drifted before attempt")
        if must_exist and (not current.exists or not current.is_file or current.is_symlink):
            raise ScopeDeniedError("approved existing target is unavailable")
        if not must_exist:
            candidate = resolver.resolve_for_write(path)
            if current.exists or current.is_symlink:
                raise ExternalWorkspaceDriftError("planned new target was externally created")
            if not candidate.parent.exists() or not candidate.parent.is_dir():
                raise ScopeDeniedError("new target parent must already exist")
        return current

    # 根据已读取的 post-state 记录 mutation，并决定当前 error 是否允许 retry
    def _record_post_state(
        self,
        *,
        path: str,
        before: PathState,
        intended: PathState,
        tool_call_id: str,
        outcome: object,
        actual: PathState,
    ) -> AuthorizationAttempt:
        # Backlog：成功写工具若报告成功但 actual == before，未来需增加 postcondition verification
        if actual == before:
            return AuthorizationAttempt(retry_allowed=True, mutation_observed=False)
        self._sequence += 1
        if actual == intended:
            classification: Literal["authorized", "abnormal-but-authorized", "inconclusive"]
            if isinstance(outcome, BaseException) or getattr(outcome, "is_error", False):
                classification = "abnormal-but-authorized"
            else:
                classification = "authorized"
            self.mutation_ledger.append(
                MutationLedgerEntry(
                    path=path,
                    operation="write_file",
                    before=before,
                    after=actual,
                    tool_call_id=tool_call_id,
                    sequence=self._sequence,
                    classification=classification,
                )
            )
            self.expected_path_states[path] = actual
            return AuthorizationAttempt(retry_allowed=False, mutation_observed=True)
        self.mutation_ledger.append(
            MutationLedgerEntry(
                path=path,
                operation="write_file",
                before=before,
                after=actual,
                tool_call_id=tool_call_id,
                sequence=self._sequence,
                classification="inconclusive",
            )
        )
        self._block_inconclusive()
        raise ScopeMutationInconclusiveError("observable mutation is outside approved state")

    # 读取 post-state、记录 mutation，并在 retry decision 前完成有界 audit
    async def audit_mutation(
        self,
        *,
        path: str,
        before: PathState,
        intended: PathState,
        tool_call_id: str,
        outcome: object,
    ) -> AuthorizationAttempt:
        resolver = WorkspacePathResolver(self.workspace_root)
        try:
            actual = await asyncio.wait_for(
                asyncio.to_thread(_read_path_state, resolver, path),
                timeout=_POST_STATE_AUDIT_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._block_inconclusive()
            raise ScopeAuditError("post-state audit failed") from exc
        return self._record_post_state(
            path=path,
            before=before,
            intended=intended,
            tool_call_id=tool_call_id,
            outcome=outcome,
            actual=actual,
        )

    # 保留同步 post-state 入口供 domain 单元测试和非管线调用使用
    def after_mutation(
        self,
        *,
        path: str,
        before: PathState,
        intended: PathState,
        tool_call_id: str,
        outcome: object,
    ) -> AuthorizationAttempt:
        resolver = WorkspacePathResolver(self.workspace_root)
        try:
            actual = _read_path_state(resolver, path)
        except Exception as exc:
            self._block_inconclusive()
            raise ScopeAuditError("post-state audit failed") from exc
        return self._record_post_state(
            path=path,
            before=before,
            intended=intended,
            tool_call_id=tool_call_id,
            outcome=outcome,
            actual=actual,
        )

    # 将 audit 自身失败标为 blocked，同时保留取消作为 primary outcome
    def mark_audit_inconclusive(self) -> None:
        self._block_inconclusive()

    # 将 execution 状态固定为不可继续的 inconclusive
    def _block_inconclusive(self) -> None:
        self.blocked = True
        self.status = "inconclusive"


@dataclass(init=False)
class ScopedExecutionContext:
    # 绑定已完成 freshness recheck 的 immutable scope 与 execution-local state
    scope: ExecutionScope
    admission_snapshot: RepositorySnapshot
    workspace_root: Path
    execution_id: str = field(default_factory=lambda: f"exec-{uuid.uuid4().hex}")
    mutation_state: ExecutionMutationState = field(init=False)

    # 拒绝绕过 freshness recheck 的任意 context 构造
    def __init__(self, *_: object, **__: object) -> None:
        raise ScopeRequiredError("execution context must come from verified scope")

    # 仅从 exact scope、snapshot 和 canonical workspace 物化 verified context
    @classmethod
    def from_verified_scope(
        cls,
        *,
        scope: ExecutionScope,
        snapshot: RepositorySnapshot,
        workspace_root: Path,
        execution_id: str | None = None,
    ) -> ScopedExecutionContext:
        from kama_claude.core.grounding import (
            RepositorySnapshot,
            SnapshotBuilder,
            workspace_identity,
        )

        if not isinstance(scope, ExecutionScope):
            raise ScopeDeniedError("execution scope is required")
        if not isinstance(snapshot, RepositorySnapshot):
            raise ScopeDeniedError("repository snapshot is required")
        canonical_root, workspace_id = workspace_identity(workspace_root)
        if workspace_id != scope.workspace_id or snapshot.workspace_id != scope.workspace_id:
            raise ScopeDeniedError("workspace identity does not match execution scope")
        if snapshot.snapshot_digest != scope.snapshot_digest:
            raise ScopeDeniedError("execution snapshot binding mismatch")
        builder = SnapshotBuilder(canonical_root)
        if not builder.is_current(snapshot):
            raise ScopeDeniedError("relevant workspace snapshot is stale")
        _validate_current_targets(
            canonical_root,
            snapshot,
            scope.files_to_modify,
            scope.files_to_create,
        )
        current_execution_id = execution_id or f"exec-{uuid.uuid4().hex}"
        state = ExecutionMutationState._capture(
            scope,
            canonical_root,
            current_execution_id,
        )
        context = object.__new__(cls)
        context.scope = scope
        context.admission_snapshot = snapshot
        context.workspace_root = canonical_root
        context.execution_id = current_execution_id
        context.mutation_state = state
        return context

    # 校验 approved target 之外的 instruction/evidence/manifest 仍与 admission 一致
    def verify_non_target_freshness(self) -> bool:
        from kama_claude.core.grounding import SnapshotBuilder

        snapshot = self.admission_snapshot
        excluded = set(self.scope.files_to_modify) | set(self.scope.files_to_create)
        builder = SnapshotBuilder(self.workspace_root)
        mappings = (
            snapshot.instruction_file_digests,
            snapshot.grounding_file_digests,
            snapshot.relevant_manifest_digests,
            snapshot.relevant_untracked_target_digests,
        )
        try:
            for expected in mappings:
                filtered = {
                    path: digest
                    for path, digest in expected.items()
                    if path not in excluded
                }
                if builder._digest_paths(filtered) != filtered:
                    return False
            for path in snapshot.planned_new_target_paths:
                if path in excluded:
                    continue
                candidate = builder._resolver.resolve_for_write(path)
                builder._access_policy.ensure_allowed(path, candidate)
                if candidate.exists() or candidate.is_symlink():
                    return False
        except (OSError, ValueError):
            return False
        return True

    @property
    # 返回 admission 时绑定的完整 repository snapshot
    def snapshot(self) -> RepositorySnapshot:
        return self.admission_snapshot

    # 在 approved execution 结束前同时审计非 target freshness 与 mutation expected state
    def terminal_audit(self) -> None:
        if self.mutation_state.blocked:
            raise ScopeMutationInconclusiveError("scoped execution is blocked")
        if not self.verify_non_target_freshness():
            self.mutation_state.mark_audit_inconclusive()
            raise ExternalWorkspaceDriftError("relevant non-target evidence drifted")
        resolver = WorkspacePathResolver(self.workspace_root)
        for logical, expected in self.mutation_state.expected_path_states.items():
            actual = _read_path_state(resolver, logical)
            if actual != expected:
                self.mutation_state.mark_audit_inconclusive()
                raise ScopeMutationInconclusiveError(
                    "approved target post-state is inconclusive"
                )


class ExecutionScopeAuthorization:
    # 将 immutable scope 和 mutation state 接入 shared invocation pipeline
    def __init__(self, context: ScopedExecutionContext) -> None:
        if context is None:
            raise ScopeRequiredError("scope-required")
        self._context = context
        self._pending: dict[int, tuple[str, PathState, PathState]] = {}

    # 在 permission 之前验证 capability、target membership 和 current pre-state
    async def authorize_call(self, *, tool_call: Any, tool: Any = None) -> None:
        del tool
        scope = self._context.scope
        if self._context.mutation_state.blocked:
            raise ScopeMutationInconclusiveError("scoped execution is blocked")
        if tool_call.name not in scope.capabilities:
            raise ScopeDeniedError("tool is outside approved capability ceiling")
        if tool_call.name not in SCOPED_CAPABILITY_CEILING:
            raise ScopeDeniedError("tool is outside scoped execution ceiling")
        if tool_call.name == "write_file":
            self._validate_write_path(str(tool_call.input.get("path", "")))

    # 在每个 mutating attempt 前保存 before state 并拒绝 drift/隐式 mkdir
    async def before_attempt(
        self,
        *,
        tool_call: Any,
        tool: Any = None,
        attempt: int,
    ) -> None:
        del tool
        if tool_call.name != "write_file":
            if self._context.mutation_state.blocked:
                raise ScopeMutationInconclusiveError("scoped execution is blocked")
            return
        if not self._context.verify_non_target_freshness():
            self._context.mutation_state.mark_audit_inconclusive()
            raise ExternalWorkspaceDriftError("relevant non-target evidence drifted")
        path = self._validate_write_path(str(tool_call.input.get("path", "")))
        expected = self._context.mutation_state.expected_path_states[path]
        baseline = self._context.mutation_state.baseline_path_states[path]
        owns_created_target = (
            path in self._context.scope.files_to_create
            and not baseline.exists
            and expected.exists
        )
        must_exist = path in self._context.scope.files_to_modify or owns_created_target
        before = self._context.mutation_state.before_mutation(
            path,
            must_exist=must_exist,
        )
        intended = PathState(
            exists=True,
            digest=hashlib.sha256(
                str(tool_call.input.get("content", "")).encode("utf-8")
            ).hexdigest(),
            is_file=True,
        )
        self._pending[attempt] = (path, before, intended)

    # 在 retry decision 前审计每个 mutating attempt 的 actual post-state
    async def after_attempt(
        self,
        *,
        tool_call: Any,
        tool: Any = None,
        attempt: int,
        outcome: object,
    ) -> AuthorizationAttempt:
        del tool
        pending = self._pending.pop(attempt, None)
        if pending is None or tool_call.name != "write_file":
            if self._context.mutation_state.blocked:
                raise ScopeMutationInconclusiveError("scoped execution is blocked")
            return AuthorizationAttempt()
        path, before, intended = pending
        return await self._context.mutation_state.audit_mutation(
            path=path,
            before=before,
            intended=intended,
            tool_call_id=tool_call.id,
            outcome=outcome,
        )

    # 校验 write_file logical path 与批准的 canonical target 完全一致
    def _validate_write_path(self, raw_path: str) -> str:
        if not raw_path:
            raise ScopeDeniedError("write target is required")
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ScopeDeniedError("write target must be workspace-relative")
        logical = path.as_posix().removeprefix("./")
        scope = self._context.scope
        if logical not in scope.files_to_modify and logical not in scope.files_to_create:
            raise ScopeDeniedError("write target is outside approved file scope")
        resolver = WorkspacePathResolver(self._context.workspace_root)
        policy = WorkspaceAccessPolicy(resolver.root)
        try:
            candidate = resolver.resolve_for_write(logical)
            policy.ensure_allowed(logical, candidate)
        except Exception as exc:
            raise ScopeDeniedError("write target is not a safe workspace path") from exc
        canonical = candidate.relative_to(resolver.root).as_posix()
        if canonical != logical:
            raise ScopeDeniedError("symlink alias is not an approved logical target")
        if not candidate.parent.exists() or not candidate.parent.is_dir():
            raise ScopeDeniedError("write target parent must already exist")
        if candidate.parent.is_symlink():
            raise ScopeDeniedError("write target parent cannot be a symlink alias")
        return logical


# 读取一个 target 的安全可比较 state，不让不存在路径抛出到 capture 阶段
def _read_path_state(resolver: WorkspacePathResolver, logical: str) -> PathState:
    try:
        candidate = resolver.resolve_for_write(logical)
        if candidate.is_symlink():
            return PathState(True, None, is_symlink=True)
        if not candidate.exists():
            return PathState(False, None)
        if not candidate.is_file():
            return PathState(True, None, is_file=False)
        return PathState(True, _file_digest(candidate), is_file=True)
    except Exception as exc:
        return PathState(False, None, error=type(exc).__name__)
