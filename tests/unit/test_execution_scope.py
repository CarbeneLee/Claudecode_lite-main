from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from kama_claude.core.approval import (
    ApprovalRecord,
    CommittedPlanEvidence,
    CommittedPlanReceipt,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.execution_scope import (
    ExecutionMutationState,
    ExecutionScope,
    ScopeDeniedError,
    ScopedExecutionContext,
    ScopeRequiredError,
)
from kama_claude.core.grounding import (
    SnapshotBuilder,
    canonical_digest,
    workspace_identity,
)
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.planning import ExactPlannerDecisionV2, build_plan_view
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.invocation import ScopedToolInvoker
from kama_claude.core.tools.registry import ToolRegistry


class _PermissionRecorder:
    # 记录 permission 是否在 scope authorize 之后被调用
    def __init__(self) -> None:
        self.calls = 0

    # 允许 generic permission flow 继续，但不应覆盖 scope denial
    async def check_and_wait(self, **_: Any) -> tuple[bool, str]:
        self.calls += 1
        return True, "auto_allow"


class _WriteTool(BaseTool):
    name = "write_file"
    description = "test writer"
    input_schema: dict[str, object] = {"type": "object"}

    # 绑定 workspace 并记录调用次数
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    # 执行最小 write_file 语义，不自动创建父目录
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        path = self.root / str(params["path"])
        path.write_text(str(params["content"]), encoding="utf-8")
        return ToolResult(content="written")


class _RetryingWriteTool(_WriteTool):
    # 返回一次可重试错误以验证 mutation audit 先于 retry decision
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        path = self.root / str(params["path"])
        path.write_text(str(params["content"]), encoding="utf-8")
        return ToolResult(
            content="temporary failure after mutation",
            is_error=True,
            error_type="transient_error",
        )


class _ExceptionWriteTool(_WriteTool):
    # 写入 intended 内容后抛出普通异常，验证异常路径也先做 post-state audit
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        (self.root / str(params["path"])).write_text(
            str(params["content"]),
            encoding="utf-8",
        )
        raise RuntimeError("failure after mutation")


class _TimeoutWriteTool(_WriteTool):
    # 写入 intended 内容后阻塞，验证 timeout 路径也先做 post-state audit
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        (self.root / str(params["path"])).write_text(
            str(params["content"]),
            encoding="utf-8",
        )
        await asyncio.sleep(0.05)
        return ToolResult(content="late")


class _CancelTool(_WriteTool):
    # 配置取消前是否写入 intended 或 unexpected 内容
    def __init__(self, root: Path, written_content: str | None) -> None:
        super().__init__(root)
        self.written_content = written_content

    # 执行写入后传播 CancelledError，或直接取消
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.written_content is not None:
            (self.root / str(params["path"])).write_text(
                self.written_content,
                encoding="utf-8",
            )
        raise asyncio.CancelledError


class _IdentityCancelTool(_WriteTool):
    # 保存 primary cancellation identity，供 secondary audit cancellation 对照
    def __init__(self, root: Path, written_content: str) -> None:
        super().__init__(root)
        self.written_content = written_content
        self.cancellation = asyncio.CancelledError("primary-tool-cancellation")

    # 写入 intended 内容后抛出预先保存的 primary cancellation 对象
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        (self.root / str(params["path"])).write_text(
            self.written_content,
            encoding="utf-8",
        )
        raise self.cancellation


class _SingleFlightWriteTool(_WriteTool):
    # 用两个 event 暴露首个 invocation 是否被第二个 invocation 并发穿透
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.started_paths: list[str] = []

    # 首次调用等待显式释放，后续调用若绕过 invoker lock 会形成 active=2
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        logical = str(params["path"])
        self.started_paths.append(logical)
        try:
            if self.calls == 1:
                self.first_started.set()
                await self.release_first.wait()
            (self.root / logical).write_text(
                str(params["content"]),
                encoding="utf-8",
            )
            return ToolResult(content="written")
        finally:
            self.active -= 1


class _ReadRetryTool(BaseTool):
    name = "read_file"
    description = "test reader"
    input_schema: dict[str, object] = {"type": "object"}

    # 初始化剩余 transient failures
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    # 返回 transient error 若干次后成功，不产生 workspace mutation
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls <= self.failures:
            return ToolResult(
                content="temporary",
                is_error=True,
                error_type="transient_error",
            )
        return ToolResult(content="ok")


class _SerializedWriteTool(_WriteTool):
    # 记录并发执行峰值，验证单一 scoped context 的调用被串行化
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.active = 0
        self.max_active = 0

    # 在写入前让并发任务交错，确保没有 single-flight 时测试会观察到重叠
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            path = self.root / str(params["path"])
            path.write_text(str(params["content"]), encoding="utf-8")
            return ToolResult(content="written")
        finally:
            self.active -= 1


def _scope(
    *,
    workspace_id: str = "test-workspace",
    snapshot_digest: str = "snapshot-digest",
    modify: tuple[str, ...] = (),
    create: tuple[str, ...] = ("new.txt",),
    capabilities: tuple[str, ...] = ("write_file",),
) -> ExecutionScope:
    return ExecutionScope._create(
        session_id="session-1",
        workspace_id=workspace_id,
        projection_key="pv1:run-1:decision-1:v1",
        decision_id="decision-1",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot_digest,
        files_to_modify=modify,
        files_to_create=create,
        capabilities=capabilities,
        dependency_changes=(),
        protocol_or_schema_changes=(),
    )


# 根据 scope targets 重建测试 snapshot，生产 context 只接受 verified materialization
def _context(
    scope: ExecutionScope,
    root: Path,
    snapshot: Any | None = None,
) -> ScopedExecutionContext:
    verified_snapshot = snapshot or SnapshotBuilder(root).capture(
        planned_existing_targets=scope.files_to_modify,
        planned_new_targets=scope.files_to_create,
    )
    return ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=verified_snapshot,
        workspace_root=root,
        execution_id="exec-1",
    )


# 构造与真实 workspace identity 和 snapshot digest 绑定的测试 scope
def _scope_for(
    root: Path,
    *,
    modify: tuple[str, ...] = (),
    create: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = ("write_file",),
    grounding_paths: tuple[str, ...] = (),
) -> tuple[ExecutionScope, Any]:
    snapshot = SnapshotBuilder(root).capture(
        grounding_paths=grounding_paths,
        planned_existing_targets=modify,
        planned_new_targets=create,
    )
    scope = _scope(
        workspace_id=snapshot.workspace_id,
        snapshot_digest=snapshot.snapshot_digest,
        modify=modify,
        create=create,
        capabilities=capabilities,
    )
    return scope, snapshot


def _call(
    name: str,
    params: dict[str, object],
    *,
    call_id: str = "tool-call-1",
) -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name=name, input=params)


# 构造最小 exact V2 decision、approved record、receipt 与 current snapshot
def _approved_fixture(
    tmp_path: Path,
    *,
    workspace_name: str = "workspace",
    planned_new: bool = False,
    relevant_manifest: bool = False,
) -> tuple[Path, ExactPlannerDecisionV2, ApprovalRecord, CommittedPlanReceipt, Any]:
    workspace = tmp_path / workspace_name
    target = workspace / "src" / "target.py"
    target.parent.mkdir(parents=True)
    if not planned_new:
        target.write_text("VALUE = 1\n", encoding="utf-8")
    manifests: tuple[str, ...] = ()
    if relevant_manifest:
        (workspace / "pyproject.toml").write_text(
            "[project]\nname = 'fixture'\n",
            encoding="utf-8",
        )
        manifests = ("pyproject.toml",)
    files_to_modify = () if planned_new else ("src/target.py",)
    files_to_create = ("src/target.py",) if planned_new else ()
    snapshot = SnapshotBuilder(workspace).capture(
        planned_existing_targets=files_to_modify,
        planned_new_targets=files_to_create,
        relevant_manifests=manifests,
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "decision_id": "decision-1",
        "version": 1,
        "goal": "change target",
        "requirements": [],
        "architecture_slice_id": "slice-1",
        "architecture_slice_version": 1,
        "architecture_slice_content_digest": "slice-digest",
        "snapshot_digest": snapshot.snapshot_digest,
        "architecture_mode": "preserve",
        "selected_approach": "edit target",
        "existing_patterns_reused": [],
        "intended_changes": [],
        "files_to_modify": list(files_to_modify),
        "files_to_create": list(files_to_create),
        "allowed_capabilities": ["read_file", "write_file"],
        "dependency_changes": [],
        "protocol_or_schema_changes": [],
        "verification_plan": [],
        "non_goals": [],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
    }
    payload["content_digest"] = canonical_digest(payload)
    decision = ExactPlannerDecisionV2.model_validate(payload)
    plan = build_plan_view(decision, top_level_run_id="run-1")
    evidence = CommittedPlanEvidence(
        session_id="session-1",
        top_level_run_id="run-1",
        planner_run_id="planner-1",
        decision_id=decision.decision_id,
        decision_version=decision.version,
        decision_content_digest=decision.content_digest,
        projection_key=plan.projection_key,
        projection_digest=plan.projection_digest,
        plan_ready_journal_event_id="ready-1",
        run_finished_journal_event_id="finished-1",
    )
    receipt = CommittedPlanReceipt.from_evidence(evidence)
    approval = ApprovalRecord.create(
        session_id="session-1",
        projection_key=plan.projection_key,
        decision_id=decision.decision_id,
        decision_version=decision.version,
        content_digest=decision.content_digest,
        commit_receipt_digest=receipt.receipt_digest,
        action="approve",
    )
    return workspace, decision, approval, receipt, snapshot


# 功能：验证 scope derivation 重新校验 exact artifacts、receipt、snapshot 与 target baseline
# 设计：使用真实 SnapshotBuilder 和 immutable model digest，证明 scope 不是从 ApprovalRecord 猜测出来的
def test_scope_derives_from_approved_exact_artifacts(tmp_path: Path) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)

    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace,
    )

    assert scope.session_id == "session-1"
    assert scope.workspace_id == snapshot.workspace_id
    assert scope.files_to_modify == ("src/target.py",)
    assert scope.commit_receipt_digest == receipt.receipt_digest


# 功能：验证 workspace stale 或非 approve record 不能 materialize ExecutionScope
# 设计：分别覆盖 approval action 和 relevant digest 两个独立 fail-closed 边界，不修改任何 authority artifact
def test_scope_derivation_fails_closed_for_stale_or_unapproved_state(tmp_path: Path) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)
    (workspace / "src" / "target.py").write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ScopeDeniedError):
        ExecutionScope.from_approved(
            decision=decision,
            approval_record=approval,
            receipt=receipt,
            snapshot=snapshot,
            workspace_root=workspace,
        )

    (workspace / "src" / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    rejected = ApprovalRecord.create(
        session_id=approval.session_id,
        projection_key=approval.projection_key,
        decision_id=approval.decision_id,
        decision_version=approval.decision_version,
        content_digest=approval.content_digest,
        commit_receipt_digest=approval.commit_receipt_digest,
        action="reject",
    )
    with pytest.raises(ScopeDeniedError):
        ExecutionScope.from_approved(
            decision=decision,
            approval_record=rejected,
            receipt=receipt,
            snapshot=snapshot,
            workspace_root=workspace,
        )


# 功能：验证缺失、pending 或 rejected approval 都不能派生执行 scope
# 设计：在同一份 committed fixture 上分别覆盖三种非批准状态，证明 receipt 不能替代用户 authority
@pytest.mark.parametrize("approval_kind", ["missing", "pending", "rejected"])
def test_scope_requires_approved_record(
    tmp_path: Path,
    approval_kind: str,
) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)
    candidate: object
    if approval_kind == "missing":
        candidate = None
    elif approval_kind == "pending":
        candidate = object()
    else:
        candidate = approval.model_copy(update={"action": "reject"})

    with pytest.raises(ScopeDeniedError):
        ExecutionScope.from_approved(
            decision=decision,
            approval_record=candidate,  # type: ignore[arg-type]
            receipt=receipt,
            snapshot=snapshot,
            workspace_root=workspace,
        )


# 功能：验证 receipt 或 decision identity 不匹配时 scope 派生 fail closed
# 设计：分别篡改 derived receipt 和 exact decision 的绑定字段，确保不接受跨 projection 或跨 decision 借用
def test_scope_rejects_receipt_and_decision_mismatch(tmp_path: Path) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)
    mismatched_receipt = receipt.model_copy(
        update={"projection_key": "pv1:run-1:other:v1"}
    )
    with pytest.raises(ScopeDeniedError):
        ExecutionScope.from_approved(
            decision=decision,
            approval_record=approval,
            receipt=mismatched_receipt,
            snapshot=snapshot,
            workspace_root=workspace,
        )

    mismatched_decision = decision.model_copy(update={"decision_id": "other"})
    with pytest.raises(ScopeDeniedError):
        ExecutionScope.from_approved(
            decision=mismatched_decision,
            approval_record=approval,
            receipt=receipt,
            snapshot=snapshot,
            workspace_root=workspace,
        )


# 功能：验证 approved scope 不能从 workspace A 移植到内容相同的 workspace B
# 设计：B 复制同路径同字节目标并配置 permission/tool recorder，断言 context 构造先拒绝且 B 零副作用
def test_verified_context_rejects_cross_workspace_scope_transplant(
    tmp_path: Path,
) -> None:
    workspace_a, decision, approval, receipt, snapshot = _approved_fixture(
        tmp_path,
        workspace_name="workspace-a",
    )
    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace_a,
    )
    workspace_b = tmp_path / "workspace-b"
    target_b = workspace_b / "src" / "target.py"
    target_b.parent.mkdir(parents=True)
    target_b.write_text("VALUE = 1\n", encoding="utf-8")
    before_b = target_b.read_bytes()
    permission = _PermissionRecorder()
    tool = _WriteTool(workspace_b)

    with pytest.raises(ScopeDeniedError):
        ScopedExecutionContext.from_verified_scope(
            scope=scope,
            snapshot=snapshot,
            workspace_root=workspace_b,
            execution_id="exec-cross-workspace",
        )

    assert target_b.read_bytes() == before_b
    assert permission.calls == 0
    assert tool.calls == 0


# 功能：验证 approved scope 在原 workspace 且 snapshot 未漂移时可物化 context
# 设计：复用同一权威 fixture，断言 canonical root、workspace identity 和 active baseline 同时绑定成功
def test_verified_context_accepts_unchanged_snapshot(tmp_path: Path) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)
    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace,
    )

    context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-current",
    )

    canonical_root, workspace_id = workspace_identity(workspace)
    assert context.workspace_root == canonical_root
    assert context.scope.workspace_id == workspace_id
    assert context.mutation_state.status == "active"
    assert context.mutation_state.baseline_path_states["src/target.py"].exists


# 功能：验证 scope 派生后 existing target 漂移会在 context materialization 时拒绝
# 设计：先成功派生 immutable scope，再改写目标，证明旧 scope 不能把漂移后的字节吸收成新 baseline
def test_verified_context_rejects_target_drift_after_scope_derivation(
    tmp_path: Path,
) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(tmp_path)
    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace,
    )
    (workspace / "src" / "target.py").write_text("DRIFTED = 1\n", encoding="utf-8")

    with pytest.raises(ScopeDeniedError):
        ScopedExecutionContext.from_verified_scope(
            scope=scope,
            snapshot=snapshot,
            workspace_root=workspace,
        )


# 功能：验证非 target 的 relevant manifest 漂移同样使 reusable scope 失效
# 设计：target 字节保持不变，只改 snapshot 纳入的 pyproject.toml，隔离全量 freshness recheck 责任
def test_verified_context_rejects_relevant_manifest_drift(
    tmp_path: Path,
) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(
        tmp_path,
        relevant_manifest=True,
    )
    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace,
    )
    (workspace / "pyproject.toml").write_text(
        "[project]\nname = 'drifted'\n",
        encoding="utf-8",
    )

    with pytest.raises(ScopeDeniedError):
        ScopedExecutionContext.from_verified_scope(
            scope=scope,
            snapshot=snapshot,
            workspace_root=workspace,
        )


# 功能：验证 scope 派生后 planned-new target 外部出现会在 context materialization 前拒绝
# 设计：先以 absent target 派生 scope，再外部创建同路径，证明 appearance 不能成为 execution baseline
def test_verified_context_rejects_planned_new_appearance(
    tmp_path: Path,
) -> None:
    workspace, decision, approval, receipt, snapshot = _approved_fixture(
        tmp_path,
        planned_new=True,
    )
    scope = ExecutionScope.from_approved(
        decision=decision,
        approval_record=approval,
        receipt=receipt,
        snapshot=snapshot,
        workspace_root=workspace,
    )
    target = workspace / "src" / "target.py"
    target.write_text("external\n", encoding="utf-8")

    with pytest.raises(ScopeDeniedError):
        ScopedExecutionContext.from_verified_scope(
            scope=scope,
            snapshot=snapshot,
            workspace_root=workspace,
        )

    assert target.read_text(encoding="utf-8") == "external\n"


# 功能：验证生产 API 不暴露 arbitrary scope 或未验证 context/mutation-state 构造路径
# 设计：直接调用 public constructor 并检查 legacy builder/capture 均不可用，锁定 authoritative derivation 边界
def test_execution_scope_public_api_requires_authoritative_derivation() -> None:
    assert not hasattr(ExecutionScope, "create")
    assert not hasattr(ExecutionMutationState, "capture")
    with pytest.raises(ScopeDeniedError):
        ExecutionScope()
    with pytest.raises(ScopeRequiredError):
        ScopedExecutionContext()


async def _invoke(
    root: Path,
    scope: ExecutionScope,
    tool: BaseTool,
    call: ToolCallBlock,
    permission: _PermissionRecorder | None = None,
    timeout: float = 120.0,
    context: ScopedExecutionContext | None = None,
) -> tuple[ScopedToolInvoker, ToolResult | None]:
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(
        registry,
        EventBus(),
        "run-1",
        context or _context(scope, root),
        timeout=timeout,
        permission_manager=permission,  # type: ignore[arg-type]
    )
    try:
        return invoker, await invoker.invoke(call)
    except asyncio.CancelledError:
        raise


# 功能：验证 ExecutionScope 只接受固定 MVP capability ceiling 且 digest 稳定
# 设计：用同一 canonical 输入创建两个 scope，既锁定白名单又排除 dict 顺序造成的 identity 漂移
def test_scope_digest_and_capability_ceiling(tmp_path: Path) -> None:
    scope_a = _scope(capabilities=("read_file", "write_file"))
    scope_b = _scope(capabilities=("write_file", "read_file"))
    other_workspace = _scope(
        workspace_id="other-workspace",
        capabilities=("read_file", "write_file"),
    )
    assert scope_a.scope_digest == scope_b.scope_digest
    assert scope_a.scope_digest != other_workspace.scope_digest
    with pytest.raises(ScopeDeniedError):
        ExecutionScope._create(
            session_id="session-1",
            workspace_id="workspace-test",
            projection_key="projection",
            decision_id="decision-1",
            decision_version=1,
            decision_content_digest="d",
            commit_receipt_digest="r",
            snapshot_digest="s",
            capabilities=("search_semantic",),
        )


# 功能：验证 scoped invoker 缺少 context 时 fail closed
# 设计：构造期直接拒绝 None，证明不存在 optional scope 退回 generic invocation 的旁路
def test_scoped_invoker_requires_non_null_context() -> None:
    with pytest.raises(ScopeRequiredError):
        ScopedToolInvoker(ToolRegistry(), EventBus(), "run-1", None)  # type: ignore[arg-type]


# 功能：验证 scope denial 发生在 PermissionManager 之前且不会写入越界目标
# 设计：用 recorder 观察 permission 调用次数，选择未批准的 create path 作为最小越界案例
async def test_scope_denial_precedes_permission(tmp_path: Path) -> None:
    permission = _PermissionRecorder()
    scope, _snapshot = _scope_for(tmp_path, create=("approved.txt",))
    _invoker, result = await _invoke(
        tmp_path,
        scope,
        _WriteTool(tmp_path),
        _call("write_file", {"path": "outside.txt", "content": "x"}),
        permission,
    )
    assert result is not None and result.is_error
    assert result.error_type == "scope_denied"
    assert permission.calls == 0
    assert not (tmp_path / "outside.txt").exists()


# 功能：验证 planned create 要求 parent 已存在且不隐式 mkdir
# 设计：缺失 parent 时在 tool.invoke 前拒绝，并断言整个目录链仍不存在
async def test_missing_parent_denied_without_mkdir(tmp_path: Path) -> None:
    scope, snapshot = _scope_for(tmp_path, create=("missing/child.txt",))
    with pytest.raises(ScopeDeniedError):
        _context(scope, tmp_path, snapshot)
    assert not (tmp_path / "missing").exists()


# 功能：验证 scoped write 拒绝指向 workspace 内部别名的 symlink logical path
# 设计：alias 与真实文件都在 workspace 内，只有 canonical/logical equality 检查才能识别该旁路
async def test_symlink_alias_is_denied(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_text("base", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(tmp_path / "real.txt")
    scope, snapshot = _scope_for(tmp_path, modify=("alias.txt",), create=())
    with pytest.raises(ScopeDeniedError):
        _context(scope, tmp_path, snapshot)
    assert (tmp_path / "real.txt").read_text(encoding="utf-8") == "base"


# 功能：验证批准的 existing target 在 execution 开始前消失时被拒绝
# 设计：scope 记录 existing baseline 后删除文件，覆盖 target existence precondition 而非仅 digest drift
async def test_existing_target_missing_is_denied(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("base", encoding="utf-8")
    scope, snapshot = _scope_for(tmp_path, modify=("target.txt",), create=())
    (tmp_path / "target.txt").unlink()
    with pytest.raises(ScopeDeniedError):
        _context(scope, tmp_path, snapshot)


# 功能：验证 planned-new target 在 execution 创建前已经存在时被拒绝
# 设计：先 capture absent baseline 再由外部创建，区分外部创建和 execution 自己的后续写入
async def test_planned_new_target_already_exists_is_denied(tmp_path: Path) -> None:
    scope, snapshot = _scope_for(tmp_path, create=("new.txt",))
    (tmp_path / "new.txt").write_text("external", encoding="utf-8")
    with pytest.raises(ScopeDeniedError):
        _context(scope, tmp_path, snapshot)


# 功能：验证 mutating exception 与 timeout 都在 retry 前记录 intended post-state
# 设计：两个 fake tool 都只调用一次，分别锁定普通异常和 wait_for timeout 的 audit 语义
@pytest.mark.parametrize("tool_kind", ["exception", "timeout"])
async def test_mutation_is_audited_before_exception_or_timeout_retry(
    tmp_path: Path,
    tool_kind: str,
) -> None:
    (tmp_path / "target.txt").write_text("base", encoding="utf-8")
    scope, _snapshot = _scope_for(tmp_path, modify=("target.txt",), create=())
    tool: BaseTool
    timeout = 120.0
    if tool_kind == "exception":
        tool = _ExceptionWriteTool(tmp_path)
    else:
        tool = _TimeoutWriteTool(tmp_path)
        timeout = 0.001
    invoker, result = await _invoke(
        tmp_path,
        scope,
        tool,
        _call("write_file", {"path": "target.txt", "content": "changed"}),
        timeout=timeout,
    )
    assert result is not None and result.is_error
    assert result.error_type in {"execution_error", "timeout"}
    assert tool.calls == 1
    assert invoker.context.mutation_state.mutation_ledger[0].classification == (
        "abnormal-but-authorized"
    )


# 功能：验证同一 execution 创建文件后可沿 expected-state ledger 再次修改
# 设计：两次真实写入之间不引入外部变化，锁定 ABSENT→CREATED→CREATED 生命周期和两条 ledger
async def test_own_new_file_writes_advance_expected_state(tmp_path: Path) -> None:
    (tmp_path / "parent").mkdir()
    scope, _snapshot = _scope_for(tmp_path, create=("parent/new.txt",))
    context = _context(scope, tmp_path)
    tool = _WriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    first = await invoker.invoke(
        _call("write_file", {"path": "parent/new.txt", "content": "one"})
    )
    second = await invoker.invoke(
        _call("write_file", {"path": "parent/new.txt", "content": "two"})
    )

    assert not first.is_error and not second.is_error
    assert len(context.mutation_state.mutation_ledger) == 2
    assert context.mutation_state.expected_path_states["parent/new.txt"].digest


# 功能：验证外部修改 expected state 后阻止后续 scoped mutation
# 设计：先完成本 execution 写入，再直接改写文件模拟外部 drift，第二次调用必须 fail closed
async def test_external_modification_between_writes_is_denied(tmp_path: Path) -> None:
    (tmp_path / "new.txt").write_text("base", encoding="utf-8")
    scope, _snapshot = _scope_for(tmp_path, modify=("new.txt",), create=())
    context = _context(scope, tmp_path)
    tool = _WriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    first = await invoker.invoke(
        _call("write_file", {"path": "new.txt", "content": "one"})
    )
    (tmp_path / "new.txt").write_text("external", encoding="utf-8")
    second = await invoker.invoke(
        _call("write_file", {"path": "new.txt", "content": "two"})
    )

    assert not first.is_error
    assert second.is_error
    assert second.error_type == "external_workspace_drift"
    assert context.mutation_state.blocked


# 功能：验证 read-only scoped retry 在无 mutation 时保留 generic retry 语义
# 设计：read_file fake 只返回 transient error，不触碰 workspace，锁定三次调用后成功
async def test_read_only_retry_without_mutation_is_preserved(tmp_path: Path) -> None:
    scope, _snapshot = _scope_for(
        tmp_path,
        create=(),
        capabilities=("read_file",),
    )
    tool = _ReadRetryTool(2)
    _invoker, result = await _invoke(
        tmp_path,
        scope,
        tool,
        _call("read_file", {"path": "missing.txt"}),
    )
    assert result is not None and not result.is_error
    assert tool.calls == 3


# 功能：验证 retryable error 产生 mutation 时先 audit 再 veto retry
# 设计：工具实际写入 approved target 后返回 transient error，断言只有一次调用和一条 ledger
async def test_retryable_error_after_mutation_is_not_retried(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("base", encoding="utf-8")
    scope, _snapshot = _scope_for(tmp_path, modify=("target.txt",), create=())
    tool = _RetryingWriteTool(tmp_path)
    _invoker, result = await _invoke(
        tmp_path,
        scope,
        tool,
        _call("write_file", {"path": "target.txt", "content": "changed"}),
    )
    assert result is not None and result.is_error
    assert result.error_type == "transient_error"
    assert tool.calls == 1


# 功能：验证取消发生在 mutation 前时原样传播且不记录 mutation
# 设计：cancel tool 不写文件，使用 pytest.raises 检查 exception identity 与空 ledger
async def test_cancel_before_mutation_propagates_without_ledger(tmp_path: Path) -> None:
    scope, _snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, None)
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.mutation_ledger == []
    assert not context.mutation_state.blocked


# 功能：验证 intended 与 before 完全相同时取消仍视为 no-op
# 设计：取消工具写回 baseline bytes，锁定 actual == intended == before 不创建 mutation ledger
async def test_cancel_same_as_before_is_noop(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("base", encoding="utf-8")
    scope, _snapshot = _scope_for(tmp_path, modify=("target.txt",), create=())
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, "base")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "target.txt", "content": "base"}))

    assert context.mutation_state.mutation_ledger == []
    assert not context.mutation_state.blocked


# 功能：验证取消发生在 intended write 后会记录 abnormal-but-authorized mutation
# 设计：写入 exact content 后取消，断言 cancellation 仍是 primary 且 ledger 保存完整 before/after
async def test_cancel_after_intended_mutation_records_authorized_ledger(
    tmp_path: Path,
) -> None:
    scope, _snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, "x")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.mutation_ledger[0].classification == (
        "abnormal-but-authorized"
    )
    assert not context.mutation_state.blocked


# 功能：验证取消后 observable unexpected mutation 被标记 inconclusive 并阻断执行
# 设计：工具写入非 intended 内容后取消，禁止把任意 observable mutation 误称为 authorized
async def test_cancel_after_unexpected_mutation_blocks_execution(tmp_path: Path) -> None:
    scope, _snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, "unexpected")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.mutation_ledger[0].classification == "inconclusive"
    assert context.mutation_state.blocked
    assert context.mutation_state.status == "inconclusive"


# 功能：验证 cancellation 期间 audit 自身失败只产生 secondary inconclusive 状态
# 设计：让 post-state reader 抛出异常，断言原始 CancelledError 仍传播且 execution 被阻断
async def test_cancel_with_audit_failure_keeps_cancellation_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, _snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, "x")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    async def _fail_audit(*_: Any, **__: Any) -> Any:
        raise OSError("audit unavailable")

    monkeypatch.setattr(
        "kama_claude.core.execution_scope.ExecutionMutationState.audit_mutation",
        _fail_audit,
    )
    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.blocked
    assert context.mutation_state.status == "inconclusive"


# 功能：验证 primary tool cancellation 遇到 secondary audit cancellation 时仍保持原对象
# 设计：tool 与 audit 分别抛出可区分 CancelledError，断言 identity、blocked 和 inconclusive 三个契约
async def test_cancel_during_post_state_audit_preserves_primary_and_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path, snapshot)
    tool = _IdentityCancelTool(tmp_path, "x")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)
    audit_cancellation = asyncio.CancelledError("secondary-audit-cancellation")

    # 模拟 post-state reader 自身被取消，确保该 secondary 不覆盖 tool primary
    async def _cancel_audit(*_: Any, **__: Any) -> Any:
        raise audit_cancellation

    monkeypatch.setattr(
        "kama_claude.core.execution_scope.ExecutionMutationState.audit_mutation",
        _cancel_audit,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert caught.value is tool.cancellation
    assert caught.value is not audit_cancellation
    assert context.mutation_state.blocked
    assert context.mutation_state.status == "inconclusive"


# 功能：验证 post-state audit 超过 bounded timeout 时标记 inconclusive
# 设计：用可控慢速 reader 和极小 timeout，不等待真实文件系统异常，锁定 audit 有界而非无限阻塞
async def test_post_state_audit_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, _snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path)
    tool = _CancelTool(tmp_path, "x")
    registry = ToolRegistry()
    registry.register(tool)

    def _slow_reader(*_: Any, **__: Any) -> Any:
        time.sleep(0.05)
        return context.mutation_state.baseline_path_states["new.txt"]

    monkeypatch.setattr(
        "kama_claude.core.execution_scope._POST_STATE_AUDIT_TIMEOUT_S",
        0.001,
    )
    monkeypatch.setattr(
        "kama_claude.core.execution_scope._read_path_state",
        _slow_reader,
    )
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.blocked
    assert context.mutation_state.status == "inconclusive"


# 功能：验证同一 ScopedToolInvoker 的两个并发请求被完整 pipeline 串行化
# 设计：首个 tool invocation 用 event 阻塞，第二个同时提交；断言 max_active=1 且 ledger provenance 不串线
async def test_scoped_tool_invoker_serializes_concurrent_requests(
    tmp_path: Path,
) -> None:
    scope, snapshot = _scope_for(
        tmp_path,
        create=("first.txt", "second.txt"),
    )
    context = _context(scope, tmp_path, snapshot)
    tool = _SingleFlightWriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    first = asyncio.create_task(
        invoker.invoke(
            _call(
                "write_file",
                {"path": "first.txt", "content": "one"},
                call_id="tool-call-first",
            )
        )
    )
    await asyncio.wait_for(tool.first_started.wait(), timeout=1.0)
    second = asyncio.create_task(
        invoker.invoke(
            _call(
                "write_file",
                {"path": "second.txt", "content": "two"},
                call_id="tool-call-second",
            )
        )
    )
    await asyncio.sleep(0)

    assert tool.calls == 1
    assert not second.done()
    tool.release_first.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert not first_result.is_error and not second_result.is_error
    assert tool.max_active == 1
    assert tool.started_paths == ["first.txt", "second.txt"]
    assert [entry.path for entry in context.mutation_state.mutation_ledger] == [
        "first.txt",
        "second.txt",
    ]
    assert [entry.tool_call_id for entry in context.mutation_state.mutation_ledger] == [
        "tool-call-first",
        "tool-call-second",
    ]


# 功能：验证 scope authorization 不能暴露 search_semantic 或未知副作用工具
# 设计：分别注册 unsupported names，统一断言 scope_denied 且不进入 tool.invoke
@pytest.mark.parametrize("name", ["search_semantic", "bash", "mcp.call", "spawn_agent"])
async def test_unsupported_scoped_capabilities_are_denied(
    tmp_path: Path,
    name: str,
) -> None:
    scope, _snapshot = _scope_for(
        tmp_path,
        create=(),
        capabilities=("read_file",),
    )
    tool = _ReadRetryTool(0)
    tool.name = name
    _invoker, result = await _invoke(tmp_path, scope, tool, _call(name, {}))
    assert result is not None and result.error_type == "scope_denied"


# 功能：验证 scope 不能从一个 workspace 移植到另一个 workspace
# 设计：使用两个真实 root 和同一份 snapshot，断言 context 在工具或 baseline 前即拒绝跨 workspace 绑定
async def test_scope_workspace_identity_is_bound_at_context_materialization(
    tmp_path: Path,
) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    (workspace_a / "target.txt").write_text("a", encoding="utf-8")
    (workspace_b / "target.txt").write_text("b", encoding="utf-8")
    scope, snapshot = _scope_for(
        workspace_a,
        modify=("target.txt",),
        create=(),
    )

    with pytest.raises(ScopeDeniedError):
        _context(scope, workspace_b, snapshot)

    context = _context(scope, workspace_a, snapshot)
    assert context.workspace_root == workspace_a.resolve()


# 功能：验证 scope 派生后的非 target relevant 内容变化会阻止 context materialization
# 设计：grounding snapshot 显式包含 README，修改该非写入 target 后仍必须由完整 freshness recheck 拒绝
async def test_context_rechecks_non_target_relevant_snapshot_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("before", encoding="utf-8")
    scope, snapshot = _scope_for(
        tmp_path,
        create=("new.txt",),
        grounding_paths=("README.md",),
    )
    (tmp_path / "README.md").write_text("after", encoding="utf-8")

    with pytest.raises(ScopeDeniedError):
        _context(scope, tmp_path, snapshot)


# 功能：验证 verified context 在同一 workspace、current snapshot 和 target state 下可成功物化
# 设计：走唯一公开 from_verified_scope 入口并检查 canonical root 与 execution-local baseline 已建立
async def test_verified_context_happy_path(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("base", encoding="utf-8")
    scope, snapshot = _scope_for(tmp_path, modify=("target.txt",), create=())

    context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=tmp_path,
        execution_id="verified-exec",
    )

    assert context.execution_id == "verified-exec"
    assert context.workspace_root == tmp_path.resolve()
    assert context.mutation_state.baseline_path_states["target.txt"].digest


# 功能：验证 scope 与 context 不暴露任意公开构造旁路
# 设计：只允许测试使用模块私有 helper，公共 API 必须给出稳定 scope 错误而不是接受未验证字段
def test_scope_and_context_public_construction_is_not_available() -> None:
    assert not hasattr(ExecutionScope, "create")
    with pytest.raises(ScopeDeniedError):
        ExecutionScope()
    with pytest.raises(ScopeRequiredError):
        ScopedExecutionContext()


# 功能：验证同一 scoped invoker 的并发调用被 single-flight 锁串行化
# 设计：两个 mutating calls 在 fake tool 内主动交错，断言 active 峰值为 1 且两次 mutation ledger 都完整记录
async def test_scoped_invoker_serializes_concurrent_calls(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("base-1", encoding="utf-8")
    (tmp_path / "second.txt").write_text("base-2", encoding="utf-8")
    scope, snapshot = _scope_for(
        tmp_path,
        modify=("first.txt", "second.txt"),
        create=(),
    )
    context = _context(scope, tmp_path, snapshot)
    tool = _SerializedWriteTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    first, second = await asyncio.gather(
        invoker.invoke(
            _call(
                "write_file",
                {"path": "first.txt", "content": "one"},
                call_id="call-first",
            )
        ),
        invoker.invoke(
            _call(
                "write_file",
                {"path": "second.txt", "content": "two"},
                call_id="call-second",
            )
        ),
    )

    assert not first.is_error and not second.is_error
    assert tool.max_active == 1
    assert {entry.path for entry in context.mutation_state.mutation_ledger} == {
        "first.txt",
        "second.txt",
    }


# 功能：验证 post-state audit 自身被取消时 execution 标记 inconclusive 但原始取消继续传播
# 设计：替换异步 audit hook 抛出 CancelledError，覆盖 cancellation secondary failure 而不吞掉 primary outcome
async def test_audit_cancellation_marks_inconclusive_and_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, snapshot = _scope_for(tmp_path, create=("new.txt",))
    context = _context(scope, tmp_path, snapshot)
    tool = _CancelTool(tmp_path, "x")
    registry = ToolRegistry()
    registry.register(tool)
    invoker = ScopedToolInvoker(registry, EventBus(), "run-1", context)

    async def _cancel_audit(*_: Any, **__: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "kama_claude.core.execution_scope.ExecutionMutationState.audit_mutation",
        _cancel_audit,
    )

    with pytest.raises(asyncio.CancelledError):
        await invoker.invoke(_call("write_file", {"path": "new.txt", "content": "x"}))

    assert context.mutation_state.blocked
    assert context.mutation_state.status == "inconclusive"
