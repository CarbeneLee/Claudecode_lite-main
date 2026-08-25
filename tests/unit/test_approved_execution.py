from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import kama_claude.core.runner as runner_module
import kama_claude.core.tools.invocation as invocation_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.execution import (
    ApprovedExecutionBinding,
    ExecutionRequestOwner,
    ExecutionSnapshot,
    ExecutionSnapshotRelation,
    ExecutionSnapshotState,
    TrustedScopedToolRegistry,
    compare_execution_snapshots,
)
from kama_claude.core.execution_scope import ExecutionScope, ScopedExecutionContext
from kama_claude.core.grounding import SnapshotBuilder, workspace_identity
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runner import AgentRunner
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin import (
    ListDirTool,
    ReadFileTool,
    SearchCodeTool,
    WriteFileTool,
)
from kama_claude.core.tools.invocation import ApprovedScopedToolInvoker
from kama_claude.core.verification import materialize_execution_completion_receipt
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver


# 功能：验证 approved execution binding 使用 immutable canonical digest
# 设计：先创建并验证 digest，再篡改字段，确保 admission authority 不接受静默修改
def test_binding_digest_and_immutability() -> None:
    binding = ApprovedExecutionBinding.create(
        session_id="sess-1",
        request_id="req-1",
        execution_id="exec-1",
        run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        approval_record_digest="approval-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest="snapshot-digest",
        workspace_id="workspace",
        admitted_at="2026-01-01T00:00:00+00:00",
    )

    binding.verify_digest()
    assert binding.binding_digest
    with pytest.raises(Exception):
        binding.model_copy(update={"run_id": "forged"}).verify_digest()


# 功能：验证 execution snapshot 只允许生命周期单调前进
# 设计：覆盖 terminal 不被 running 覆盖以及 conflicting terminal fail-closed
def test_execution_snapshot_monotonic_relations() -> None:
    running = ExecutionSnapshot(
        execution_id="exec-1",
        run_id="run-1",
        request_id="req-1",
        projection_key="pv1:run-1:decision:v1",
        status="running",
        status_revision=2,
    )
    completed = running.model_copy(
        update={"status": "completed_unverified", "status_revision": 3}
    )
    admitted_after_running = running.model_copy(
        update={"status": "admitted", "status_revision": 3}
    )
    stale_running = running.model_copy(update={"status_revision": 2})
    conflicting = running.model_copy(
        update={"status": "failed", "status_revision": 3}
    )

    assert compare_execution_snapshots(running, admitted_after_running) is ExecutionSnapshotRelation.STALE
    assert compare_execution_snapshots(completed, stale_running) is ExecutionSnapshotRelation.STALE
    assert compare_execution_snapshots(completed, completed) is ExecutionSnapshotRelation.IDEMPOTENT
    assert compare_execution_snapshots(completed, conflicting) is ExecutionSnapshotRelation.CONFLICT


# 功能：验证 snapshot 状态机不会用更高 revision 的 admitted 覆盖 running
# 设计：通过真实 merge 路径检查 relation 与保留快照，排除只测试纯比较函数的间接证据
def test_execution_snapshot_state_rejects_running_to_admitted() -> None:
    owner = ExecutionRequestOwner(
        client_object=object(),
        session_id="sess-1",
        daemon_instance_id="daemon-1",
        projection_key="pv1:run-1:decision:v1",
        request_id="request-1",
        execution_id="exec-1",
    )
    running = ExecutionSnapshot(
        execution_id="exec-1",
        run_id="run-1",
        request_id="request-1",
        projection_key=owner.projection_key,
        status="running",
        status_revision=2,
    )
    admitted = running.model_copy(update={"status": "admitted", "status_revision": 3})
    state = ExecutionSnapshotState(owner)

    assert state.merge(running) == ExecutionSnapshotRelation.NEWER
    assert state.merge(admitted) == ExecutionSnapshotRelation.STALE
    assert state.snapshot == running


# 功能：验证 conflicting terminal claim 进入 unknown 并只安排一个 authority refresh
# 设计：状态机接收同一 owner 的两个 terminal snapshot，断言冲突不使用 last-arrival-wins
def test_execution_snapshot_conflict_is_coalesced() -> None:
    owner = ExecutionRequestOwner(
        client_object=object(),
        session_id="sess-1",
        daemon_instance_id="daemon-1",
        projection_key="pv1:run-1:decision:v1",
        request_id="request-1",
        execution_id="exec-1",
    )
    state = ExecutionSnapshotState(owner)
    running = ExecutionSnapshot(
        execution_id="exec-1",
        run_id="run-1",
        request_id="request-1",
        projection_key=owner.projection_key,
        status="running",
        status_revision=1,
    )
    approved = running.model_copy(
        update={"status": "completed_unverified", "status_revision": 2}
    )
    failed = running.model_copy(update={"status": "failed", "status_revision": 2})
    assert state.merge(running) == "newer"
    assert state.merge(approved) == "newer"
    assert state.merge(failed) == "conflict"
    assert state.status == "conflicted/unknown"
    epoch = state.begin_refresh()
    assert epoch == 1
    assert state.begin_refresh() is None
    assert state.apply_authoritative(approved, epoch=epoch)
    assert state.status == "completed_unverified"


# 功能：验证 trusted scoped registry 只暴露四个 exact builtin 工具且构造后不可替换
# 设计：使用真实 workspace resolver 和 access policy，检查名称、类型与 sealed mutation rejection
def test_trusted_scoped_registry_is_sealed(tmp_path: Path) -> None:
    resolver = WorkspacePathResolver(tmp_path)
    policy = WorkspaceAccessPolicy(resolver.root)
    registry = TrustedScopedToolRegistry.create(resolver, policy)

    assert set(registry.tool_names()) == {
        "read_file",
        "list_dir",
        "search_code",
        "write_file",
    }
    assert type(registry.get("read_file")) is ReadFileTool
    assert type(registry.get("list_dir")) is ListDirTool
    assert type(registry.get("search_code")) is SearchCodeTool
    assert type(registry.get("write_file")) is WriteFileTool
    schemas_before = registry.tool_schemas()
    original = registry.get("write_file")

    class FakeWrite(BaseTool):
        name = "write_file"
        description = "fake"
        input_schema: dict[str, object] = {"type": "object"}

        async def invoke(self, params: dict[str, object]) -> ToolResult:
            del params
            return ToolResult(content="fake")

    with pytest.raises(Exception):
        registry.register(FakeWrite())  # type: ignore[attr-defined]
    assert registry.get("write_file") is original
    assert registry.tool_schemas() == schemas_before


# 功能：验证 execution binding 在 SessionStore 中 create-once 且 status 只能单调更新
# 设计：真实文件存储覆盖重复相同 request、冲突 request 和 terminal status 防回退
def test_store_binding_create_once_and_status_monotonic(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    binding = ApprovedExecutionBinding.create(
        session_id="sess-1",
        request_id="req-1",
        execution_id="exec-1",
        run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        approval_record_digest="approval-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest="snapshot-digest",
        workspace_id="workspace",
        admitted_at="2026-01-01T00:00:00+00:00",
    )
    store.write_approved_execution_binding(binding)
    assert store.read_approved_execution_binding("sess-1", "req-1") == binding
    store.write_approved_execution_binding(binding)
    conflicting = ApprovedExecutionBinding.create(
        session_id=binding.session_id,
        request_id=binding.request_id,
        execution_id=binding.execution_id,
        run_id="run-2",
        projection_key=binding.projection_key,
        decision_id=binding.decision_id,
        decision_version=binding.decision_version,
        decision_content_digest=binding.decision_content_digest,
        approval_record_digest=binding.approval_record_digest,
        commit_receipt_digest=binding.commit_receipt_digest,
        snapshot_digest=binding.snapshot_digest,
        workspace_id=binding.workspace_id,
        admitted_at=binding.admitted_at,
    )
    with pytest.raises(ValueError, match="binding conflict"):
        store.write_approved_execution_binding(conflicting)

    store.write_execution_status(
        "sess-1",
        "req-1",
        status="running",
        status_revision=1,
        reason=None,
    )
    with pytest.raises(ValueError, match="status regression"):
        store.write_execution_status(
            "sess-1",
            "req-1",
            status="admitted",
            status_revision=2,
            reason=None,
        )
    store.write_execution_status(
        "sess-1",
        "req-1",
        status="completed_unverified",
        status_revision=2,
        reason="execution_completed_unverified",
    )
    with pytest.raises(ValueError, match="status regression"):
        store.write_execution_status(
            "sess-1",
            "req-1",
            status="running",
            status_revision=3,
            reason=None,
        )
    reconciled = store.write_execution_status(
        "sess-1",
        "req-1",
        status="completed_unverified",
        status_revision=4,
        reason="journal-authority",
        authoritative=True,
    )
    assert reconciled.status == "completed_unverified"


class _FiniteApprovedProvider:
    # 为 approved runner 提供一次 end_turn 后立即拒绝额外调用的有限状态机
    def __init__(self) -> None:
        self.calls = 0

    # 精确验证 approved runner 只进行一次 provider interaction 且只暴露四个工具
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, bus, run_id, step, system
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("unexpected approved provider call")
        assert [schema["name"] for schema in tool_schemas] == [
            "read_file",
            "list_dir",
            "search_code",
            "write_file",
        ]
        return LlmResponse(stop_reason="end_turn", text="completed")


class _SingleOutOfScopeWriteProvider:
    # 为 hard scope terminal 测试提供一次越界写请求并拒绝额外 provider 调用
    def __init__(self) -> None:
        self.calls = 0

    # 第一次返回越界 write，任何第二次调用都立即失败
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, tool_schemas, bus, run_id, step, system
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("unexpected approved provider call")
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCallBlock(
                    id="out-of-scope-write",
                    name="write_file",
                    input={"path": "outside.txt", "content": "blocked"},
                )
            ],
        )


# 为 approved unknown-tool 测试提供一次非法工具请求并拒绝额外 provider 调用
class _SingleUnknownApprovedProvider:
    # 初始化要注入的非法工具名和 provider 调用计数
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.calls = 0

    # 第一次请求非法工具，任何第二次 provider 调用都立即失败
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, tool_schemas, bus, run_id, step, system
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("unexpected approved provider call")
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCallBlock(
                    id="unknown-approved-tool",
                    name=self.tool_name,
                    input={},
                )
            ],
        )


# 功能：验证 approved 请求 Bash 或伪造工具名都会成为 hard scope terminal
# 设计：有限 provider 对两种非法名称各只允许一次调用，锁定无 fallback、无第二次请求和无副作用
@pytest.mark.parametrize("tool_name", ["bash", "definitely_not_a_tool"])
@pytest.mark.asyncio
async def test_approved_unknown_tool_is_hard_scope_terminal(
    tmp_path: Path,
    tool_name: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _SingleUnknownApprovedProvider(tool_name)
    snapshot = SnapshotBuilder(workspace).capture()
    _, workspace_id = workspace_identity(workspace)
    scope = ExecutionScope._create(
        session_id="sess-approved",
        workspace_id=workspace_id,
        projection_key="pv1:run-unknown:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot.snapshot_digest,
        capabilities=("read_file",),
    )
    execution_context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-unknown",
    )
    resolver = WorkspacePathResolver(workspace)
    registry = TrustedScopedToolRegistry.create(resolver, WorkspaceAccessPolicy(resolver.root))
    invoker = ApprovedScopedToolInvoker(
        registry,
        EventBus(),
        "run-unknown",
        execution_context,
    )
    loop = AgentLoop(provider, invoker, EventBus())

    context = ExecutionContext(run_id="run-unknown", goal="unknown", max_steps=4)
    await loop.run(context)

    assert provider.calls == 1
    assert context.status == "failed"
    assert context.reason == "scope_denied"
    assert invoker.terminal_reason() == "scope_denied"
    assert not any(workspace.iterdir())


# 功能：验证 approved invoker 不会回退到 generic invoke_tool 且保留 trusted schema
# 设计：直接替换 generic 函数为立即失败哨兵，再执行真实 read_file invocation
@pytest.mark.asyncio
async def test_approved_invoker_does_not_use_generic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("trusted", encoding="utf-8")
    snapshot = SnapshotBuilder(workspace).capture()
    _, workspace_id = workspace_identity(workspace)
    scope = ExecutionScope._create(
        session_id="sess-approved",
        workspace_id=workspace_id,
        projection_key="pv1:run-read:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot.snapshot_digest,
        capabilities=("read_file",),
    )
    execution_context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-read",
    )
    resolver = WorkspacePathResolver(workspace)
    registry = TrustedScopedToolRegistry.create(resolver, WorkspaceAccessPolicy(resolver.root))
    invoker = ApprovedScopedToolInvoker(
        registry,
        EventBus(),
        "run-read",
        execution_context,
    )

    async def fail_generic(*args: object, **kwargs: object) -> ToolResult:
        del args, kwargs
        raise AssertionError("approved path used generic invoke_tool")

    monkeypatch.setattr(invocation_module, "invoke_tool", fail_generic)
    result = await invoker.invoke(
        ToolCallBlock(
            id="read-note",
            name="read_file",
            input={"path": "note.txt"},
        )
    )

    assert result.is_error is False
    assert result.content == "trusted"
    assert [schema["name"] for schema in invoker.tool_schemas()] == [
        "read_file",
        "list_dir",
        "search_code",
        "write_file",
    ]


# 功能：验证 hard scope denial 会锁存 terminal 并阻止 AgentLoop 第二次请求 provider
# 设计：有限状态 provider 只允许一次越界写，真实 approved invoker 在授权层拒绝且不触发工具副作用
@pytest.mark.asyncio
async def test_approved_invoker_scope_terminal_stops_agent_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allowed = workspace / "allowed.txt"
    allowed.write_text("before", encoding="utf-8")
    snapshot = SnapshotBuilder(workspace).capture(planned_existing_targets=("allowed.txt",))
    _, workspace_id = workspace_identity(workspace)
    scope = ExecutionScope._create(
        session_id="sess-approved",
        workspace_id=workspace_id,
        projection_key="pv1:run-hard:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot.snapshot_digest,
        files_to_modify=("allowed.txt",),
        capabilities=("write_file",),
    )
    execution_context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-hard",
    )
    resolver = WorkspacePathResolver(workspace)
    registry = TrustedScopedToolRegistry.create(resolver, WorkspaceAccessPolicy(resolver.root))
    bus = EventBus()
    invoker = ApprovedScopedToolInvoker(registry, bus, "run-hard", execution_context)
    provider = _SingleOutOfScopeWriteProvider()
    loop = AgentLoop(provider, invoker, bus)

    context = ExecutionContext(run_id="run-hard", goal="write", max_steps=4)
    await loop.run(context)

    assert provider.calls == 1
    assert context.status == "failed"
    assert context.reason == "scope_denied"
    assert invoker.terminal_reason() == "scope_denied"
    assert not (workspace / "outside.txt").exists()


# 功能：验证真实 AgentRunner approved path 使用完整 summary、sealed registry 和 unverified terminal
# 设计：有限 provider 只允许一次 end_turn，真实 journal 检查 RunStarted 在 RunFinished 前且无额外 child/tool
@pytest.mark.asyncio
async def test_agent_runner_approved_path_is_bounded_and_unverified(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-approved",
        mode="chat",
        status="active",
        title="approved",
        created_at="t",
        updated_at="t",
        workspace_root=workspace,
    )
    session_store.write_meta(session)
    snapshot = SnapshotBuilder(workspace).capture()
    _, workspace_id = workspace_identity(workspace)
    scope = ExecutionScope._create(
        session_id=session.id,
        workspace_id=workspace_id,
        projection_key="pv1:run-approved:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot.snapshot_digest,
        capabilities=("read_file",),
    )
    context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-approved",
    )
    journal = EventJournalCoordinator()
    await journal.register_session(session.id, session_store.session_dir(session.id))
    await journal.register_run(
        "run-approved",
        session_store.runs_dir(session.id) / "run-approved",
        session_id=session.id,
    )
    provider = _FiniteApprovedProvider()
    runner = AgentRunner(
        KamaConfig(),
        workspace_root=workspace,
        provider=provider,  # type: ignore[arg-type]
        journal=journal,
    )
    binding = ApprovedExecutionBinding.create(
        session_id=session.id,
        request_id="request-approved",
        execution_id=context.execution_id,
        run_id="run-approved",
        projection_key=scope.projection_key,
        decision_id=scope.decision_id,
        decision_version=scope.decision_version,
        decision_content_digest=scope.decision_content_digest,
        approval_record_digest="approval-digest",
        commit_receipt_digest=scope.commit_receipt_digest,
        snapshot_digest=scope.snapshot_digest,
        workspace_id=scope.workspace_id,
    )
    session_store.write_approved_execution_binding(binding)

    outcome = await runner.run_approved(
        summary='{"planner_decision":{"decision_id":"decision"}}',
        run_id="run-approved",
        session=session,
        store=session_store,
        execution_context=context,
        execution_binding=binding,
    )

    assert outcome.status == "completed_unverified"
    assert provider.calls == 1
    replay = await journal.read_replay(
        "run:run-approved",
        after_seq=0,
        high_watermark=journal.high_watermark("run:run-approved"),
    )
    assert replay.records[0].event["type"] == "run.started"
    assert replay.records[-1].event["type"] == "run.finished"
    assert replay.records[-1].event["execution_status"] == "completed_unverified"
    completion = session_store.read_execution_completion_receipt(
        session.id,
        binding.request_id,
    )
    assert completion is not None
    assert completion.run_finished_event_id == replay.records[-1].event_id
    artifact = session_store.read_execution_output_snapshot(
        session.id,
        completion.snapshot_manifest_digest,
    )
    assert artifact.manifest.manifest_digest == completion.snapshot_manifest_digest
    assert artifact.artifact_dir.parent == session_store.verification_snapshot_root(session.id)
    session_store.execution_completion_receipt_path(
        session.id,
        binding.request_id,
    ).write_text("corrupt\n", encoding="utf-8")
    await journal.close()
    reopened = EventJournalCoordinator()
    await reopened.register_session(session.id, session_store.session_dir(session.id))
    await reopened.register_run(
        "run-approved",
        session_store.runs_dir(session.id) / "run-approved",
        session_id=session.id,
    )
    reopened_store = SessionStore(session_store.session_dir(session.id).parent)
    restored = await materialize_execution_completion_receipt(
        store=reopened_store,
        journal=reopened,
        session_id=session.id,
        request_id=binding.request_id,
    )
    assert restored.snapshot_manifest_digest == artifact.manifest.manifest_digest
    assert (
        reopened_store.read_execution_completion_receipt(
            session.id,
            binding.request_id,
        )
        == restored
    )
    await reopened.close()


# 功能：验证 snapshot 阶段取消会写 durable cancelled terminal、无 receipt 并传播原始取消
# 设计：有限 provider 先 end_turn，再用受控 snapshot await 精确制造取消窗口
@pytest.mark.asyncio
async def test_agent_runner_snapshot_cancellation_is_terminal_and_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-cancel",
        mode="chat",
        status="active",
        title="cancel",
        created_at="t",
        updated_at="t",
        workspace_root=workspace,
    )
    session_store.write_meta(session)
    snapshot = SnapshotBuilder(workspace).capture()
    _, workspace_id = workspace_identity(workspace)
    scope = ExecutionScope._create(
        session_id=session.id,
        workspace_id=workspace_id,
        projection_key="pv1:run-cancel:decision:v1",
        decision_id="decision",
        decision_version=1,
        decision_content_digest="decision-digest",
        commit_receipt_digest="receipt-digest",
        snapshot_digest=snapshot.snapshot_digest,
        capabilities=("read_file",),
    )
    execution_context = ScopedExecutionContext.from_verified_scope(
        scope=scope,
        snapshot=snapshot,
        workspace_root=workspace,
        execution_id="exec-cancel",
    )
    journal = EventJournalCoordinator()
    await journal.register_session(session.id, session_store.session_dir(session.id))
    await journal.register_run(
        "run-cancel",
        session_store.runs_dir(session.id) / "run-cancel",
        session_id=session.id,
    )
    runner = AgentRunner(
        KamaConfig(),
        workspace_root=workspace,
        provider=_FiniteApprovedProvider(),  # type: ignore[arg-type]
        journal=journal,
    )
    binding = ApprovedExecutionBinding.create(
        session_id=session.id,
        request_id="request-cancel",
        execution_id=execution_context.execution_id,
        run_id="run-cancel",
        projection_key=scope.projection_key,
        decision_id=scope.decision_id,
        decision_version=scope.decision_version,
        decision_content_digest=scope.decision_content_digest,
        approval_record_digest="approval-digest",
        commit_receipt_digest=scope.commit_receipt_digest,
        snapshot_digest=scope.snapshot_digest,
        workspace_id=scope.workspace_id,
    )
    session_store.write_approved_execution_binding(binding)
    snapshot_started = asyncio.Event()

    async def blocked_snapshot(**_kwargs: object) -> object:
        snapshot_started.set()
        await asyncio.Event().wait()
        return object()

    monkeypatch.setattr(runner_module, "capture_execution_output_snapshot_async", blocked_snapshot)
    task = asyncio.create_task(
        runner.run_approved(
            summary='{"planner_decision":{"decision_id":"decision"}}',
            run_id="run-cancel",
            session=session,
            store=session_store,
            execution_context=execution_context,
            execution_binding=binding,
        )
    )
    await asyncio.wait_for(snapshot_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replay = await journal.read_replay(
        "run:run-cancel",
        after_seq=0,
        high_watermark=journal.high_watermark("run:run-cancel"),
    )
    assert replay.records[-1].event["status"] == "cancelled"
    assert replay.records[-1].event["execution_status"] == "cancelled"
    assert session_store.read_execution_completion_receipt(session.id, binding.request_id) is None
    await journal.close()
