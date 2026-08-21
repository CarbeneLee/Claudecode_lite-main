from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kama_claude.core.approval import (
    ApprovalAction,
    ApprovalError,
    ApprovalRecord,
    ApprovalService,
    materialize_committed_plan_receipt,
)
from kama_claude.core.bus.envelope import INVALID_PARAMS, HandlerError
from kama_claude.core.bus.events import (
    SessionAgentModeChangedEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.execution import (
    ApprovedExecutionBinding,
    ExecutionStatus,
    ExecutionStatusProjection,
)
from kama_claude.core.execution_scope import ExecutionScope, ScopedExecutionContext
from kama_claude.core.runs import new_run_id
from kama_claude.core.session.model import (
    MAX_AGENT_MODE_REVISION,
    AgentMode,
    AgentModeSnapshot,
    Session,
    SessionMode,
)
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader

if TYPE_CHECKING:
    from kama_claude.core.bus.commands import (
        PlanApprovalResult,
        PlanExecuteResult,
        PlanGetApprovalResult,
        PlanGetExecutionResult,
    )
    from kama_claude.core.events.journal import EventJournalCoordinator
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.planning import ExactPlannerDecisionV2
    from kama_claude.core.runner import AgentRunner

logger = logging.getLogger(__name__)

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012
SESSION_INTERRUPTED = -32013
SESSION_INVALID_MODE = -32014


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 判断以斜杠开头的消息是否更像合法绝对路径目标，而不是未知 slash skill
def _looks_like_absolute_path_goal(content: str) -> bool:
    token = content.split(None, 1)[0]
    return token.startswith("/") and token.count("/") >= 2


class SessionManager:
    # 初始化会话管理器，接入文件存储、runner 工厂、事件总线和可选的 LLM provider（用于手动压缩）
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[Path], AgentRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
        journal: EventJournalCoordinator | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._provider = provider
        self._journal = journal
        self._approval = ApprovalService(store, journal, bus)
        if journal is not None:
            journal_handler = getattr(journal, "handle", None)
            if callable(journal_handler):
                bus.subscribe(journal_handler)
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 每个 session 至多一个后台 run task；send_message 注册后立即返回
        self._running_runs: dict[str, asyncio.Task[None]] = {}
        # 重启后没有 terminal journal 的 session 进入显式 interrupted 状态
        self._interrupted_sessions: set[str] = set()
        # approved execution task map is lifecycle tracking only, not admission authority
        self._approved_execution_tasks: dict[str, asyncio.Task[None]] = {}

    # 在 daemon wiring 完成且任何 session 创建前注入 shared journal coordinator
    def attach_journal(self, journal: EventJournalCoordinator) -> None:
        if self._sessions:
            raise RuntimeError("journal must be attached before session creation")
        self._journal = journal
        self._approval = ApprovalService(self._store, journal, self._bus)
        journal_handler = getattr(journal, "handle", None)
        if callable(journal_handler):
            self._bus.subscribe(journal_handler)

    # 创建新 session 并写入 meta.json
    async def create(
        self,
        mode: SessionMode,
        title: str = "",
        *,
        workspace_root: Path,
        agent_mode: AgentMode = "direct",
    ) -> Session:
        if agent_mode not in ("direct", "plan"):
            raise HandlerError(SESSION_INVALID_MODE, "invalid agent_mode")
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        ts = _now()
        session = Session(
            id=sid,
            mode=mode,
            agent_mode=agent_mode,
            status="active",
            title=title,
            created_at=ts,
            updated_at=ts,
            workspace_root=workspace_root,
            run_ids=[],
        )
        session_path = self._store.ensure_session_dir(sid)
        if self._journal is not None:
            await self._journal.register_session(sid, session_path)
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=ts))
        return session

    # 重启后加载持久化 session，校验 mode 并依据 terminal journal 收敛 active_run_id
    async def reconcile_persisted_sessions(self) -> None:
        for sid in self._store.list_session_ids():
            session = self._store.read_meta(sid)
            if session.agent_mode not in ("direct", "plan"):
                raise HandlerError(SESSION_INVALID_MODE, "invalid persisted agent_mode")
            if session.mode not in ("chat", "one_shot"):
                raise HandlerError(SESSION_INVALID_MODE, "invalid persisted session mode")
            session_path = self._store.ensure_session_dir(sid)
            if self._journal is not None:
                has_stream = getattr(self._journal, "has_stream", None)
                register_session = getattr(self._journal, "register_session", None)
                if (
                    callable(register_session)
                    and callable(has_stream)
                    and not has_stream(f"session:{sid}")
                ):
                    await register_session(sid, session_path)
                for run_id in session.run_ids:
                    run_path = self._store.runs_dir(sid) / run_id
                    run_path.mkdir(parents=True, exist_ok=True)
                    has_run_stream = (
                        callable(has_stream) and has_stream(f"run:{run_id}")
                    )
                    register_run = getattr(self._journal, "register_run", None)
                    if callable(register_run) and not has_run_stream:
                        await register_run(
                            run_id,
                            run_path,
                            session_id=sid,
                        )
            if session.active_run_id is not None:
                terminal = False
                if self._journal is not None:
                    terminal_query = getattr(self._journal, "has_terminal_run", None)
                    if callable(terminal_query):
                        terminal = terminal_query(session.active_run_id)
                        if inspect.isawaitable(terminal):
                            terminal = await terminal
                if terminal:
                    session.active_run_id = None
                    session.status = (
                        "closed" if session.mode == "one_shot" else "waiting_for_input"
                    )
                    session.updated_at = _now()
                    self._store.write_meta(session)
                else:
                    self._interrupted_sessions.add(sid)
            # approved bindings survive daemon restart; never rerun an admitted task automatically
            for binding in self._store.list_approved_execution_bindings(sid):
                cached = self._store.read_execution_status(sid, binding.request_id)
                terminal = False
                if self._journal is not None:
                    terminal_query = getattr(self._journal, "has_terminal_run", None)
                    if callable(terminal_query):
                        terminal = terminal_query(binding.run_id)
                        if inspect.isawaitable(terminal):
                            terminal = await terminal
                if terminal:
                    await self._reconcile_execution_status(binding, cached)
                elif cached is None or cached.status in ("admitted", "running"):
                    next_revision = (cached.status_revision + 1) if cached else 0
                    self._store.write_execution_status(
                        sid,
                        binding.request_id,
                        status="interrupted",
                        status_revision=next_revision,
                        reason="daemon-restarted-before-terminal",
                    )
            self._sessions[sid] = session
            self._locks[sid] = asyncio.Lock()

    # 处理用户消息，追加 thread 并启动一次 agent run
    async def send_message(
        self,
        sid: str,
        content: str,
        *,
        run_id: str | None = None,
    ) -> str:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if session.active_run_id is not None:
            if sid in self._interrupted_sessions:
                raise HandlerError(SESSION_INTERRUPTED, "session has an interrupted run")
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            if session.active_run_id is not None:
                if sid in self._interrupted_sessions:
                    raise HandlerError(SESSION_INTERRUPTED, "session has an interrupted run")
                raise HandlerError(SESSION_BUSY, "session busy")
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")

            if content.startswith("/"):
                parts = content[1:].split(None, 1)
                skill_name = parts[0] if parts else ""
                is_absolute_path_goal = _looks_like_absolute_path_goal(content)
                if not is_absolute_path_goal and not SkillLoader(
                    session.workspace_root
                ).resolve(skill_name):
                    raise HandlerError(INVALID_PARAMS, "unknown slash skill")

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )

            if not session.title:
                session.title = content[:40]

            run_id = run_id or new_run_id()
            run_path = self._store.runs_dir(sid) / run_id
            run_path.mkdir(parents=True, exist_ok=True)
            if self._journal is not None:
                await self._journal.register_run(
                    run_id,
                    run_path,
                    session_id=sid,
                )
            session.run_ids.append(run_id)
            session.active_run_id = run_id
            self._interrupted_sessions.discard(sid)
            session.updated_at = _now()
            self._store.write_meta(session)

            # Skill 解析：检测 "/" 前缀，展开为系统提示覆盖和工具白名单
            goal = content
            system_prompt_override: str | None = None
            tool_whitelist: list[str] | None = None
            if content.startswith("/"):
                skill_loader = SkillLoader(session.workspace_root)
                parts = content[1:].split(None, 1)
                skill_name = parts[0]
                arguments = parts[1] if len(parts) > 1 else ""
                skill = skill_loader.resolve(skill_name)
                if skill is not None:
                    goal = skill_loader.render_prompt(skill, arguments)
                    system_prompt_override = skill.system_prompt_template
                    tool_whitelist = skill.allowed_tools or None
                    await self._bus.publish(
                        SkillInvokedEvent(
                            skill_name=skill_name,
                            arguments=arguments,
                            run_id=run_id,
                            ts=_now(),
                        )
                    )

            # run 在后台 task 中执行：send_message 立即返回，CLI 主循环不再被阻塞
            task = asyncio.create_task(
                self._run_and_finalize(
                    sid,
                    run_id,
                    goal,
                    agent_mode=session.agent_mode,
                    system_prompt_override=system_prompt_override,
                    tool_whitelist=tool_whitelist,
                )
            )
            self._running_runs[sid] = task
            return run_id

    # 在后台执行 agent run；完成后在 lock 内收敛 session 状态并发布事件
    async def _run_and_finalize(
        self,
        sid: str,
        run_id: str,
        goal: str,
        *,
        agent_mode: AgentMode,
        system_prompt_override: str | None,
        tool_whitelist: list[str] | None,
    ) -> None:
        try:
            try:
                session = self._get_session(sid)
                runner = self._runner_factory(session.workspace_root)
                kwargs: dict[str, Any] = {
                    "run_id": run_id,
                    "session": session,
                    "store": self._store,
                    "system_prompt_override": system_prompt_override,
                    "tool_whitelist": tool_whitelist,
                }
                if agent_mode == "plan":
                    kwargs["agent_mode"] = agent_mode
                await runner.run_and_capture(goal, **kwargs)
            except asyncio.CancelledError:
                # close()/shutdown 已收敛状态，不再发布 waiting_for_input
                raise
            except Exception:
                # runner 通常已发布 run.finished；无 terminal journal 时保留 interrupted 边界
                logger.exception("run failed sid=%s run_id=%s", sid, run_id)
            if self._journal is not None:
                terminal_query = getattr(self._journal, "has_terminal_run", None)
                if callable(terminal_query):
                    terminal_probe = terminal_query(run_id)
                    has_terminal = (
                        await terminal_probe
                        if inspect.isawaitable(terminal_probe)
                        else terminal_probe
                    )
                    if not has_terminal:
                        lock = self._locks[sid]
                        async with lock:
                            session = self._get_session(sid)
                            if session.status == "closed":
                                return
                            self._interrupted_sessions.add(sid)
                            session.updated_at = _now()
                            self._store.write_meta(session)
                        return
            session = self._get_session(sid)
            lock = self._locks[sid]
            async with lock:
                if session.status == "closed":
                    return
                if session.active_run_id == run_id:
                    session.active_run_id = None
                self._interrupted_sessions.discard(sid)
                session.updated_at = _now()
                if session.mode == "one_shot":
                    session.status = "closed"
                    await self._bus.publish(
                        SessionClosedEvent(session_id=sid, ts=session.updated_at)
                    )
                else:
                    session.status = "waiting_for_input"
                    await self._bus.publish(
                        SessionWaitingForInputEvent(
                            session_id=sid,
                            last_run_id=run_id,
                            ts=session.updated_at,
                        )
                    )
                self._store.write_meta(session)
        finally:
            self._running_runs.pop(sid, None)

    # 关闭指定 session 并更新 meta.json；取消仍在执行的后台 run
    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            session.status = "closed"
            session.active_run_id = None
            self._interrupted_sessions.discard(sid)
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))
        run_task = self._running_runs.get(sid)
        if run_task is not None and not run_task.done():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)

    # 当前仍有未完成 run 的 task 列表（测试与 shutdown 使用）
    def active_run_tasks(self) -> list[asyncio.Task[None]]:
        return [t for t in self._running_runs.values() if not t.done()]

    # 取消全部后台 run（daemon shutdown 路径）
    async def cancel_active_runs(self) -> None:
        tasks = self.active_run_tasks()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # 手动压缩指定 session 的 thread，将摘要持久化写入 thread.jsonl
    async def compact(self, sid: str, focus: str = "") -> Any:
        self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(-32020, "provider not available for compaction")
        async with lock:
            from kama_claude.core.bus.commands import SessionCompactResult
            from kama_claude.core.compact.compactor import Compactor
            messages = self._store.read_messages(sid)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(self._bus, session_dir, sid)
            result = await compactor.compact_messages(messages, self._provider, focus=focus)
            if result is None:
                raise HandlerError(-32021, "compaction failed or not beneficial")
            self._store.write_compacted(sid, [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant", "content": "Understood, I'll continue from this summary."},
            ])
            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    # 读取指定 session 的完整 thread 历史
    async def get_history(
        self,
        sid: str,
        *,
        include_projection_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        self._get_session(sid)
        if include_projection_metadata:
            return self._store.read_history_projection(sid)
        return self._store.read_messages(sid)

    # 在 session lock 内切换下一条消息使用的 agent mode，并持久化变更事件
    async def set_agent_mode(self, sid: str, agent_mode: AgentMode) -> AgentModeSnapshot:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked() or session.active_run_id is not None:
            raise HandlerError(SESSION_BUSY, "session busy")
        if session.status == "closed":
            raise HandlerError(SESSION_CLOSED, "session already closed")
        if agent_mode not in ("direct", "plan"):
            raise HandlerError(SESSION_INVALID_MODE, "invalid agent_mode")
        async with lock:
            if session.active_run_id is not None:
                raise HandlerError(SESSION_BUSY, "session busy")
            if str(session.status) == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            previous = session.agent_mode
            if previous == agent_mode:
                return AgentModeSnapshot(previous, session.agent_mode_revision)
            next_revision = session.agent_mode_revision + 1
            if next_revision > MAX_AGENT_MODE_REVISION:
                raise HandlerError(SESSION_INVALID_MODE, "agent_mode revision overflow")
            candidate = replace(
                session,
                agent_mode=agent_mode,
                agent_mode_revision=next_revision,
                updated_at=_now(),
            )
            self._store.write_meta(candidate)
            session.agent_mode = candidate.agent_mode
            session.agent_mode_revision = candidate.agent_mode_revision
            session.updated_at = candidate.updated_at
            try:
                await self._bus.publish(
                    SessionAgentModeChangedEvent(
                        session_id=sid,
                        previous_mode=previous,
                        agent_mode=agent_mode,
                        revision=next_revision,
                        ts=session.updated_at,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("agent mode notification failed sid=%s", sid)
            return AgentModeSnapshot(agent_mode, next_revision)

    # 读取 daemon 持久化的当前 agent mode
    async def get_agent_mode(self, sid: str) -> AgentModeSnapshot:
        session = self._get_session(sid)
        return AgentModeSnapshot(session.agent_mode, session.agent_mode_revision)

    # 读取 committed plan 的 approval snapshot，不改变 session 状态
    async def get_approval(self, sid: str, projection_key: str) -> PlanGetApprovalResult:
        from kama_claude.core.bus.commands import PlanGetApprovalResult

        session = self._get_session(sid)
        snapshot = await self._approval.get_snapshot(sid, projection_key)
        receipt = await materialize_committed_plan_receipt(
            store=self._store,
            journal=self._journal,
            session_id=sid,
            projection_key=projection_key,
        )
        return PlanGetApprovalResult(
            session_id=session.id,
            projection_key=projection_key,
            status=snapshot.status,
            decision_id=receipt.decision_id,
            decision_version=receipt.decision_version,
            content_digest=receipt.decision_content_digest,
            commit_receipt_digest=receipt.receipt_digest,
            action=snapshot.action,
            record_digest=snapshot.record_digest,
        )

    # 在 session lock 内创建一次 immutable user approval record
    async def resolve_approval(
        self,
        sid: str,
        projection_key: str,
        *,
        action: ApprovalAction,
        decision_id: str,
        decision_version: int,
        content_digest: str,
        commit_receipt_digest: str,
    ) -> PlanApprovalResult:
        from kama_claude.core.bus.commands import PlanApprovalResult

        self._get_session(sid)
        lock = self._locks[sid]
        async with lock:
            snapshot = await self._approval.resolve(
                session_id=sid,
                projection_key=projection_key,
                action=action,
                decision_id=decision_id,
                decision_version=decision_version,
                content_digest=content_digest,
                commit_receipt_digest=commit_receipt_digest,
            )
            receipt = await materialize_committed_plan_receipt(
                store=self._store,
                journal=self._journal,
                session_id=sid,
                projection_key=projection_key,
            )
            return PlanApprovalResult(
                session_id=sid,
                projection_key=projection_key,
                status=snapshot.status,
                decision_id=receipt.decision_id,
                decision_version=receipt.decision_version,
                content_digest=receipt.decision_content_digest,
                commit_receipt_digest=receipt.receipt_digest,
                action=snapshot.action,
                record_digest=snapshot.record_digest,
            )

    # 在 session admission lock 内重新验证 exact approval 并 create-once 启动 approved execution
    async def execute_approved_plan(
        self,
        sid: str,
        projection_key: str,
        request_id: str,
    ) -> PlanExecuteResult:
        from kama_claude.core.bus.commands import PlanExecuteResult
        from kama_claude.core.grounding import RepositorySnapshot
        from kama_claude.core.planning import (
            decode_exact_planner_decision,
            render_planner_decision_execution_summary,
        )

        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            existing = self._store.read_approved_execution_binding(sid, request_id)
            if existing is not None:
                if existing.projection_key != projection_key:
                    raise HandlerError(INVALID_PARAMS, "execution request binding conflict")
                status = self._store.read_execution_status(sid, request_id)
                status = await self._reconcile_execution_status(existing, status)
                return PlanExecuteResult(
                    session_id=sid,
                    request_id=request_id,
                    execution_id=existing.execution_id,
                    run_id=existing.run_id,
                    projection_key=existing.projection_key,
                    status=status.status,
                    status_revision=status.status_revision,
                    status_digest=None,
                    reason=status.reason,
                )
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            if session.agent_mode != "plan":
                raise HandlerError(INVALID_PARAMS, "plan mode is required")
            if sid in self._interrupted_sessions:
                raise HandlerError(SESSION_INTERRUPTED, "session has an interrupted run")
            if session.active_run_id is not None:
                if sid in self._interrupted_sessions:
                    raise HandlerError(SESSION_INTERRUPTED, "session has an interrupted run")
                raise HandlerError(SESSION_BUSY, "session busy")
            admission_committed = False
            try:
                raw_approval = self._store.read_approval_record(sid, projection_key)
                if raw_approval is None:
                    raise ValueError("approval is missing")
                approval = ApprovalRecord.model_validate(raw_approval)
                approval.verify_digest()
                if approval.action != "approve":
                    raise ValueError("approval is not approved")
                receipt = await materialize_committed_plan_receipt(
                    store=self._store,
                    journal=self._journal,
                    session_id=sid,
                    projection_key=projection_key,
                )
                payload = self._store.read_decision(
                    sid,
                    receipt.decision_id,
                    receipt.decision_version,
                )
                decision = decode_exact_planner_decision(payload)
                if (
                    approval.projection_key != projection_key
                    or approval.decision_id != decision.decision_id
                    or approval.decision_version != decision.version
                    or approval.content_digest != decision.content_digest
                    or approval.commit_receipt_digest != receipt.receipt_digest
                ):
                    raise ValueError("approved artifacts do not bind to exact decision")
                grounding = self._store.read_grounding(sid)
                if not isinstance(grounding, dict):
                    raise ValueError("repository snapshot is missing")
                snapshots = grounding.get("snapshots")
                if not isinstance(snapshots, list):
                    raise ValueError("repository snapshot is missing")
                snapshot = next(
                    (
                        RepositorySnapshot.model_validate(item)
                        for item in snapshots
                        if isinstance(item, dict)
                        and item.get("snapshot_digest") == decision.snapshot_digest
                    ),
                    None,
                )
                if snapshot is None:
                    raise ValueError("exact repository snapshot is missing")
                scope = ExecutionScope.from_approved(
                    decision=decision,
                    approval_record=approval,
                    receipt=receipt,
                    snapshot=snapshot,
                    workspace_root=session.workspace_root,
                )
                execution_id = f"exec-{uuid.uuid4().hex}"
                context = ScopedExecutionContext.from_verified_scope(
                    scope=scope,
                    snapshot=snapshot,
                    workspace_root=session.workspace_root,
                    execution_id=execution_id,
                )
                summary = render_planner_decision_execution_summary(decision)
                run_id = new_run_id()
                run_path = self._store.runs_dir(sid) / run_id
                run_path.mkdir(parents=True, exist_ok=True)
                if self._journal is None:
                    raise ValueError("execution journal is unavailable")
                await self._journal.register_run(run_id, run_path, session_id=sid)
                binding = ApprovedExecutionBinding.create(
                    session_id=sid,
                    request_id=request_id,
                    execution_id=execution_id,
                    run_id=run_id,
                    projection_key=projection_key,
                    decision_id=decision.decision_id,
                    decision_version=decision.version,
                    decision_content_digest=decision.content_digest,
                    approval_record_digest=approval.record_digest,
                    commit_receipt_digest=receipt.receipt_digest,
                    snapshot_digest=decision.snapshot_digest,
                    workspace_id=scope.workspace_id,
                )
                # Admission commit point: after this write the request is consumed forever.
                self._store.write_approved_execution_binding(binding)
                admission_committed = True
                self._store.write_execution_status(
                    sid,
                    request_id,
                    status="admitted",
                    status_revision=0,
                    reason=None,
                )
                candidate_session = replace(
                    session,
                    run_ids=[*session.run_ids, run_id],
                    active_run_id=run_id,
                    updated_at=_now(),
                )
                try:
                    self._store.write_meta(candidate_session)
                except Exception as exc:
                    # binding 已提交但 session authority 未提交，保守解释为 interrupted
                    self._interrupted_sessions.add(sid)
                    try:
                        self._store.write_execution_status(
                            sid,
                            request_id,
                            status="interrupted",
                            status_revision=1,
                            reason="session-meta-write-failed",
                        )
                    except Exception:
                        logger.exception(
                            "approved admission status could not be reconciled execution_id=%s",
                            execution_id,
                        )
                    raise HandlerError(
                        INVALID_PARAMS,
                        "execution-admission-interrupted",
                    ) from exc
                session.run_ids = candidate_session.run_ids
                session.active_run_id = candidate_session.active_run_id
                session.updated_at = candidate_session.updated_at
                try:
                    task = asyncio.create_task(
                        self._run_approved_and_finalize(
                            sid=sid,
                            binding=binding,
                            decision=decision,
                            summary=summary,
                            context=context,
                        ),
                        name=f"approved-execution:{execution_id}",
                    )
                except asyncio.CancelledError:
                    self._interrupted_sessions.add(sid)
                    try:
                        self._store.write_execution_status(
                            sid,
                            request_id,
                            status="interrupted",
                            status_revision=1,
                            reason="execution-task-not-started",
                        )
                    except Exception:
                        logger.exception(
                            "approved task cancellation status could not be persisted "
                            "execution_id=%s",
                            execution_id,
                        )
                    raise
                except Exception as exc:
                    self._interrupted_sessions.add(sid)
                    try:
                        self._store.write_execution_status(
                            sid,
                            request_id,
                            status="interrupted",
                            status_revision=1,
                            reason="execution-task-not-started",
                        )
                    except Exception:
                        logger.exception(
                            "approved task failure status could not be persisted execution_id=%s",
                            execution_id,
                        )
                    raise HandlerError(
                        INVALID_PARAMS,
                        "execution-admission-interrupted",
                    ) from exc
                self._approved_execution_tasks[execution_id] = task
                self._running_runs[sid] = task
                return PlanExecuteResult(
                    session_id=sid,
                    request_id=request_id,
                    execution_id=execution_id,
                    run_id=run_id,
                    projection_key=projection_key,
                    status="admitted",
                    status_revision=0,
                )
            except HandlerError:
                raise
            except (
                ApprovalError,
                OSError,
                PermissionError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                if admission_committed:
                    try:
                        current = self._store.read_execution_status(sid, request_id)
                        revision = (current.status_revision + 1) if current else 1
                        self._store.write_execution_status(
                            sid,
                            request_id,
                            status="interrupted",
                            status_revision=revision,
                            reason="execution-admission-uncertain",
                        )
                    except Exception:
                        logger.exception(
                            "approved admission reconciliation failed request_id=%s",
                            request_id,
                        )
                    raise HandlerError(
                        INVALID_PARAMS,
                        "execution-admission-interrupted",
                    ) from exc
                raise HandlerError(INVALID_PARAMS, str(exc)) from exc

    # 执行 approved task 并把 provider/runtime terminal 映射为 durable execution status
    async def _run_approved_and_finalize(
        self,
        *,
        sid: str,
        binding: ApprovedExecutionBinding,
        decision: ExactPlannerDecisionV2,
        summary: str,
        context: ScopedExecutionContext,
    ) -> None:
        del decision
        status: ExecutionStatus = "failed"
        reason: str | None = "approved-execution-failed"
        terminal_durable = False
        try:
            self._store.write_execution_status(
                sid,
                binding.request_id,
                status="running",
                status_revision=1,
                reason=None,
            )
            runner = self._runner_factory(self._get_session(sid).workspace_root)
            run_approved = getattr(runner, "run_approved", None)
            if not callable(run_approved):
                raise RuntimeError("approved runner is unavailable")
            outcome = await run_approved(
                summary=summary,
                run_id=binding.run_id,
                session=self._get_session(sid),
                store=self._store,
                execution_context=context,
            )
            outcome_status = getattr(outcome, "status", None)
            if outcome_status in ("success", "completed_unverified"):
                status = "completed_unverified"
                reason = "execution_completed_unverified"
            elif outcome_status in {
                "cancelled",
                "scope_denied",
                "inconclusive",
            }:
                status = outcome_status
                reason = getattr(outcome, "reason", None) or outcome_status
            else:
                status = "failed"
                reason = getattr(outcome, "reason", None) or "approved-execution-failed"
        except asyncio.CancelledError:
            status = "cancelled"
            reason = "cancelled"
            raise
        except Exception:
            logger.exception("approved execution failed execution_id=%s", binding.execution_id)
        finally:
            # runner factory/task failure must still leave one durable terminal record
            if self._journal is not None:
                try:
                    terminal_query = getattr(self._journal, "has_terminal_run", None)
                    has_terminal = (
                        terminal_query(binding.run_id)
                        if callable(terminal_query)
                        else False
                    )
                    if inspect.isawaitable(has_terminal):
                        has_terminal = await has_terminal
                    terminal_durable = bool(has_terminal)
                    if not has_terminal:
                        from kama_claude.core.bus.events import RunFinishedEvent

                        await self._journal.publish_required_durable(
                            RunFinishedEvent(
                                run_id=binding.run_id,
                                status=(
                                    "success"
                                    if status == "completed_unverified"
                                    else "cancelled"
                                    if status == "cancelled"
                                    else "failed"
                                ),
                                reason=reason,
                                steps=0,
                                ts=_now(),
                                execution_id=binding.execution_id,
                                execution_status=status,
                            )
                        )
                        terminal_durable = True
                except Exception:
                    logger.exception(
                        "approved execution terminal fallback failed execution_id=%s",
                        binding.execution_id,
                    )
                    terminal_durable = False
            try:
                current = self._store.read_execution_status(sid, binding.request_id)
                next_revision = (current.status_revision + 1) if current is not None else 1
                persisted_status = status if terminal_durable else "interrupted"
                persisted_reason = (
                    reason
                    if terminal_durable
                    else "terminal-journal-unavailable"
                )
                self._store.write_execution_status(
                    sid,
                    binding.request_id,
                    status=persisted_status,
                    status_revision=next_revision,
                    reason=persisted_reason,
                )
            except Exception:
                logger.exception(
                    "approved execution status persistence failed execution_id=%s",
                    binding.execution_id,
                )
            lock = self._locks[sid]
            try:
                async with lock:
                    session = self._get_session(sid)
                    if session.active_run_id == binding.run_id:
                        session.active_run_id = None
                    if session.status != "closed":
                        session.status = (
                            "closed"
                            if session.mode == "one_shot"
                            else "waiting_for_input"
                        )
                    session.updated_at = _now()
                    self._store.write_meta(session)
            except Exception:
                logger.exception(
                    "approved execution session finalization failed execution_id=%s",
                    binding.execution_id,
                )
            self._approved_execution_tasks.pop(binding.execution_id, None)
            self._running_runs.pop(sid, None)

    # 按 request identity 查询 approved execution authority/status，不依赖 live task map
    async def get_execution(self, sid: str, request_id: str) -> PlanGetExecutionResult:
        from kama_claude.core.bus.commands import PlanGetExecutionResult

        self._get_session(sid)
        binding = self._store.read_approved_execution_binding(sid, request_id)
        if binding is None:
            raise HandlerError(INVALID_PARAMS, "approved execution not found")
        status = self._store.read_execution_status(sid, request_id)
        reconciled = await self._reconcile_execution_status(binding, status)
        return PlanGetExecutionResult(
            session_id=sid,
            request_id=request_id,
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            projection_key=binding.projection_key,
            status=reconciled.status,
            status_revision=reconciled.status_revision,
            status_digest=None,
            reason=reconciled.reason,
        )

    # 从 durable run journal 推导 terminal execution status，避免 status cache 成为第二 authority
    async def _reconcile_execution_status(
        self,
        binding: ApprovedExecutionBinding,
        cached: ExecutionStatusProjection | None,
    ) -> ExecutionStatusProjection:
        if self._journal is not None and self._journal.has_stream(f"run:{binding.run_id}"):
            high = self._journal.high_watermark(f"run:{binding.run_id}")
            if high > 0:
                replay = await self._journal.read_replay(
                    f"run:{binding.run_id}",
                    after_seq=0,
                    high_watermark=high,
                )
                for record in reversed(replay.records):
                    event = record.event
                    if event.get("type") != "run.finished":
                        continue
                    raw_status_value = event.get("execution_status")
                    if raw_status_value is None:
                        raw_status: ExecutionStatus = (
                            "completed_unverified"
                            if event.get("status") == "success"
                            else "failed"
                        )
                    elif raw_status_value in {
                        "completed_unverified",
                        "failed",
                        "cancelled",
                        "scope_denied",
                        "inconclusive",
                        "interrupted",
                    }:
                        raw_status = cast(ExecutionStatus, raw_status_value)
                    else:
                        raw_status = "failed"
                    if raw_status not in {
                        "completed_unverified",
                        "failed",
                        "cancelled",
                        "scope_denied",
                        "inconclusive",
                        "interrupted",
                    }:
                        raw_status = "failed"
                    durable_reason = event.get("reason")
                    if (
                        cached is not None
                        and cached.status == raw_status
                        and cached.reason == durable_reason
                    ):
                        return cached
                    revision = (cached.status_revision + 1) if cached is not None else 1
                    return self._store.write_execution_status(
                        binding.session_id,
                        binding.request_id,
                        status=raw_status,
                        status_revision=revision,
                        reason=durable_reason,
                        authoritative=True,
                    )
        if cached is not None:
            return cached
        return self._store.write_execution_status(
            binding.session_id,
            binding.request_id,
            status="interrupted",
            status_revision=0,
            reason="execution-not-started-or-interrupted",
        )

    # 从内存索引取 session，不存在时抛 JSON-RPC 结构化错误
    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "session not found")
        return session
