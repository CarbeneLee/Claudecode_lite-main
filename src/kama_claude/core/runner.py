from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kama_claude.core.approval import materialize_committed_plan_receipt
from kama_claude.core.bus.events import (
    GitRunDiffEvent,
    PermissionRequestedEvent,
    PlannerDecisionReadyEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
)
from kama_claude.core.compact.compactor import Compactor
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus, EventHandler
from kama_claude.core.events.journal import EventJournalCoordinator, JournalError
from kama_claude.core.execution import ExecutionStatus, TrustedScopedToolRegistry
from kama_claude.core.execution_scope import (
    ExternalWorkspaceDriftError,
    ScopedExecutionContext,
    ScopeMutationInconclusiveError,
)
from kama_claude.core.git.errors import DirtyWorkspaceError, GitError, GitUnavailableError
from kama_claude.core.git.manager import GitManager
from kama_claude.core.git.tools import (
    GitCheckpointTool,
    GitCommitTool,
    GitDiffTool,
    GitRollbackTool,
    GitStatusTool,
)
from kama_claude.core.grounding import (
    RepositoryInstructionLoader,
    render_repository_instructions,
)
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.loop import AgentLoop
from kama_claude.core.mcp.server import McpServerManager
from kama_claude.core.memory.loader import load_context_file
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.planning import (
    ExactPlannerDecisionV2,
    build_plan_view,
    planner_failure_message,
    render_planner_decision,
)
from kama_claude.core.runs import RUNS_DIR, new_run_id
from kama_claude.core.sandbox.executors import build_executor
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.semantic.service import SemanticRetrievalService
from kama_claude.core.semantic.tools import SearchSemanticTool
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import (
    AgentResultTool,
    SpawnAgentTool,
    TrustedPlannerSuccess,
    _cancel_background_tasks,
)
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.builtin import (
    BashTool,
    ListDirTool,
    NoteSaveTool,
    ReadFileTool,
    SearchCodeTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from kama_claude.core.tools.invocation import ApprovedScopedToolInvoker, DirectToolInvoker
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.trace.provider import TracingProvider
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_LOGGER = logging.getLogger(__name__)


def _now() -> str:  # 生成当前 UTC 时间的 ISO 格式字符串，用于事件时间戳。
    return datetime.now(UTC).isoformat()


# 返回str而非datetime对象，方便JSONL序列化和日志记录。


@dataclass
class RunOutcome:
    status: str  # 运行状态，例如 "success" 或 "failure"
    result: str  # 运行结果的文本输出，可能为 None
    reason: str | None  # 运行失败的原因，如果有的话


# 一次运行后的不可变快照结果


class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: KamaConfig,  # 唯一必须参数
        *,  # 通过*传递的可选参数
        workspace_root: Path,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        journal: EventJournalCoordinator | None = None,
        sandbox_manager: SandboxManager | None = None,
        git_manager: GitManager | None = None,
        semantic_service: SemanticRetrievalService | None = None,
    ) -> None:
        self._config = config
        self._path_resolver = WorkspacePathResolver(workspace_root)
        self._workspace_root = self._path_resolver.root
        self._access_policy = WorkspaceAccessPolicy(self._workspace_root)
        self._bus = bus or EventBus()
        self._provider = provider
        self._extra_handlers: list[EventHandler] = extra_handlers or []
        self._runs_dir = runs_dir or RUNS_DIR
        self._trace = trace
        self._permission_manager = permission_manager
        self._mcp_manager = mcp_manager
        self._sandbox_manager = sandbox_manager
        self._git_manager = git_manager
        self._semantic_service = semantic_service
        self._journal = journal or EventJournalCoordinator()
        self._owns_journal = journal is None
        journal_handler = getattr(self._journal, "handle", None)
        if callable(journal_handler):
            self._bus.subscribe(journal_handler)
        # 跨 run 共享的后台 subagent 任务注册表
        self._task_registry = BackgroundTaskRegistry()
        self._background_tasks: dict[str, set[asyncio.Task[None]]] = {}

    # 构建工具注册表，注入 TaskManager（任务工具共享同一实例）；可选注入 SpawnAgentTool
    def _build_registry(  # 条件化工具注入
        self,
        task_manager: TaskManager,
        *,
        session: Session | None = None,
        store: SessionStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        bus: EventBus | None = None,
        child_runs_dir: Path | None = None,
        session_id: str = "",
        tool_whitelist: list[str] | None = None,
        git_head: str | None = None,
    ) -> ToolRegistry:
        allowed: set[str] | None = set(tool_whitelist) if tool_whitelist else None

        def _ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        search_tool = SearchCodeTool(self._path_resolver, self._access_policy)
        for t in [
            ReadFileTool(self._path_resolver, self._access_policy),
            BashTool(
                build_executor(self._sandbox_manager, workspace_root=self._workspace_root),
                workspace_root=self._workspace_root,
            ),
            WriteFileTool(self._path_resolver, self._access_policy),
            ListDirTool(self._path_resolver, self._access_policy),
            search_tool,
        ]:
            if _ok(t.name):
                registry.register(t)
        # search_semantic 可选：注入 service 时注册，降级回退复用同一 search_code 实例
        if self._semantic_service is not None and _ok("search_semantic"):
            registry.register(
                SearchSemanticTool(
                    self._semantic_service,
                    fallback=search_tool,
                    degradation=self._config.semantic.degradation,
                )
            )
        for t in [
            TaskCreateTool(task_manager),
            TaskUpdateTool(task_manager),
            TaskListTool(task_manager),
            TaskGetTool(task_manager),
        ]:
            if _ok(t.name):
                registry.register(t)
        if session is not None and store is not None and run_id is not None:
            note_tool = NoteSaveTool(store, session.id, run_id)
            if _ok(note_tool.name):
                registry.register(note_tool)
        if provider is not None and bus is not None and run_id is not None:
            runs_dir = child_runs_dir or self._runs_dir
            if _ok("spawn_agent"):
                registry.register(
                    self._make_spawn_agent_tool(
                        provider=provider,
                        bus=bus,
                        run_id=run_id,
                        child_runs_dir=runs_dir,
                        session_id=session_id,
                        store=store,
                        git_head=git_head,
                    )
                )
            if _ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
        if self._mcp_manager is not None:
            for mcp_tool in self._mcp_manager.get_tools():
                if _ok(mcp_tool.name):
                    registry.register(mcp_tool)
        # git 工具仅在注入 git manager 且 run_id 已知时注册（非 git 仓库降级为无 git 能力）
        if self._git_manager is not None and run_id is not None:
            for git_tool in [
                GitStatusTool(self._git_manager),
                GitDiffTool(self._git_manager),
                GitCheckpointTool(self._git_manager, run_id),
                GitCommitTool(self._git_manager, run_id),
                GitRollbackTool(self._git_manager, run_id),
            ]:
                if _ok(git_tool.name):
                    registry.register(git_tool)
        return registry

    # 复用 Direct 与 planning path 共用的最小 SpawnAgent wiring，不构造完整 Direct registry
    def _make_spawn_agent_tool(
        self,
        *,
        provider: LLMProvider,
        bus: EventBus,
        run_id: str,
        child_runs_dir: Path,
        session_id: str,
        store: SessionStore | None,
        git_head: str | None,
        planning_only: bool = False,
    ) -> SpawnAgentTool:
        return SpawnAgentTool(
            provider=provider,
            workspace_root=self._workspace_root,
            parent_bus=bus,
            parent_run_id=run_id,
            permission_manager=self._permission_manager,
            max_steps=self._config.agent.max_steps,
            task_registry=self._task_registry,
            runs_dir=child_runs_dir,
            session_id=session_id,
            depth=0,
            journal=self._journal,
            background_tasks=self._background_tasks,
            sandbox_manager=self._sandbox_manager,
            store=store,
            semantic_service=self._semantic_service,
            semantic_degradation=self._config.semantic.degradation,
            git_head=git_head,
            planning_only=planning_only,
        )

    # 执行一次完整的 agent run（委托给 run_and_capture，忽略返回值）
    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    # 执行 agent run 并返回 RunOutcome（含最终文字结果）
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        agent_mode: str = "direct",
        # 无 session 时创建最小历史，作为一次性执行
        session: Session | None = None,
        # 有 session 时将运行结果存入 SessionStore，实现上下文延续
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        run_id = run_id or new_run_id()
        if session is not None and store is not None:
            run_path = store.runs_dir(session.id) / run_id
            history = store.read_messages(session.id)
            notes = store.read_notes(session.id)
        else:
            run_path = self._runs_dir / run_id
            history = [{"role": "user", "content": goal}]
            notes = ""
        run_path.mkdir(parents=True, exist_ok=True)

        global_ctx = load_context_file(Path("~/.kama/context.md").expanduser())
        project_ctx = load_context_file(self._workspace_root / ".kama" / "context.md")
        root_instructions = RepositoryInstructionLoader(self._workspace_root).load([])

        # TaskManager 存储在 run_path / ".tasks"，每个 run 相互隔离
        task_manager = TaskManager(run_path / ".tasks")

        bus = self._bus
        for h in self._extra_handlers:
            bus.subscribe(h)

        stream_id = f"run:{run_id}"
        planning_session_journal = (
            agent_mode == "plan"
            and session is not None
            and store is not None
        )
        if not planning_session_journal and not self._journal.has_stream(stream_id):
            await self._journal.register_run(
                run_id,
                run_path,
                session_id=(session.id if session is not None and not self._owns_journal else None),
            )

        # 创建包含本次运行基本信息和状态的执行上下文
        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            # max_steps 来自 agent 配置
            max_steps=self._config.agent.max_steps,
            # messages 使用完整对话历史进行预填充
            prefill_messages=history,
            session_notes=notes,
            global_context=global_ctx,
            project_context=project_ctx,
            repository_instructions=render_repository_instructions(
                root_instructions.root_sources
            ),
            system_prompt_override=system_prompt_override,
        )
        prefill_len = len(history)  # 避免重复存储历史消息

        await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

        if agent_mode == "plan":
            return await self._run_planning_and_capture(
                goal=goal,
                run_id=run_id,
                run_path=run_path,
                context=context,
                session=session,
                store=store,
                bus=bus,
            )

        # git preflight（P4）：ensure_ready fail-open → dirty ASK → task 分支 + baseline
        session_id_str = session.id if session is not None else ""
        git_manager = self._git_manager
        git_enabled = False
        git_head: str | None = None
        if git_manager is not None:
            try:
                await git_manager.ensure_ready()
            except GitUnavailableError:
                # fail-open：非 git 仓库（或无 git 可用）时降级为无 git 能力的 run
                logging.getLogger(__name__).warning(
                    "git preflight skipped run_id=%s failure_category=fail_open",
                    run_id,
                )
            else:
                git_enabled = True
        step_checkpoint_handler: EventHandler | None = None
        if git_enabled and git_manager is not None:
            gm = git_manager
            try:
                status = await gm.status()
                dirty = status.dirty
                if dirty:
                    allowed = await self._request_git_snapshot(bus, run_id, session_id_str)
                    if not allowed:
                        raise DirtyWorkspaceError(
                            "workspace has uncommitted changes; approve snapshot "
                            "baseline or handle manually"
                        )
                # 先切 task 分支再固化快照：dirty 改动随 checkout 带到 agent 分支，
                # pre-run 提交不再落在 main（零污染）；快照后树干净，baseline 用
                # force 让 ref 指向 HEAD（不产生空提交），保证 step-0 始终存在
                await gm.ensure_task_branch(run_id)
                if dirty:
                    await gm.snapshot_pre_run(run_id)
                baseline = await gm.create_checkpoint(run_id, 0, "baseline", force=True)
                if baseline is not None:
                    git_head = baseline.commit_sha
            except DirtyWorkspaceError:
                # 用户拒绝快照：run 直接失败，工作树保持原样
                await bus.publish(
                    RunFinishedEvent(
                        run_id=run_id,
                        status="failed",
                        reason="dirty_workspace",
                        steps=0,
                        ts=_now(),
                    )
                )
                raise
            except GitError:
                # 其余 git 故障同样 fail-open：本次 run 降级为无 git 能力
                logging.getLogger(__name__).warning(
                    "git preflight degraded run_id=%s failure_category=fail_open",
                    run_id,
                )
                git_enabled = False
            if gm.config.checkpoint_mode == "per_step":
                # per_step 模式：每个 step 结束自动落 checkpoint（auto-step-N）
                async def _step_checkpoint(event: BaseModel) -> None:
                    if isinstance(event, StepFinishedEvent) and event.run_id == run_id:
                        await gm.create_checkpoint(run_id, event.step, f"auto-step-{event.step}")

                step_checkpoint_handler = _step_checkpoint
                bus.subscribe(_step_checkpoint)

        cancelled_error: asyncio.CancelledError | None = None
        try:
            provider: LLMProvider = self._provider or AnthropicProvider(
                self._config.llm.default_model
            )
            if self._trace is not None:
                provider = TracingProvider(
                    provider,
                    self._trace,
                    include_payload=self._config.trace.include_llm_payload,
                )
            session_id_str = session.id if session is not None else ""
            child_runs_dir = (
                store.runs_dir(session.id)
                if session is not None and store is not None
                else self._runs_dir
            )
            registry = self._build_registry(
                task_manager,
                session=session,
                store=store,
                run_id=run_id,
                provider=provider,
                bus=bus,
                child_runs_dir=child_runs_dir,
                session_id=session_id_str,
                tool_whitelist=tool_whitelist,
                git_head=git_head,
            )
            session_dir = (
                store.session_dir(session.id)
                if session is not None and store is not None
                else run_path
            )
            compactor = Compactor(bus, session_dir, session_id_str)
            direct_invoker = DirectToolInvoker(
                registry,
                bus,
                run_id,
                permission_manager=self._permission_manager,
                session_id=session_id_str,
            )
            loop = AgentLoop(
                provider,
                direct_invoker,
                bus,
                compactor=compactor,
                compact_threshold=self._config.compaction.auto_threshold,  # 触发压缩上下文的阈值
            )
            await loop.run(context)
        # CancelledError 单独处理，并保存对象供 terminal barrier 后原样恢复
        except asyncio.CancelledError as exc:
            cancelled_error = exc
            if not context.is_done():
                context.mark_failed("cancelled")
        except Exception:
            logging.getLogger(__name__).error(
                "agent run failed run_id=%s step=%d "
                "failure_role=primary failure_category=propagated_exception",
                run_id,
                context.step,
            )
            if not context.is_done():
                context.mark_failed("llm_error")

        try:
            await _cancel_background_tasks(self._background_tasks.pop(run_id, set()))
        except asyncio.CancelledError as exc:
            if cancelled_error is None:
                cancelled_error = exc
                if not context.is_done() or context.status == "success":
                    context.mark_failed("cancelled")
            else:
                logging.getLogger(__name__).error(
                    "background cleanup cancellation treated as secondary run_id=%s role=cleanup",
                    run_id,
                )

        if step_checkpoint_handler is not None:
            bus.unsubscribe(step_checkpoint_handler)
        if git_enabled and git_manager is not None:
            await self._git_run_end(bus, run_id, git_manager, context, cancelled_error)

        terminal_failure: asyncio.CancelledError | Exception | None = None
        try:
            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )
        except asyncio.CancelledError as exc:
            terminal_failure = exc
        except Exception as exc:
            terminal_failure = exc

        if session is not None and store is not None:
            store.append_messages(session.id, context.messages[prefill_len:], run_id=run_id)

        if cancelled_error is not None:
            if terminal_failure is not None:
                logging.getLogger(__name__).error(
                    "terminal journal failure treated as secondary run_id=%s role=terminal",
                    run_id,
                )
            raise cancelled_error
        if terminal_failure is not None:
            raise terminal_failure

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
        )

    # 在已由 SessionManager 注册的顶层 run 内确定性执行 trusted Planner
    async def _run_planning_and_capture(
        self,
        *,
        goal: str,
        run_id: str,
        run_path: Path,
        context: ExecutionContext,
        session: Session | None,
        store: SessionStore | None,
        bus: EventBus,
    ) -> RunOutcome:
        cancelled_error: asyncio.CancelledError | None = None
        plan = None
        rendered_plan = ""
        planner_run_id = ""
        try:
            if session is None or store is None or self._journal is None:
                context.result = (
                    "Planner failed to produce a valid terminal PlannerDecision:\n"
                    "projection-incomplete"
                )
                context.mark_failed("projection-incomplete")
            else:
                provider: LLMProvider = self._provider or AnthropicProvider(
                    self._config.llm.default_model
                )
                if self._trace is not None:
                    provider = TracingProvider(
                        provider,
                        self._trace,
                        include_payload=self._config.trace.include_llm_payload,
                    )
                spawn_tool = self._make_spawn_agent_tool(
                    provider=provider,
                    bus=bus,
                    run_id=run_id,
                    child_runs_dir=store.runs_dir(session.id),
                    session_id=session.id,
                    store=store,
                    git_head=None,
                    planning_only=True,
                )
                planner_result = await spawn_tool.run_trusted_planner_foreground(
                    goal=goal,
                    description="trusted planner",
                )
                if isinstance(planner_result, TrustedPlannerSuccess):
                    decision_payload = store.read_decision(
                        session.id,
                        planner_result.decision_identity.decision_id,
                        planner_result.decision_identity.version,
                    )
                    decision = ExactPlannerDecisionV2.model_validate(decision_payload)
                    plan = build_plan_view(decision, top_level_run_id=run_id)
                    rendered_plan = render_planner_decision(plan)
                    planner_run_id = planner_result.planner_run_id
                    await self._journal.publish_required_durable(
                        PlannerDecisionReadyEvent(
                            event_id=f"plan-ready:{plan.projection_key}",
                            run_id=run_id,
                            planner_run_id=planner_result.planner_run_id,
                            session_id=session.id,
                            plan=plan,
                            ts=_now(),
                        )
                    )
                    context.result = rendered_plan
                    context.mark_success()
                else:
                    context.result = planner_failure_message(
                        planner_result.failure_reason
                    )
                    context.mark_failed(planner_result.failure_reason)
        except asyncio.CancelledError as exc:
            cancelled_error = exc
            context.result = "Planner run cancelled."
            context.mark_failed("cancelled")
        except JournalError:
            _LOGGER.exception("planning durable projection failed run_id=%s", run_id)
            context.result = (
                "Planner failed to produce a valid terminal PlannerDecision:\n"
                "plan-event-append-failed"
            )
            context.mark_failed("plan-event-append-failed")
        except Exception:
            _LOGGER.exception("planning run failed run_id=%s", run_id)
            context.result = (
                "Planner failed to produce a valid terminal PlannerDecision:\n"
                "artifact-corrupt"
            )
            context.mark_failed("artifact-corrupt")

        if (
            session is not None
            and store is not None
            and context.result
            and context.status == "failed"
        ):
            store.append_message(session.id, "assistant", context.result, run_id=run_id)

        terminal_failure: BaseException | None = None
        try:
            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )
        except asyncio.CancelledError as exc:
            terminal_failure = exc
        except Exception as exc:
            terminal_failure = exc

        if (
            terminal_failure is None
            and cancelled_error is None
            and context.status == "success"
            and plan is not None
            and session is not None
            and store is not None
        ):
            try:
                # 等待 session stream 的 run.finished durable 后再写 derived thread projection
                await self._journal.flush_all()
                try:
                    await materialize_committed_plan_receipt(
                        store=store,
                        journal=self._journal,
                        session_id=session.id,
                        projection_key=plan.projection_key,
                    )
                except Exception:
                    _LOGGER.warning(
                        "committed-plan-receipt-unavailable run_id=%s projection_key=%s",
                        run_id,
                        plan.projection_key,
                        exc_info=True,
                    )
                try:
                    store.append_plan_projection(
                        session.id,
                        rendered_plan,
                        run_id=run_id,
                        projection_key=plan.projection_key,
                        projection_digest=plan.projection_digest,
                        metadata={
                            "plan_key": plan.projection_key,
                            "decision_key": plan.decision_key,
                            "run_id": run_id,
                            "planner_run_id": planner_run_id,
                            "decision_id": plan.decision_id,
                            "decision_version": plan.decision_version,
                            "decision_content_digest": plan.decision_content_digest,
                        },
                    )
                except Exception:
                    _LOGGER.warning(
                        "plan-thread-projection-failed run_id=%s projection_key=%s",
                        run_id,
                        plan.projection_key,
                        exc_info=True,
                    )
            except Exception:
                _LOGGER.warning(
                    "committed-plan-post-commit-failed run_id=%s projection_key=%s",
                    run_id,
                    plan.projection_key,
                    exc_info=True,
                )

        if cancelled_error is not None:
            if terminal_failure is not None:
                _LOGGER.error(
                    "planning terminal journal failure treated as secondary run_id=%s",
                    run_id,
                )
            raise cancelled_error
        if terminal_failure is not None:
            raise terminal_failure
        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
        )

    # 使用 fresh context 和 sealed scoped registry 执行已批准的完整 decision summary
    async def run_approved(
        self,
        *,
        summary: str,
        run_id: str,
        session: Session,
        store: SessionStore,
        execution_context: ScopedExecutionContext,
    ) -> RunOutcome:
        if not summary:
            raise ValueError("approved execution summary is empty")
        run_path = store.runs_dir(session.id) / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        provider: LLMProvider = self._provider or AnthropicProvider(
            self._config.llm.default_model
        )
        if self._trace is not None:
            provider = TracingProvider(
                provider,
                self._trace,
                include_payload=self._config.trace.include_llm_payload,
            )
        registry = TrustedScopedToolRegistry.create(
            self._path_resolver,
            self._access_policy,
        )
        invoker = ApprovedScopedToolInvoker(
            registry,
            self._bus,
            run_id,
            execution_context,
            permission_manager=self._permission_manager,
            session_id=session.id,
        )
        global_ctx = load_context_file(Path("~/.kama/context.md").expanduser())
        project_ctx = load_context_file(self._workspace_root / ".kama" / "context.md")
        instructions = RepositoryInstructionLoader(self._workspace_root).load(
            [*execution_context.scope.files_to_modify, *execution_context.scope.files_to_create]
        )
        context = ExecutionContext(
            run_id=run_id,
            goal="Execute the approved plan",
            max_steps=self._config.agent.max_steps,
            prefill_messages=[{"role": "user", "content": summary}],
            global_context=global_ctx,
            project_context=project_ctx,
            repository_instructions=render_repository_instructions(
                instructions.root_sources
            ),
            system_prompt_override=(
                "You are the approved execution agent. Execute only the exact plan "
                "provided in the user message, using only the available scoped tools. "
                "Do not broaden targets, capabilities, or workspace. Verification is "
                "unavailable in this run; report completion without claiming tests passed."
            ),
        )
        await self._journal.publish_required_durable(
            RunStartedEvent(
                run_id=run_id,
                goal="Execute the approved plan",
                ts=_now(),
                execution_id=execution_context.execution_id,
                execution_status="running",
            )
        )
        loop = AgentLoop(
            provider,
            invoker,
            self._bus,
            compactor=None,
            compact_threshold=0.0,
        )
        cancelled_error: asyncio.CancelledError | None = None
        try:
            await loop.run(context)
            if context.status == "success":
                try:
                    execution_context.terminal_audit()
                except ExternalWorkspaceDriftError:
                    context.mark_failed("scope_denied")
                except ScopeMutationInconclusiveError:
                    context.mark_failed("inconclusive")
        except asyncio.CancelledError as exc:
            cancelled_error = exc
            context.mark_failed("cancelled")
        except Exception:
            _LOGGER.exception("approved execution failed run_id=%s", run_id)
            context.mark_failed("failed")

        if context.status == "success":
            execution_status: ExecutionStatus = "completed_unverified"
        elif context.reason == "scope_denied":
            execution_status = "scope_denied"
        elif context.reason == "inconclusive":
            execution_status = "inconclusive"
        elif context.reason == "cancelled":
            execution_status = "cancelled"
        else:
            execution_status = "failed"
        reason = (
            "execution_completed_unverified"
            if execution_status == "completed_unverified"
            else context.reason or "approved-execution-failed"
        )
        if cancelled_error is not None:
            execution_status = "cancelled"
            reason = "cancelled"
        await self._journal.publish_required_durable(
            RunFinishedEvent(
                run_id=run_id,
                status="success" if execution_status == "completed_unverified" else "failed",
                reason=reason,
                steps=context.step,
                ts=_now(),
                execution_id=execution_context.execution_id,
                execution_status=execution_status,
            )
        )
        if cancelled_error is not None:
            raise cancelled_error
        return RunOutcome(
            status=execution_status,
            result=context.result,
            reason=None if execution_status == "completed_unverified" else reason,
        )

    # dirty 工作树快照审批：走权限通道向用户 ASK，无权限系统时默认拒绝
    async def _request_git_snapshot(self, bus: EventBus, run_id: str, session_id: str) -> bool:
        if self._permission_manager is None:
            return False

        async def _emit(raw: dict[str, Any]) -> None:
            await bus.publish(PermissionRequestedEvent(**raw, run_id=run_id))

        allowed, _decision = await self._permission_manager.check_and_wait(
            tool_use_id=f"pre-run:{run_id}",
            tool_name="git_pre_run_snapshot",
            params={
                "reason": "workspace has uncommitted changes; snapshot them into the run baseline?"
            },
            session_id=session_id,
            event_emitter=_emit,
        )
        return allowed

    # run 结束的 git 收尾：失败自动回滚（取消时保留 refs）+ 发布最终 diff 摘要
    async def _git_run_end(
        self,
        bus: EventBus,
        run_id: str,
        git_manager: GitManager,
        context: ExecutionContext,
        cancelled_error: asyncio.CancelledError | None,
    ) -> None:
        if (
            git_manager.config.auto_rollback_on_fail
            and context.status == "failed"
            and cancelled_error is None
        ):
            baseline = await git_manager.get_checkpoint(run_id, 0)
            if baseline is not None:
                await git_manager.restore(baseline)
        try:
            git_diff = await git_manager.diff()
            await bus.publish(GitRunDiffEvent(run_id=run_id, stat=git_diff.stat, ts=_now()))
        except GitError:
            # diff 摘要失败不影响已完成的 run（fail-open）
            logging.getLogger(__name__).warning(
                "git run-end diff failed run_id=%s failure_category=fail_open",
                run_id,
            )
