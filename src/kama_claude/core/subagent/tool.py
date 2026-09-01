from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from kama_claude.core.agents.loader import AgentProfile, AgentProfileLoader
from kama_claude.core.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator, JournalError
from kama_claude.core.grounding import (
    ArchitectureSliceService,
    ArchitectureSliceSubmitTool,
    RepositoryInstructionLoader,
    ToolObservationCollector,
    architecture_slice_result_payload,
    render_repository_instructions,
)
from kama_claude.core.loop import AgentLoop
from kama_claude.core.memory.loader import load_context_file
from kama_claude.core.planning import (
    PlannerDecisionService,
    PlannerDecisionSubmitTool,
    SubmittedDecisionIdentity,
    planner_failure_message,
    render_planner_decision_execution_summary,
)
from kama_claude.core.runs import new_run_id
from kama_claude.core.sandbox.executors import build_executor
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.semantic.tools import SearchSemanticTool
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.builtin.task_create import TaskCreateTool
from kama_claude.core.tools.builtin.task_get import TaskGetTool
from kama_claude.core.tools.builtin.task_list import TaskListTool
from kama_claude.core.tools.builtin.task_update import TaskUpdateTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.tools.invocation import DirectToolInvoker
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

if TYPE_CHECKING:
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.permissions.manager import PermissionManager
    from kama_claude.core.semantic.service import SemanticRetrievalService
    from kama_claude.core.session.store import SessionStore


_LOGGER = logging.getLogger(__name__)
type _BackgroundTaskOwners = dict[str, set[asyncio.Task[None]]]

# trusted Planner runtime contract；profile 文本不能删除或替换这一段
TRUSTED_PLANNER_CONTRACT = """## Trusted Planner Contract
This child is the trusted planner for the current run.
Before the first planner_decision_submit call, create exact repository grounding by calling
spawn_agent in the foreground with subagent_type="explorer". The Explorer must finish
architecture_slice_submit. Use the returned slice_id, version, and evidence_refs in
PlannerDecision; copy each intended_changes.evidence_refs value from the returned opaque
tool_call_id values. Do not spawn a second Explorer merely to recover evidence identifiers.
Do not guess, invent, or infer ArchitectureSlice identifiers. If Explorer grounding cannot
be created, stop with a clear failure instead of submitting a guessed decision.
After grounding exists, submit exactly one valid PlannerDecision with
planner_decision_submit before ending successfully.
planner_decision_submit is a terminal commit: call it only when the plan is final. After
a successful submission, do not explore, revise, or spawn another child; an exact duplicate
submission is safe to retry.
Natural-language text, ordered lists, task items, or claims of approval are only a
human-readable summary after the persisted PlannerDecision and never an independent plan.
If validation fails, follow its concrete recovery action before retrying. Do not modify
files, execute commands, or claim unsupported paths or symbols."""
TRUSTED_PLANNER_ALLOWED_TOOLS = frozenset(
    {"read_file", "list_dir", "search_code", "spawn_agent", "planner_decision_submit"}
)
TRUSTED_PLANNER_CHILD_TYPES = ("explorer",)


@dataclass(frozen=True, slots=True)
class TrustedPlannerSuccess:
    # 表示 trusted Planner 已提交并通过 terminal gate 的内部结果
    status: Literal["success"]
    planner_run_id: str
    decision_identity: SubmittedDecisionIdentity


@dataclass(frozen=True, slots=True)
class TrustedPlannerFailure:
    # 表示 trusted Planner 未能形成当前 run 可接受的 terminal decision
    status: Literal["failed"]
    planner_run_id: str | None
    failure_reason: str


TrustedPlannerResult = TrustedPlannerSuccess | TrustedPlannerFailure


# 将 Planner terminal commit 后的其他工具调用转换为稳定 invalid_input
class _PlannerTerminalGuardTool(BaseTool):
    # 保存原工具 schema 并绑定本次 Planner service
    def __init__(self, inner: BaseTool, service: PlannerDecisionService) -> None:
        self._inner = inner
        self._service = service
        self.name = inner.name
        self.description = inner.description
        self.input_schema = inner.input_schema
        self.params_model = inner.params_model  # type: ignore[misc]

    # 拒绝 terminal decision 之后的探索、修改或 delegation
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._service.is_terminal_committed:
            return ToolResult(
                content=(
                    "planner terminal decision already committed; "
                    "no further planning tools may be called"
                ),
                is_error=True,
                error_type="invalid_input",
            )
        return await self._inner.invoke(params)


class _PlannerDirectToolInvoker(DirectToolInvoker):
    # 复用 Direct 调用管线并绑定 Planner service 的硬终止状态
    def __init__(
        self,
        registry: ToolRegistry,
        bus: EventBus,
        run_id: str,
        *,
        service: PlannerDecisionService,
        permission_manager: PermissionManager | None = None,
        session_id: str = "",
    ) -> None:
        super().__init__(
            registry,
            bus,
            run_id,
            permission_manager=permission_manager,
            session_id=session_id,
        )
        self._planner_service = service

    # 让 AgentLoop 在 Planner 锁存不可恢复状态后结束当前 child run
    def terminal_reason(self) -> str | None:
        return self._planner_service.invocation_terminal_reason


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 取消并等待同一 parent 拥有的后台 children，重复 cancellation 后仍先完成清理
async def _cancel_background_tasks(tasks: set[asyncio.Task[None]]) -> None:
    if not tasks:
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    waiter = asyncio.gather(*tasks, return_exceptions=True)
    primary: asyncio.CancelledError | None = None
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError as exc:
            if primary is None:
                primary = exc
    outcomes = waiter.result()
    failures = sum(
        isinstance(outcome, Exception)
        for outcome in outcomes
    )
    if failures:
        _LOGGER.error(
            "background child cleanup failures=%d role=secondary",
            failures,
        )
    if primary is not None:
        raise primary


class SpawnAgentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    prompt: str
    run_in_background: bool = False
    subagent_type: str = ""
    exploration_level: Literal["light", "standard", "deep"] | None = None


# 按当前 runtime allowlist 构造模型实际可见的子角色 schema
def _spawn_agent_input_schema(
    allowed_subagent_types: list[str] | None,
) -> dict[str, Any]:
    role_schema: dict[str, Any] = {
        "type": "string",
        "description": (
            "Agent role profile (planner/explorer/executor/reviewer). "
            "Leave empty for default."
        ),
    }
    if allowed_subagent_types:
        allowed = sorted(set(allowed_subagent_types))
        role_schema["enum"] = allowed
        role_schema["description"] = (
            "Only allowed in this context: " + ", ".join(allowed) + "."
        )
    elif allowed_subagent_types == []:
        role_schema["description"] = "No subagent types are allowed in this context."
    return {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "3-5 word task description shown in progress display",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete task description including all context the sub-agent needs. "
                    "The sub-agent cannot see the parent conversation, so be explicit."
                ),
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "When true, returns immediately with a run_id; "
                    "use agent_result to poll."
                ),
            },
            "subagent_type": role_schema,
            "exploration_level": {
                "type": ["string", "null"],
                "enum": ["light", "standard", "deep", None],
                "description": "Optional repository exploration depth hint.",
            },
        },
        "required": ["description", "prompt"],
    }


# 在隔离的冷启动上下文中派生子 agent，支持前台阻塞和后台并行两种模式
class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn an isolated sub-agent to handle a self-contained sub-task. "
        "The sub-agent starts with a clean context containing only the provided prompt — "
        "it does not inherit the current conversation history. "
        "Use run_in_background=true to run in parallel; retrieve result later with agent_result."
    )
    input_schema: dict[str, Any] = _spawn_agent_input_schema(None)
    params_model = SpawnAgentParams

    # 构造 SpawnAgentTool；depth=0 表示根 agent，最大允许嵌套深度为 2
    def __init__(
        self,
        provider: LLMProvider,
        workspace_root: Path,
        parent_bus: EventBus,
        parent_run_id: str,
        permission_manager: PermissionManager | None,
        max_steps: int,
        task_registry: BackgroundTaskRegistry,
        runs_dir: Path,
        session_id: str,
        depth: int = 0,
        journal: EventJournalCoordinator | None = None,
        background_tasks: _BackgroundTaskOwners | None = None,
        sandbox_manager: SandboxManager | None = None,
        store: SessionStore | None = None,
        semantic_service: SemanticRetrievalService | None = None,
        semantic_degradation: str = "literal_fallback",
        allowed_subagent_types: list[str] | None = None,
        git_head: str | None = None,
        planning_only: bool = False,
    ) -> None:
        self.input_schema = _spawn_agent_input_schema(allowed_subagent_types)
        self._provider = provider
        self._path_resolver = WorkspacePathResolver(workspace_root)
        self._workspace_root = self._path_resolver.root
        self._access_policy = WorkspaceAccessPolicy(self._workspace_root)
        self._profile_loader = AgentProfileLoader(self._workspace_root)
        self._parent_bus = parent_bus
        self._parent_run_id = parent_run_id
        self._permission_manager = permission_manager
        self._max_steps = max_steps
        self._task_registry = task_registry
        self._runs_dir = runs_dir
        self._session_id = session_id
        self._depth = depth
        self._journal = journal
        self._sandbox_manager = sandbox_manager
        self._store = store
        self._semantic_service = semantic_service
        self._semantic_degradation = semantic_degradation
        self._git_head = git_head
        self._planning_only = planning_only
        self._allowed_subagent_types = (
            None if allowed_subagent_types is None else set(allowed_subagent_types)
        )
        self._background_tasks = background_tasks if background_tasks is not None else {}

    # 派生子 agent，前台时阻塞直到完成并返回结果，后台时立即返回 run_id
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        trusted_requested = params.get("subagent_type") == "planner"
        try:
            result = await self._invoke_impl(params, return_internal=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not trusted_requested:
                raise
            reason = (
                "plan-event-append-failed"
                if isinstance(exc, JournalError)
                else "planner-contract-failure"
            )
            return ToolResult(
                content=planner_failure_message(reason),
                is_error=True,
                error_type="command_failed",
            )
        assert isinstance(result, ToolResult)
        return result

    # 以 runtime-only typed API 执行一个前台 trusted Planner，不暴露可变 capture side channel
    async def run_trusted_planner_foreground(
        self,
        *,
        goal: str,
        description: str = "planner",
    ) -> TrustedPlannerResult:
        result = await self._invoke_impl(
            {
                "description": description,
                "prompt": goal,
                "run_in_background": False,
                "subagent_type": "planner",
            },
            return_internal=True,
        )
        if isinstance(result, TrustedPlannerSuccess | TrustedPlannerFailure):
            return result
        return TrustedPlannerFailure(
            status="failed",
            planner_run_id=None,
            failure_reason="planner-contract-failure",
        )

    # 共享普通 tool adapter 与 planning runtime 的 child 执行实现
    async def _invoke_impl(
        self,
        params: dict[str, object],
        *,
        return_internal: bool,
    ) -> ToolResult | TrustedPlannerResult:
        p = SpawnAgentParams.model_validate(params)

        if (
            self._allowed_subagent_types is not None
            and p.subagent_type not in self._allowed_subagent_types
        ):
            allowed = ", ".join(sorted(self._allowed_subagent_types)) or "none"
            return ToolResult(
                content=f"subagent type is not allowed; allowed subagent types: {allowed}",
                is_error=True,
                error_type="invalid_input",
            )

        if self._depth >= 2:
            return ToolResult(
                content="Subagent nesting limit (2) reached; cannot spawn further subagents.",
                is_error=True,
                error_type="invalid_input",
            )

        profile: AgentProfile | None = None
        if p.subagent_type:
            profile = self._profile_loader.load(p.subagent_type)
            if profile is None:
                return ToolResult(
                    content=f"subagent profile is unavailable: {p.subagent_type}",
                    is_error=True,
                    error_type="invalid_input",
                )
        trusted_planner = p.subagent_type == "planner"
        if trusted_planner and (self._store is None or not self._session_id):
            if return_internal:
                return TrustedPlannerFailure(
                    status="failed",
                    planner_run_id=None,
                    failure_reason="projection-incomplete",
                )
            return ToolResult(
                content="trusted planner requires a session-backed planning store",
                is_error=True,
                error_type="invalid_input",
            )
        if profile is not None and profile.name == "explorer" and p.run_in_background:
            return ToolResult(
                content="repository explorer must run in foreground to return a typed slice",
                is_error=True,
                error_type="invalid_input",
            )

        child_run_id = new_run_id()
        global_ctx = load_context_file(Path("~/.kama/context.md").expanduser())
        project_ctx = load_context_file(
            self._workspace_root / ".kama" / "context.md"
        )
        root_instructions = RepositoryInstructionLoader(self._workspace_root).load([])
        child_goal = p.prompt
        if p.subagent_type == "explorer" and p.exploration_level is not None:
            child_goal += f"\n\nExploration level hint: {p.exploration_level}."
        effective_profile_prompt = profile.system_prompt if profile else None
        if trusted_planner:
            effective_profile_prompt = "\n\n".join(
                part for part in (effective_profile_prompt, TRUSTED_PLANNER_CONTRACT) if part
            )
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=child_goal,
            max_steps=self._max_steps,
            global_context=global_ctx,
            project_context=project_ctx,
            repository_instructions=render_repository_instructions(
                root_instructions.root_sources
            ),
            system_prompt_override=effective_profile_prompt,
        )

        child_bus = EventBus()

        # 将子 bus 所有事件桥接到父 bus，TUI 据此渲染嵌套进度
        async def _bridge(event: BaseModel) -> None:
            if self._planning_only and getattr(event, "type", "") == "llm.token":
                return
            await self._parent_bus.publish(event)

        child_bus.subscribe(_bridge)

        slice_service: ArchitectureSliceService | None = None
        planner_service: PlannerDecisionService | None = None
        if profile is not None and profile.name == "explorer":
            collector = ToolObservationCollector()
            child_bus.subscribe(collector.handle)
            slice_service = ArchitectureSliceService(
                workspace_root=self._workspace_root,
                run_id=child_run_id,
                goal=p.prompt,
                collector=collector,
                session_id=self._session_id,
                store=self._store,
                git_head=self._git_head,
            )
        if trusted_planner and self._store is not None and self._session_id:
            planner_service = PlannerDecisionService(
                workspace_root=self._workspace_root,
                session_id=self._session_id,
                store=self._store,
                goal=p.prompt,
                run_id=child_run_id,
            )
        child_registry = self._build_child_registry(
            child_bus,
            child_run_id,
            profile,
            slice_service=slice_service,
            planner_service=planner_service,
        )
        if planner_service is None:
            direct_invoker = DirectToolInvoker(
                child_registry,
                child_bus,
                child_run_id,
                permission_manager=self._permission_manager,
                session_id=self._session_id,
            )
        else:
            direct_invoker = _PlannerDirectToolInvoker(
                child_registry,
                child_bus,
                child_run_id,
                service=planner_service,
                permission_manager=self._permission_manager,
                session_id=self._session_id,
            )
        child_loop = AgentLoop(
            self._provider,
            direct_invoker,
            child_bus,
        )

        child_run_path = self._runs_dir / child_run_id
        child_run_path.mkdir(parents=True, exist_ok=True)

        try:
            if self._journal is not None:
                await self._journal.register_run(
                    child_run_id,
                    child_run_path,
                    session_id=self._session_id or None,
                )

            await self._parent_bus.publish(
                SubagentStartedEvent(
                    run_id=child_run_id,
                    parent_run_id=self._parent_run_id,
                    description=p.description,
                    ts=_now(),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not trusted_planner:
                raise
            reason = (
                "plan-event-append-failed"
                if isinstance(exc, JournalError)
                else "planner-contract-failure"
            )
            if return_internal:
                return TrustedPlannerFailure(
                    status="failed",
                    planner_run_id=None,
                    failure_reason=reason,
                )
            return ToolResult(
                content=planner_failure_message(reason),
                is_error=True,
                error_type="command_failed",
            )

        if p.run_in_background:
            lifecycle_entered = asyncio.Event()
            task: asyncio.Task[None] = asyncio.create_task(
                self._run_background(
                    child_loop,
                    child_context,
                    child_bus,
                    child_run_path,
                    child_run_id,
                    lifecycle_entered,
                    planner_service,
                )
            )
            self._task_registry.register(child_run_id, task, child_context)
            self._background_tasks.setdefault(self._parent_run_id, set()).add(task)
            try:
                await lifecycle_entered.wait()
            except asyncio.CancelledError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    _LOGGER.exception(
                        "background cleanup failure treated as secondary run_id=%s",
                        child_run_id,
                    )
                raise
            return ToolResult(
                content=(
                    f"Subagent started in background. run_id={child_run_id}. "
                    f"Use agent_result(run_id='{child_run_id}') to retrieve result."
                )
            )

        try:
            await self._run_child(
                child_loop,
                child_context,
                child_bus,
                child_run_path,
                child_run_id,
                planner_service=planner_service,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if return_internal and trusted_planner:
                reason = (
                    "plan-event-append-failed"
                    if isinstance(exc, JournalError)
                    else "planner-contract-failure"
                )
                return TrustedPlannerFailure(
                    status="failed",
                    planner_run_id=child_run_id,
                    failure_reason=reason,
                )
            if trusted_planner:
                reason = (
                    "plan-event-append-failed"
                    if isinstance(exc, JournalError)
                    else "planner-contract-failure"
                )
                return ToolResult(
                    content=planner_failure_message(reason),
                    is_error=True,
                    error_type="command_failed",
                )
            raise

        if return_internal and trusted_planner:
            assert planner_service is not None
            if child_context.status == "success" and planner_service.is_terminal_committed:
                identity = planner_service.terminal_decision
                if identity is not None:
                    return TrustedPlannerSuccess(
                        status="success",
                        planner_run_id=child_run_id,
                        decision_identity=identity,
                    )
            terminal_reason = planner_service.terminal_failure_reason()
            if terminal_reason is None:
                terminal_reason = child_context.reason or "planner-contract-failure"
            return TrustedPlannerFailure(
                status="failed",
                planner_run_id=child_run_id,
                failure_reason=terminal_reason,
            )

        if trusted_planner and planner_service is not None:
            if child_context.status != "success" or not planner_service.is_terminal_committed:
                reason = planner_service.terminal_failure_reason() or "planner-contract-failure"
                return ToolResult(
                    content=planner_failure_message(reason),
                    is_error=True,
                    error_type="command_failed",
                )
            try:
                # /orchestrate 读取完整 immutable V2 decision，而不是 bounded PlanView 或 child 原文
                summary = render_planner_decision_execution_summary(
                    planner_service.read_terminal_decision()
                )
            except ValueError as exc:
                if str(exc) == "planner-result-too-large":
                    return ToolResult(
                        content=planner_failure_message("planner-result-too-large"),
                        is_error=True,
                        error_type="command_failed",
                    )
                return ToolResult(
                    content=planner_failure_message("artifact-corrupt"),
                    is_error=True,
                    error_type="command_failed",
                )
            except Exception:
                return ToolResult(
                    content=planner_failure_message("artifact-corrupt"),
                    is_error=True,
                    error_type="command_failed",
                )
            return ToolResult(content=summary)

        if slice_service is not None:
            architecture_slice = slice_service.submitted
            if architecture_slice is None:
                completeness: Literal["partial", "blocked"] = (
                    "partial"
                    if child_context.reason == "exceeded_max_steps"
                    else "blocked"
                )
                reason = child_context.reason or "explorer ended without a valid submit"
                architecture_slice = slice_service.record_incomplete(
                    completeness,
                    reason,
                )
            return ToolResult(
                content=json.dumps(
                    architecture_slice_result_payload(architecture_slice),
                    sort_keys=True,
                )
            )

        if child_context.status == "success":
            return ToolResult(
                content=child_context.result or "Subagent completed with no text output."
            )
        return ToolResult(
            content=(child_context.result or "Subagent failed to complete the delegated task."),
            is_error=True,
            error_type="command_failed",
        )

    # 运行 child 并在所有 started 后终态发布一次 finished，再恢复原异常控制流
    async def _run_child(
        self,
        loop: AgentLoop,
        context: ExecutionContext,
        bus: EventBus,
        run_path: Path,
        run_id: str,
        lifecycle_entered: asyncio.Event | None = None,
        planner_service: PlannerDecisionService | None = None,
    ) -> None:
        primary_failure: BaseException | None = None
        delivery_failure: BaseException | None = None
        try:
            if lifecycle_entered is not None:
                lifecycle_entered.set()
            await loop.run(context)
        except asyncio.CancelledError as exc:
            context.mark_failed("cancelled")
            primary_failure = exc
        except Exception as exc:
            _LOGGER.exception("subagent execution failed run_id=%s", run_id)
            context.mark_failed("subagent_error")
            primary_failure = exc

        if (
            planner_service is not None
            and not isinstance(primary_failure, asyncio.CancelledError)
        ):
            terminal_reason = planner_service.terminal_failure_reason()
            if terminal_reason is None and context.status != "success":
                terminal_reason = "planner-contract-failure"
            if terminal_reason is not None:
                context.result = planner_failure_message(terminal_reason)
                context.mark_failed(terminal_reason)

        try:
            await _cancel_background_tasks(self._background_tasks.pop(run_id, set()))
        except asyncio.CancelledError as exc:
            if primary_failure is None:
                context.mark_failed("cancelled")
                primary_failure = exc
            else:
                _LOGGER.error(
                    "nested background cleanup cancellation role=secondary run_id=%s",
                    run_id,
                )

        try:
            await self._parent_bus.publish(
                SubagentFinishedEvent(
                    run_id=run_id,
                    parent_run_id=self._parent_run_id,
                    status=context.status,
                    ts=_now(),
                )
            )
        except asyncio.CancelledError as exc:
            delivery_failure = exc
        except Exception as exc:
            delivery_failure = exc

        if primary_failure is not None:
            if (
                planner_service is not None
                and not isinstance(primary_failure, asyncio.CancelledError)
            ):
                return
            if delivery_failure is not None:
                _LOGGER.error(
                    "finished delivery failure treated as secondary run_id=%s",
                    run_id,
                    exc_info=(
                        type(delivery_failure),
                        delivery_failure,
                        delivery_failure.__traceback__,
                    ),
                )
            raise primary_failure
        if delivery_failure is not None:
            raise delivery_failure

    # 后台任务协程：写事件文件，运行 loop，发布完成事件
    async def _run_background(
        self,
        loop: AgentLoop,
        context: ExecutionContext,
        bus: EventBus,
        run_path: Path,
        run_id: str,
        lifecycle_entered: asyncio.Event,
        planner_service: PlannerDecisionService | None = None,
    ) -> None:
        await self._run_child(
            loop,
            context,
            bus,
            run_path,
            run_id,
            lifecycle_entered,
            planner_service,
        )

    # 构造子 registry；基于角色配置过滤工具，深度允许时注册嵌套 SpawnAgentTool
    def _build_child_registry(
        self,
        child_bus: EventBus,
        child_run_id: str,
        profile: AgentProfile | None,
        *,
        slice_service: ArchitectureSliceService | None = None,
        planner_service: PlannerDecisionService | None = None,
    ) -> ToolRegistry:
        from kama_claude.core.task.manager import TaskManager

        allowed: set[str] | None = (
            set(profile.allowed_tools) if profile is not None else None
        )
        if planner_service is not None:
            allowed = (
                set(TRUSTED_PLANNER_ALLOWED_TOOLS)
                if profile is None
                else set(profile.allowed_tools) & set(TRUSTED_PLANNER_ALLOWED_TOOLS)
            )

        effective_child_types: list[str] | None = (
            profile.allowed_subagent_types if profile is not None else None
        )
        if planner_service is not None:
            if profile is None:
                effective_child_types = list(TRUSTED_PLANNER_CHILD_TYPES)
            else:
                requested_child_types = profile.allowed_subagent_types
                effective_child_types = [
                    name
                    for name in TRUSTED_PLANNER_CHILD_TYPES
                    if requested_child_types is not None
                    and name in requested_child_types
                ]

        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()

        def _register(tool: BaseTool) -> None:
            guarded = (
                _PlannerTerminalGuardTool(tool, planner_service)
                if planner_service is not None and tool.name != "planner_decision_submit"
                else tool
            )
            registry.register(guarded)

        _all_tools = [
            ReadFileTool(self._path_resolver, self._access_policy),
            BashTool(
                build_executor(
                    self._sandbox_manager, workspace_root=self._workspace_root
                ),
                workspace_root=self._workspace_root,
            ),
            WriteFileTool(self._path_resolver, self._access_policy),
            ListDirTool(self._path_resolver, self._access_policy),
            SearchCodeTool(self._path_resolver, self._access_policy),
        ]
        for t in _all_tools:
            if _allowed(t.name):
                _register(t)

        if self._semantic_service is not None and _allowed("search_semantic"):
            _register(
                SearchSemanticTool(
                    self._semantic_service,
                    fallback=SearchCodeTool(self._path_resolver, self._access_policy),
                    degradation=self._semantic_degradation,
                )
            )
        if slice_service is not None and _allowed("architecture_slice_submit"):
            _register(ArchitectureSliceSubmitTool(slice_service))
        if planner_service is not None and _allowed("planner_decision_submit"):
            _register(PlannerDecisionSubmitTool(planner_service))

        child_task_manager = TaskManager(self._runs_dir / child_run_id / ".tasks")
        for t in [
            TaskCreateTool(child_task_manager),
            TaskUpdateTool(child_task_manager),
            TaskListTool(child_task_manager),
            TaskGetTool(child_task_manager),
        ]:
            if _allowed(t.name):
                _register(t)

        if self._depth < 1:
            nested = SpawnAgentTool(
                provider=self._provider,
                workspace_root=self._workspace_root,
                parent_bus=child_bus,
                parent_run_id=child_run_id,
                permission_manager=self._permission_manager,
                max_steps=self._max_steps,
                task_registry=self._task_registry,
                runs_dir=self._runs_dir,
                session_id=self._session_id,
                depth=self._depth + 1,
                journal=self._journal,
                background_tasks=self._background_tasks,
                sandbox_manager=self._sandbox_manager,
                store=self._store,
                semantic_service=self._semantic_service,
                semantic_degradation=self._semantic_degradation,
                git_head=self._git_head,
                allowed_subagent_types=(
                    effective_child_types
                ),
                planning_only=self._planning_only,
            )
            if _allowed("spawn_agent"):
                _register(nested)
            if _allowed("agent_result"):
                _register(AgentResultTool(self._task_registry))

        return registry


class AgentResultParams(BaseModel):
    run_id: str


# 查询后台 subagent 的执行状态和最终结果
class AgentResultTool(BaseTool):
    name = "agent_result"
    description = (
        "Retrieve the result of a background sub-agent previously started with spawn_agent. "
        "Returns 'still running' if the sub-agent has not yet completed."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by spawn_agent(run_in_background=true)",
            },
        },
        "required": ["run_id"],
    }
    params_model = AgentResultParams

    # 初始化，持有共享的后台任务注册表
    def __init__(self, task_registry: BackgroundTaskRegistry) -> None:
        self._task_registry = task_registry

    # 查询指定 run_id 的后台任务状态，返回结果或错误
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = AgentResultParams.model_validate(params)
        entry = self._task_registry.get(p.run_id)
        if entry is None:
            return ToolResult(
                content=f"Unknown background subagent run_id: {p.run_id}.",
                is_error=True,
                error_type="not_found",
            )
        task, context = entry
        if not task.done():
            return ToolResult(content="still running")
        if task.cancelled():
            return ToolResult(
                content="Subagent was cancelled.",
                is_error=True,
                error_type="command_failed",
            )
        exc = task.exception()
        if exc is not None:
            return ToolResult(
                content="Subagent execution failed.",
                is_error=True,
                error_type="execution_error",
            )
        if context.status == "failed":
            return ToolResult(
                content=(
                    context.result
                    or "Subagent failed to complete the delegated task."
                ),
                is_error=True,
                error_type="command_failed",
            )
        return ToolResult(content=context.result or "Subagent completed with no text result.")
