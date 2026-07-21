from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from kama_claude.core.agents.loader import AgentProfile, AgentProfileLoader
from kama_claude.core.bus.events import SubagentFinishedEvent, SubagentStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.loop import AgentLoop
from kama_claude.core.memory.loader import load_context_file
from kama_claude.core.runs import new_run_id
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
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

if TYPE_CHECKING:
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.permissions.manager import PermissionManager


_LOGGER = logging.getLogger(__name__)
type _BackgroundTaskOwners = dict[str, set[asyncio.Task[None]]]


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


# 在隔离的冷启动上下文中派生子 agent，支持前台阻塞和后台并行两种模式
class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn an isolated sub-agent to handle a self-contained sub-task. "
        "The sub-agent starts with a clean context containing only the provided prompt — "
        "it does not inherit the current conversation history. "
        "Use run_in_background=true to run in parallel; retrieve result later with agent_result."
    )
    input_schema: dict[str, Any] = {
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
            "subagent_type": {
                "type": "string",
                "description": (
                    "Agent role profile (planner/executor/reviewer). "
                    "Leave empty for default."
                ),
            },
        },
        "required": ["description", "prompt"],
    }
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
    ) -> None:
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
        self._background_tasks = background_tasks if background_tasks is not None else {}

    # 派生子 agent，前台时阻塞直到完成并返回结果，后台时立即返回 run_id
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = SpawnAgentParams.model_validate(params)

        if self._depth >= 2:
            return ToolResult(
                content="Subagent nesting limit (2) reached; cannot spawn further subagents.",
                is_error=True,
                error_type="invalid_input",
            )

        profile: AgentProfile | None = None
        if p.subagent_type:
            profile = self._profile_loader.load(p.subagent_type)

        child_run_id = new_run_id()
        global_ctx = load_context_file(Path("~/.kama/context.md").expanduser())
        project_ctx = load_context_file(
            self._workspace_root / ".kama" / "context.md"
        )
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=p.prompt,
            max_steps=self._max_steps,
            global_context=global_ctx,
            project_context=project_ctx,
            system_prompt_override=profile.system_prompt if profile else None,
        )

        child_bus = EventBus()

        # 将子 bus 所有事件桥接到父 bus，TUI 据此渲染嵌套进度
        async def _bridge(event: BaseModel) -> None:
            await self._parent_bus.publish(event)

        child_bus.subscribe(_bridge)

        child_registry = self._build_child_registry(child_bus, child_run_id, profile)
        child_loop = AgentLoop(
            self._provider,
            child_registry,
            child_bus,
            permission_manager=self._permission_manager,
            session_id=self._session_id,
        )

        child_run_path = self._runs_dir / child_run_id
        child_run_path.mkdir(parents=True, exist_ok=True)

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

        await self._run_child(
            child_loop,
            child_context,
            child_bus,
            child_run_path,
            child_run_id,
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
    ) -> None:
        await self._run_child(
            loop,
            context,
            bus,
            run_path,
            run_id,
            lifecycle_entered,
        )

    # 构造子 registry；基于角色配置过滤工具，深度允许时注册嵌套 SpawnAgentTool
    def _build_child_registry(
        self,
        child_bus: EventBus,
        child_run_id: str,
        profile: AgentProfile | None,
    ) -> ToolRegistry:
        from kama_claude.core.task.manager import TaskManager

        allowed: set[str] | None = (
            set(profile.allowed_tools) if profile and profile.allowed_tools else None
        )

        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        _all_tools = [
            ReadFileTool(self._path_resolver, self._access_policy),
            BashTool(self._workspace_root),
            WriteFileTool(self._path_resolver, self._access_policy),
            ListDirTool(self._path_resolver, self._access_policy),
            SearchCodeTool(self._path_resolver, self._access_policy),
        ]
        for t in _all_tools:
            if _allowed(t.name):
                registry.register(t)

        child_task_manager = TaskManager(self._runs_dir / child_run_id / ".tasks")
        for t in [
            TaskCreateTool(child_task_manager),
            TaskUpdateTool(child_task_manager),
            TaskListTool(child_task_manager),
            TaskGetTool(child_task_manager),
        ]:
            if _allowed(t.name):
                registry.register(t)

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
            )
            if _allowed("spawn_agent"):
                registry.register(nested)
            if _allowed("agent_result"):
                registry.register(AgentResultTool(self._task_registry))

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
