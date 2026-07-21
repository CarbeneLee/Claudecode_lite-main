from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.bus.events import RunFinishedEvent, RunStartedEvent
from kama_claude.core.compact.compactor import Compactor
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus, EventHandler
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.loop import AgentLoop
from kama_claude.core.mcp.server import McpServerManager
from kama_claude.core.memory.loader import load_context_file
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.runs import RUNS_DIR, new_run_id
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import (
    AgentResultTool,
    SpawnAgentTool,
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
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.trace.provider import TracingProvider
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver


def _now() -> str: # 生成当前 UTC 时间的 ISO 格式字符串，用于事件时间戳。
    return datetime.now(UTC).isoformat()
# 返回str而非datetime对象，方便JSONL序列化和日志记录。

@dataclass
class RunOutcome:
    status: str # 运行状态，例如 "success" 或 "failure"
    result: str  # 运行结果的文本输出，可能为 None
    reason: str | None # 运行失败的原因，如果有的话
# 一次运行后的不可变快照结果

class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: KamaConfig, #唯一必须参数
        *, # 通过*传递的可选参数
        workspace_root: Path,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        journal: EventJournalCoordinator | None = None,
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
        self._journal = journal or EventJournalCoordinator()
        self._owns_journal = journal is None
        if self._owns_journal:
            self._bus.subscribe(self._journal.handle)
        # 跨 run 共享的后台 subagent 任务注册表
        self._task_registry = BackgroundTaskRegistry()
        self._background_tasks: dict[str, set[asyncio.Task[None]]] = {}

    # 构建工具注册表，注入 TaskManager（任务工具共享同一实例）；可选注入 SpawnAgentTool
    def _build_registry( #条件化工具注入
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
    ) -> ToolRegistry:
        allowed: set[str] | None = set(tool_whitelist) if tool_whitelist else None

        def _ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        for t in [
            ReadFileTool(self._path_resolver, self._access_policy),
            BashTool(self._workspace_root),
            WriteFileTool(self._path_resolver, self._access_policy),
            ListDirTool(self._path_resolver, self._access_policy),
            SearchCodeTool(self._path_resolver, self._access_policy),
        ]:
            if _ok(t.name):
                registry.register(t)
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
                    SpawnAgentTool(
                        provider=provider,
                        workspace_root=self._workspace_root,
                        parent_bus=bus,
                        parent_run_id=run_id,
                        permission_manager=self._permission_manager,
                        max_steps=self._config.agent.max_steps,
                        task_registry=self._task_registry,
                        runs_dir=runs_dir,
                        session_id=session_id,
                        depth=0, # 防止agent无限递归调用自身，depth=0表示这是顶层agent
                        journal=self._journal,
                        background_tasks=self._background_tasks,
                    )
                )
            if _ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
        if self._mcp_manager is not None:
            for mcp_tool in self._mcp_manager.get_tools():
                if _ok(mcp_tool.name):
                    registry.register(mcp_tool)
        return registry

    # 执行一次完整的 agent run（委托给 run_and_capture，忽略返回值）
    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    # 执行 agent run 并返回 RunOutcome（含最终文字结果）
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
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

        # TaskManager 存储在 run_path / ".tasks"，每个 run 相互隔离
        task_manager = TaskManager(run_path / ".tasks")

        bus = self._bus
        for h in self._extra_handlers:
            bus.subscribe(h)

        stream_id = f"run:{run_id}"
        if not self._journal.has_stream(stream_id):
            await self._journal.register_run(
                run_id,
                run_path,
                session_id=(
                    session.id
                    if session is not None and not self._owns_journal
                    else None
                ),
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
            system_prompt_override=system_prompt_override,
        )
        prefill_len = len(history) #避免重复存储历史消息

        await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

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
            )
            session_dir = (
                store.session_dir(session.id)
                if session is not None and store is not None
                else run_path
            )
            compactor = Compactor(bus, session_dir, session_id_str)
            loop = AgentLoop(
                provider, registry, bus,
                permission_manager=self._permission_manager,
                compactor=compactor,
                compact_threshold=self._config.compaction.auto_threshold, #触发压缩上下文的阈值
                session_id=session_id_str,
            )
            await loop.run(context)
        # CancelledError 单独处理，并保存对象供 terminal barrier 后原样恢复
        except asyncio.CancelledError as exc:
            cancelled_error = exc
            if not context.is_done():
                context.mark_failed("cancelled")
        except Exception:
            logging.getLogger(__name__).exception(
                "agent run failed run_id=%s step=%d", run_id, context.step
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
                    "background cleanup cancellation treated as secondary "
                    "run_id=%s role=cleanup",
                    run_id,
                )

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
