from __future__ import annotations

import asyncio
import datetime
import logging
import signal
import subprocess
import time
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import kama_claude
from kama_claude.core.approval import (
    ApprovalConflict,
    ApprovalError,
    ApprovalRecordCorrupt,
)
from kama_claude.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    EchoCommand,
    EchoResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    EventUnsubscribeCommand,
    EventUnsubscribeResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PlanApprovalResult,
    PlanApproveCommand,
    PlanExecuteCommand,
    PlanExecuteResult,
    PlanGetApprovalCommand,
    PlanGetApprovalResult,
    PlanGetExecutionCommand,
    PlanGetExecutionResult,
    PlanRejectCommand,
    PongResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetAgentModeCommand,
    SessionGetAgentModeResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
    SessionSetAgentModeCommand,
    SessionSetAgentModeResult,
)
from kama_claude.core.bus.envelope import INVALID_PARAMS, HandlerError
from kama_claude.core.config import KamaConfig, get_config
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.git.manager import GitManager
from kama_claude.core.llm.provider import AnthropicProvider
from kama_claude.core.logging_setup import setup_logging
from kama_claude.core.mcp.server import McpServerManager
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.permissions.storage import load_policy_file
from kama_claude.core.runner import AgentRunner
from kama_claude.core.runs import new_run_id
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.semantic.service import SemanticRetrievalService
from kama_claude.core.session import SessionManager, SessionStore
from kama_claude.core.trace.record import TraceRecord
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from kama_claude.core.transport.socket_server import (
    HandlerOutcome,
    SocketServer,
    get_connection_context,
)
from kama_claude.core.workspace.context import WorkspaceContext
from kama_claude.core.workspace.errors import INVALID_WORKSPACE, InvalidWorkspaceError
from kama_claude.core.workspace.validation import validate_workspace_root

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


# 供语义索引检测分支切换：返回当前 HEAD sha；非 git 目录/异常返回 None（跳过检查）
def git_head_provider(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


class CoreApp:
    # 初始化单个 daemon 实例的稳定身份与运行时组件引用
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._daemon_instance_id = uuid.uuid4().hex
        self._bus = EventBus()
        self._broadcaster: IpcEventBroadcaster | None = None
        self._journal: EventJournalCoordinator | None = None
        self._trace: TraceWriter | None = None
        self._config: KamaConfig | None = None
        # 保留兼容性观察字段；session-backed run task 只由 SessionManager 拥有
        self._running_runs: set[asyncio.Task[Any]] = set()
        self._sessions: SessionManager | None = None
        self._permission_manager: PermissionManager | None = None
        self._mcp_manager: McpServerManager | None = None
        self._contexts: dict[Path, WorkspaceContext] = {}

    # 处理 core.ping 请求，返回服务版本、运行时长和接收时间
    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        client = params.get("client", "unknown")
        logger.debug("ping from %s", client)
        return PongResult(
            server_version=kama_claude.__version__,
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    # 处理 core.echo 请求，校验消息参数并返回原始 message
    async def _echo_handler(self, params: dict[str, Any]) -> EchoResult:
        cmd = EchoCommand.model_validate(params)
        logger.debug("echo message=%s", cmd.message)
        return EchoResult(
            server_version=kama_claude.__version__,
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
            message=cmd.message,
        )

    # 将 EventBus 事件写入 trace（作为 EventBus 订阅者）
    async def _trace_event_handler(self, event: BaseModel) -> None:
        assert self._trace is not None
        event_dict = event.model_dump()
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE",
                layer="event",
                kind="event",
                run_id=event_dict.get("run_id"),
                data=event_dict,
            )
        )

    # 启动一次 agent run：在 durable owner 注册后返回 run_id
    async def _agent_run_handler(self, params: dict[str, Any]) -> AgentRunResult:
        assert self._sessions is not None
        cmd = AgentRunCommand.model_validate(params)
        try:
            workspace_root = validate_workspace_root(cmd.workspace_root)
        except InvalidWorkspaceError as exc:
            raise HandlerError(
                INVALID_WORKSPACE,
                "invalid workspace_root",
                {"reason": exc.reason},
            ) from exc
        session = await self._sessions.create(
            mode="one_shot",
            title=cmd.goal[:40],
            workspace_root=workspace_root,
            agent_mode=cmd.agent_mode,
        )
        run_id = new_run_id()
        try:
            await self._sessions.send_message(session.id, cmd.goal, run_id=run_id)
        except BaseException:
            await self._close_unexposed_session(session.id)
            raise
        return AgentRunResult(run_id=run_id)

    # 将未向客户端暴露的 one-shot session 收敛到 durable closed 终态
    async def _close_unexposed_session(self, session_id: str) -> None:
        assert self._sessions is not None
        close_task = asyncio.create_task(self._sessions.close(session_id))
        await self._await_secondary_task(close_task, role="session_startup_cleanup")

    # 返回 workspace 对应的运行时上下文（sandbox + git 管理器）；按配置启停，同 workspace 复用实例
    def _workspace_context_for(self, workspace_root: Path) -> WorkspaceContext:
        assert self._config is not None
        context = self._contexts.get(workspace_root)
        if context is None:
            context = WorkspaceContext(
                root=workspace_root,
                sandbox=(
                    SandboxManager(
                        config=self._config.sandbox,
                        workspace_root=workspace_root,
                    )
                    if self._config.sandbox.enabled
                    else None
                ),
                git=(
                    GitManager(
                        config=self._config.git,
                        workspace_root=workspace_root,
                    )
                    if self._config.git.enabled
                    else None
                ),
                semantic=(
                    SemanticRetrievalService(
                        config=self._config.semantic,
                        workspace_root=workspace_root,
                        git_head_provider=git_head_provider,
                    )
                    if self._config.semantic.enabled
                    else None
                ),
            )
            self._contexts[workspace_root] = context
        return context

    # 从 workspace context 取全部管理器（单次构建 context，供 runner_factory 展开注入）
    def _workspace_managers_for(self, workspace_root: Path) -> dict[str, Any]:
        context = self._workspace_context_for(workspace_root)
        return {
            "sandbox_manager": context.sandbox,
            "git_manager": context.git,
            "semantic_service": context.semantic,
        }

    @staticmethod
    # 屏蔽重复取消并等待 cleanup task 终态，失败时只记录脱敏 secondary
    async def _await_secondary_task(
        task: asyncio.Task[Any],
        *,
        role: str,
        log_failure: bool = True,
    ) -> None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            if log_failure:
                logger.error("startup cleanup failed role=%s", role)

    # 创建 chat 或 one_shot session，并返回 session_id
    async def _session_create_handler(self, params: dict[str, Any]) -> SessionCreateResult:
        assert self._sessions is not None
        cmd = SessionCreateCommand.model_validate(params)
        try:
            workspace_root = validate_workspace_root(cmd.workspace_root)
        except InvalidWorkspaceError as exc:
            raise HandlerError(
                INVALID_WORKSPACE,
                "invalid workspace_root",
                {"reason": exc.reason},
            ) from exc
        session = await self._sessions.create(
            mode=cmd.mode,
            title=cmd.title,
            workspace_root=workspace_root,
            agent_mode=cmd.agent_mode,
        )
        return SessionCreateResult(session_id=session.id, status=session.status)

    # 向 session 发送一条用户消息并等待 durable run registration 完成
    async def _session_send_handler(self, params: dict[str, Any]) -> SessionSendMessageResult:
        assert self._sessions is not None
        cmd = SessionSendMessageCommand.model_validate(params)
        run_id = await self._sessions.send_message(cmd.session_id, cmd.content)
        return SessionSendMessageResult(run_id=run_id)

    # 返回 session 的完整 Anthropic messages 历史
    async def _session_history_handler(self, params: dict[str, Any]) -> SessionGetHistoryResult:
        assert self._sessions is not None
        cmd = SessionGetHistoryCommand.model_validate(params)
        messages = await self._sessions.get_history(
            cmd.session_id,
            include_projection_metadata=cmd.include_projection_metadata,
        )
        return SessionGetHistoryResult(messages=messages)

    # 返回 daemon 持久化的 session agent mode
    async def _session_get_agent_mode_handler(
        self,
        params: dict[str, Any],
    ) -> SessionGetAgentModeResult:
        assert self._sessions is not None
        cmd = SessionGetAgentModeCommand.model_validate(params)
        snapshot = await self._sessions.get_agent_mode(cmd.session_id)
        return SessionGetAgentModeResult(
            agent_mode=snapshot.agent_mode,
            revision=snapshot.revision,
        )

    # 返回 committed plan 的当前 approval authority snapshot
    async def _plan_get_approval_handler(self, params: dict[str, Any]) -> PlanGetApprovalResult:
        assert self._sessions is not None
        cmd = PlanGetApprovalCommand.model_validate(params)
        try:
            result = await self._sessions.get_approval(cmd.session_id, cmd.projection_key)
            return result
        except ApprovalRecordCorrupt as exc:
            raise HandlerError(INVALID_PARAMS, "approval-record-corrupt") from exc
        except ApprovalError as exc:
            raise HandlerError(INVALID_PARAMS, str(exc)) from exc

    # 创建一次 immutable approve authority record
    async def _plan_approve_handler(self, params: dict[str, Any]) -> PlanApprovalResult:
        assert self._sessions is not None
        cmd = PlanApproveCommand.model_validate(params)
        try:
            return await self._sessions.resolve_approval(
                cmd.session_id,
                cmd.projection_key,
                action="approve",
                decision_id=cmd.decision_id,
                decision_version=cmd.decision_version,
                content_digest=cmd.content_digest,
                commit_receipt_digest=cmd.commit_receipt_digest,
            )
        except ApprovalRecordCorrupt as exc:
            raise HandlerError(INVALID_PARAMS, "approval-record-corrupt") from exc
        except ApprovalConflict as exc:
            raise HandlerError(INVALID_PARAMS, "approval-already-resolved-conflict") from exc
        except ApprovalError as exc:
            raise HandlerError(INVALID_PARAMS, str(exc)) from exc

    # 创建一次 immutable reject authority record
    async def _plan_reject_handler(self, params: dict[str, Any]) -> PlanApprovalResult:
        assert self._sessions is not None
        cmd = PlanRejectCommand.model_validate(params)
        try:
            return await self._sessions.resolve_approval(
                cmd.session_id,
                cmd.projection_key,
                action="reject",
                decision_id=cmd.decision_id,
                decision_version=cmd.decision_version,
                content_digest=cmd.content_digest,
                commit_receipt_digest=cmd.commit_receipt_digest,
            )
        except ApprovalRecordCorrupt as exc:
            raise HandlerError(INVALID_PARAMS, "approval-record-corrupt") from exc
        except ApprovalConflict as exc:
            raise HandlerError(INVALID_PARAMS, "approval-already-resolved-conflict") from exc
        except ApprovalError as exc:
            raise HandlerError(INVALID_PARAMS, str(exc)) from exc

    # 在 daemon admission lock 内创建一次 approved execution binding
    async def _plan_execute_handler(self, params: dict[str, Any]) -> PlanExecuteResult:
        assert self._sessions is not None
        cmd = PlanExecuteCommand.model_validate(params)
        return await self._sessions.execute_approved_plan(
            cmd.session_id,
            cmd.projection_key,
            cmd.request_id,
        )

    # 通过 request identity 读取 approved execution authority/status
    async def _plan_get_execution_handler(
        self,
        params: dict[str, Any],
    ) -> PlanGetExecutionResult:
        assert self._sessions is not None
        cmd = PlanGetExecutionCommand.model_validate(params)
        return await self._sessions.get_execution(cmd.session_id, cmd.request_id)

    # 在 session lock 内切换下一条消息的 agent mode
    async def _session_set_agent_mode_handler(
        self,
        params: dict[str, Any],
    ) -> SessionSetAgentModeResult:
        assert self._sessions is not None
        cmd = SessionSetAgentModeCommand.model_validate(params)
        snapshot = await self._sessions.set_agent_mode(
            cmd.session_id,
            cmd.agent_mode,
        )
        return SessionSetAgentModeResult(
            agent_mode=snapshot.agent_mode,
            revision=snapshot.revision,
        )

    # 接收客户端权限审批响应，resolve 对应挂起的 Future
    async def _permission_respond_handler(self, params: dict[str, Any]) -> PermissionRespondResult:
        cmd = PermissionRespondCommand.model_validate(params)
        logger.info(
            "permission.respond received tool_use_id=%s decision=%s",
            cmd.tool_use_id,
            cmd.decision,
        )
        if self._permission_manager is None:
            logger.error("permission.respond: PermissionManager not initialized")
            return PermissionRespondResult()
        self._permission_manager.respond(cmd.tool_use_id, cmd.decision)
        return PermissionRespondResult()

    # 手动压缩 session thread，将摘要持久化写入 thread.jsonl
    async def _session_compact_handler(self, params: dict[str, Any]) -> SessionCompactResult:
        assert self._sessions is not None
        cmd = SessionCompactCommand.model_validate(params)
        result = await self._sessions.compact(cmd.session_id, cmd.focus)
        return result  # type: ignore[no-any-return]

    # 关闭 session 并返回 closed 状态
    async def _session_close_handler(self, params: dict[str, Any]) -> SessionCloseResult:
        assert self._sessions is not None
        cmd = SessionCloseCommand.model_validate(params)
        await self._sessions.close(cmd.session_id)
        return SessionCloseResult(status="closed")

    # 注册 live 或 response-first durable replay subscription
    async def _subscribe_handler(
        self,
        params: dict[str, Any],
    ) -> EventSubscribeResult | HandlerOutcome:
        cmd = EventSubscribeCommand.model_validate(params)
        context = get_connection_context()
        context.ensure_open_for_subscription()
        assert self._broadcaster is not None
        durable_stream: str | None = None
        after_seq: int | None = cmd.after_seq
        if cmd.replay_from_run is not None:
            durable_stream = f"run:{cmd.replay_from_run}"
            after_seq = cmd.after_seq or 0
        elif cmd.after_seq is not None:
            if not cmd.scope.startswith(("run:", "session:")):
                raise HandlerError(
                    INVALID_PARAMS,
                    "after_seq requires run: or session: scope",
                )
            durable_stream = cmd.scope

        if durable_stream is None:
            sub_id = self._broadcaster.subscribe(context, cmd.topics, cmd.scope)
            return EventSubscribeResult(
                subscription_id=sub_id,
                daemon_instance_id=self._broadcaster.daemon_instance_id,
            )

        assert self._journal is not None
        prepared = await self._broadcaster.prepare_durable_subscription(
            self._journal,
            context,
            topics=cmd.topics,
            stream_id=durable_stream,
            after_seq=after_seq or 0,
        )
        result = EventSubscribeResult(
            subscription_id=prepared.subscription_id,
            daemon_instance_id=self._broadcaster.daemon_instance_id,
            replayed_count=0,
            stream_id=prepared.stream_id,
            accepted_after_seq=prepared.accepted_after_seq,
            high_watermark_seq=prepared.high_watermark_seq,
        )
        return HandlerOutcome(result=result, post_response=prepared.activation)

    # 删除当前 connection 自己拥有的 subscription_id
    async def _unsubscribe_handler(
        self,
        params: dict[str, Any],
    ) -> EventUnsubscribeResult:
        cmd = EventUnsubscribeCommand.model_validate(params)
        context = get_connection_context()
        assert self._broadcaster is not None
        removed = self._broadcaster.unsubscribe(context, cmd.subscription_id)
        return EventUnsubscribeResult(removed=removed)

    # 先停止连接请求和 detached runs，再关闭 MCP、journal 与 trace
    async def _shutdown(self, server: SocketServer) -> None:
        if self._sessions is not None:
            await self._sessions.cancel_active_runs()
        await server.stop()
        if self._mcp_manager is not None:
            await self._mcp_manager.stop_all()
        for context in self._contexts.values():
            await context.close()
        if self._journal is not None:
            await self._journal.close()
        if self._trace is not None:
            await self._trace.stop()

    # 启动守护进程：加载配置、初始化日志、启动 trace、启动 TCP 服务器，并等待退出信号
    async def run(self) -> None:
        self._start_time = time.monotonic()
        self._config = get_config()
        setup_logging(self._config)

        if self._config.trace.enabled:
            trace_path = Path(self._config.trace.file).expanduser()
            # daemon.jsonl 记录所有事件和 LLM 请求响应的 trace，便于调试和回放。
            # TraceWriter 是异步上下文管理器，自动处理文件打开和关闭。
            self._trace = TraceWriter(trace_path)
            await self._trace.start()
            self._bus.subscribe(self._trace_event_handler)

        policy_file = Path("~/.kama/policy.toml").expanduser()
        self._permission_manager = PermissionManager(
            policy_file=policy_file,
            timeout_s=self._config.permission.timeout_s,
        )
        logger.info(
            "permission manager: timeout_s=%.1f  persistent=%d entries",
            self._config.permission.timeout_s,
            len(load_policy_file(policy_file)),
        )

        self._broadcaster = IpcEventBroadcaster(
            trace=self._trace,
            daemon_instance_id=self._daemon_instance_id,
        )
        self._journal = EventJournalCoordinator(
            on_durable=self._broadcaster.publish_durable,
            on_live_only=self._broadcaster.publish_live_only,
            on_stream_failure=self._broadcaster.fail_stream,
        )
        sessions_root = Path("~/.kama/sessions").expanduser()
        store = SessionStore(sessions_root)
        assert self._config is not None
        compact_provider = AnthropicProvider(self._config.llm.default_model)

        self._mcp_manager = McpServerManager()
        if self._config.mcp.servers:
            logger.info("mcp: starting %d server(s)", len(self._config.mcp.servers))
            await self._mcp_manager.start_all(self._config.mcp.servers)

        self._sessions = SessionManager(
            store,
            runner_factory=lambda workspace_root: AgentRunner(
                self._config,  # type: ignore[arg-type]
                workspace_root=workspace_root,
                bus=self._bus,
                trace=self._trace,
                permission_manager=self._permission_manager,
                mcp_manager=self._mcp_manager,
                journal=self._journal,
                **self._workspace_managers_for(workspace_root),
            ),
            bus=self._bus,
            provider=compact_provider,
        )
        self._sessions.attach_journal(self._journal)
        await self._sessions.reconcile_persisted_sessions()

        server = SocketServer(
            self._config.host,
            self._config.port,
            self._broadcaster,
            trace=self._trace,
        )
        server.register("core.ping", self._ping_handler)
        server.register("core.echo", self._echo_handler)
        server.register("agent.run", self._agent_run_handler)
        server.register("event.subscribe", self._subscribe_handler)
        server.register("event.unsubscribe", self._unsubscribe_handler)
        server.register("session.create", self._session_create_handler)
        server.register("session.send_message", self._session_send_handler)
        server.register("session.get_history", self._session_history_handler)
        server.register("session.get_agent_mode", self._session_get_agent_mode_handler)
        server.register("session.set_agent_mode", self._session_set_agent_mode_handler)
        server.register("plan.get_approval", self._plan_get_approval_handler)
        server.register("plan.approve", self._plan_approve_handler)
        server.register("plan.reject", self._plan_reject_handler)
        server.register("plan.execute", self._plan_execute_handler)
        server.register("plan.get_execution", self._plan_get_execution_handler)
        server.register("session.close", self._session_close_handler)
        server.register("permission.respond", self._permission_respond_handler)
        server.register("session.compact", self._session_compact_handler)

        addr = await server.start()
        logger.info("kama-core %s listening addr=%s", kama_claude.__version__, addr)
        logger.info("config: %s", self._config)

        loop = asyncio.get_running_loop()
        shutdown = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, shutdown.set)
        loop.add_signal_handler(signal.SIGTERM, shutdown.set)

        await shutdown.wait()

        logger.info("shutting down")
        await self._shutdown(server)


# 同步入口：启动 CoreApp 事件循环
def run() -> None:
    asyncio.run(CoreApp().run())
