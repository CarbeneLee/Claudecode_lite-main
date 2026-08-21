from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kama_claude.core.approval import (
    ApprovalRequestOwner,
    ApprovalSnapshot,
    ApprovalSnapshotRelation,
    ApprovalSnapshotState,
    CommittedApprovalTarget,
    approval_snapshot_from_payload,
)
from kama_claude.core.config import KamaConfig
from kama_claude.core.plan_view import PlanReadyCommitReducer, PlanViewV1
from kama_claude.core.session.model import (
    AgentModeSnapshot,
    ModeSnapshotRelation,
    compare_agent_mode_snapshots,
)
from kama_claude.core.transport.socket_client import (
    EventDelivery,
    IpcError,
    SocketClient,
)

_CHAT_TOPICS = [
    "session.*",
    "run.*",
    "tool.*",
    "llm.token",
    "permission.*",
    "planner.*",
    "plan.*",
]
_MAX_RECONNECT_ATTEMPTS = 3
_RECONNECT_DELAY_S = 0.1
# 单次 RPC 的最长等待：daemon 侧 send_message 不再阻塞到 run 完成，
# 此超时仅作为断线兜底，避免主循环无限冻结
_RPC_TIMEOUT_S = 5.0

log = logging.getLogger(__name__)

type DeliveryConsumer = Callable[[EventDelivery], Awaitable[None]]
type RpcOutcome = Literal["ok", "disconnected", "ipc_error"]
type ChatPhase = Literal["create", "subscribe", "interactive"]

_DECISION_MAP: dict[str, str] = {
    "y": "allow_once",
    "a": "always_allow",
    "n": "deny_once",
    "d": "always_deny",
}


class ChatPrinter:
    # 初始化 chat 模式的流式输出状态和待审批权限请求
    def __init__(self) -> None:
        self._inline = False
        self.agent_mode = "direct"
        self.agent_mode_revision = 0
        self._plan_reducer = PlanReadyCommitReducer()
        self.on_plan_committed: Callable[[PlanViewV1], None] | None = None
        self.pending_permission_id: str | None = None
        # run 激活期间输入 y/a/n/d 会被缓冲，等待下一条权限请求自动生效
        self.active_run = False
        self.buffered_decision: str | None = None

    # 若当前 LLM token 尚未换行，则补一个换行
    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

    # 应用已由 daemon 确认的 mode snapshot，并按需显示变更
    def apply_mode_snapshot(self, snapshot: AgentModeSnapshot, *, announce: bool) -> None:
        self.agent_mode = snapshot.agent_mode
        self.agent_mode_revision = snapshot.revision
        if announce:
            self._ensure_newline()
            print(f"[mode: {self.agent_mode}]")

    # 输出已经通过 run.finished(success) 提交的结构化 PlanView
    def _print_committed_plans(self, event: dict[str, Any]) -> None:
        try:
            plans = self._plan_reducer.ingest(event)
        except (KeyError, TypeError, ValueError):
            log.warning("invalid PlanReady event ignored by chat renderer")
            return
        for plan in plans:
            if isinstance(plan, PlanViewV1) and self.on_plan_committed is not None:
                self.on_plan_committed(plan)
            payload = plan.model_dump(mode="json")
            self._ensure_newline()
            print(f"[plan] {payload.get('goal', '')}")
            print(f"  approach: {payload.get('selected_approach', '')}")
            for key, label in (
                ("intended_changes", "intended changes"),
                ("files_to_modify", "files to modify"),
                ("files_to_create", "files to create"),
                ("unresolved_questions", "unresolved"),
                ("assumptions", "assumptions"),
                ("dependency_changes", "dependency changes"),
                ("protocol_or_schema_changes", "protocol/schema changes"),
                ("verification_plan", "verification"),
            ):
                values = payload.get(key) or []
                if values:
                    print(f"  {label}: {values}")

    # 按事件类型打印 chat 输出、等待提示和权限审批请求
    async def handle(self, event: dict[str, Any]) -> None:
        t = event.get("type", "")
        if t == "llm.token":
            print(event.get("token", ""), end="", flush=True)
            self._inline = True
        elif t == "tool.call_started":
            self._ensure_newline()
            print(f"[tool] {event.get('tool_name', '')}")
        elif t == "permission.requested":
            self._ensure_newline()
            tool_name = str(event.get("tool_name", ""))
            param_preview = str(event.get("param_preview", ""))
            tool_use_id = str(event.get("tool_use_id", ""))
            print(f"[permission] {tool_name}  {param_preview}")
            print("  y=allow once  a=always allow  n=deny once  d=always deny")
            self.pending_permission_id = tool_use_id
        elif t == "run.started":
            self.active_run = True
        elif t == "run.finished":
            self._ensure_newline()
            self.active_run = False
            status = str(event.get("status", ""))
            if status == "cancelled":
                print("[run cancelled]")
            elif status not in ("", "success"):
                reason = str(event.get("reason") or "unknown")
                print(f"[run failed: {reason}]")
            self._print_committed_plans(event)

        elif t == "session.waiting_for_input":
            self._ensure_newline()
            self.pending_permission_id = None
            self.active_run = False
            self.buffered_decision = None
            print("[waiting for input]")
        elif t == "session.closed":
            self._ensure_newline()
            self.pending_permission_id = None
            self.active_run = False
            self.buffered_decision = None
            print("session closed.")
        elif t == "planner.decision_ready":
            self._print_committed_plans(event)
        elif t == "plan.approval_changed":
            # approval notification 由 delivery state 校验并合并
            return

    # 以已验证 snapshot 显示 approval authority 状态，不直接信任 raw notification 文本
    def show_approval(self, snapshot: ApprovalSnapshot) -> None:
        self._ensure_newline()
        print(f"[approval: {snapshot.status}] {snapshot.projection_key}")


# 严格解析新的 mode RPC response，不为缺失字段合成默认 authority
def _parse_mode_snapshot(value: dict[str, Any]) -> AgentModeSnapshot:
    return AgentModeSnapshot(
        agent_mode=value["agent_mode"],
        revision=value["revision"],
    )


@dataclass(slots=True)
class _ChatDeliveryState:
    printer: ChatPrinter
    stream_id: str
    daemon_instance_id: str | None
    daemon_changed: asyncio.Event
    permission_arrived: asyncio.Event
    cursor: int = 0
    rendered_event_ids: set[str] = field(default_factory=set)
    mode_snapshot: AgentModeSnapshot | None = None
    mode_refresh_required: asyncio.Event = field(default_factory=asyncio.Event)
    mode_refresh_in_flight: bool = False
    approval_targets: dict[str, CommittedApprovalTarget] = field(default_factory=dict)
    approval_states: dict[str, ApprovalSnapshotState] = field(default_factory=dict)
    approval_refresh_required: asyncio.Event = field(default_factory=asyncio.Event)
    approval_refresh_in_flight: bool = False
    approval_client: SocketClient | None = None

    # 将 printer 的 committed PlanView 回调接到 approval identity 状态
    def __post_init__(self) -> None:
        self.printer.on_plan_committed = self.register_committed_plan

    # 为新 committed projection 记录精确 approval target，并安排 authority 查询
    def register_committed_plan(self, plan: PlanViewV1) -> None:
        session_id = self.stream_id.removeprefix("session:")
        if not session_id:
            return
        target = CommittedApprovalTarget.from_plan(session_id, plan)
        owner = ApprovalRequestOwner(
            client_identity="cli",
            session_id=session_id,
            daemon_instance_id=self.daemon_instance_id or "",
            projection_key=plan.projection_key,
        )
        state = self.approval_states.get(plan.projection_key)
        if state is None or state.owner != owner:
            state = ApprovalSnapshotState(owner)
            self.approval_states[plan.projection_key] = state
        self.approval_targets[plan.projection_key] = target
        self.approval_refresh_required.set()

    # 读取一个 target 的状态，拒绝跨 session 或模糊 projection 选择
    def select_approval_target(
        self,
        projection_key: str | None = None,
    ) -> tuple[CommittedApprovalTarget, ApprovalSnapshotState] | None:
        if projection_key is None:
            if len(self.approval_targets) != 1:
                return None
            projection_key = next(iter(self.approval_targets))
        target = self.approval_targets.get(projection_key)
        state = self.approval_states.get(projection_key)
        if target is None or state is None:
            return None
        return target, state

    # 合并 approval payload 前验证 committed decision identity
    def _parse_approval_payload(
        self,
        target: CommittedApprovalTarget,
        payload: dict[str, Any],
    ) -> ApprovalSnapshot | None:
        if not target.matches_payload(payload):
            log.warning(
                "approval identity mismatch ignored projection_key=%s",
                target.projection_key,
            )
            return None
        try:
            return approval_snapshot_from_payload(payload)
        except (KeyError, TypeError, ValueError):
            log.warning(
                "invalid approval snapshot ignored projection_key=%s",
                target.projection_key,
            )
            return None

    # 处理非权威 approval notification，只设置冲突刷新信号而不发 RPC
    def handle_approval_event(self, event: dict[str, Any]) -> None:
        projection_key = str(event.get("projection_key", ""))
        selected = self.select_approval_target(projection_key)
        if selected is None:
            return
        target, state = selected
        payload = dict(event)
        payload["decision_id"] = target.decision_id
        payload["decision_version"] = target.decision_version
        payload["content_digest"] = target.decision_content_digest
        snapshot = self._parse_approval_payload(target, payload)
        if snapshot is None:
            return
        relation = state.merge(snapshot)
        if relation is ApprovalSnapshotRelation.CONFLICT:
            self.approval_refresh_required.set()
        if relation in (
            ApprovalSnapshotRelation.APPLY,
            ApprovalSnapshotRelation.CONFLICT,
        ):
            self.printer.show_approval(state.snapshot)

    # 以 authority GET 结果直接替换状态，冲突刷新不再进入普通 merge
    def apply_approval_authority(
        self,
        projection_key: str,
        payload: dict[str, Any],
        *,
        epoch: int | None,
    ) -> bool:
        selected = self.select_approval_target(projection_key)
        if selected is None:
            return False
        target, state = selected
        snapshot = self._parse_approval_payload(target, payload)
        if snapshot is None:
            return False
        if epoch is None:
            applied = state.seed_authoritative_snapshot(snapshot, owner=state.owner)
        else:
            applied = state.apply_authoritative_snapshot(
                snapshot,
                epoch=epoch,
                owner=state.owner,
            )
        if applied:
            self.printer.show_approval(snapshot)
        return applied

    # 将 approve/reject RPC 结果走普通 snapshot merge，并保留冲突刷新语义
    def merge_approval_response(self, payload: dict[str, Any]) -> ApprovalSnapshotRelation | None:
        projection_key = str(payload.get("projection_key", ""))
        selected = self.select_approval_target(projection_key)
        if selected is None:
            return None
        target, state = selected
        snapshot = self._parse_approval_payload(target, payload)
        if snapshot is None:
            return None
        relation = state.merge(snapshot)
        if relation is ApprovalSnapshotRelation.CONFLICT:
            self.approval_refresh_required.set()
        if relation in (
            ApprovalSnapshotRelation.APPLY,
            ApprovalSnapshotRelation.CONFLICT,
        ):
            self.printer.show_approval(state.snapshot)
        return relation

    # 清除一次 committed view 的 approval 状态，避免新 session 继承旧 target
    def reset_approval_view(self) -> None:
        self.approval_targets.clear()
        self.approval_states.clear()
        self.approval_refresh_required.clear()

    # 将当前 authority snapshot 转为 approve/reject command 的精确参数
    def approval_command_params(
        self,
        projection_key: str | None,
        action: Literal["approve", "reject"],
    ) -> tuple[str, dict[str, Any]] | None:
        selected = self.select_approval_target(projection_key)
        if selected is None:
            return None
        target, state = selected
        receipt_digest = state.snapshot.commit_receipt_digest
        if not receipt_digest:
            return None
        return (
            f"plan.{action}",
            {
                "session_id": target.session_id,
                "projection_key": target.projection_key,
                "decision_id": target.decision_id,
                "decision_version": target.decision_version,
                "content_digest": target.decision_content_digest,
                "commit_receipt_digest": receipt_digest,
            },
        )

    # 根据 daemon response 设置当前 authoritative mode snapshot
    def apply_authority(self, snapshot: AgentModeSnapshot, *, announce: bool) -> None:
        relation = compare_agent_mode_snapshots(self.mode_snapshot, snapshot)
        if relation in (
            ModeSnapshotRelation.NEWER,
            ModeSnapshotRelation.EQUAL_CONFLICT,
        ):
            self.mode_snapshot = snapshot
            self.printer.apply_mode_snapshot(snapshot, announce=announce)

    # 对一次 mode event 应用单调规则，冲突只设置 deferred refresh 信号
    def handle_mode_event(self, event: dict[str, Any]) -> None:
        try:
            incoming = AgentModeSnapshot(
                agent_mode=event["agent_mode"],
                revision=event.get("revision", 0),
            )
        except (TypeError, ValueError):
            log.warning("invalid agent mode event ignored by chat")
            return
        relation = compare_agent_mode_snapshots(self.mode_snapshot, incoming)
        if relation is ModeSnapshotRelation.NEWER:
            self.mode_snapshot = incoming
            self.printer.apply_mode_snapshot(incoming, announce=True)
        elif relation is ModeSnapshotRelation.EQUAL_CONFLICT:
            self.mode_refresh_required.set()

    # 校验 delivery 身份，成功渲染后才原子推进当前 session cursor
    async def handle(self, delivery: EventDelivery) -> None:
        if self.daemon_instance_id is not None and (
            delivery.daemon_instance_id != self.daemon_instance_id
        ):
            self.daemon_changed.set()
            return
        if delivery.stream_id != self.stream_id:
            return
        if delivery.event_id is None or delivery.event_id not in self.rendered_event_ids:
            if delivery.event.get("type") == "session.agent_mode_changed":
                self.handle_mode_event(delivery.event)
            elif delivery.event.get("type") == "plan.approval_changed":
                self.handle_approval_event(delivery.event)
            else:
                await self.printer.handle(delivery.event)
            if delivery.event_id is not None:
                self.rendered_event_ids.add(delivery.event_id)
        if delivery.event.get("type") == "permission.requested":
            self.permission_arrived.set()
        if delivery.seq is not None:
            self.cursor = max(self.cursor, delivery.seq)


# 清理被放弃 chat view 的 mode snapshot 与 deferred refresh signal
def _reset_chat_mode_view(state: _ChatDeliveryState, printer: ChatPrinter) -> None:
    state.mode_snapshot = None
    state.mode_refresh_required.clear()
    state.reset_approval_view()
    state.approval_client = None
    printer.apply_mode_snapshot(AgentModeSnapshot("direct", 0), announce=False)


@dataclass(frozen=True, slots=True)
class _RpcResult:
    outcome: RpcOutcome
    value: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _OwnedConnection:
    client: SocketClient
    loop_task: asyncio.Task[None] | None = None
    closed: bool = False

    # 幂等取消读循环并关闭 socket，确保同一连接只有一个 cleanup owner
    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.loop_task is not None and not self.loop_task.done():
            self.loop_task.cancel()
        if self.loop_task is not None:
            await asyncio.gather(self.loop_task, return_exceptions=True)
        try:
            await self.client.close()
        except (ConnectionError, OSError):
            log.warning("secondary cleanup failed role=chat_connection_close")


class _DeliveryGate:
    # 初始化 subscribe response 前的 delivery 缓冲门闩
    def __init__(self, consumer: DeliveryConsumer) -> None:
        self._consumer = consumer
        self._pending: list[EventDelivery] = []
        self._ready = False
        self._lock = asyncio.Lock()

    # 身份尚未确认时缓冲 delivery，确认后串行交给真实 handler
    async def handle(self, delivery: EventDelivery) -> None:
        async with self._lock:
            if not self._ready:
                self._pending.append(delivery)
                return
            await self._consumer(delivery)

    # 打开门闩并按到达顺序处理 response 前已缓冲的 delivery
    async def open(self) -> None:
        async with self._lock:
            self._ready = True
            pending = self._pending
            self._pending = []
            for delivery in pending:
                await self._consumer(delivery)

    # 丢弃尚未确认 daemon/session 身份的 delivery
    async def discard(self) -> None:
        async with self._lock:
            self._pending = []


# 判断 CancelledError 是否来自当前调用方而不是 EOF 取消的 RPC future
def _caller_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


# 等待已创建的 cleanup task 终态，并屏蔽 primary 后到达的重复 cancellation
async def _await_cleanup_task(
    cleanup_task: asyncio.Future[Any],
    *,
    role: str,
) -> None:
    current = asyncio.current_task()
    baseline_cancels = current.cancelling() if current is not None else 0
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if current is not None:
                while current.cancelling() > baseline_cancels:
                    current.uncancel()
            continue
        except Exception:
            log.warning("secondary cleanup failed role=%s", role)
            return
    try:
        cleanup_task.result()
    except asyncio.CancelledError:
        log.warning("secondary cleanup cancelled role=%s", role)
    except Exception:
        log.warning("secondary cleanup failed role=%s", role)


# 屏蔽 cleanup task，若此处首次取消则等待终态后原样传播
async def _run_cleanup(
    awaitable: Awaitable[Any],
    *,
    role: str,
) -> None:
    cleanup_task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        await _await_cleanup_task(cleanup_task, role=role)
        raise


# 停止消费 stdin 结果、尽力关闭 daemon 侧会话并关闭当前连接
async def _cleanup_chat_resources(
    owner: _OwnedConnection,
    input_task: asyncio.Task[str] | None,
    session_id: str,
) -> None:
    if input_task is not None and not input_task.done():
        input_task.cancel()
    if input_task is not None:
        await asyncio.gather(input_task, return_exceptions=True)
    if session_id:
        # 尽力而为：让 daemon 取消后台 run 并关闭会话；失败仅依赖连接断开兜底
        try:
            await asyncio.wait_for(
                owner.client.send_command(
                    "session.close", {"session_id": session_id}
                ),
                timeout=_RPC_TIMEOUT_S,
            )
        except (ConnectionError, OSError, TimeoutError, IpcError, RuntimeError):
            log.warning("session.close interrupted during chat cleanup")
    await owner.close()


# 将 RPC 的协议失败与 transport disconnect 分类，caller cancellation 保持原样
async def _send_rpc(
    client: SocketClient,
    method: str,
    params: dict[str, Any],
) -> _RpcResult:
    try:
        value = await asyncio.wait_for(
            client.send_command(method, params),
            timeout=_RPC_TIMEOUT_S,
        )
        return _RpcResult("ok", value)
    except asyncio.CancelledError:
        if _caller_is_cancelling():
            raise
        return _RpcResult("disconnected")
    except (ConnectionError, OSError):
        return _RpcResult("disconnected")
    except TimeoutError:
        return _RpcResult("disconnected")
    except IpcError:
        return _RpcResult("ipc_error")


# 在 delivery callback 之外查询所有 committed projection 的 approval authority
async def _refresh_chat_approvals(
    client: SocketClient,
    state: _ChatDeliveryState,
) -> None:
    if state.approval_refresh_in_flight:
        return
    state.approval_refresh_in_flight = True
    state.approval_refresh_required.clear()
    try:
        for projection_key in tuple(state.approval_targets):
            selected = state.select_approval_target(projection_key)
            if selected is None:
                continue
            _target, approval_state = selected
            captured_daemon_id = state.daemon_instance_id
            captured_target = state.approval_targets.get(projection_key)
            epoch = approval_state.begin_refresh()
            result = await _send_rpc(
                client,
                "plan.get_approval",
                {
                    "session_id": _target.session_id,
                    "projection_key": _target.projection_key,
                },
            )
            if result.outcome == "ok":
                if (
                    state.daemon_instance_id != captured_daemon_id
                    or (
                        state.approval_client is not None
                        and state.approval_client is not client
                    )
                    or state.approval_targets.get(projection_key) is not captured_target
                    or state.approval_states.get(projection_key) is not approval_state
                ):
                    if epoch is not None:
                        approval_state.fail_refresh(epoch=epoch)
                    continue
                applied = state.apply_approval_authority(
                    projection_key,
                    result.value,
                    epoch=epoch,
                )
                if not applied and epoch is not None:
                    approval_state.fail_refresh(epoch=epoch)
            elif epoch is not None:
                approval_state.fail_refresh(epoch=epoch)
    finally:
        state.approval_refresh_in_flight = False


# 等待 socket 断线、daemon 身份异常、权限请求或一行用户输入，断线语义优先
async def _wait_for_chat_activity(
    loop_task: asyncio.Task[None],
    input_task: asyncio.Task[str],
    daemon_changed: asyncio.Event,
    extra_events: Sequence[asyncio.Event] = (),
) -> str:
    daemon_changed_task = asyncio.create_task(daemon_changed.wait())
    extra_tasks = [asyncio.create_task(ev.wait()) for ev in extra_events]
    cancellation_seen = False
    try:
        await asyncio.wait(
            {loop_task, input_task, daemon_changed_task, *extra_tasks},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if daemon_changed.is_set():
            return "daemon_changed"
        if loop_task.done():
            return "disconnected"
        if len(extra_events) > 1 and extra_events[1].is_set():
            return "mode_refresh"
        if len(extra_events) > 2 and extra_events[2].is_set():
            return "approval_refresh"
        for event in extra_events:
            if event.is_set():
                return "permission"
        return "input"
    except asyncio.CancelledError:
        cancellation_seen = True
        if not daemon_changed_task.done():
            daemon_changed_task.cancel()
        for extra_task in extra_tasks:
            if not extra_task.done():
                extra_task.cancel()
        cleanup_task = asyncio.ensure_future(
            asyncio.gather(daemon_changed_task, *extra_tasks, return_exceptions=True)
        )
        await _await_cleanup_task(cleanup_task, role="chat_activity_waiter")
        raise
    finally:
        if not cancellation_seen:
            if not daemon_changed_task.done():
                daemon_changed_task.cancel()
            for extra_task in extra_tasks:
                if not extra_task.done():
                    extra_task.cancel()
            await asyncio.gather(daemon_changed_task, *extra_tasks, return_exceptions=True)


# 建立 owner 的 socket、delivery gate 与读循环
async def _activate_connection(
    owner: _OwnedConnection,
    state: _ChatDeliveryState,
) -> _DeliveryGate:
    await owner.client.connect()
    state.approval_client = owner.client
    state.daemon_changed = asyncio.Event()
    state.permission_arrived = asyncio.Event()
    gate = _DeliveryGate(state.handle)
    owner.client.on_delivery(gate.handle)
    owner.loop_task = asyncio.create_task(owner.client.run_event_loop())
    return gate


# 在 daemon 线程中读取 stdin，避免阻塞 socket event loop；
# daemon 线程保证退出时解释器不会 join 阻塞的 input() 而挂死
def _spawn_stdin_reader(prompt: str, future: asyncio.Future[str]) -> threading.Thread:
    loop = future.get_loop()

    def set_result(value: str) -> None:
        if not future.done():
            future.set_result(value)

    def set_error(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def worker() -> None:
        try:
            value = input(prompt)
        except (EOFError, KeyboardInterrupt):
            loop.call_soon_threadsafe(set_error, EOFError())
        except Exception as exc:
            loop.call_soon_threadsafe(set_error, exc)
        else:
            # 必须经 call_soon_threadsafe（写自管道唤醒 select）：直接
            # set_result 只入就绪队列，事件循环空闲阻塞时输入会被吞掉
            loop.call_soon_threadsafe(set_result, value)

    thread = threading.Thread(target=worker, daemon=True, name="kama-chat-stdin")
    thread.start()
    return thread


async def _readline(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    _spawn_stdin_reader(prompt, future)
    return await future


# 异步核心：以显式 phase 和 RPC outcome 管理 session-scoped chat 的有界重连
async def _chat_async(config: KamaConfig) -> int:
    printer = ChatPrinter()
    state = _ChatDeliveryState(printer, "", None, asyncio.Event(), asyncio.Event())
    owner = _OwnedConnection(SocketClient(config.host, config.port))
    gate: _DeliveryGate | None = None
    input_task: asyncio.Task[str] | None = None
    session_id = ""
    phase: ChatPhase = "create"
    reconnect_attempts = 0
    fresh_view_fallbacks = 0
    discard_pending_input = False
    cleanup_claimed = False

    # 关闭当前 owner 并在总上限内建立下一条完整 delivery 连接
    async def reconnect() -> bool:
        nonlocal owner, gate, reconnect_attempts
        if gate is not None:
            await gate.discard()
        await _run_cleanup(owner.close(), role="chat_reconnect_close")
        while reconnect_attempts < _MAX_RECONNECT_ATTEMPTS:
            reconnect_attempts += 1
            if _RECONNECT_DELAY_S:
                await asyncio.sleep(_RECONNECT_DELAY_S)
            owner = _OwnedConnection(SocketClient(config.host, config.port))
            try:
                gate = await _activate_connection(owner, state)
            except (ConnectionRefusedError, OSError):
                await _run_cleanup(owner.close(), role="chat_failed_connect")
                continue
            return True
        return False

    # 有预算地结束旧视图；只有已经存在的 input task 才标记为 stale
    def begin_fresh_view(
        notice: str,
        *,
        daemon_instance_id: str | None = None,
    ) -> bool:
        nonlocal session_id, phase, fresh_view_fallbacks, discard_pending_input
        if fresh_view_fallbacks >= 1:
            print("error: session unavailable after fallback", file=sys.stderr)
            return False
        fresh_view_fallbacks += 1
        print(notice)
        printer.pending_permission_id = None
        printer.active_run = False
        printer.buffered_decision = None
        printer._plan_reducer = PlanReadyCommitReducer()
        if input_task is not None:
            discard_pending_input = True
        session_id = ""
        state.stream_id = ""
        state.cursor = 0
        state.rendered_event_ids.clear()
        state.daemon_instance_id = daemon_instance_id
        _reset_chat_mode_view(state, printer)
        phase = "create"
        return True

    try:
        try:
            gate = await _activate_connection(owner, state)
        except (ConnectionRefusedError, OSError):
            print(f"error: core not running ({config.host}:{config.port})", file=sys.stderr)
            return 1

        while True:
            assert gate is not None

            if phase == "create":
                created = await _send_rpc(
                    owner.client,
                    "session.create",
                    {
                        "mode": "chat",
                        "workspace_root": str(Path.cwd().resolve()),
                    },
                )
                if created.outcome == "disconnected":
                    print(
                        "error: session creation outcome unknown; not retried",
                        file=sys.stderr,
                    )
                    return 1
                if created.outcome == "ipc_error":
                    print("error: session creation failed", file=sys.stderr)
                    return 1
                session_id = str(created.value.get("session_id", ""))
                if not session_id:
                    print("error: daemon returned an empty session id", file=sys.stderr)
                    return 1
                state.stream_id = f"session:{session_id}"
                print(f"[session: {session_id}]")
                phase = "subscribe"
                continue

            if phase == "subscribe":
                subscribed = await _send_rpc(
                    owner.client,
                    "event.subscribe",
                    {
                        "topics": _CHAT_TOPICS,
                        "scope": state.stream_id,
                        "after_seq": state.cursor,
                    },
                )
                if subscribed.outcome == "disconnected":
                    if state.daemon_instance_id is None:
                        await gate.discard()
                        if not begin_fresh_view(
                            "[session handshake interrupted; old session view ended]"
                        ):
                            return 1
                    if not await reconnect():
                        print("error: reconnect attempts exhausted", file=sys.stderr)
                        return 1
                    continue
                if subscribed.outcome == "ipc_error":
                    await gate.discard()
                    if not begin_fresh_view(
                        "[session unavailable; old session view ended]"
                    ):
                        return 1
                    continue

                resumed_daemon_id = str(
                    subscribed.value.get("daemon_instance_id", "")
                )
                if not resumed_daemon_id:
                    print("error: daemon returned an empty identity", file=sys.stderr)
                    return 1
                if (
                    state.daemon_instance_id is not None
                    and resumed_daemon_id != state.daemon_instance_id
                ):
                    await gate.discard()
                    if not begin_fresh_view(
                        "[daemon restarted; old session view ended]",
                        daemon_instance_id=resumed_daemon_id,
                    ):
                        return 1
                    continue

                state.daemon_instance_id = resumed_daemon_id
                mode_result = await _send_rpc(
                    owner.client,
                    "session.get_agent_mode",
                    {"session_id": session_id},
                )
                if mode_result.outcome != "ok":
                    await gate.discard()
                    if not await reconnect():
                        print("error: reconnect attempts exhausted", file=sys.stderr)
                        return 1
                    continue
                try:
                    state.apply_authority(
                        _parse_mode_snapshot(mode_result.value),
                        announce=False,
                    )
                except (KeyError, TypeError, ValueError):
                    await gate.discard()
                    print("error: invalid mode snapshot", file=sys.stderr)
                    if not await reconnect():
                        return 1
                    continue
                try:
                    await gate.open()
                except Exception:
                    print("error: event delivery failed", file=sys.stderr)
                    return 1
                if state.approval_targets:
                    await _refresh_chat_approvals(owner.client, state)
                phase = "interactive"
                if input_task is None:
                    input_task = asyncio.create_task(_readline("> "))
                continue

            assert owner.loop_task is not None
            assert input_task is not None
            outcome = await _wait_for_chat_activity(
                owner.loop_task,
                input_task,
                state.daemon_changed,
                extra_events=[
                    state.permission_arrived,
                    state.mode_refresh_required,
                    state.approval_refresh_required,
                ],
            )
            if outcome == "mode_refresh":
                state.mode_refresh_required.clear()
                if not state.mode_refresh_in_flight:
                    state.mode_refresh_in_flight = True
                    try:
                        mode_result = await _send_rpc(
                            owner.client,
                            "session.get_agent_mode",
                            {"session_id": session_id},
                        )
                        if mode_result.outcome == "ok":
                            try:
                                state.apply_authority(
                                    _parse_mode_snapshot(mode_result.value),
                                    announce=True,
                                )
                            except (KeyError, TypeError, ValueError):
                                log.warning("invalid mode refresh response")
                    finally:
                        state.mode_refresh_in_flight = False
                continue
            if outcome == "approval_refresh":
                await _refresh_chat_approvals(owner.client, state)
                continue
            if outcome == "permission":
                state.permission_arrived.clear()
                if (
                    printer.buffered_decision is not None
                    and printer.pending_permission_id is not None
                ):
                    queued_decision = printer.buffered_decision
                    printer.buffered_decision = None
                    tool_use_id = printer.pending_permission_id
                    printer.pending_permission_id = None
                    sent = await _send_rpc(
                        owner.client,
                        "permission.respond",
                        {"tool_use_id": tool_use_id, "decision": queued_decision},
                    )
                    if sent.outcome == "ok":
                        continue
                    print("[permission] response interrupted; not retried")
                    if sent.outcome == "ipc_error":
                        continue
                    phase = "subscribe"
                    if not await reconnect():
                        print("error: reconnect attempts exhausted", file=sys.stderr)
                        return 1
                continue
            if outcome != "input":
                if owner.loop_task.done() and not owner.loop_task.cancelled():
                    if owner.loop_task.exception() is not None:
                        print("error: event delivery failed", file=sys.stderr)
                        return 1
                if printer.pending_permission_id is not None:
                    printer.pending_permission_id = None
                    discard_pending_input = True
                    print("[permission] connection lost; pending decision discarded")
                if outcome == "daemon_changed":
                    if not begin_fresh_view(
                        "[daemon restarted; old session view ended]"
                    ):
                        return 1
                else:
                    phase = "subscribe"
                if not await reconnect():
                    print("error: reconnect attempts exhausted", file=sys.stderr)
                    return 1
                continue

            if discard_pending_input:
                try:
                    input_task.result()
                except (EOFError, KeyboardInterrupt):
                    return 0
                discard_pending_input = False
                input_task = asyncio.create_task(_readline("> "))
                continue

            try:
                line = input_task.result()
            except (EOFError, KeyboardInterrupt):
                closed = await _send_rpc(
                    owner.client,
                    "session.close",
                    {"session_id": session_id},
                )
                if closed.outcome != "ok":
                    print("[session close interrupted]")
                return 0

            content = line.strip()
            input_task = asyncio.create_task(_readline("> "))
            if not content:
                continue

            if printer.pending_permission_id:
                decision = _DECISION_MAP.get(content.lower())
                if decision is None:
                    print(
                        "  enter y (allow once), a (always allow), "
                        "n (deny once), d (always deny)"
                    )
                    continue
                tool_use_id = printer.pending_permission_id
                printer.pending_permission_id = None
                sent = await _send_rpc(
                    owner.client,
                    "permission.respond",
                    {"tool_use_id": tool_use_id, "decision": decision},
                )
                if sent.outcome == "ok":
                    continue
                print("[permission] response interrupted; not retried")
                if sent.outcome == "ipc_error":
                    continue
                phase = "subscribe"
                if not await reconnect():
                    print("error: reconnect attempts exhausted", file=sys.stderr)
                    return 1
                continue

            if content in ("/plan", "/direct"):
                requested_mode = "plan" if content == "/plan" else "direct"
                mode_result = await _send_rpc(
                    owner.client,
                    "session.set_agent_mode",
                    {"session_id": session_id, "agent_mode": requested_mode},
                )
                if mode_result.outcome == "ok":
                    try:
                        state.apply_authority(
                            _parse_mode_snapshot(mode_result.value),
                            announce=True,
                        )
                    except (KeyError, TypeError, ValueError):
                        print("[mode change failed]", file=sys.stderr)
                    continue
                print("[mode change failed]", file=sys.stderr)
                if mode_result.outcome == "disconnected":
                    phase = "subscribe"
                    if not await reconnect():
                        print("error: reconnect attempts exhausted", file=sys.stderr)
                        return 1
                continue

            if content == "/mode":
                mode_result = await _send_rpc(
                    owner.client,
                    "session.get_agent_mode",
                    {"session_id": session_id},
                )
                if mode_result.outcome == "ok":
                    try:
                        state.apply_authority(
                            _parse_mode_snapshot(mode_result.value),
                            announce=True,
                        )
                    except (KeyError, TypeError, ValueError):
                        print("[mode query failed]", file=sys.stderr)
                    continue
                print("[mode query failed]", file=sys.stderr)
                continue

            if content == "/approve" or content.startswith("/approve "):
                action: Literal["approve", "reject"] | None = "approve"
            elif content == "/reject" or content.startswith("/reject "):
                action = "reject"
            else:
                action = None
            if action is not None:
                parts = content.split(maxsplit=1)
                projection_key = parts[1].strip() if len(parts) == 2 else None
                selected = state.select_approval_target(projection_key)
                if selected is None:
                    print("[approval] exact projection key required or unavailable")
                    continue
                _target, approval_state = selected
                if not approval_state.snapshot.commit_receipt_digest:
                    await _refresh_chat_approvals(owner.client, state)
                command = state.approval_command_params(projection_key, action)
                if command is None:
                    print("[approval] authority unavailable; retry after PlanReady")
                    continue
                method, params = command
                resolved = await _send_rpc(owner.client, method, params)
                if resolved.outcome != "ok":
                    print("[approval] request failed", file=sys.stderr)
                    continue
                state.merge_approval_response(resolved.value)
                continue

            if printer.active_run:
                decision = _DECISION_MAP.get(content.lower())
                if decision is not None:
                    printer.buffered_decision = decision
                    print(
                        f"  [{decision} queued; applied to the next permission prompt]"
                    )
                    continue
                print(
                    "[run in progress] only y/a/n/d can be queued while "
                    "the agent is running",
                    file=sys.stderr,
                )
                continue

            sent = await _send_rpc(
                owner.client,
                "session.send_message",
                {"session_id": session_id, "content": content},
            )
            if sent.outcome == "ok":
                printer.active_run = True
                continue
            print("[message] delivery interrupted; not retried")
            if sent.outcome == "ipc_error":
                continue
            phase = "subscribe"
            if not await reconnect():
                print("error: reconnect attempts exhausted", file=sys.stderr)
                return 1
    except asyncio.CancelledError:
        cleanup_claimed = True
        cleanup_task = asyncio.create_task(
            _cleanup_chat_resources(owner, input_task, session_id)
        )
        await _await_cleanup_task(cleanup_task, role="chat_cancelled")
        raise
    except Exception:
        print("error: chat client failed", file=sys.stderr)
        return 1
    finally:
        if not cleanup_claimed:
            cleanup_claimed = True
            await _run_cleanup(
                _cleanup_chat_resources(owner, input_task, session_id),
                role="chat_final",
            )


# 执行 kama chat 命令
def cmd_chat(config: KamaConfig) -> None:
    try:
        exit_code = asyncio.run(_chat_async(config))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)
