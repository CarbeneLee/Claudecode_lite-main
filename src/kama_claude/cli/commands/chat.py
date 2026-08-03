from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kama_claude.core.config import KamaConfig
from kama_claude.core.transport.socket_client import (
    EventDelivery,
    IpcError,
    SocketClient,
)

_CHAT_TOPICS = ["session.*", "run.*", "tool.*", "llm.token", "permission.*"]
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
        self.pending_permission_id: str | None = None
        # run 激活期间输入 y/a/n/d 会被缓冲，等待下一条权限请求自动生效
        self.active_run = False
        self.buffered_decision: str | None = None

    # 若当前 LLM token 尚未换行，则补一个换行
    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

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
            self.active_run = False
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


@dataclass(slots=True)
class _ChatDeliveryState:
    printer: ChatPrinter
    stream_id: str
    daemon_instance_id: str | None
    daemon_changed: asyncio.Event
    permission_arrived: asyncio.Event
    cursor: int = 0

    # 校验 delivery 身份，成功渲染后才原子推进当前 session cursor
    async def handle(self, delivery: EventDelivery) -> None:
        if self.daemon_instance_id is not None and (
            delivery.daemon_instance_id != self.daemon_instance_id
        ):
            self.daemon_changed.set()
            return
        if delivery.stream_id != self.stream_id:
            return
        await self.printer.handle(delivery.event)
        if delivery.event.get("type") == "permission.requested":
            self.permission_arrived.set()
        if delivery.seq is not None:
            self.cursor = max(self.cursor, delivery.seq)


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


# 停止消费 stdin 结果并关闭当前连接；无法强制终止已阻塞的 executor input 线程
async def _cleanup_chat_resources(
    owner: _OwnedConnection,
    input_task: asyncio.Task[str] | None,
) -> None:
    if input_task is not None and not input_task.done():
        input_task.cancel()
    if input_task is not None:
        await asyncio.gather(input_task, return_exceptions=True)
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
    state.daemon_changed = asyncio.Event()
    state.permission_arrived = asyncio.Event()
    gate = _DeliveryGate(state.handle)
    owner.client.on_delivery(gate.handle)
    owner.loop_task = asyncio.create_task(owner.client.run_event_loop())
    return gate


# 在线程池中读取 stdin，避免阻塞 socket event loop
async def _readline(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


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
        if input_task is not None:
            discard_pending_input = True
        session_id = ""
        state.stream_id = ""
        state.cursor = 0
        state.daemon_instance_id = daemon_instance_id
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
                try:
                    await gate.open()
                except Exception:
                    print("error: event delivery failed", file=sys.stderr)
                    return 1
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
                extra_events=[state.permission_arrived],
            )
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
        cleanup_task = asyncio.create_task(_cleanup_chat_resources(owner, input_task))
        await _await_cleanup_task(cleanup_task, role="chat_cancelled")
        raise
    except Exception:
        print("error: chat client failed", file=sys.stderr)
        return 1
    finally:
        if not cleanup_claimed:
            cleanup_claimed = True
            await _run_cleanup(
                _cleanup_chat_resources(owner, input_task),
                role="chat_final",
            )


# 执行 kama chat 命令
def cmd_chat(config: KamaConfig) -> None:
    try:
        exit_code = asyncio.run(_chat_async(config))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)
