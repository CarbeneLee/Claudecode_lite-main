from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kama_claude.core.config import KamaConfig
from kama_claude.core.transport.socket_client import (
    EventDelivery,
    IpcError,
    SocketClient,
)

_RUN_TOPICS = ["run.*", "step.*", "tool.*", "llm.token", "llm.usage"]
_MAX_RECONNECT_ATTEMPTS = 3
_RECONNECT_DELAY_S = 0.1
_DAEMON_RESTART_ERROR = (
    "error: daemon restarted; active run continuation is not guaranteed; "
    "historical state unavailable after restart"
)

log = logging.getLogger(__name__)

type DeliveryConsumer = Callable[[EventDelivery], Awaitable[None]]


class StdoutPrinter:
    # 接收 dict 格式的事件并将运行进度格式化打印到终端
    def __init__(self) -> None:
        self._inline = False  # True while LLM tokens are mid-line
        self._run_start: float = 0.0

    # 若当前行有未换行的 token，补一个换行符
    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

    # 根据事件 type 字段分发并格式化打印到 stdout/stderr
    async def handle(self, event: dict[str, Any]) -> None:
        t = event.get("type", "")

        if t == "run.started":
            self._run_start = time.monotonic()
            print(f"[run] {event.get('run_id', '')}")

        elif t == "step.started":
            self._ensure_newline()
            print(f"[step {event.get('step')}] planning...")

        elif t == "llm.token":
            print(event.get("token", ""), end="", flush=True)
            self._inline = True

        elif t == "tool.call_started":
            self._ensure_newline()
            params_str = json.dumps(event.get("params", {}), ensure_ascii=False)
            print(f"[tool] {event.get('tool_name', '')} {params_str}")

        elif t == "tool.call_finished":
            print(f"[tool] {event.get('tool_name', '')} ✓  {event.get('elapsed_ms')}ms")

        elif t == "tool.call_failed":
            print(
                f"[tool] {event.get('tool_name', '')} ✗  {event.get('error_message', '')}",
                file=sys.stderr,
            )

        elif t == "step.finished":
            self._ensure_newline()
            print(f"[step {event.get('step')}] done")

        elif t == "run.finished":
            self._ensure_newline()
            elapsed = time.monotonic() - self._run_start
            print(f"[run] {event.get('status', '')}  {event.get('steps')} steps  {elapsed:.1f}s")


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


# 取消读循环并关闭当前 IPC 连接
async def _close_connection(
    client: SocketClient,
    loop_task: asyncio.Task[None] | None,
) -> None:
    if loop_task is not None and not loop_task.done():
        loop_task.cancel()
    if loop_task is not None:
        await asyncio.gather(loop_task, return_exceptions=True)
    try:
        await client.close()
    except (ConnectionError, OSError):
        log.warning("secondary cleanup failed role=run_connection_close")


# 在首次 cancellation 已保存时屏蔽重复 cancellation 直到 secondary cleanup 终态
async def _await_secondary_cleanup(
    awaitable: Awaitable[Any],
    *,
    role: str,
) -> None:
    cleanup_task = asyncio.ensure_future(awaitable)
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


# 清理连接且不允许重复 cancellation 覆盖已捕获的首次 cancellation
async def _cleanup_cancelled_connection(
    client: SocketClient,
    loop_task: asyncio.Task[None] | None,
) -> None:
    await _await_secondary_cleanup(
        _close_connection(client, loop_task),
        role="run_connection",
    )


# 等待目标 run 结束、daemon 身份异常或连接读循环终止
async def _wait_for_run_outcome(
    loop_task: asyncio.Task[None],
    finished: asyncio.Event,
    daemon_changed: asyncio.Event,
) -> str:
    finished_task = asyncio.create_task(finished.wait())
    daemon_changed_task = asyncio.create_task(daemon_changed.wait())
    try:
        await asyncio.wait(
            {loop_task, finished_task, daemon_changed_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if finished.is_set():
            return "finished"
        if daemon_changed.is_set():
            return "daemon_changed"
        return "disconnected"
    except asyncio.CancelledError:
        for task in (finished_task, daemon_changed_task):
            if not task.done():
                task.cancel()
        await _await_secondary_cleanup(
            asyncio.gather(
                finished_task,
                daemon_changed_task,
                return_exceptions=True,
            ),
            role="run_outcome_waiters",
        )
        raise
    finally:
        for task in (finished_task, daemon_changed_task):
            if not task.done():
                task.cancel()
        if any(not task.done() for task in (finished_task, daemon_changed_task)):
            await asyncio.gather(
                finished_task,
                daemon_changed_task,
                return_exceptions=True,
            )


# 判断 CancelledError 是否来自当前调用方，而不是断线时被取消的 RPC future
def _caller_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


# 异步核心：先启动 run，再按 run stream 游标订阅并有界重连
async def _run_async(goal: str, config: KamaConfig) -> int:
    printer = StdoutPrinter()
    finished = asyncio.Event()
    daemon_changed = asyncio.Event()
    exit_code = 0
    run_id = ""
    stream_id = ""
    cursor = 0
    daemon_instance_id: str | None = None
    client = SocketClient(config.host, config.port)
    loop_task: asyncio.Task[None] | None = None

    # 处理完整 delivery，成功渲染后才推进当前 run stream 的 cursor
    async def on_delivery(delivery: EventDelivery) -> None:
        nonlocal cursor, exit_code
        if daemon_instance_id is not None and (
            delivery.daemon_instance_id != daemon_instance_id
        ):
            daemon_changed.set()
            return
        if delivery.stream_id != stream_id:
            return
        event = delivery.event
        if event.get("run_id") == run_id:
            await printer.handle(event)
        if delivery.seq is not None:
            cursor = max(cursor, delivery.seq)
        if event.get("type") == "run.finished" and event.get("run_id") == run_id:
            if event.get("status") != "success":
                exit_code = 1
            finished.set()

    try:
        try:
            await client.connect()
        except (ConnectionRefusedError, OSError):
            await _close_connection(client, None)
            print(f"error: core not running ({config.host}:{config.port})", file=sys.stderr)
            return 1

        gate = _DeliveryGate(on_delivery)
        client.on_delivery(gate.handle)
        loop_task = asyncio.create_task(client.run_event_loop())
        try:
            started = await client.send_command(
                "agent.run",
                {
                    "goal": goal,
                    "workspace_root": str(Path.cwd().resolve()),
                },
            )
        except asyncio.CancelledError:
            if _caller_is_cancelling():
                raise
            print("error: connection lost before run id was received", file=sys.stderr)
            await _close_connection(client, loop_task)
            return 1
        except IpcError:
            print("error: IPC command failed", file=sys.stderr)
            await _close_connection(client, loop_task)
            return 1
        except (ConnectionError, OSError):
            print("error: connection lost before run id was received", file=sys.stderr)
            await _close_connection(client, loop_task)
            return 1

        run_id = str(started.get("run_id", ""))
        if not run_id:
            print("error: daemon returned an empty run id", file=sys.stderr)
            await _close_connection(client, loop_task)
            return 1
        stream_id = f"run:{run_id}"
        reconnect_attempts = 0

        while True:
            subscribe_succeeded = False
            try:
                subscribed = await client.send_command(
                    "event.subscribe",
                    {
                        "topics": _RUN_TOPICS,
                        "scope": stream_id,
                        "after_seq": cursor,
                    },
                )
                subscribe_succeeded = True
            except asyncio.CancelledError:
                if _caller_is_cancelling():
                    raise
            except IpcError:
                print("error: unable to resume run stream", file=sys.stderr)
                await _close_connection(client, loop_task)
                return 1
            except (ConnectionError, OSError):
                subscribe_succeeded = False

            if subscribe_succeeded:
                current_daemon_id = str(subscribed.get("daemon_instance_id", ""))
                if not current_daemon_id:
                    print("error: subscribe response omitted daemon identity", file=sys.stderr)
                    await _close_connection(client, loop_task)
                    return 1
                if daemon_instance_id is None:
                    daemon_instance_id = current_daemon_id
                elif current_daemon_id != daemon_instance_id:
                    print(_DAEMON_RESTART_ERROR, file=sys.stderr)
                    await _close_connection(client, loop_task)
                    return 1

                try:
                    await gate.open()
                except Exception:
                    print("error: event delivery failed", file=sys.stderr)
                    await _close_connection(client, loop_task)
                    return 1

                outcome = await _wait_for_run_outcome(
                    loop_task,
                    finished,
                    daemon_changed,
                )
                if outcome == "finished":
                    await _close_connection(client, loop_task)
                    return exit_code
                if outcome == "daemon_changed":
                    print(_DAEMON_RESTART_ERROR, file=sys.stderr)
                    await _close_connection(client, loop_task)
                    return 1
                if not loop_task.cancelled():
                    loop_error = loop_task.exception()
                    if loop_error is not None:
                        print("error: event delivery failed", file=sys.stderr)
                        await _close_connection(client, loop_task)
                        return 1

            await _close_connection(client, loop_task)
            if reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
                print("error: reconnect attempts exhausted", file=sys.stderr)
                return 1

            while True:
                reconnect_attempts += 1
                if _RECONNECT_DELAY_S > 0:
                    await asyncio.sleep(_RECONNECT_DELAY_S)
                client = SocketClient(config.host, config.port)
                loop_task = None
                gate = _DeliveryGate(on_delivery)
                client.on_delivery(gate.handle)
                try:
                    await client.connect()
                except (ConnectionRefusedError, OSError):
                    await client.close()
                    if reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
                        print("error: reconnect attempts exhausted", file=sys.stderr)
                        return 1
                    continue
                loop_task = asyncio.create_task(client.run_event_loop())
                break
    except asyncio.CancelledError:
        await _cleanup_cancelled_connection(client, loop_task)
        raise


# 执行 kama run --goal "..." 命令
def cmd_run(goal: str, config: KamaConfig) -> None:
    try:
        exit_code = asyncio.run(_run_async(goal, config))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)
