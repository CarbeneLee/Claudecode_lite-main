from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from kama_claude.core.bus.events import (
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.search_code import SearchCodeParams
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry
from tests.unit._search_code_test_support import _call, _collect_events, _tool


async def _wait_thread_event(event: threading.Event) -> None:
    await asyncio.to_thread(event.wait)


# 让出两个 event-loop tick 以确保已调度的取消被协程观察
async def _event_loop_barrier() -> None:
    loop = asyncio.get_running_loop()
    for _ in range(2):
        reached = loop.create_future()
        loop.call_soon(reached.set_result, None)
        await reached


# 功能：验证单次取消会等待合作式 worker 终态后再原样传播
# 设计：真实 asyncio.Task 与 threading.Event 建立 entered/terminal 顺序，并检查中央事件只有 started
async def test_single_cancellation_waits_for_worker_terminal_without_terminal_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool(tmp_path)
    entered = threading.Event()
    terminal = threading.Event()
    cancellation_seen: list[asyncio.CancelledError] = []

    # 阻塞到 stop_event 被取消路径设置，随后记录 worker 终态
    def _cooperative_search(
        params: SearchCodeParams,
        stop_event: threading.Event,
    ) -> ToolResult:
        entered.set()
        stop_event.wait()
        terminal.set()
        return ToolResult(content="must not escape cancellation")

    monkeypatch.setattr(tool, "_search_sync", _cooperative_search)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events = _collect_events(bus)

    # 捕获 invoke_tool 向调用者重新抛出的取消对象
    async def _invoke_and_observe() -> ToolResult:
        try:
            return await invoke_tool(
                registry,
                _call({"query": "needle"}),
                bus,
                run_id="run-cancel",
            )
        except asyncio.CancelledError as exc:
            assert terminal.is_set()
            cancellation_seen.append(exc)
            raise

    task = asyncio.create_task(_invoke_and_observe())
    await _wait_thread_event(entered)
    task.cancel("outer-cancellation")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert terminal.is_set()
    assert cancellation_seen == [exc_info.value]
    assert [type(event) for event in events] == [ToolCallStartedEvent]


# 功能：验证 cleanup 期间重复取消不会遗弃 worker 或覆盖首个取消对象
# 设计：两次 task.cancel 之间用 stop-seen/release Event 控制 worker，证明 terminal 前 task 仍未完成
async def test_repeated_cancellation_keeps_waiting_and_preserves_first_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool(tmp_path)
    entered = threading.Event()
    stop_seen = threading.Event()
    release = threading.Event()
    terminal = threading.Event()
    cancellation_seen: list[asyncio.CancelledError] = []

    # 取消后停在可控 release gate，便于对 cleanup 发送第二次取消
    def _gated_search(
        params: SearchCodeParams,
        stop_event: threading.Event,
    ) -> ToolResult:
        entered.set()
        stop_event.wait()
        stop_seen.set()
        release.wait()
        terminal.set()
        return ToolResult(content="worker terminal")

    monkeypatch.setattr(tool, "_search_sync", _gated_search)

    # 直接观察工具取消边界重新抛出的对象
    async def _invoke_and_observe() -> ToolResult:
        try:
            return await tool.invoke({"query": "needle"})
        except asyncio.CancelledError as exc:
            assert terminal.is_set()
            cancellation_seen.append(exc)
            raise

    task = asyncio.create_task(_invoke_and_observe())
    await _wait_thread_event(entered)
    task.cancel("first-cancellation")
    await _wait_thread_event(stop_seen)
    assert not task.done()

    task.cancel("second-cancellation")
    await _event_loop_barrier()
    assert not task.done()
    assert not terminal.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert terminal.is_set()
    assert cancellation_seen == [exc_info.value]
    assert exc_info.value.args == ("first-cancellation",)
