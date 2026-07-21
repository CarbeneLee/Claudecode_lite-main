from __future__ import annotations

import asyncio
import gc
import json
from typing import cast

import pytest

from kama_claude.core.bus.events import RunStartedEvent, StepStartedEvent
from kama_claude.core.transport.connection import (
    MAX_EVENT_FRAMES,
    ConnectionContext,
    ConnectionState,
)
from kama_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster


class _RecordingWriter:
    # 初始化写入记录、drain gate 与同步事件
    def __init__(
        self,
        drain_gate: asyncio.Event | None = None,
        drain_error: Exception | None = None,
    ) -> None:
        self.frames: list[bytes] = []
        self.drain_gate = drain_gate
        self.drain_error = drain_error
        self.drain_started = asyncio.Event()
        self.drained = asyncio.Event()

    # 记录由 ConnectionContext writer task 发送的完整帧
    def write(self, data: bytes) -> None:
        self.frames.append(data)

    # 在可选 gate 上制造确定性慢客户端
    async def drain(self) -> None:
        self.drain_started.set()
        if self.drain_error is not None:
            raise self.drain_error
        if self.drain_gate is not None:
            await self.drain_gate.wait()
        self.drained.set()

    # 提供 ConnectionContext cleanup 所需的同步关闭接口
    def close(self) -> None:
        return None

    # 提供 ConnectionContext cleanup 所需的异步 reap 接口
    async def wait_closed(self) -> None:
        return None

    # 返回稳定 peername 供 trace callback 使用
    def get_extra_info(self, name: str, default: object = None) -> object:
        if name == "peername":
            return ("127.0.0.1", 7001)
        return default


# 创建带唯一 writer task 的真实 ConnectionContext 测试夹具
def _make_context(
    connection_id: str,
    *,
    drain_gate: asyncio.Event | None = None,
    drain_error: Exception | None = None,
    start: bool = True,
) -> tuple[ConnectionContext, _RecordingWriter]:
    writer = _RecordingWriter(drain_gate, drain_error)
    context = ConnectionContext(
        cast(asyncio.StreamWriter, writer),
        connection_id=connection_id,
    )
    if start:
        context.start()
    return context, writer


# 构造稳定的 run.started 测试事件
def _run_started(run_id: str = "r1") -> RunStartedEvent:
    return RunStartedEvent(run_id=run_id, goal="test", ts="2026-01-01T00:00:00Z")


# 解析 writer 收到的最后一条 EventPushEnvelope
def _last_envelope(writer: _RecordingWriter) -> dict[str, object]:
    return cast(dict[str, object], json.loads(writer.frames[-1]))


# 功能：验证 broadcaster 只向 ConnectionContext event queue 入队而不直接写 socket
# 设计：使用真实 context/writer task，等待 drain 后解析 envelope，锁定 broadcaster 与 I/O 的职责分离
async def test_subscriber_receives_matching_event_through_connection_queue() -> None:
    broadcaster = IpcEventBroadcaster()
    context, writer = _make_context("conn-match")
    broadcaster.subscribe(context, topics=["run.*"])

    await broadcaster.handle(_run_started())
    await writer.drained.wait()
    await context.close("test complete")

    data = _last_envelope(writer)
    assert data["kind"] == "event"
    assert cast(dict[str, object], data["event"])["type"] == "run.started"


# 功能：验证无订阅时 handle 不向任何 connection queue 写入数据
# 设计：创建未订阅的真实 context，发布后断言 writer 无帧并正常 cleanup
async def test_no_subscription_no_write() -> None:
    broadcaster = IpcEventBroadcaster()
    context, writer = _make_context("conn-empty")

    await broadcaster.handle(_run_started())
    await context.close("test complete")

    assert writer.frames == []


# 功能：验证 topic glob 匹配 step.started 并过滤 run.started
# 设计：同一 context 连续发布两种事件，最终只保留一条 step envelope
async def test_topic_glob_matches_step_not_run() -> None:
    broadcaster = IpcEventBroadcaster()
    context, writer = _make_context("conn-topic")
    broadcaster.subscribe(context, topics=["step.*"])

    await broadcaster.handle(
        StepStartedEvent(run_id="r1", step=1, ts="2026-01-01T00:00:00Z")
    )
    await broadcaster.handle(_run_started())
    await writer.drained.wait()
    await context.close("test complete")

    assert len(writer.frames) == 1
    assert cast(dict[str, object], _last_envelope(writer)["event"])["type"] == "step.started"


# 功能：验证 global scope 接收不同 run_id 且 run scope 只接收目标 run
# 设计：两个真实 context 共享 broadcaster，分别断言全局两帧与局部一帧
async def test_scope_filters_preserve_existing_semantics() -> None:
    broadcaster = IpcEventBroadcaster()
    global_context, global_writer = _make_context("conn-global")
    run_context, run_writer = _make_context("conn-run")
    broadcaster.subscribe(global_context, topics=["run.*"], scope="global")
    broadcaster.subscribe(run_context, topics=["run.*"], scope="run:r1")

    await broadcaster.handle(_run_started("r1"))
    await broadcaster.handle(_run_started("r2"))
    while len(global_writer.frames) < 2 or len(run_writer.frames) < 1:
        await asyncio.sleep(0)
    await global_context.close("test complete")
    await run_context.close("test complete")

    assert len(global_writer.frames) == 2
    assert len(run_writer.frames) == 1


# 功能：验证慢客户端 A 不阻塞 EventBus handler 或客户端 B 的 drain
# 设计：A 的 drain 由 Event gate 锁住，B 无 gate；handle 必须先返回且 B 完成，再释放 A
async def test_slow_client_does_not_block_publish_or_other_client() -> None:
    broadcaster = IpcEventBroadcaster()
    slow_gate = asyncio.Event()
    slow_context, slow_writer = _make_context("conn-slow", drain_gate=slow_gate)
    fast_context, fast_writer = _make_context("conn-fast")
    broadcaster.subscribe(slow_context, topics=["run.*"])
    broadcaster.subscribe(fast_context, topics=["run.*"])
    publish_done = asyncio.Event()

    # 发布结束后设置独立 Event，避免依赖 task.done 的调度瞬间
    async def publish() -> None:
        await broadcaster.handle(_run_started())
        publish_done.set()

    publish_task = asyncio.create_task(publish())
    try:
        await slow_writer.drain_started.wait()
        await asyncio.wait_for(asyncio.shield(publish_done.wait()), timeout=0.5)
        await asyncio.wait_for(fast_writer.drained.wait(), timeout=0.5)
        assert not slow_writer.drained.is_set()
    finally:
        slow_gate.set()
        await publish_task
        await slow_context.close("test complete")
        await fast_context.close("test complete")


# 功能：验证被丢弃的 live FrameReceipt 失败不会产生未观察 Future 异常
# 设计：安装 loop exception handler 并让真实 context writer drain 失败，GC 后检查无 Future exception warning
async def test_failed_live_delivery_receipt_exception_is_observed() -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop_issues: list[dict[str, object]] = []

    # 捕获 asyncio 对未观察 Future exception 的诊断
    def capture_loop_issue(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, object],
    ) -> None:
        loop_issues.append(context)

    # 在内层协程退出后释放 context/task/frame 引用，使未观察 Future 诊断确定发生
    async def exercise_failed_delivery() -> None:
        broadcaster = IpcEventBroadcaster()
        context, _writer = _make_context(
            "conn-live-receipt",
            drain_error=RuntimeError("live-delivery-secret"),
        )
        broadcaster.subscribe(context, topics=["run.*"])
        await broadcaster.handle(_run_started())
        while context.state is not ConnectionState.CLOSED:
            await asyncio.sleep(0)
        broadcaster.unsubscribe_all(context)
        await context.close("writer failure")

    loop.set_exception_handler(capture_loop_issue)
    try:
        await exercise_failed_delivery()
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not any(
        issue.get("message") == "Future exception was never retrieved"
        for issue in loop_issues
    )


# 功能：验证 A event queue overflow 只清理 A，B 仍收到同一事件
# 设计：不启动 A writer 并预填 512 帧，触发第 513 帧；B 使用正常 writer 证明隔离
async def test_event_overflow_closes_only_affected_connection() -> None:
    broadcaster = IpcEventBroadcaster()
    overflow_context, _overflow_writer = _make_context("conn-overflow", start=False)
    fast_context, fast_writer = _make_context("conn-survivor")
    queued = [
        overflow_context.enqueue_event(_run_started(f"queued-{index}"))
        for index in range(MAX_EVENT_FRAMES)
    ]
    broadcaster.subscribe(overflow_context, topics=["run.*"])
    broadcaster.subscribe(fast_context, topics=["run.*"])

    await broadcaster.handle(_run_started("live"))
    await fast_writer.drained.wait()
    await overflow_context.close("overflow")
    await asyncio.gather(
        *(receipt.written for receipt in queued),
        return_exceptions=True,
    )
    assert fast_context.state is ConnectionState.OPEN
    await fast_context.close("test complete")

    assert overflow_context.state is ConnectionState.CLOSED
    assert cast(dict[str, object], _last_envelope(fast_writer)["event"])["run_id"] == "live"


# 功能：验证 subscription_id 只能由所属 connection 删除且重复删除稳定返回 false
# 设计：A/B 各有真实订阅，A 删除 B 的 ID 不泄漏存在性，再验证自身 true/false 序列
async def test_unsubscribe_enforces_connection_ownership() -> None:
    broadcaster = IpcEventBroadcaster()
    context_a, _writer_a = _make_context("conn-owner-a")
    context_b, _writer_b = _make_context("conn-owner-b")
    sub_a = broadcaster.subscribe(context_a, topics=["run.*"])
    sub_b = broadcaster.subscribe(context_b, topics=["run.*"])

    assert broadcaster.unsubscribe(context_a, sub_b) is False
    assert broadcaster.unsubscribe(context_a, sub_a) is True
    assert broadcaster.unsubscribe(context_a, sub_a) is False
    assert broadcaster.unsubscribe(context_b, sub_b) is True
    await context_a.close("test complete")
    await context_b.close("test complete")


# 功能：验证 closing/closed connection 不能创建新的 zombie subscription
# 设计：先完成真实 context close，再调用 subscribe 并断言稳定拒绝
async def test_subscribe_rejects_closed_connection() -> None:
    broadcaster = IpcEventBroadcaster()
    context, _writer = _make_context("conn-closed")
    await context.close("peer EOF")

    with pytest.raises(ConnectionError, match="not open"):
        broadcaster.subscribe(context, topics=["run.*"])
