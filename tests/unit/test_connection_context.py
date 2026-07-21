from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from importlib import import_module
from typing import cast

import pytest
from pydantic import BaseModel

from kama_claude.core.transport import connection as connection_module
from kama_claude.core.transport.connection import (
    MAX_CONTROL_FRAMES,
    MAX_EVENT_FRAMES,
    MAX_OUTBOUND_FRAME_BYTES,
    ConnectionContext,
    ConnectionState,
)


class _Frame(BaseModel):
    value: str


class _RecordingWriter:
    # 初始化记录缓冲区和可选 drain gate
    def __init__(
        self,
        drain_gate: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
        drain_error: Exception | None = None,
    ) -> None:
        self.frames: list[bytes] = []
        self.write_tasks: list[asyncio.Task[object] | None] = []
        self.drain_gate = drain_gate
        self.close_gate = close_gate
        self.drain_error = drain_error
        self.close_calls = 0
        self.wait_closed_calls = 0
        self.wait_closed_started = asyncio.Event()

    # 记录写入帧以及执行写入的唯一 asyncio task
    def write(self, data: bytes) -> None:
        self.frames.append(data)
        self.write_tasks.append(asyncio.current_task())

    # 在可选 Event gate 上阻塞 drain，支持确定性背压测试
    async def drain(self) -> None:
        if self.drain_error is not None:
            raise self.drain_error
        if self.drain_gate is not None:
            await self.drain_gate.wait()

    # 记录连接关闭调用
    def close(self) -> None:
        self.close_calls += 1

    # 记录等待连接关闭调用
    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1
        self.wait_closed_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()

    # 返回稳定 peername 供安全日志与 trace 使用
    def get_extra_info(self, name: str, default: object = None) -> object:
        if name == "peername":
            return ("127.0.0.1", 7000)
        return default


# 将记录型 writer 转为 ConnectionContext 所需的 StreamWriter 协议对象
def _writer(value: _RecordingWriter) -> asyncio.StreamWriter:
    return cast(asyncio.StreamWriter, value)


# 将已写 NDJSON 帧还原为测试 value 顺序
def _values(writer: _RecordingWriter) -> list[str]:
    return [json.loads(frame)["value"] for frame in writer.frames]


# 构造编码后 value bytes 恰好填满单帧余量且同时包含中文和 emoji 的字符串
def _exact_multibyte_value() -> str:
    empty_size = len(b'{"value":""}\n')
    target_value_bytes = MAX_OUTBOUND_FRAME_BYTES - empty_size
    unit = "中🧪"
    unit_bytes = len(unit.encode("utf-8"))
    return unit * (target_value_bytes // unit_bytes) + "a" * (
        target_value_bytes % unit_bytes
    )


# 功能：验证 Phase 7A 提供独立的 connection lifecycle 模块
# 设计：先用模块发现断言制造稳定 RED，避免缺失模块导致 pytest collection error
def test_connection_context_module_exists() -> None:
    spec = importlib.util.find_spec("kama_claude.core.transport.connection")

    assert spec is not None


# 功能：验证 connection 模块暴露状态、回执和上下文三个生命周期构件
# 设计：使用运行时属性检查制造清晰 RED，避免尚未定义符号造成 collection error
def test_connection_context_exposes_lifecycle_api() -> None:
    module = import_module("kama_claude.core.transport.connection")

    assert getattr(module, "ConnectionState", None) is not None
    assert getattr(module, "FrameReceipt", None) is not None
    assert getattr(module, "ConnectionContext", None) is not None


# 功能：验证 control 与 event 并发排队后只有一个 writer task 执行所有 socket write
# 设计：记录 asyncio.current_task identity 并等待真实 written receipts，直接锁定单 writer 生命周期边界
async def test_single_writer_task_serializes_control_and_event_frames() -> None:
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-single")
    context.start()

    receipts = [
        context.enqueue_control(_Frame(value="control-1")),
        context.enqueue_event(_Frame(value="event-1")),
        context.enqueue_control(_Frame(value="control-2")),
    ]
    await asyncio.gather(*(receipt.written for receipt in receipts))
    await context.close("test complete")

    assert len(set(writer.write_tasks)) == 1
    assert _values(writer) == ["control-1", "control-2", "event-1"]
    assert context.state is ConnectionState.CLOSED


# 功能：验证连续八个 control 后必须发送一个已经等待的 event
# 设计：在 writer 启动前确定性预填两类队列，避免依赖调度 sleep 来观察 weighted priority
async def test_control_burst_yields_to_waiting_event_after_eight_frames() -> None:
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-weighted")
    receipts = [
        context.enqueue_control(_Frame(value=f"control-{index}"))
        for index in range(10)
    ]
    event_receipt = context.enqueue_event(_Frame(value="event"))

    context.start()
    await asyncio.gather(
        *(receipt.written for receipt in receipts),
        event_receipt.written,
    )
    await context.close("test complete")

    assert _values(writer) == [
        *(f"control-{index}" for index in range(8)),
        "event",
        "control-8",
        "control-9",
    ]


# 功能：验证 connection close 会取消并等待全部 connection-owned request tasks
# 设计：用 Event 永久阻塞真实 task，再以 close 驱动取消，断言 task terminal 与 writer reap 均完成
async def test_close_cancels_and_joins_owned_request_tasks() -> None:
    blocker = asyncio.Event()
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-owned")
    context.start()
    request_task = context.create_request_task(blocker.wait())

    await context.close("peer EOF")

    assert request_task.cancelled()
    assert context.state is ConnectionState.CLOSED
    assert writer.close_calls == 1
    assert writer.wait_closed_calls == 1


# 功能：验证 event queue 达到 frame cap 时仍保留独立 control capacity
# 设计：writer 启动前填满 event queue，再加入 ping-like control，证明两类计数不会互相借用
async def test_full_event_queue_does_not_consume_control_capacity() -> None:
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-reserved")
    event_receipts = [
        context.enqueue_event(_Frame(value=f"event-{index}"))
        for index in range(MAX_EVENT_FRAMES)
    ]
    control_receipt = context.enqueue_control(_Frame(value="control"))

    context.start()
    await asyncio.gather(
        control_receipt.written,
        *(receipt.written for receipt in event_receipts),
    )
    await context.close("test complete")

    assert _values(writer)[0] == "control"


# 功能：验证 control frame-count cap 在 limit 接受并在 limit+1 关闭连接
# 设计：writer 启动前填入 128 个小帧消除消费竞态，再断言第 129 个稳定 overflow
async def test_control_frame_limit_uses_limit_plus_one_sentinel() -> None:
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-control-cap")
    receipts = [
        context.enqueue_control(_Frame(value=f"control-{index}"))
        for index in range(MAX_CONTROL_FRAMES)
    ]

    with pytest.raises(ConnectionError, match="control queue overflow"):
        context.enqueue_control(_Frame(value="overflow"))
    await context.close("overflow")
    await asyncio.gather(
        *(receipt.written for receipt in receipts),
        return_exceptions=True,
    )

    assert context.state is ConnectionState.CLOSED


# 功能：验证 control 与 event 队列都按已编码 JSONL bytes 执行精确容量边界
# 设计：同一组中文和 emoji 帧先在 exact cap 下成功，再把 cap 降一字节触发 overflow，隔离 byte check 与 frame check
@pytest.mark.parametrize("role", ["control", "event"])
async def test_queue_byte_limit_accepts_exact_and_rejects_one_byte_over(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    frame = _Frame(value="中🧪")
    payload_size = len('{"value":"中🧪"}\n'.encode())
    cap_name = "MAX_CONTROL_BYTES" if role == "control" else "MAX_EVENT_BYTES"
    enqueue_name = "enqueue_control" if role == "control" else "enqueue_event"

    monkeypatch.setattr(connection_module, cap_name, payload_size * 2)
    exact_writer = _RecordingWriter()
    exact = ConnectionContext(_writer(exact_writer), connection_id=f"{role}-exact")
    first = getattr(exact, enqueue_name)(frame)
    second = getattr(exact, enqueue_name)(frame)
    exact.start()
    await asyncio.gather(first.written, second.written)
    await exact.close("exact queue bytes")

    monkeypatch.setattr(connection_module, cap_name, payload_size * 2 - 1)
    overflow_writer = _RecordingWriter()
    overflow = ConnectionContext(
        _writer(overflow_writer),
        connection_id=f"{role}-overflow",
    )
    accepted = getattr(overflow, enqueue_name)(frame)
    with pytest.raises(ConnectionError, match=f"{role} queue overflow"):
        getattr(overflow, enqueue_name)(frame)
    await overflow.close("queue byte overflow")
    await asyncio.gather(accepted.written, return_exceptions=True)

    assert exact.state is ConnectionState.CLOSED
    assert overflow.state is ConnectionState.CLOSED


# 功能：验证单帧上限包含最终 UTF-8 JSONL newline 且使用 byte count
# 设计：用中文和 emoji 构造恰好 1 MiB 的 frame，再加一个 ASCII byte，杀死 character-count mutation
async def test_single_frame_limit_counts_final_utf8_jsonl_bytes() -> None:
    exact_value = _exact_multibyte_value()
    writer = _RecordingWriter()
    context = ConnectionContext(_writer(writer), connection_id="conn-frame-exact")
    exact_receipt = context.enqueue_control(_Frame(value=exact_value))
    context.start()

    await exact_receipt.written
    await context.close("exact complete")
    assert len(writer.frames[0]) == MAX_OUTBOUND_FRAME_BYTES

    overflow_writer = _RecordingWriter()
    overflow = ConnectionContext(
        _writer(overflow_writer),
        connection_id="conn-frame-overflow",
    )
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        overflow.enqueue_control(_Frame(value=exact_value + "a"))
    await overflow.close("oversize")
    assert overflow_writer.frames == []


# 功能：验证 overflow 同步关闭 admission gate 并由统一 finalizer 恰好清理一次
# 设计：不让出 event loop 就检查 CLOSING 与订阅拒绝，再 await terminal，锁定 close trigger 的原子可见性
async def test_overflow_synchronously_closes_admission_and_runs_finalizer() -> None:
    finalized: list[ConnectionContext] = []
    writer = _RecordingWriter()
    context = ConnectionContext(
        _writer(writer),
        connection_id="conn-overflow-finalizer",
        on_close=finalized.append,
    )

    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        context.enqueue_event(_Frame(value=_exact_multibyte_value() + "a"))

    assert context.state is ConnectionState.CLOSING
    with pytest.raises(ConnectionError, match="not open"):
        context.ensure_open_for_subscription()
    await context.close("oversize")

    assert finalized == [context]
    assert context.state is ConnectionState.CLOSED


# 功能：验证 drain timeout 使当前 receipt 失败并只关闭所属 connection
# 设计：用永不释放的 Event gate 和临时极短 timeout，避免真实 sleep 或网络背压
async def test_drain_timeout_fails_receipt_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connection_module, "WRITER_DRAIN_TIMEOUT_S", 0.01)
    writer = _RecordingWriter(drain_gate=asyncio.Event())
    context = ConnectionContext(_writer(writer), connection_id="conn-timeout")
    context.start()
    receipt = context.enqueue_control(_Frame(value="blocked"))

    with pytest.raises(ConnectionError, match="connection delivery failed"):
        await receipt.written
    await context.close("writer timeout")

    assert context.state is ConnectionState.CLOSED
    assert writer.close_calls == 1


# 功能：验证 writer 技术异常中的不可信文本不会进入连接交付日志
# 设计：让 fake drain 抛出含 secret 的异常并检查 caplog，直接锁定 traceback 泄漏而不依赖真实 socket
async def test_writer_failure_log_redacts_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "delivery-secret-must-not-leak"
    writer = _RecordingWriter(drain_error=RuntimeError(secret))
    context = ConnectionContext(_writer(writer), connection_id="conn-redaction")
    context.start()
    receipt = context.enqueue_control(_Frame(value="safe"))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ConnectionError, match="connection delivery failed"):
            await receipt.written
        await context.close("writer failure")

    assert "connection_id=conn-redaction" in caplog.text
    assert "role=writer" in caplog.text
    assert secret not in caplog.text


# 功能：验证 close 等待期间收到 cancellation 仍先完成 CLOSED 再恢复取消
# 设计：用 wait_closed gate 精确注入外层取消，释放后断言 cleanup terminal 与 CancelledError 都保留
async def test_close_cancellation_waits_for_terminal_cleanup() -> None:
    close_gate = asyncio.Event()
    writer = _RecordingWriter(close_gate=close_gate)
    context = ConnectionContext(_writer(writer), connection_id="conn-cancel-close")
    context.start()
    closing = asyncio.create_task(context.close("peer EOF"))
    await writer.wait_closed_started.wait()

    closing.cancel()
    close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert context.state is ConnectionState.CLOSED
    with pytest.raises(ConnectionError, match="not open"):
        context.ensure_open_for_subscription()
