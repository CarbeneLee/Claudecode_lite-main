from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_CONTROL_FRAMES = 128
MAX_CONTROL_BYTES = 1024 * 1024
MAX_EVENT_FRAMES = 512
MAX_EVENT_BYTES = 4 * 1024 * 1024
MAX_OUTBOUND_FRAME_BYTES = 1024 * 1024
CONTROL_BURST_MAX = 8
WRITER_DRAIN_TIMEOUT_S = 5.0


class ConnectionState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class FrameReceipt:
    enqueued: asyncio.Future[None]
    written: asyncio.Future[None]


@dataclass(frozen=True)
class _OutboundFrame:
    payload: bytes
    receipt: FrameReceipt
    on_written: Callable[[], None] | None


class ConnectionContext:
    # 初始化单连接 writer、独立队列与 owned task 注册表
    def __init__(
        self,
        writer: asyncio.StreamWriter,
        *,
        connection_id: str,
        on_close: Callable[[ConnectionContext], None] | None = None,
    ) -> None:
        self._writer = writer
        self._connection_id = connection_id
        self._on_close = on_close
        self._state = ConnectionState.OPEN
        self._control: deque[_OutboundFrame] = deque()
        self._events: deque[_OutboundFrame] = deque()
        self._control_bytes = 0
        self._event_bytes = 0
        self._frames_ready = asyncio.Event()
        self._writer_task: asyncio.Task[None] | None = None
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._control_burst = 0

    @property
    # 返回当前连接状态
    def state(self) -> ConnectionState:
        return self._state

    @property
    # 返回不含业务参数的连接标识
    def connection_id(self) -> str:
        return self._connection_id

    @property
    # 返回连接 peername 供安全 trace 元数据使用
    def peername(self) -> object:
        return self._writer.get_extra_info("peername", "<unknown>")

    # 启动并复用该连接唯一的 socket writer task
    def start(self) -> asyncio.Task[None]:
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(
                self._writer_loop(),
                name=f"connection-writer:{self._connection_id}",
            )
        return self._writer_task

    # 将 JSON-RPC control frame 放入独立有界队列
    def enqueue_control(
        self,
        frame: BaseModel,
        *,
        on_written: Callable[[], None] | None = None,
    ) -> FrameReceipt:
        return self._enqueue(frame, is_control=True, on_written=on_written)

    # 将 replay 或 live event frame 放入独立有界队列
    def enqueue_event(
        self,
        frame: BaseModel,
        *,
        on_written: Callable[[], None] | None = None,
    ) -> FrameReceipt:
        return self._enqueue(frame, is_control=False, on_written=on_written)

    # 创建并登记 connection-owned request task
    def create_request_task(
        self,
        coroutine: Coroutine[Any, Any, Any],
    ) -> asyncio.Task[Any]:
        if self._state is not ConnectionState.OPEN:
            coroutine.close()
            raise ConnectionError("connection is not open")
        task = asyncio.create_task(coroutine)
        self._request_tasks.add(task)
        task.add_done_callback(self._request_tasks.discard)
        return task

    # 在建立新订阅前确认连接仍处于 OPEN
    def ensure_open_for_subscription(self) -> None:
        if self._state is not ConnectionState.OPEN:
            raise ConnectionError("connection is not open")

    # 幂等触发后台连接清理并返回唯一 close task
    def initiate_close(self, reason: str) -> asyncio.Task[None]:
        if self._close_task is None:
            self._state = ConnectionState.CLOSING
            self._close_task = asyncio.create_task(
                self._close_impl(reason),
                name=f"connection-close:{self._connection_id}",
            )
        return self._close_task

    # 等待幂等清理完成，外层取消时仍先让连接到达 CLOSED
    async def close(self, reason: str) -> None:
        close_task = self.initiate_close(reason)
        primary_cancel: asyncio.CancelledError | None = None
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as exc:
                if primary_cancel is None:
                    primary_cancel = exc
        await close_task
        if primary_cancel is not None:
            raise primary_cancel

    # 对单帧执行有界 UTF-8 JSONL 编码
    @staticmethod
    def _encode_frame(frame: BaseModel) -> bytes:
        encoded = bytearray()
        for chunk in ConnectionContext._iter_json(frame.model_dump(mode="json")):
            chunk_bytes = chunk.encode("utf-8")
            if len(encoded) + len(chunk_bytes) + 1 > MAX_OUTBOUND_FRAME_BYTES:
                raise ValueError("outbound frame exceeds 1 MiB")
            encoded.extend(chunk_bytes)
        encoded.extend(b"\n")
        return bytes(encoded)

    # 递归生成小块 compact JSON，避免先物化无界序列化结果
    @staticmethod
    def _iter_json(value: Any) -> Iterator[str]:
        if value is None:
            yield "null"
            return
        if value is True:
            yield "true"
            return
        if value is False:
            yield "false"
            return
        if isinstance(value, str):
            yield from ConnectionContext._iter_json_string(value)
            return
        if isinstance(value, int):
            yield str(value)
            return
        if isinstance(value, float):
            yield "null" if not math.isfinite(value) else repr(value)
            return
        if isinstance(value, list):
            yield "["
            for index, item in enumerate(value):
                if index:
                    yield ","
                yield from ConnectionContext._iter_json(item)
            yield "]"
            return
        if isinstance(value, dict):
            yield "{"
            for index, (key, item) in enumerate(value.items()):
                if index:
                    yield ","
                yield from ConnectionContext._iter_json_string(str(key))
                yield ":"
                yield from ConnectionContext._iter_json(item)
            yield "}"
            return
        raise TypeError(f"unsupported outbound JSON type: {type(value).__name__}")

    # 分块转义 JSON 字符串，限制单次临时字符串大小
    @staticmethod
    def _iter_json_string(value: str) -> Iterator[str]:
        escapes = {
            '"': '\\"',
            "\\": "\\\\",
            "\b": "\\b",
            "\f": "\\f",
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
        }
        yield '"'
        plain: list[str] = []
        for character in value:
            escaped = escapes.get(character)
            if escaped is None and ord(character) < 0x20:
                escaped = f"\\u{ord(character):04x}"
            if escaped is None:
                plain.append(character)
                if len(plain) == 4096:
                    yield "".join(plain)
                    plain.clear()
                continue
            if plain:
                yield "".join(plain)
                plain.clear()
            yield escaped
        if plain:
            yield "".join(plain)
        yield '"'

    # 原子校验队列容量并返回双阶段 frame receipt
    def _enqueue(
        self,
        frame: BaseModel,
        *,
        is_control: bool,
        on_written: Callable[[], None] | None,
    ) -> FrameReceipt:
        if self._state is not ConnectionState.OPEN:
            raise ConnectionError("connection is not open")
        try:
            payload = self._encode_frame(frame)
        except ValueError:
            self.initiate_close("outbound frame too large")
            raise

        queue = self._control if is_control else self._events
        used_bytes = self._control_bytes if is_control else self._event_bytes
        max_frames = MAX_CONTROL_FRAMES if is_control else MAX_EVENT_FRAMES
        max_bytes = MAX_CONTROL_BYTES if is_control else MAX_EVENT_BYTES
        if len(queue) >= max_frames or used_bytes + len(payload) > max_bytes:
            role = "control" if is_control else "event"
            self.initiate_close(f"{role} queue overflow")
            raise ConnectionError(f"{role} queue overflow")

        loop = asyncio.get_running_loop()
        receipt = FrameReceipt(
            enqueued=loop.create_future(),
            written=loop.create_future(),
        )
        queue.append(
            _OutboundFrame(
                payload=payload,
                receipt=receipt,
                on_written=on_written,
            )
        )
        if is_control:
            self._control_bytes += len(payload)
        else:
            self._event_bytes += len(payload)
        receipt.enqueued.set_result(None)
        self._frames_ready.set()
        return receipt

    # 按 control burst 权重从两个队列选择下一帧
    async def _next_frame(self) -> _OutboundFrame:
        while True:
            if self._control and (
                self._control_burst < CONTROL_BURST_MAX or not self._events
            ):
                frame = self._control.popleft()
                self._control_bytes -= len(frame.payload)
                self._control_burst += 1
                return frame
            if self._events:
                frame = self._events.popleft()
                self._event_bytes -= len(frame.payload)
                self._control_burst = 0
                return frame
            self._frames_ready.clear()
            if not self._control and not self._events:
                await self._frames_ready.wait()

    # 作为唯一 socket writer 顺序写入并在 drain 后完成 receipt
    async def _writer_loop(self) -> None:
        current: _OutboundFrame | None = None
        try:
            while True:
                current = await self._next_frame()
                self._writer.write(current.payload)
                await asyncio.wait_for(
                    self._writer.drain(),
                    timeout=WRITER_DRAIN_TIMEOUT_S,
                )
                if current.on_written is not None:
                    current.on_written()
                if not current.receipt.written.done():
                    current.receipt.written.set_result(None)
                current = None
        except asyncio.CancelledError:
            if current is not None:
                self._fail_receipt(current.receipt)
            raise
        except Exception:
            if current is not None:
                self._fail_receipt(current.receipt)
            logger.warning(
                "connection delivery failed connection_id=%s role=writer",
                self._connection_id,
            )
            self.initiate_close("writer failure")

    # 以安全稳定异常完成未写出的 frame receipt
    @staticmethod
    def _fail_receipt(receipt: FrameReceipt) -> None:
        if not receipt.written.done():
            receipt.written.set_exception(ConnectionError("connection delivery failed"))

    # 失败并清空两个 outbound queue
    def _fail_pending_frames(self) -> None:
        for queue in (self._control, self._events):
            while queue:
                self._fail_receipt(queue.popleft().receipt)
        self._control_bytes = 0
        self._event_bytes = 0

    # 执行一次 request cancel/join、writer reap 与 socket close
    async def _close_impl(self, reason: str) -> None:
        if self._state is ConnectionState.CLOSED:
            return
        self._state = ConnectionState.CLOSING
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:
                logger.warning(
                    "connection finalizer failed connection_id=%s role=finalizer",
                    self._connection_id,
                )
        request_tasks = list(self._request_tasks)
        for task in request_tasks:
            task.cancel()
        if request_tasks:
            await asyncio.gather(*request_tasks, return_exceptions=True)

        writer_task = self._writer_task
        if writer_task is not None and not writer_task.done():
            writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)
        self._fail_pending_frames()

        try:
            self._writer.close()
        except Exception:
            logger.warning(
                "connection close failed connection_id=%s role=close",
                self._connection_id,
            )
        try:
            await self._writer.wait_closed()
        except Exception:
            logger.warning(
                "connection reap failed connection_id=%s role=wait_closed",
                self._connection_id,
            )
        self._state = ConnectionState.CLOSED
        logger.debug(
            "connection closed connection_id=%s reason=%s",
            self._connection_id,
            reason,
        )
