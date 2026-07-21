from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from kama_claude.core.bus.events import Event
from kama_claude.core.events.writer import EventWriter

logger = logging.getLogger(__name__)

JOURNAL_QUEUE_MAX_FRAMES = 4096
JOURNAL_QUEUE_MAX_BYTES = 16 * 1024 * 1024
JOURNAL_MAX_SINGLE_FRAME_BYTES = 1024 * 1024
JOURNAL_BATCH_MAX_FRAMES = 256
JOURNAL_BATCH_MAX_BYTES = 256 * 1024
JOURNAL_BATCH_MAX_DELAY_MS = 5
JOURNAL_SHUTDOWN_TIMEOUT_S = 10.0

MAX_REPLAY_EVENTS = 10_000
MAX_REPLAY_BYTES = 16 * 1024 * 1024
MAX_REPLAY_LINE_BYTES = 1024 * 1024
REPLAY_READ_CHUNK_BYTES = 64 * 1024

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


class _V2JournalRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[2]
    event_id: str = Field(min_length=1)
    stream_id: str
    seq: int = Field(ge=1)
    event: dict[str, Any]


class JournalError(RuntimeError):
    pass


class DuplicateStreamOwnerError(JournalError):
    pass


class JournalCorruptionError(JournalError):
    pass


class JournalCapacityError(JournalError):
    pass


class UnknownStreamError(JournalError):
    pass


class StreamLifecycle(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class StreamOwnerToken:
    stream_id: str
    value: str


@dataclass(frozen=True)
class JournalRecord:
    event_id: str
    stream_id: str
    seq: int
    event: dict[str, Any]
    serialized: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class ReplayBatch:
    records: tuple[JournalRecord, ...]
    examined_bytes: int


@dataclass(frozen=True)
class JournalPublishOutcome:
    event_id: str
    streams: tuple[str, ...]


@dataclass
class _QueuedRecord:
    record: JournalRecord
    durable: asyncio.Future[None]
    terminal: bool


@dataclass
class _StreamState:
    stream_id: str
    path: Path
    legacy_path: Path | None
    owner: StreamOwnerToken
    next_seq: int
    durable_seq: int
    lifecycle: StreamLifecycle = StreamLifecycle.OPEN
    queue: deque[_QueuedRecord] = field(default_factory=deque)
    reserved_frames: int = 0
    queue_bytes: int = 0
    peak_queue_frames: int = 0
    peak_queue_bytes: int = 0
    pending: set[asyncio.Future[None]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task[None] | None = None


class RunSessionIdentityRegistry:
    # 初始化不可变 run→session 映射和已注册 session 集合
    def __init__(self) -> None:
        self._run_sessions: dict[str, str | None] = {}
        self._sessions: set[str] = set()

    # 在 session 第一个事件前登记其 durable identity
    def register_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            raise DuplicateStreamOwnerError(f"duplicate stream owner session:{session_id}")
        self._sessions.add(session_id)

    # 在 run 第一个事件前登记不可变的可选 parent session
    def register_run(self, run_id: str, session_id: str | None) -> None:
        if run_id in self._run_sessions:
            raise DuplicateStreamOwnerError(f"duplicate stream owner run:{run_id}")
        if session_id is not None and session_id not in self._sessions:
            raise UnknownStreamError(f"session:{session_id} is not registered")
        self._run_sessions[run_id] = session_id

    # 返回 run 的 parent session；未知 run 与 direct run 分开表达
    def lookup_run(self, run_id: str) -> tuple[bool, str | None]:
        if run_id not in self._run_sessions:
            return False, None
        return True, self._run_sessions[run_id]


class EventJournalCoordinator:
    # 初始化 per-stream journal owner、identity registry 与 durable 回调
    def __init__(
        self,
        *,
        on_durable: Callable[[JournalRecord], Awaitable[None]] | None = None,
        on_live_only: Callable[[BaseModel], Awaitable[None]] | None = None,
        on_stream_failure: Callable[[str], None] | None = None,
    ) -> None:
        self._streams: dict[str, _StreamState] = {}
        self._identities = RunSessionIdentityRegistry()
        self._on_durable = on_durable
        self._on_live_only = on_live_only
        self._on_stream_failure = on_stream_failure
        self._accepting = True

    @property
    # 暴露 identity registry 供 lifecycle 注册测试与只读查询
    def identities(self) -> RunSessionIdentityRegistry:
        return self._identities

    # 注册 session v2 journal 的唯一 writer owner
    async def register_session(
        self,
        session_id: str,
        session_path: Path,
    ) -> StreamOwnerToken:
        self._identities.register_session(session_id)
        try:
            return await self._register_stream(
                f"session:{session_id}",
                session_path / "events.v2.jsonl",
                legacy_path=None,
            )
        except asyncio.CancelledError:
            self._identities._sessions.discard(session_id)
            raise
        except Exception:
            self._identities._sessions.discard(session_id)
            raise

    # 注册 run v2 journal、legacy prefix 与不可变 run→session mapping
    async def register_run(
        self,
        run_id: str,
        run_path: Path,
        *,
        session_id: str | None,
    ) -> StreamOwnerToken:
        self._identities.register_run(run_id, session_id)
        try:
            return await self._register_stream(
                f"run:{run_id}",
                run_path / "events.v2.jsonl",
                legacy_path=run_path / "events.jsonl",
            )
        except asyncio.CancelledError:
            self._identities._run_sessions.pop(run_id, None)
            raise
        except Exception:
            self._identities._run_sessions.pop(run_id, None)
            raise

    # 注册 stream 前验证 legacy/v2 前缀并启动唯一 worker
    async def _register_stream(
        self,
        stream_id: str,
        path: Path,
        *,
        legacy_path: Path | None,
    ) -> StreamOwnerToken:
        if not self._accepting:
            raise JournalError("journal coordinator is closing")
        if stream_id in self._streams:
            raise DuplicateStreamOwnerError(f"duplicate stream owner {stream_id}")
        last_seq = await _validate_stream_preserving_cancellation(
            self._validate_existing_stream,
            stream_id,
            path,
            legacy_path,
        )
        owner = StreamOwnerToken(stream_id=stream_id, value=uuid.uuid4().hex)
        state = _StreamState(
            stream_id=stream_id,
            path=path,
            legacy_path=legacy_path,
            owner=owner,
            next_seq=last_seq + 1,
            durable_seq=last_seq,
        )
        self._streams[stream_id] = state
        state.worker = asyncio.create_task(
            self._worker_loop(state),
            name=f"event-journal:{stream_id}",
        )
        return owner

    @staticmethod
    # 验证已有 legacy/v2 文件并返回最后一个完整 durable seq
    def _validate_existing_stream(
        stream_id: str,
        path: Path,
        legacy_path: Path | None,
    ) -> int:
        legacy_records, legacy_bytes = _read_legacy_records(
            stream_id,
            legacy_path,
            stop_event=None,
            max_bytes=MAX_REPLAY_BYTES,
        )
        v2_records, _ = _read_v2_records(
            stream_id,
            path,
            expected_first_seq=len(legacy_records) + 1,
            stop_event=None,
            max_bytes=MAX_REPLAY_BYTES - legacy_bytes,
        )
        _reject_cross_format_duplicate_identity(legacy_records, v2_records)
        _normalize_v2_tail(path)
        if v2_records:
            return v2_records[-1].seq
        return len(legacy_records)

    # 将一个 domain event 原子预留到全部目标 stream queue
    async def handle(self, event: BaseModel) -> None:
        if not self._accepting:
            raise JournalError("journal coordinator is closing")
        targets, terminal_targets = self._route_event(event)
        event_id = f"evt-{uuid.uuid4().hex}"
        if not targets:
            if self._on_live_only is not None:
                await self._on_live_only(event)
            return

        states = [self._streams[target] for target in sorted(targets)]
        for state in states:
            await state.lock.acquire()
        queued: list[_QueuedRecord] = []
        limiting_state: _StreamState | None = None
        try:
            event_data = event.model_dump(mode="json")
            for state in states:
                try:
                    if state.lifecycle is not StreamLifecycle.OPEN:
                        raise JournalError(
                            f"stream {state.stream_id} is {state.lifecycle.value}"
                        )
                    record = _build_v2_record(
                        event_id=event_id,
                        stream_id=state.stream_id,
                        seq=state.next_seq,
                        event=event_data,
                    )
                    if len(record.serialized) > JOURNAL_MAX_SINGLE_FRAME_BYTES:
                        raise JournalCapacityError("journal frame exceeds 1 MiB")
                    if (
                        state.reserved_frames >= JOURNAL_QUEUE_MAX_FRAMES
                        or state.queue_bytes + len(record.serialized)
                        > JOURNAL_QUEUE_MAX_BYTES
                    ):
                        raise JournalCapacityError(
                            f"journal queue capacity exceeded {state.stream_id}"
                        )
                except JournalCapacityError:
                    limiting_state = state
                    raise
                receipt = asyncio.get_running_loop().create_future()
                queued.append(
                    _QueuedRecord(
                        record=record,
                        durable=receipt,
                        terminal=state.stream_id in terminal_targets,
                    )
                )

            for state, item in zip(states, queued, strict=True):
                state.queue.append(item)
                state.reserved_frames += 1
                state.queue_bytes += len(item.record.serialized)
                state.peak_queue_frames = max(
                    state.peak_queue_frames,
                    state.reserved_frames,
                )
                state.peak_queue_bytes = max(
                    state.peak_queue_bytes,
                    state.queue_bytes,
                )
                state.next_seq += 1
                state.pending.add(item.durable)
                item.durable.add_done_callback(state.pending.discard)
                item.durable.add_done_callback(self._observe_receipt)
                if item.terminal:
                    state.lifecycle = StreamLifecycle.CLOSING
                state.ready.set()
        except JournalCapacityError:
            assert limiting_state is not None
            limiting_state.lifecycle = StreamLifecycle.DEGRADED
            limiting_state.ready.set()
            self._notify_stream_failure(limiting_state.stream_id)
            raise
        finally:
            for state in reversed(states):
                state.lock.release()

        terminal_receipts = [item.durable for item in queued if item.terminal]
        if terminal_receipts:
            await _await_preserving_cancellation(terminal_receipts)

    # 依据冻结 routing table 计算 durable targets 与 terminal stream
    def _route_event(self, event: BaseModel) -> tuple[set[str], set[str]]:
        data = event.model_dump(mode="json")
        event_type = str(data.get("type", ""))
        targets: set[str] = set()
        terminals: set[str] = set()

        if event_type == "core.started":
            return targets, terminals
        if event_type.startswith("session."):
            session_id = str(data.get("session_id", ""))
            stream_id = f"session:{session_id}"
            if stream_id not in self._streams:
                raise UnknownStreamError(f"unknown durable stream {stream_id}")
            targets.add(stream_id)
            if event_type == "session.closed":
                terminals.add(stream_id)
            return targets, terminals

        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return targets, terminals

        run_ids = [run_id]
        if event_type.startswith("subagent."):
            parent_run_id = data.get("parent_run_id")
            if isinstance(parent_run_id, str) and parent_run_id:
                run_ids.append(parent_run_id)

        mapped_sessions: set[str] = set()
        for target_run_id in run_ids:
            stream_id = f"run:{target_run_id}"
            if stream_id in self._streams:
                targets.add(stream_id)
            else:
                raise UnknownStreamError(f"unknown durable stream {stream_id}")
            known, mapped_session_id = self._identities.lookup_run(target_run_id)
            if known and mapped_session_id is not None:
                mapped_sessions.add(mapped_session_id)

        direct_session = data.get("session_id")
        if isinstance(direct_session, str) and direct_session:
            if mapped_sessions and mapped_sessions != {direct_session}:
                raise JournalError("run/session identity disagreement")
            mapped_sessions.add(direct_session)
        if len(mapped_sessions) > 1:
            raise JournalError("parent/child session identity disagreement")
        for session_id in mapped_sessions:
            stream_id = f"session:{session_id}"
            if stream_id not in self._streams:
                raise UnknownStreamError(f"unknown durable stream {stream_id}")
            targets.add(stream_id)

        if event_type == "run.finished":
            terminals.add(f"run:{run_id}")
        elif event_type == "subagent.finished":
            terminals.add(f"run:{run_id}")
        return targets, terminals

    # 顺序批量追加并在 durable 后触发 live delivery callback
    async def _worker_loop(self, state: _StreamState) -> None:
        try:
            while True:
                await state.ready.wait()
                state.ready.clear()
                if not state.queue:
                    if state.lifecycle in {
                        StreamLifecycle.CLOSING,
                        StreamLifecycle.CLOSED,
                        StreamLifecycle.DEGRADED,
                    }:
                        break
                    continue

                await asyncio.sleep(JOURNAL_BATCH_MAX_DELAY_MS / 1000)
                batch: list[_QueuedRecord] = []
                batch_bytes = 0
                async with state.lock:
                    while state.queue and len(batch) < JOURNAL_BATCH_MAX_FRAMES:
                        candidate = state.queue[0]
                        size = len(candidate.record.serialized)
                        if batch and batch_bytes + size > JOURNAL_BATCH_MAX_BYTES:
                            break
                        batch.append(state.queue.popleft())
                        batch_bytes += size
                    if state.queue:
                        state.ready.set()

                await _append_and_flush_preserving_cancellation(
                    state.stream_id,
                    state.path,
                    tuple(item.record.serialized for item in batch),
                )
                async with state.lock:
                    for item in batch:
                        state.reserved_frames -= 1
                        state.queue_bytes -= len(item.record.serialized)
                        state.durable_seq = item.record.seq
                        if not item.durable.done():
                            item.durable.set_result(None)
                for item in batch:
                    if self._on_durable is not None:
                        try:
                            await self._on_durable(item.record)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.warning(
                                "durable live delivery failed stream_id=%s role=live",
                                state.stream_id,
                            )
                if state.lifecycle in {
                    StreamLifecycle.CLOSING,
                    StreamLifecycle.DEGRADED,
                } and not state.queue:
                    break
            if state.lifecycle is not StreamLifecycle.DEGRADED:
                state.lifecycle = StreamLifecycle.CLOSED
        except asyncio.CancelledError:
            self._fail_stream(state, JournalError("journal worker cancelled"))
            raise
        except Exception:
            logger.error(
                "journal worker failed stream_id=%s role=append",
                state.stream_id,
            )
            self._fail_stream(state, JournalError("journal append failed"))

    # 将 stream 标记 degraded 并失败所有尚未 durable 的 receipt
    def _fail_stream(self, state: _StreamState, error: Exception) -> None:
        state.lifecycle = StreamLifecycle.DEGRADED
        while state.queue:
            item = state.queue.popleft()
            if not item.durable.done():
                item.durable.set_exception(error)
        state.queue_bytes = 0
        state.reserved_frames = 0
        for receipt in list(state.pending):
            if not receipt.done():
                receipt.set_exception(error)
        self._notify_stream_failure(state.stream_id)

    # 通知 transport 隔离该 degraded stream 的 active/catching-up subscriptions
    def _notify_stream_failure(self, stream_id: str) -> None:
        if self._on_stream_failure is None:
            return
        try:
            self._on_stream_failure(stream_id)
        except Exception:
            logger.error(
                "stream failure cleanup failed stream_id=%s role=secondary",
                stream_id,
            )

    @staticmethod
    # 观察非 terminal durable receipt 的失败，避免 fire-and-forget Future warning
    def _observe_receipt(receipt: asyncio.Future[None]) -> None:
        if receipt.cancelled():
            return
        receipt.exception()

    # 返回 stream 当前已 flush 的 high watermark
    def high_watermark(self, stream_id: str) -> int:
        return self._require_stream(stream_id).durable_seq

    # 返回 stream 当前 lifecycle 供 shutdown、subscription 和审计使用
    def stream_lifecycle(self, stream_id: str) -> StreamLifecycle:
        return self._require_stream(stream_id).lifecycle

    # 返回不含 payload/path 的 stream queue 审计计数
    def stream_metrics(self, stream_id: str) -> dict[str, int]:
        state = self._require_stream(stream_id)
        return {
            "reserved_frames": state.reserved_frames,
            "reserved_bytes": state.queue_bytes,
            "peak_frames": state.peak_queue_frames,
            "peak_bytes": state.peak_queue_bytes,
            "durable_seq": state.durable_seq,
        }

    # 返回 durable stream 是否已由唯一 owner 注册
    def has_stream(self, stream_id: str) -> bool:
        return stream_id in self._streams

    # 在 stream lock 内捕获 durable watermark 并同步登记 catch-up observer
    async def capture_high_watermark(
        self,
        stream_id: str,
        registrar: Callable[[int], None],
    ) -> int:
        state = self._require_stream(stream_id)
        async with state.lock:
            if state.lifecycle is StreamLifecycle.DEGRADED:
                raise JournalError(f"stream {stream_id} is degraded")
            high_watermark = state.durable_seq
            registrar(high_watermark)
            return high_watermark

    # 在 thread 中完整读取并验证指定 replay 区间
    async def read_replay(
        self,
        stream_id: str,
        *,
        after_seq: int,
        high_watermark: int,
        stop_event: threading.Event | None = None,
    ) -> ReplayBatch:
        state = self._require_stream(stream_id)
        stop = stop_event or threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                _read_replay_batch,
                state.stream_id,
                state.path,
                state.legacy_path,
                after_seq,
                high_watermark,
                stop,
            ),
            name=f"event-replay-reader:{stream_id}",
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            stop.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    logger.error(
                        "replay reader cleanup failed stream_id=%s role=secondary",
                        stream_id,
                    )
            if not worker.cancelled():
                try:
                    worker.result()
                except Exception:
                    logger.error(
                        "replay reader failed during cancellation stream_id=%s role=secondary",
                        stream_id,
                    )
            raise

    # 等待所有当前已入队 record 达到 durable 或失败终态
    async def flush_all(self) -> None:
        receipts = [
            receipt
            for state in self._streams.values()
            for receipt in tuple(state.pending)
        ]
        if receipts:
            outcomes = await asyncio.gather(*receipts, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    raise outcome

    # 停止新 publication，排空 worker 并保持已关闭 stream 可 replay
    async def close(self) -> None:
        if not self._accepting and all(
            state.worker is None or state.worker.done()
            for state in self._streams.values()
        ):
            return
        self._accepting = False
        for state in self._streams.values():
            async with state.lock:
                if state.lifecycle is StreamLifecycle.OPEN:
                    state.lifecycle = StreamLifecycle.CLOSING
                state.ready.set()
        workers = [
            state.worker
            for state in self._streams.values()
            if state.worker is not None
        ]
        if not workers:
            return

        primary_cancel: asyncio.CancelledError | None = None
        try:
            _done, pending = await asyncio.wait(
                workers,
                timeout=JOURNAL_SHUTDOWN_TIMEOUT_S,
            )
        except asyncio.CancelledError as exc:
            primary_cancel = exc
            pending = {worker for worker in workers if not worker.done()}

        if pending:
            pending_states = [
                state
                for state in self._streams.values()
                if state.worker in pending
            ]
            stream_ids = [state.stream_id for state in pending_states]
            counters = {
                state.stream_id: self.stream_metrics(state.stream_id)
                for state in pending_states
            }
            if primary_cancel is None:
                logger.error(
                    "journal shutdown timeout streams=%s counters=%s role=shutdown",
                    stream_ids,
                    counters,
                )
            else:
                logger.error(
                    "journal shutdown cancelled streams=%s counters=%s role=shutdown",
                    stream_ids,
                    counters,
                )
            for worker in pending:
                worker.cancel()

        waiter = asyncio.gather(*workers, return_exceptions=True)
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError as exc:
                if primary_cancel is None:
                    primary_cancel = exc
        await waiter
        if primary_cancel is not None:
            raise primary_cancel

    # 取已注册 stream，未知时抛稳定内部错误
    def _require_stream(self, stream_id: str) -> _StreamState:
        try:
            return self._streams[stream_id]
        except KeyError as exc:
            raise UnknownStreamError(f"unknown durable stream {stream_id}") from exc


# 用 compact UTF-8 JSONL 构造不可变 v2 record
def _build_v2_record(
    *,
    event_id: str,
    stream_id: str,
    seq: int,
    event: dict[str, Any],
) -> JournalRecord:
    wrapper = {
        "schema_version": 2,
        "event_id": event_id,
        "stream_id": stream_id,
        "seq": seq,
        "event": event,
    }
    serialized = _bounded_json_line(wrapper, JOURNAL_MAX_SINGLE_FRAME_BYTES)
    return JournalRecord(
        event_id=event_id,
        stream_id=stream_id,
        seq=seq,
        event=event,
        serialized=serialized,
    )


# 使用 iterencode 在 limit+1 处停止，避免先物化无界 JSON 字节串
def _bounded_json_line(value: dict[str, Any], limit: int) -> bytes:
    encoded = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        chunk_bytes = chunk.encode("utf-8")
        if len(encoded) + len(chunk_bytes) + 1 > limit:
            raise JournalCapacityError("journal frame exceeds 1 MiB")
        encoded.extend(chunk_bytes)
    encoded.extend(b"\n")
    return bytes(encoded)


# 在指定剩余 budget 内读取实际 bytes，并在每个 chunk 前后检查 stop
def _read_file_bytes(
    path: Path,
    stop_event: threading.Event | None,
    *,
    max_bytes: int,
) -> bytes:
    if not path.exists():
        return b""
    data = bytearray()
    with path.open("rb") as file:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise asyncio.CancelledError
            read_size = min(
                REPLAY_READ_CHUNK_BYTES,
                max_bytes - len(data) + 1,
            )
            chunk = file.read(read_size)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise JournalCapacityError("replay exceeds 16 MiB")
            if stop_event is not None and stop_event.is_set():
                raise asyncio.CancelledError
    return bytes(data)


# 将 JSONL 拆成完整行，并允许仅最后一个无换行但合法的 JSON row
def _iter_complete_rows(data: bytes, *, role: str) -> Iterator[tuple[int, bytes]]:
    if not data:
        return
    parts = data.split(b"\n")
    has_final_newline = data.endswith(b"\n")
    complete_count = len(parts) - 1 if has_final_newline else len(parts)
    for index in range(complete_count):
        row = parts[index]
        if not row:
            continue
        if len(row) > MAX_REPLAY_LINE_BYTES:
            raise JournalCapacityError("replay line exceeds 1 MiB")
        if not has_final_newline and index == complete_count - 1:
            try:
                json.loads(row)
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning("ignored truncated journal tail role=%s", role)
                return
        yield index + 1, row


# 用 strict Event union 校验 payload 并拒绝具体 event model 的未知字段
def _validate_event_payload(raw: object, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise JournalCorruptionError(f"{role} journal Event must be an object")
    try:
        event = _EVENT_ADAPTER.validate_python(raw, strict=True)
    except ValidationError as exc:
        raise JournalCorruptionError(f"invalid Event in {role} journal") from exc
    if set(raw) - set(type(event).model_fields):
        raise JournalCorruptionError(f"unknown Event field in {role} journal")
    return event.model_dump(mode="json")


# 在 owner open 前把合法无换行末行补 newline，或移除不完整 crash tail
def _normalize_v2_tail(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    tail_start = data.rfind(b"\n") + 1
    tail = data[tail_start:]
    try:
        json.loads(tail)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as file:
            file.truncate(tail_start)
            file.flush()
        return
    with path.open("ab") as file:
        file.write(b"\n")
        file.flush()


# 拒绝 v2 wrapper 复用同 stream legacy synthetic identity
def _reject_cross_format_duplicate_identity(
    legacy: list[JournalRecord],
    v2: list[JournalRecord],
) -> None:
    legacy_ids = {record.event_id for record in legacy}
    if any(record.event_id in legacy_ids for record in v2):
        raise JournalCorruptionError("duplicate event identity across legacy/v2 journal")


# 解析并校验 legacy raw Event rows，生成稳定 synthetic identity
def _read_legacy_records(
    stream_id: str,
    path: Path | None,
    *,
    stop_event: threading.Event | None,
    max_bytes: int,
) -> tuple[list[JournalRecord], int]:
    if path is None:
        return [], 0
    data = _read_file_bytes(path, stop_event, max_bytes=max_bytes)
    records: list[JournalRecord] = []
    for _line_no, row in _iter_complete_rows(data, role="legacy"):
        try:
            raw = json.loads(row)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError("complete invalid row in legacy journal") from exc
        if not isinstance(raw, dict):
            raise JournalCorruptionError("legacy journal row must be an Event object")
        if "schema_version" in raw or "event_id" in raw or "seq" in raw:
            raise JournalCorruptionError("mixed legacy/v2 journal")
        seq = len(records) + 1
        event_data = _validate_event_payload(raw, role="legacy")
        canonical = json.dumps(
            event_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(
            b"kamaclaude-legacy-v1\0"
            + stream_id.encode("utf-8")
            + b"\0"
            + str(seq).encode("ascii")
            + b"\0"
            + canonical
        ).hexdigest()[:32]
        records.append(
            JournalRecord(
                event_id=f"legacy-{digest}",
                stream_id=stream_id,
                seq=seq,
                event=event_data,
                serialized=row + b"\n",
            )
        )
    return records, len(data)


# 解析并严格校验 v2 schema、stream、seq continuity、identity 与 Event
def _read_v2_records(
    stream_id: str,
    path: Path,
    *,
    expected_first_seq: int,
    stop_event: threading.Event | None,
    max_bytes: int,
) -> tuple[list[JournalRecord], int]:
    data = _read_file_bytes(path, stop_event, max_bytes=max_bytes)
    records: list[JournalRecord] = []
    event_ids: set[str] = set()
    expected_seq = expected_first_seq
    for _line_no, row in _iter_complete_rows(data, role="v2"):
        try:
            raw = json.loads(row)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError("complete invalid row in v2 journal") from exc
        try:
            wrapper = _V2JournalRow.model_validate(raw, strict=True)
        except ValidationError as exc:
            raise JournalCorruptionError("invalid strict v2 journal row") from exc
        event_id = wrapper.event_id
        row_stream = wrapper.stream_id
        seq = wrapper.seq
        if event_id in event_ids:
            raise JournalCorruptionError("duplicate event identity in v2 journal")
        if row_stream != stream_id:
            raise JournalCorruptionError("v2 stream_id mismatch")
        if seq != expected_seq:
            raise JournalCorruptionError("v2 seq continuity failure")
        event_data = _validate_event_payload(wrapper.event, role="v2")
        records.append(
            JournalRecord(
                event_id=event_id,
                stream_id=stream_id,
                seq=seq,
                event=event_data,
                serialized=row + b"\n",
            )
        )
        event_ids.add(event_id)
        expected_seq += 1
    return records, len(data)


# 完整读取 legacy+v2 并只在全部校验通过后构造目标 ReplayBatch
def _read_replay_batch(
    stream_id: str,
    v2_path: Path,
    legacy_path: Path | None,
    after_seq: int,
    high_watermark: int,
    stop_event: threading.Event,
) -> ReplayBatch:
    legacy, legacy_bytes = _read_legacy_records(
        stream_id,
        legacy_path,
        stop_event=stop_event,
        max_bytes=MAX_REPLAY_BYTES,
    )
    v2, v2_bytes = _read_v2_records(
        stream_id,
        v2_path,
        expected_first_seq=len(legacy) + 1,
        stop_event=stop_event,
        max_bytes=MAX_REPLAY_BYTES - legacy_bytes,
    )
    _reject_cross_format_duplicate_identity(legacy, v2)
    examined = legacy_bytes + v2_bytes
    if examined > MAX_REPLAY_BYTES:
        raise JournalCapacityError("replay exceeds 16 MiB")
    selected = tuple(
        record
        for record in (*legacy, *v2)
        if after_seq < record.seq <= high_watermark
    )
    if len(selected) > MAX_REPLAY_EVENTS:
        raise JournalCapacityError("replay exceeds 10000 events")
    if high_watermark > 0:
        available = len(legacy) + len(v2)
        if high_watermark > available:
            raise JournalCorruptionError("missing high watermark")
    return ReplayBatch(records=selected, examined_bytes=examined)


# 取消 stream 注册时等待底层校验 thread 终态并保留原 cancellation
async def _validate_stream_preserving_cancellation(
    validator: Callable[[str, Path, Path | None], int],
    stream_id: str,
    path: Path,
    legacy_path: Path | None,
) -> int:
    validation_task = asyncio.create_task(
        asyncio.to_thread(validator, stream_id, path, legacy_path),
        name=f"event-journal-register:{stream_id}",
    )
    try:
        return await asyncio.shield(validation_task)
    except asyncio.CancelledError as primary:
        while not validation_task.done():
            try:
                await asyncio.shield(validation_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not validation_task.cancelled():
            try:
                validation_task.result()
            except Exception:
                logger.error(
                    "journal validation failed during cancellation "
                    "stream_id=%s role=secondary",
                    stream_id,
                )
        raise primary


# 取消 journal worker 时仍等待底层写盘 thread 终态并优先恢复原 cancellation
async def _append_and_flush_preserving_cancellation(
    stream_id: str,
    path: Path,
    rows: tuple[bytes, ...],
) -> None:
    append_task = asyncio.create_task(
        asyncio.to_thread(EventWriter(path).append_and_flush, rows),
        name=f"event-journal-append:{stream_id}",
    )
    try:
        await asyncio.shield(append_task)
    except asyncio.CancelledError as primary:
        while not append_task.done():
            try:
                await asyncio.shield(append_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not append_task.cancelled():
            try:
                append_task.result()
            except Exception:
                logger.error(
                    "journal append failed during cancellation "
                    "stream_id=%s role=secondary",
                    stream_id,
                )
        raise primary


# 等待 terminal receipts 时保留首次 cancellation 对象并让清理到达终态
async def _await_preserving_cancellation(
    receipts: list[asyncio.Future[None]],
) -> None:
    waiter = asyncio.gather(*receipts, return_exceptions=True)
    primary: asyncio.CancelledError | None = None
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError as exc:
            if primary is None:
                primary = exc
    outcomes = await waiter
    if primary is not None:
        raise primary
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            raise outcome
