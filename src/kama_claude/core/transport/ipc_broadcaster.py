from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from typing import Literal

from pydantic import BaseModel

from kama_claude.core.bus.envelope import EventPushEnvelope
from kama_claude.core.events.journal import (
    EventJournalCoordinator,
    JournalRecord,
)
from kama_claude.core.trace.record import TraceRecord
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.transport.connection import ConnectionContext

logger = logging.getLogger(__name__)

CATCHUP_BUFFER_MAX_FRAMES = 512
CATCHUP_BUFFER_MAX_BYTES = 4 * 1024 * 1024
CATCHUP_MAX_SINGLE_FRAME_BYTES = 1024 * 1024
MAX_SUBSCRIPTIONS_PER_CONNECTION = 16
MAX_TOTAL_SUBSCRIPTIONS = 256
_GLOBAL_EVENT_DEDUP_LIMIT = 16_384


# 生成 UTC ISO timestamp 供安全 delivery trace 使用
def _now() -> str:
    return datetime.now(UTC).isoformat()


class _SubscriptionPhase(StrEnum):
    CATCHING_UP = "catching_up"
    ACTIVE = "active"
    TERMINAL = "terminal"


@dataclass
class _Subscription:
    sub_id: str
    context: ConnectionContext
    topics: list[str]
    scope: str
    stream_id: str | None
    phase: _SubscriptionPhase
    high_watermark: int = 0
    catchup: deque[JournalRecord] = field(default_factory=deque)
    catchup_bytes: int = 0
    replay_task: asyncio.Task[None] | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class PreparedSubscription:
    subscription_id: str
    stream_id: str
    accepted_after_seq: int
    high_watermark_seq: int
    activation: _ReplayActivation


class _ReplayActivation:
    # 保存 response barrier 后启动 replay 所需的私有引用
    def __init__(
        self,
        broadcaster: IpcEventBroadcaster,
        coordinator: EventJournalCoordinator,
        subscription_id: str,
        after_seq: int,
    ) -> None:
        self._broadcaster = broadcaster
        self._coordinator = coordinator
        self._subscription_id = subscription_id
        self._after_seq = after_seq
        self._resolved = False

    # response written 后仅创建 connection-owned replay task，不阻塞 request 返回
    async def on_written(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._broadcaster._activate_replay(
            self._coordinator,
            self._subscription_id,
            self._after_seq,
        )

    # response delivery 失败时删除 subscription/catch-up 且不启动 reader
    async def on_failure(self, exc: BaseException) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._broadcaster._terminate_subscription(self._subscription_id)


class IpcEventBroadcaster:
    # 初始化 connection-owned subscriptions 与安全 delivery trace
    def __init__(self, trace: TraceWriter | None = None) -> None:
        self._subscriptions: list[_Subscription] = []
        self._trace = trace
        self._global_event_ids: set[str] = set()
        self._global_event_order: deque[str] = deque()

    # 注册 live-only 或已完成 catch-up 的普通订阅
    def subscribe(
        self,
        context: ConnectionContext,
        topics: list[str],
        scope: str = "global",
    ) -> str:
        context.ensure_open_for_subscription()
        self._enforce_subscription_limits(context)
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        stream_id = scope if scope.startswith(("run:", "session:")) else None
        self._subscriptions.append(
            _Subscription(
                sub_id=sub_id,
                context=context,
                topics=topics,
                scope=scope,
                stream_id=stream_id,
                phase=_SubscriptionPhase.ACTIVE,
            )
        )
        return sub_id

    # 在 durable watermark capture 的同一临界区登记 catching-up subscription
    async def prepare_durable_subscription(
        self,
        coordinator: EventJournalCoordinator,
        context: ConnectionContext,
        *,
        topics: list[str],
        stream_id: str,
        after_seq: int,
    ) -> PreparedSubscription:
        context.ensure_open_for_subscription()
        self._enforce_subscription_limits(context)
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        sub = _Subscription(
            sub_id=sub_id,
            context=context,
            topics=topics,
            scope=stream_id,
            stream_id=stream_id,
            phase=_SubscriptionPhase.CATCHING_UP,
        )

        # callback 不得 await，确保 watermark 与 observer registration 原子
        def register(high_watermark: int) -> None:
            context.ensure_open_for_subscription()
            self._enforce_subscription_limits(context)
            sub.high_watermark = high_watermark
            self._subscriptions.append(sub)

        high_watermark = await coordinator.capture_high_watermark(stream_id, register)
        accepted = min(max(after_seq, 0), high_watermark)
        return PreparedSubscription(
            subscription_id=sub_id,
            stream_id=stream_id,
            accepted_after_seq=accepted,
            high_watermark_seq=high_watermark,
            activation=_ReplayActivation(self, coordinator, sub_id, accepted),
        )

    # 由 journal worker flush 成功后推送 stream-aware live 或暂存 catch-up
    async def publish_durable(self, record: JournalRecord) -> None:
        deliver_global = self._remember_global_event(record.event_id)
        event_type = str(record.event.get("type", ""))
        for sub in list(self._subscriptions):
            if sub.phase is _SubscriptionPhase.TERMINAL:
                continue
            is_global = sub.scope == "global"
            if is_global:
                if not deliver_global:
                    continue
            elif sub.stream_id != record.stream_id:
                continue
            if not self._matches_topic(event_type, sub.topics):
                continue

            if sub.phase is _SubscriptionPhase.CATCHING_UP:
                if record.seq <= sub.high_watermark:
                    continue
                if not self._buffer_catchup(sub, record):
                    self._terminate_subscription(sub.sub_id)
                    sub.context.initiate_close("catch-up buffer overflow")
                continue
            try:
                self._enqueue_delivery(sub, record, delivery="live", wait=False)
            except (ConnectionError, ValueError):
                continue

    # 仅向 global subscriptions 发送不具有 durable stream 的事件
    async def publish_live_only(self, event: BaseModel) -> None:
        event_dict = event.model_dump(mode="json")
        event_type = str(event_dict.get("type", ""))
        for sub in list(self._subscriptions):
            if (
                sub.phase is not _SubscriptionPhase.ACTIVE
                or sub.scope != "global"
                or not self._matches_topic(event_type, sub.topics)
            ):
                continue
            envelope = EventPushEnvelope(
                subscription_id=sub.sub_id,
                delivery="live",
                event=event_dict,
            )
            try:
                receipt = sub.context.enqueue_event(envelope)
                receipt.written.add_done_callback(self._observe_delivery)
            except (ConnectionError, ValueError):
                self.unsubscribe_all(sub.context)

    # 保留旧 handler 名称供非 durable global-only EventBus 接线兼容
    async def handle(self, event: BaseModel) -> None:
        await self.publish_live_only(event)

    # response barrier 成功后创建 connection-owned replay lifecycle task
    def _activate_replay(
        self,
        coordinator: EventJournalCoordinator,
        sub_id: str,
        after_seq: int,
    ) -> None:
        sub = self._find(sub_id)
        if sub is None or sub.phase is not _SubscriptionPhase.CATCHING_UP:
            return
        sub.replay_task = sub.context.create_request_task(
            self._run_replay(coordinator, sub, after_seq)
        )

    # 完整校验 ReplayBatch 后依次发送 replay，再无竞态切换 catch-up/live
    async def _run_replay(
        self,
        coordinator: EventJournalCoordinator,
        sub: _Subscription,
        after_seq: int,
    ) -> None:
        assert sub.stream_id is not None
        try:
            batch = await coordinator.read_replay(
                sub.stream_id,
                after_seq=after_seq,
                high_watermark=sub.high_watermark,
                stop_event=sub.stop_event,
            )
            for record in batch.records:
                if not self._matches_topic(
                    str(record.event.get("type", "")),
                    sub.topics,
                ):
                    continue
                await self._enqueue_delivery(
                    sub,
                    record,
                    delivery="replay",
                    wait=True,
                )

            while True:
                current = self._find(sub.sub_id)
                if current is None or current.phase is _SubscriptionPhase.TERMINAL:
                    return
                if not current.catchup:
                    current.phase = _SubscriptionPhase.ACTIVE
                    return
                record = current.catchup.popleft()
                current.catchup_bytes -= self._delivery_size(current, record, "live")
                await self._enqueue_delivery(
                    current,
                    record,
                    delivery="live",
                    wait=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("replay lifecycle failed sub_id=%s role=replay", sub.sub_id)
            self._terminate_subscription(sub.sub_id)
            sub.context.initiate_close("replay validation failure")

    # 将完整 delivery frame 入 event queue，可选等待真实 socket drain
    def _enqueue_delivery(
        self,
        sub: _Subscription,
        record: JournalRecord,
        *,
        delivery: Literal["replay", "live"],
        wait: bool,
    ) -> asyncio.Future[None]:
        envelope = EventPushEnvelope(
            subscription_id=sub.sub_id,
            delivery=delivery,
            event_id=record.event_id,
            stream_id=record.stream_id,
            seq=record.seq,
            event=record.event,
        )
        try:
            receipt = sub.context.enqueue_event(
                envelope,
                on_written=partial(
                    self._emit_trace,
                    sub.context,
                    sub.sub_id,
                    str(record.event.get("type", "")),
                    record.event.get("run_id"),
                ),
            )
        except (ConnectionError, ValueError):
            self.unsubscribe_all(sub.context)
            raise
        if wait:
            return receipt.written
        receipt.written.add_done_callback(self._observe_delivery)
        return receipt.written

    @staticmethod
    # 计算完整 envelope JSONL bytes 作为 catch-up 精确容量依据
    def _delivery_size(
        sub: _Subscription,
        record: JournalRecord,
        delivery: Literal["replay", "live"],
    ) -> int:
        envelope = EventPushEnvelope(
            subscription_id=sub.sub_id,
            delivery=delivery,
            event_id=record.event_id,
            stream_id=record.stream_id,
            seq=record.seq,
            event=record.event,
        )
        return len(ConnectionContext._encode_frame(envelope))

    # 按 frame/byte/single-frame 三类上限原子追加 catch-up record
    def _buffer_catchup(self, sub: _Subscription, record: JournalRecord) -> bool:
        try:
            size = self._delivery_size(sub, record, "live")
        except ValueError:
            return False
        if size > CATCHUP_MAX_SINGLE_FRAME_BYTES:
            return False
        if (
            len(sub.catchup) >= CATCHUP_BUFFER_MAX_FRAMES
            or sub.catchup_bytes + size > CATCHUP_BUFFER_MAX_BYTES
        ):
            return False
        sub.catchup.append(record)
        sub.catchup_bytes += size
        return True

    # 仅由所属 connection 删除指定 subscription_id
    def unsubscribe(self, context: ConnectionContext, sub_id: str) -> bool:
        sub = self._find(sub_id)
        if sub is None or sub.context is not context:
            return False
        self._terminate_subscription(sub_id)
        return True

    # 移除指定 connection 的全部订阅和 replay reader
    def unsubscribe_all(self, context: ConnectionContext) -> None:
        for sub in list(self._subscriptions):
            if sub.context is context:
                self._terminate_subscription(sub.sub_id)

    # 终止 degraded durable stream 的 subscriptions 并关闭所属连接
    def fail_stream(self, stream_id: str) -> None:
        for sub in list(self._subscriptions):
            if sub.stream_id != stream_id:
                continue
            context = sub.context
            self._terminate_subscription(sub.sub_id)
            context.initiate_close("durable stream failure")

    # 标记 subscription terminal、停止 reader并释放 catch-up 内存
    def _terminate_subscription(self, sub_id: str) -> None:
        sub = self._find(sub_id)
        if sub is None:
            return
        sub.phase = _SubscriptionPhase.TERMINAL
        sub.stop_event.set()
        sub.catchup.clear()
        sub.catchup_bytes = 0
        if sub.replay_task is not None and not sub.replay_task.done():
            sub.replay_task.cancel()
        self._subscriptions.remove(sub)

    # 等待指定 replay lifecycle task 到达终态，供确定性测试和 shutdown 使用
    async def wait_subscription(self, sub_id: str) -> None:
        sub = self._find(sub_id)
        if sub is None or sub.replay_task is None:
            return
        await asyncio.shield(sub.replay_task)

    # 返回 subscription 是否仍受 broadcaster 管理
    def has_subscription(self, sub_id: str) -> bool:
        return self._find(sub_id) is not None

    # 查找 subscription，内部不存在时返回 None
    def _find(self, sub_id: str) -> _Subscription | None:
        return next(
            (sub for sub in self._subscriptions if sub.sub_id == sub_id),
            None,
        )

    # 强制执行 per-connection 与全局 subscription 数量上限
    def _enforce_subscription_limits(self, context: ConnectionContext) -> None:
        if len(self._subscriptions) >= MAX_TOTAL_SUBSCRIPTIONS:
            raise ConnectionError("global subscription limit reached")
        owned = sum(sub.context is context for sub in self._subscriptions)
        if owned >= MAX_SUBSCRIPTIONS_PER_CONNECTION:
            raise ConnectionError("connection subscription limit reached")

    # 只让同一 event_id 的第一个 durable stream 回调触发 global live
    def _remember_global_event(self, event_id: str) -> bool:
        if event_id in self._global_event_ids:
            return False
        self._global_event_ids.add(event_id)
        self._global_event_order.append(event_id)
        if len(self._global_event_order) > _GLOBAL_EVENT_DEDUP_LIMIT:
            expired = self._global_event_order.popleft()
            self._global_event_ids.discard(expired)
        return True

    @staticmethod
    # 观察 fire-and-forget receipt 终态，避免未读取 Future exception
    def _observe_delivery(future: asyncio.Future[None]) -> None:
        if future.cancelled():
            return
        future.exception()

    # 在对应 event frame 成功 drain 后写入安全 trace 元数据
    def _emit_trace(
        self,
        context: ConnectionContext,
        sub_id: str,
        event_type: str,
        run_id: object,
    ) -> None:
        if self._trace is None:
            return
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→CLIENT",
                layer="ipc",
                kind="push",
                run_id=run_id if isinstance(run_id, str) else None,
                client_id=str(context.peername),
                data={"sub_id": sub_id, "event_type": event_type},
            )
        )

    @staticmethod
    # 检查事件类型是否匹配订阅 topic glob
    def _matches_topic(event_type: str, topics: list[str]) -> bool:
        return any(fnmatch.fnmatch(event_type, pattern) for pattern in topics)
