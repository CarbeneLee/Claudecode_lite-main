from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

from pydantic import BaseModel

from kama_claude.core.bus.envelope import EventPushEnvelope
from kama_claude.core.trace.record import TraceRecord
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.transport.connection import ConnectionContext

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _Subscription:
    sub_id: str
    context: ConnectionContext
    topics: list[str]
    scope: str


class IpcEventBroadcaster:
    def __init__(self, trace: TraceWriter | None = None) -> None:
        self._subscriptions: list[_Subscription] = []
        self._trace = trace

    # 注册一个客户端订阅，返回 subscription_id
    def subscribe(
        self,
        context: ConnectionContext,
        topics: list[str],
        scope: str = "global",
    ) -> str:
        context.ensure_open_for_subscription()
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        sub = _Subscription(
            sub_id=sub_id,
            context=context,
            topics=topics,
            scope=scope,
        )
        self._subscriptions.append(sub)
        return sub_id

    # 仅由所属 connection 删除指定 subscription_id
    def unsubscribe(self, context: ConnectionContext, sub_id: str) -> bool:
        for index, sub in enumerate(self._subscriptions):
            if sub.context is context and sub.sub_id == sub_id:
                del self._subscriptions[index]
                return True
        return False

    # 移除指定 connection 拥有的全部订阅
    def unsubscribe_all(self, context: ConnectionContext) -> None:
        self._subscriptions = [
            sub for sub in self._subscriptions if sub.context is not context
        ]

    # 将事件推送到所有匹配的订阅客户端，写入失败时延迟清理死连接
    async def handle(self, event: BaseModel) -> None:
        event_dict = event.model_dump()
        event_type: str = event_dict.get("type", "")
        run_id: str | None = event_dict.get("run_id")

        for sub in list(self._subscriptions):
            if not self._matches_topic(event_type, sub.topics):
                continue
            if not self._matches_scope(run_id, sub.scope):
                continue
            try:
                envelope = EventPushEnvelope(event=event_dict)
                receipt = sub.context.enqueue_event(
                    envelope,
                    on_written=partial(
                        self._emit_trace,
                        sub.context,
                        sub.sub_id,
                        event_type,
                        run_id,
                    )
                )
                receipt.written.add_done_callback(self._observe_delivery)
            except (ConnectionError, ValueError):
                logger.warning(
                    "event enqueue failed connection_id=%s sub_id=%s role=live",
                    sub.context.connection_id,
                    sub.sub_id,
                )
                self.unsubscribe_all(sub.context)

    # 观察 fire-and-forget live receipt 的终态，避免失败 Future 产生全局 warning
    @staticmethod
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
        run_id: str | None,
    ) -> None:
        if self._trace is None:
            return
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→CLIENT",
                layer="ipc",
                kind="push",
                run_id=run_id,
                client_id=str(context.peername),
                data={"sub_id": sub_id, "event_type": event_type},
            )
        )

    # 检查事件类型是否匹配订阅的 topic 列表（支持 fnmatch glob 模式）
    @staticmethod
    def _matches_topic(event_type: str, topics: list[str]) -> bool:
        return any(fnmatch.fnmatch(event_type, pattern) for pattern in topics)

    # 检查事件 run_id 是否匹配订阅的 scope（global 全通，run:<id> 精确匹配）
    @staticmethod
    def _matches_scope(run_id: str | None, scope: str) -> bool:
        if scope == "global":
            return True
        if scope.startswith("run:"):
            return run_id == scope[4:]
        return False
