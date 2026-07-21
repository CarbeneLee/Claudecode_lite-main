from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from kama_claude.core.bus.events import RunStartedEvent
from kama_claude.core.events import journal as journal_module
from kama_claude.core.events.journal import (
    EventJournalCoordinator,
    JournalCapacityError,
    JournalError,
    JournalRecord,
)
from kama_claude.core.transport import ipc_broadcaster as broadcaster_module
from kama_claude.core.transport.connection import (
    ConnectionContext,
    ConnectionState,
    FrameReceipt,
)
from kama_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from kama_claude.core.transport.socket_server import (
    HandlerOutcome,
    SocketServer,
)


class _ActivationRecorder:
    # 初始化 activation/failure 信号与异常引用
    def __init__(self) -> None:
        self.written = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure: BaseException | None = None
        self.calls: list[str] = []

    # 记录 response written 后的唯一 activation
    async def on_written(self) -> None:
        self.calls.append("written")
        self.written.set()

    # 记录 response delivery 失败并保留原异常引用
    async def on_failure(self, exc: BaseException) -> None:
        self.calls.append("failure")
        self.failure = exc
        self.failed.set()


class _ReceiptContext:
    # 初始化可由测试显式完成 written future 的 control queue 替身
    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.receipt = FrameReceipt(
            enqueued=loop.create_future(),
            written=loop.create_future(),
        )
        self.receipt.enqueued.set_result(None)
        self.enqueued = asyncio.Event()
        self.frame: object | None = None

    @property
    # 返回稳定 peername 供 trace 分支兼容
    def peername(self) -> object:
        return ("127.0.0.1", 7007)

    # 记录 control frame 并返回受控 receipt
    def enqueue_control(self, frame: object, *, on_written: object = None) -> FrameReceipt:
        self.frame = frame
        self.enqueued.set()
        return self.receipt


class _RecordingWriter:
    # 初始化完整 frame 记录与同步关闭状态
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    # 记录 ConnectionContext 唯一 writer 发送的完整 frame
    def write(self, data: bytes) -> None:
        self.frames.append(data)

    # 模拟立即完成 drain
    async def drain(self) -> None:
        return None

    # 提供同步 close 接口
    def close(self) -> None:
        return None

    # 提供异步 wait_closed 接口
    async def wait_closed(self) -> None:
        return None

    # 返回稳定 peername
    def get_extra_info(self, name: str, default: object = None) -> object:
        if name == "peername":
            return ("127.0.0.1", 7010)
        return default


class _GatedWatermarkCoordinator:
    # 初始化并发 watermark 调用的双参与门闩
    def __init__(self) -> None:
        self.calls = 0
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()

    # 等两个 prepare 都通过前置检查后再依次执行注册回调
    async def capture_high_watermark(
        self,
        stream_id: str,
        registrar: object,
    ) -> int:
        self.calls += 1
        if self.calls == 2:
            self.all_entered.set()
        await self.release.wait()
        cast(Any, registrar)(0)
        return 0


# 创建使用真实 outbound queue/writer task 的 ConnectionContext
def _connection() -> tuple[ConnectionContext, _RecordingWriter]:
    writer = _RecordingWriter()
    context = ConnectionContext(
        cast(asyncio.StreamWriter, writer),
        connection_id="conn-replay",
    )
    context.start()
    return context, writer


# 构造可区分 seq 的 run.started 事件
def _event(goal: str) -> RunStartedEvent:
    return RunStartedEvent(
        run_id="run-1",
        goal=goal,
        ts="2026-07-21T00:00:00Z",
    )


# 构造可用于 catch-up 容量审计的固定 durable record
def _record(goal: str = "catch-up") -> JournalRecord:
    event = _event(goal).model_dump(mode="json")
    return JournalRecord(
        event_id="evt-fixed",
        stream_id="run:run-1",
        seq=1,
        event=event,
        serialized=b"unused\n",
    )


# 构造一条固定 JSON-RPC request line
def _request_line() -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "subscribe-1",
            "method": "test.subscribe",
            "params": {},
        }
    ).encode() + b"\n"


# 功能：验证 post-response activation 严格等待 success frame 的 written receipt
# 设计：直接控制真实 Future，在 enqueue 与 written 之间断言 activation 尚未发生，排除 create_task 调度偶然性
async def test_handler_outcome_activates_only_after_response_written() -> None:
    activation = _ActivationRecorder()
    context = _ReceiptContext()
    server = SocketServer("127.0.0.1", 0)

    # 返回带 activation 的稳定 subscribe outcome
    async def handler(params: dict[str, Any]) -> HandlerOutcome:
        return HandlerOutcome(result={"subscription_id": "sub-1"}, post_response=activation)

    server.register("test.subscribe", handler)
    request = asyncio.create_task(
        server._handle_line(_request_line(), cast(Any, context))
    )
    await context.enqueued.wait()

    assert not activation.written.is_set()
    context.receipt.written.set_result(None)
    await request

    assert activation.calls == ["written"]


# 功能：验证 response write failure 只调用 on_failure 且传递同一个异常对象
# 设计：让 receipt.written 以唯一异常失败，断言 identity 与单次调用，防止失败后错误启动 replay
async def test_handler_outcome_response_failure_cleans_up_without_activation() -> None:
    activation = _ActivationRecorder()
    context = _ReceiptContext()
    server = SocketServer("127.0.0.1", 0)
    failure = ConnectionError("response-barrier-secret")

    # 返回带 activation 的稳定 subscribe outcome
    async def handler(params: dict[str, Any]) -> HandlerOutcome:
        return HandlerOutcome(result={"subscription_id": "sub-1"}, post_response=activation)

    server.register("test.subscribe", handler)
    request = asyncio.create_task(
        server._handle_line(_request_line(), cast(Any, context))
    )
    await context.enqueued.wait()
    context.receipt.written.set_exception(failure)
    await request

    assert activation.calls == ["failure"]
    assert activation.failure is failure
    assert not activation.written.is_set()


# 功能：验证等待 response written 时 request cancellation 会先调用 on_failure 再原样传播
# 设计：保存外层 CancelledError 引用并直接 await request，锁定 cleanup 顺序与 cancellation identity
async def test_handler_outcome_cancellation_reports_failure_and_preserves_identity() -> None:
    activation = _ActivationRecorder()
    context = _ReceiptContext()
    server = SocketServer("127.0.0.1", 0)

    # 返回带 activation 的稳定 subscribe outcome
    async def handler(params: dict[str, Any]) -> HandlerOutcome:
        return HandlerOutcome(result={"subscription_id": "sub-1"}, post_response=activation)

    server.register("test.subscribe", handler)
    request = asyncio.create_task(
        server._handle_line(_request_line(), cast(Any, context))
    )
    await context.enqueued.wait()
    request.cancel("response-wait-cancel")

    try:
        await request
    except asyncio.CancelledError as caught:
        observed = caught
    else:
        raise AssertionError("request cancellation did not propagate")

    assert activation.calls == ["failure"]
    assert activation.failure is observed
    assert not activation.written.is_set()


# 功能：验证 high watermark 后到达的 durable live 先进入 catch-up，replay 完成后再按 seq 发送
# 设计：使用真实 coordinator、broadcaster 和 ConnectionContext，在 activation 前写入第二条事件并解析最终 delivery 顺序
async def test_replay_then_catchup_preserves_stream_order(tmp_path: Path) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    await coordinator.handle(_event("before-watermark"))
    await coordinator.flush_all()
    context, writer = _connection()

    prepared = await broadcaster.prepare_durable_subscription(
        coordinator,
        context,
        topics=["run.*"],
        stream_id="run:run-1",
        after_seq=0,
    )
    await coordinator.handle(_event("after-watermark"))
    await coordinator.flush_all()
    assert writer.frames == []

    await prepared.activation.on_written()
    await broadcaster.wait_subscription(prepared.subscription_id)
    await context.close("test complete")
    await coordinator.close()

    envelopes = [json.loads(frame) for frame in writer.frames]
    assert [envelope["seq"] for envelope in envelopes] == [1, 2]
    assert [envelope["delivery"] for envelope in envelopes] == ["replay", "live"]
    assert envelopes[0]["event"]["goal"] == "before-watermark"
    assert envelopes[1]["event"]["goal"] == "after-watermark"


# 功能：验证 response failure 会删除 catching-up subscription 且不会启动 replay reader
# 设计：在 prepare 后直接调用 activation.on_failure，以 coordinator reader spy 证明 disk replay 零调用
async def test_response_failure_removes_prepared_subscription_before_replay(
    tmp_path: Path,
) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    await coordinator.handle(_event("history"))
    await coordinator.flush_all()
    context, writer = _connection()
    read_calls = 0
    original = coordinator.read_replay

    # 记录 response failure 后是否曾启动真实 replay reader
    async def counted_read(*args: object, **kwargs: object) -> object:
        nonlocal read_calls
        read_calls += 1
        return await original(*args, **kwargs)

    coordinator.read_replay = cast(Any, counted_read)
    prepared = await broadcaster.prepare_durable_subscription(
        coordinator,
        context,
        topics=["run.*"],
        stream_id="run:run-1",
        after_seq=0,
    )

    await prepared.activation.on_failure(ConnectionError("response failed"))
    await coordinator.handle(_event("later"))
    await coordinator.flush_all()
    await context.close("test complete")
    await coordinator.close()

    assert read_calls == 0
    assert writer.frames == []
    assert not broadcaster.has_subscription(prepared.subscription_id)


# 功能：验证第 513 条 catch-up record 只关闭所属慢连接而不影响同 stream 的 active 客户端
# 设计：一个 subscription 保持 catching_up，另一个正常 drain，连续 durable publish 锁定独立 frame cap 与隔离
async def test_catchup_frame_overflow_closes_only_owning_connection(
    tmp_path: Path,
) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    catching_context, _catching_writer = _connection()
    active_context, active_writer = _connection()
    prepared = await broadcaster.prepare_durable_subscription(
        coordinator,
        catching_context,
        topics=["run.*"],
        stream_id="run:run-1",
        after_seq=0,
    )
    broadcaster.subscribe(active_context, ["run.*"], scope="run:run-1")

    for index in range(broadcaster_module.CATCHUP_BUFFER_MAX_FRAMES + 1):
        await coordinator.handle(_event(f"live-{index}"))
    await coordinator.flush_all()
    while catching_context.state is not ConnectionState.CLOSED:
        await asyncio.sleep(0)
    while len(active_writer.frames) < broadcaster_module.CATCHUP_BUFFER_MAX_FRAMES + 1:
        await asyncio.sleep(0)

    assert not broadcaster.has_subscription(prepared.subscription_id)
    assert active_context.state is ConnectionState.OPEN
    await active_context.close("test complete")
    await coordinator.close()


# 功能：验证 catch-up byte cap 精确接受等号边界且拒绝下一条完整 frame
# 设计：用真实 envelope 编码长度设定临时 cap，直接核对 deque 和精确 bytes 计数
async def test_catchup_byte_limit_accepts_exact_boundary(
) -> None:
    broadcaster = IpcEventBroadcaster()
    context, _writer = _connection()
    sub_id = broadcaster.subscribe(context, ["run.*"], scope="run:run-1")
    sub = broadcaster._find(sub_id)
    assert sub is not None
    record = _record()
    size = broadcaster._delivery_size(sub, record, "live")
    sub.catchup_bytes = broadcaster_module.CATCHUP_BUFFER_MAX_BYTES - size - 1
    assert broadcaster._buffer_catchup(sub, record)
    assert sub.catchup_bytes == broadcaster_module.CATCHUP_BUFFER_MAX_BYTES - 1

    sub.catchup.clear()
    sub.catchup_bytes = broadcaster_module.CATCHUP_BUFFER_MAX_BYTES - size
    assert broadcaster._buffer_catchup(sub, record)
    assert sub.catchup_bytes == broadcaster_module.CATCHUP_BUFFER_MAX_BYTES

    sub.catchup.clear()
    sub.catchup_bytes = broadcaster_module.CATCHUP_BUFFER_MAX_BYTES - size + 1
    assert not broadcaster._buffer_catchup(sub, record)
    assert not sub.catchup
    assert sub.catchup_bytes == broadcaster_module.CATCHUP_BUFFER_MAX_BYTES - size + 1

    broadcaster.unsubscribe_all(context)
    await context.close("test complete")


# 功能：验证单条 catch-up envelope 超过 single-frame byte cap 时零部分缓冲
# 设计：以真实编码 frame 长度减一作为 cap，断言 deque 与 byte counter 均保持为零
async def test_catchup_single_frame_byte_limit_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = IpcEventBroadcaster()
    context, _writer = _connection()
    sub_id = broadcaster.subscribe(context, ["run.*"], scope="run:run-1")
    sub = broadcaster._find(sub_id)
    assert sub is not None
    record = _record()
    size = broadcaster._delivery_size(sub, record, "live")
    monkeypatch.setattr(broadcaster_module, "CATCHUP_MAX_SINGLE_FRAME_BYTES", size - 1)

    assert not broadcaster._buffer_catchup(sub, record)
    assert not sub.catchup
    assert sub.catchup_bytes == 0

    broadcaster.unsubscribe_all(context)
    await context.close("test complete")


# 功能：验证单连接精确可拥有 16 个 subscription，第 17 个被拒绝
# 设计：使用同一真实 ConnectionContext 连续注册到冻结上限，排除全局 cap 干扰
async def test_per_connection_subscription_limit_exact_boundary() -> None:
    broadcaster = IpcEventBroadcaster()
    context, _writer = _connection()

    for _ in range(broadcaster_module.MAX_SUBSCRIPTIONS_PER_CONNECTION):
        broadcaster.subscribe(context, ["run.*"])
    with pytest.raises(ConnectionError, match="connection subscription limit"):
        broadcaster.subscribe(context, ["run.*"])

    broadcaster.unsubscribe_all(context)
    await context.close("test complete")


# 功能：验证全局精确可注册 256 个 subscription，再增加时 fail closed
# 设计：用 16 个连接各占满 16 个名额，再由新连接触发全局上限以区分两类计数
async def test_global_subscription_limit_exact_boundary() -> None:
    broadcaster = IpcEventBroadcaster()
    connections = [_connection()[0] for _ in range(17)]

    for context in connections[:16]:
        for _ in range(broadcaster_module.MAX_SUBSCRIPTIONS_PER_CONNECTION):
            broadcaster.subscribe(context, ["run.*"])
    with pytest.raises(ConnectionError, match="global subscription limit"):
        broadcaster.subscribe(connections[-1], ["run.*"])

    for context in connections:
        broadcaster.unsubscribe_all(context)
        await context.close("test complete")


# 功能：验证并发 durable prepare 不能越过单连接 subscription 上限
# 设计：先占一个名额，再让两个 prepare 同时停在 watermark 门闩后竞争最后一个名额
async def test_concurrent_prepare_enforces_per_connection_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(broadcaster_module, "MAX_SUBSCRIPTIONS_PER_CONNECTION", 2)
    broadcaster = IpcEventBroadcaster()
    context, _writer = _connection()
    broadcaster.subscribe(context, ["run.*"])
    coordinator = _GatedWatermarkCoordinator()

    # 创建一个在受控 watermark 回调中完成的 durable prepare
    async def prepare() -> object:
        return await broadcaster.prepare_durable_subscription(
            cast(Any, coordinator),
            context,
            topics=["run.*"],
            stream_id="run:run-1",
            after_seq=0,
        )

    tasks = [asyncio.create_task(prepare()), asyncio.create_task(prepare())]
    await coordinator.all_entered.wait()
    coordinator.release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(isinstance(outcome, ConnectionError) for outcome in outcomes) == 1
    broadcaster.unsubscribe_all(context)
    await context.close("test complete")


# 功能：验证并发 durable prepare 不能越过全局 subscription 上限
# 设计：用三个连接隔离 per-connection cap，让两个 prepare 竞争唯一剩余的全局名额
async def test_concurrent_prepare_enforces_global_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(broadcaster_module, "MAX_TOTAL_SUBSCRIPTIONS", 2)
    broadcaster = IpcEventBroadcaster()
    contexts = [_connection()[0] for _ in range(3)]
    broadcaster.subscribe(contexts[0], ["run.*"])
    coordinator = _GatedWatermarkCoordinator()

    # 为指定连接创建受控 durable prepare
    async def prepare(context: ConnectionContext) -> object:
        return await broadcaster.prepare_durable_subscription(
            cast(Any, coordinator),
            context,
            topics=["run.*"],
            stream_id="run:run-1",
            after_seq=0,
        )

    tasks = [
        asyncio.create_task(prepare(contexts[1])),
        asyncio.create_task(prepare(contexts[2])),
    ]
    await coordinator.all_entered.wait()
    coordinator.release.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    assert sum(isinstance(outcome, ConnectionError) for outcome in outcomes) == 1
    for context in contexts:
        broadcaster.unsubscribe_all(context)
        await context.close("test complete")


# 功能：验证 replay 文件在 registration 后出现完整 corrupt row 时零 replay frame 并关闭连接
# 设计：watermark 指向合法 seq=1，再追加带换行坏行，activation 后等待真实 context terminal并断言 writer 空
async def test_replay_validation_failure_sends_zero_partial_frames(tmp_path: Path) -> None:
    run_path = tmp_path / "run-1"
    run_path.mkdir(parents=True)
    valid = {
        "schema_version": 2,
        "event_id": "evt-valid",
        "stream_id": "run:run-1",
        "seq": 1,
        "event": _event("valid").model_dump(mode="json"),
    }
    v2_path = run_path / "events.v2.jsonl"
    v2_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    await coordinator.register_run("run-1", run_path, session_id=None)
    context, writer = _connection()
    prepared = await broadcaster.prepare_durable_subscription(
        coordinator,
        context,
        topics=["run.*"],
        stream_id="run:run-1",
        after_seq=0,
    )
    with v2_path.open("ab") as file:
        file.write(b"{\n")

    await prepared.activation.on_written()
    while context.state is not ConnectionState.CLOSED:
        await asyncio.sleep(0)

    assert writer.frames == []
    assert not broadcaster.has_subscription(prepared.subscription_id)
    await coordinator.close()


# 功能：验证 overlapping run/session subscriptions 各自收到相同 event_id 与独立 stream seq
# 设计：同一 connection 注册两个 active scope并解析两帧，证明 server 不跨 subscription 去重
async def test_overlapping_stream_subscriptions_preserve_both_cursors(
    tmp_path: Path,
) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    session_path = tmp_path / "sess-1"
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run(
        "run-1",
        session_path / "runs" / "run-1",
        session_id="sess-1",
    )
    context, writer = _connection()
    broadcaster.subscribe(context, ["run.*"], scope="run:run-1")
    broadcaster.subscribe(context, ["run.*"], scope="session:sess-1")

    await coordinator.handle(_event("overlap"))
    await coordinator.flush_all()
    while len(writer.frames) < 2:
        await asyncio.sleep(0)
    await context.close("test complete")
    await coordinator.close()

    deliveries = [json.loads(frame) for frame in writer.frames]
    assert {delivery["stream_id"] for delivery in deliveries} == {
        "run:run-1",
        "session:sess-1",
    }
    assert {delivery["seq"] for delivery in deliveries} == {1}
    assert len({delivery["event_id"] for delivery in deliveries}) == 1


# 功能：验证一个 durable live 连接 enqueue 失败不会阻止后续健康订阅者收到同一事件
# 设计：先注册后关闭第一个真实 ConnectionContext，再发布一次 durable record并检查第二个 writer 的完整 frame
async def test_durable_live_fanout_isolates_failed_connection(tmp_path: Path) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    failed_context, _failed_writer = _connection()
    healthy_context, healthy_writer = _connection()
    broadcaster.subscribe(failed_context, ["run.*"], scope="run:run-1")
    broadcaster.subscribe(healthy_context, ["run.*"], scope="run:run-1")
    await failed_context.close("inject enqueue failure")

    await coordinator.handle(_event("fanout isolation"))
    await coordinator.flush_all()
    await asyncio.sleep(0)

    assert healthy_writer.frames
    envelope = json.loads(healthy_writer.frames[0])
    assert envelope["event"]["goal"] == "fanout isolation"
    assert envelope["delivery"] == "live"
    await healthy_context.close("test complete")
    await coordinator.close()


# 功能：验证 durable stream degraded 会移除其 subscription 并关闭所属连接
# 设计：预占单 run frame cap 后触发真实 coordinator callback，检查 broadcaster ownership 与 ConnectionContext 终态
async def test_degraded_stream_closes_owned_subscription(tmp_path: Path) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_stream_failure=broadcaster.fail_stream)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    context, _writer = _connection()
    sub_id = broadcaster.subscribe(context, ["run.*"], scope="run:run-1")
    state = cast(Any, coordinator)._streams["run:run-1"]
    state.reserved_frames = journal_module.JOURNAL_QUEUE_MAX_FRAMES

    with pytest.raises(JournalCapacityError, match="capacity"):
        await coordinator.handle(_event("overflow"))
    while context.state is not ConnectionState.CLOSED:
        await asyncio.sleep(0)

    assert not broadcaster.has_subscription(sub_id)
    await coordinator.close()


# 功能：验证 degraded durable stream 拒绝新 subscription，不暴露不连续 cursor
# 设计：先用真实 queue cap 将 stream 降级，再走 watermark registrar 路径断言零 subscription 提交
async def test_degraded_stream_rejects_new_durable_subscription(tmp_path: Path) -> None:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_stream_failure=broadcaster.fail_stream)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    state = cast(Any, coordinator)._streams["run:run-1"]
    state.reserved_frames = journal_module.JOURNAL_QUEUE_MAX_FRAMES
    with pytest.raises(JournalCapacityError, match="capacity"):
        await coordinator.handle(_event("degrade"))
    context, _writer = _connection()

    with pytest.raises(JournalError, match="degraded"):
        await broadcaster.prepare_durable_subscription(
            coordinator,
            context,
            topics=["run.*"],
            stream_id="run:run-1",
            after_seq=0,
        )

    assert cast(Any, broadcaster)._subscriptions == []
    await context.close("test complete")
    await coordinator.close()
