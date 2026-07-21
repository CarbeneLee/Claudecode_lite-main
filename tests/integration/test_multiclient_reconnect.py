from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from kama_claude.core.app import CoreApp
from kama_claude.core.bus.events import RunFinishedEvent, RunStartedEvent
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.session.manager import SessionManager
from kama_claude.core.session.store import SessionStore
from kama_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from kama_claude.core.transport.socket_server import SocketServer


# 申请一个只用于当前测试进程的 loopback 端口
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


# 发送固定 event.subscribe JSON-RPC request
async def _subscribe(writer: asyncio.StreamWriter, run_id: str) -> None:
    writer.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "subscribe-1",
                "method": "event.subscribe",
                "params": {
                    "topics": ["run.*"],
                    "scope": f"run:{run_id}",
                    "after_seq": 0,
                },
            }
        ).encode()
        + b"\n"
    )
    await writer.drain()


# 构建只注册 subscribe handler 的真实 TCP server 与 v2 journal 组合
async def _server(
    tmp_path: Path,
) -> tuple[SocketServer, EventJournalCoordinator, str, int]:
    broadcaster = IpcEventBroadcaster()
    coordinator = EventJournalCoordinator(on_durable=broadcaster.publish_durable)
    run_id = "run-response-first"
    await coordinator.register_run(run_id, tmp_path / run_id, session_id=None)
    await coordinator.handle(
        RunStartedEvent(
            run_id=run_id,
            goal="history",
            ts="2026-07-21T00:00:00Z",
        )
    )
    await coordinator.flush_all()
    app = CoreApp()
    app._broadcaster = broadcaster
    app._journal = coordinator
    port = _free_port()
    server = SocketServer("127.0.0.1", port, broadcaster)
    server.register("event.subscribe", app._subscribe_handler)
    await server.start()
    return server, coordinator, run_id, port


# 功能：验证真实 TCP 中 subscribe success frame 严格先于第一条 replay EventPushEnvelope
# 设计：同一 socket 连续读取两条 NDJSON，第一条必须含 result/watermark，第二条才允许是 replay delivery
async def test_subscribe_response_is_written_before_first_replay_frame(
    tmp_path: Path,
) -> None:
    server, coordinator, run_id, port = await _server(tmp_path)
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await _subscribe(writer, run_id)

        response = json.loads(await reader.readline())
        replay = json.loads(await reader.readline())

        assert response["id"] == "subscribe-1"
        assert response["result"]["stream_id"] == f"run:{run_id}"
        assert response["result"]["high_watermark_seq"] == 1
        assert replay["kind"] == "event"
        assert replay["delivery"] == "replay"
        assert replay["seq"] == 1
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await coordinator.close()


# 功能：验证 replay reader 被阻塞时 subscribe response 仍已 written 且 event queue 尚无 replay
# 设计：用 Event gate 包住 coordinator.read_replay，先读取响应并确认 reader entered，再释放 gate读取首事件
async def test_slow_replay_reader_does_not_delay_subscribe_response(
    tmp_path: Path,
) -> None:
    server, coordinator, run_id, port = await _server(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = coordinator.read_replay

    # 在真实 replay reader 前设置 gate，分离 response written 与磁盘读取时序
    async def gated_read(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    coordinator.read_replay = cast(Any, gated_read)
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        await _subscribe(writer, run_id)

        response = json.loads(await reader.readline())
        await entered.wait()

        assert response["result"]["subscription_id"].startswith("sub-")
        assert response["result"]["high_watermark_seq"] == 1
        release.set()
        replay = json.loads(await reader.readline())
        assert replay["delivery"] == "replay"
    finally:
        release.set()
        writer.close()
        await writer.wait_closed()
        await server.stop()
        await coordinator.close()


# 功能：验证 daemon shutdown 先取消并等待 connection request 的 terminal publish，再关闭 journal
# 设计：真实 TCP handler 在 cancellation 中写 run.finished；调用 CoreApp shutdown 后检查 terminal watermark与 CLOSED lifecycle
async def test_shutdown_drains_request_terminal_before_journal_close(
    tmp_path: Path,
) -> None:
    run_id = "run-shutdown"
    coordinator = EventJournalCoordinator()
    await coordinator.register_run(run_id, tmp_path / run_id, session_id=None)
    app = CoreApp()
    app._journal = coordinator
    entered = asyncio.Event()
    port = _free_port()
    server = SocketServer("127.0.0.1", port)

    # cancellation handler 模拟 session request 中 Runner 的 terminal journal barrier
    async def cancellable_handler(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await coordinator.handle(
                RunFinishedEvent(
                    run_id=run_id,
                    status="failed",
                    steps=0,
                    reason="cancelled",
                    ts="2026-07-21T00:00:00Z",
                )
            )
            raise

    server.register("test.block", cancellable_handler)
    await server.start()
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "shutdown-1",
                "method": "test.block",
                "params": {},
            }
        ).encode()
        + b"\n"
    )
    await writer.drain()
    await entered.wait()

    await app._shutdown(server)

    assert coordinator.high_watermark(f"run:{run_id}") == 1
    assert coordinator.stream_lifecycle(f"run:{run_id}").value == "closed"
    writer.close()
    await writer.wait_closed()


# 功能：验证 agent.run response 只在 run durable owner 完成注册后返回
# 设计：仅阻塞第二次（run）现有文件校验，用 thread gate 证明 handler 不会提前暴露 run_id
async def test_agent_run_response_waits_for_run_stream_registration(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original = EventJournalCoordinator._validate_existing_stream

    # 保留 session 注册快速路径，只在 run owner 校验处停顿
    def gated_validate(
        stream_id: str,
        path: Path,
        legacy_path: Path | None,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            release.wait(timeout=5)
        return original(stream_id, path, legacy_path)

    class _Runner:
        # 提供 SessionManager 需要的最小 runner 边界
        async def run_and_capture(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        EventJournalCoordinator,
        "_validate_existing_stream",
        staticmethod(gated_validate),
    )
    bus = CoreApp()._bus
    coordinator = EventJournalCoordinator()
    bus.subscribe(coordinator.handle)
    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory=lambda workspace_root: cast(Any, _Runner()),
        bus=bus,
        journal=coordinator,
    )
    app = CoreApp()
    app._bus = bus
    app._journal = coordinator
    app._sessions = manager
    handler = asyncio.create_task(
        app._agent_run_handler(
            {"goal": "registration handshake", "workspace_root": str(tmp_path)}
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)

    try:
        assert not handler.done()
    finally:
        release.set()
    result = await handler

    assert coordinator.has_stream(f"run:{result.run_id}")
    if app._running_runs:
        await asyncio.gather(*app._running_runs)
    await coordinator.close()


# 功能：验证 agent.run 注册等待期取消会回收 run task 并终结未暴露 session
# 设计：用 thread gate 卡住 run 文件校验，取消 handler 后断言等待底层终态且 session journal closed
async def test_agent_run_registration_cancellation_closes_unexposed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    original = EventJournalCoordinator._validate_existing_stream

    # 只阻塞 run owner 校验，保留 session owner 的真实 durable lifecycle
    def gated_validate(
        stream_id: str,
        path: Path,
        legacy_path: Path | None,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            release.wait(timeout=5)
        return original(stream_id, path, legacy_path)

    class _Runner:
        # 若注册取消正确，runner 不应被调用
        async def run_and_capture(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("runner must not start before registration")

    monkeypatch.setattr(
        EventJournalCoordinator,
        "_validate_existing_stream",
        staticmethod(gated_validate),
    )
    app = CoreApp()
    coordinator = EventJournalCoordinator()
    app._bus.subscribe(coordinator.handle)
    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory=lambda workspace_root: cast(Any, _Runner()),
        bus=app._bus,
        journal=coordinator,
    )
    app._journal = coordinator
    app._sessions = manager
    handler = asyncio.create_task(
        app._agent_run_handler(
            {"goal": "cancel registration", "workspace_root": str(tmp_path)}
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    session = next(iter(cast(Any, manager)._sessions.values()))
    primary_cancel = object()
    handler.cancel(primary_cancel)
    await asyncio.sleep(0)

    try:
        assert not handler.done()
        handler.cancel(object())
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await handler

    assert caught.value.args == (primary_cancel,)
    assert not app._running_runs
    assert session.status == "closed"
    assert coordinator.stream_lifecycle(f"session:{session.id}").value == "closed"
    assert cast(Any, coordinator).identities._run_sessions == {}
    await coordinator.close()


# 功能：验证 run owner 注册异常原样传播并终结未暴露 session
# 设计：在第二次 stream 校验抛固定异常对象，同时锁定 identity、run mapping 和 session terminal
async def test_agent_run_registration_failure_closes_unexposed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = PermissionError("registration-secret")
    calls = 0
    original = EventJournalCoordinator._validate_existing_stream

    # 保留 session 注册，仅在 run owner 校验处抛出指定异常
    def failing_validate(
        stream_id: str,
        path: Path,
        legacy_path: Path | None,
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original(stream_id, path, legacy_path)

    class _Runner:
        # 注册失败时 runner 不应进入执行路径
        async def run_and_capture(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("runner must not start after registration failure")

    monkeypatch.setattr(
        EventJournalCoordinator,
        "_validate_existing_stream",
        staticmethod(failing_validate),
    )
    app = CoreApp()
    coordinator = EventJournalCoordinator()
    app._bus.subscribe(coordinator.handle)
    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory=lambda workspace_root: cast(Any, _Runner()),
        bus=app._bus,
        journal=coordinator,
    )
    app._journal = coordinator
    app._sessions = manager

    with pytest.raises(PermissionError) as caught:
        await app._agent_run_handler(
            {"goal": "fail registration", "workspace_root": str(tmp_path)}
        )

    session = next(iter(cast(Any, manager)._sessions.values()))
    assert caught.value is failure
    assert not app._running_runs
    assert session.status == "closed"
    assert coordinator.stream_lifecycle(f"session:{session.id}").value == "closed"
    assert cast(Any, coordinator).identities._run_sessions == {}
    await coordinator.close()
