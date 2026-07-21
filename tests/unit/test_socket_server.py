from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any, cast

import pytest

from kama_claude.core.transport.socket_server import (
    SocketServer,
    get_connection_context,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# 向真实 TCP writer 发送一条 JSON-RPC NDJSON 请求
async def _send_request(
    writer: asyncio.StreamWriter,
    method: str,
    params: dict[str, Any] | None = None,
) -> None:
    writer.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": method,
                "params": params or {},
            }
        ).encode()
        + b"\n"
    )
    await writer.drain()


# 功能：验证客户端断开后 SocketServer 按 ConnectionContext 清理全部 owned subscriptions
# 设计：内联 broadcaster 捕获 unsubscribe_all，并用 Event 锁定 cleanup 已在连接终态后执行
async def test_broadcaster_unsubscribe_called_on_disconnect() -> None:
    unsubscribed = asyncio.Event()

    class MockBroadcaster:
        # 记录 connection-scoped cleanup 调用
        def unsubscribe_all(self, context: object) -> None:
            unsubscribed.set()

    port = _free_port()
    server = SocketServer("127.0.0.1", port, broadcaster=MockBroadcaster())  # type: ignore[arg-type]
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await writer.wait_closed()

        await asyncio.wait_for(unsubscribed.wait(), timeout=2.0)
    finally:
        await server.stop()


# 功能：验证不传入 broadcaster 时 SocketServer 仍可正常启动和停止（backward-compatible 默认值）
# 设计：直接实例化 SocketServer(host, port)（无 broadcaster），start/stop 不抛异常即为通过；
#       回归测试确保新参数的默认值 None 不破坏现有调用方
async def test_no_broadcaster_server_starts_and_stops() -> None:
    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    await server.start()
    await server.stop()


# 功能：验证客户端 EOF 会取消并等待该连接仍在执行的 request handler
# 设计：用真实 TCP 和 Event barrier 阻塞 handler，断开后等待 CancelledError 路径而不依赖 sleep
async def test_disconnect_cancels_and_joins_blocked_request_task() -> None:
    entered = asyncio.Event()
    blocker = asyncio.Event()
    cancelled = asyncio.Event()

    # 阻塞到连接 cleanup 取消当前 request task
    async def blocked_handler(params: dict[str, Any]) -> dict[str, bool]:
        entered.set()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise asyncio.CancelledError

    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    server.register("test.blocked", blocked_handler)
    await server.start()
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)

    try:
        await _send_request(writer, "test.blocked")
        await entered.wait()
        writer.close()

        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    finally:
        blocker.set()
        await server.stop()


# 功能：验证 server.stop 会使所有活跃 connection-owned request tasks 到达终态
# 设计：保持客户端 socket 打开并阻塞 handler，由 stop 主动驱动连接清理和 task cancellation
async def test_server_stop_awaits_active_connection_cleanup() -> None:
    entered = asyncio.Event()
    blocker = asyncio.Event()
    cancelled = asyncio.Event()

    # 阻塞到 server.stop 取消当前 request task
    async def blocked_handler(params: dict[str, Any]) -> dict[str, bool]:
        entered.set()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise asyncio.CancelledError

    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    server.register("test.blocked", blocked_handler)
    await server.start()
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)

    try:
        await _send_request(writer, "test.blocked")
        await entered.wait()

        await server.stop()

        assert cancelled.is_set()
    finally:
        blocker.set()
        writer.close()


# 功能：验证 server.stop 返回前已删除所有 active connection 的 owned subscriptions
# 设计：用不响应 writer.close 的 fake reader 固定暂停 connection finally，隔离并锁定 stop 自身的同步 cleanup 责任
async def test_server_stop_removes_subscriptions_before_reader_loop_exits() -> None:
    class BlockingReader:
        # 初始化 reader 进入信号和 EOF gate
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.eof = asyncio.Event()

        # 阻塞读取直到测试显式释放 EOF
        async def readline(self) -> bytes:
            self.entered.set()
            await self.eof.wait()
            return b""

    class RecordingWriter:
        # 记录 close/wait_closed 而不改变 fake reader 状态
        def __init__(self) -> None:
            self.close_calls = 0

        # 提供 writer loop 所需的同步写接口
        def write(self, data: bytes) -> None:
            return None

        # 提供 writer loop 所需的异步 drain 接口
        async def drain(self) -> None:
            return None

        # 记录连接关闭但不释放 reader
        def close(self) -> None:
            self.close_calls += 1

        # 模拟底层 writer 已完成关闭
        async def wait_closed(self) -> None:
            return None

        # 返回稳定 peername
        def get_extra_info(self, name: str, default: object = None) -> object:
            if name == "peername":
                return ("127.0.0.1", 7002)
            return default

    class RecordingBroadcaster:
        # 初始化 connection cleanup 调用记录
        def __init__(self) -> None:
            self.unsubscribed: list[object] = []

        # 记录幂等的 connection-scoped unsubscribe_all
        def unsubscribe_all(self, context: object) -> None:
            self.unsubscribed.append(context)

    reader = BlockingReader()
    writer = RecordingWriter()
    broadcaster = RecordingBroadcaster()
    port = _free_port()
    server = SocketServer(
        "127.0.0.1",
        port,
        broadcaster=cast(Any, broadcaster),
    )
    await server.start()
    connection_task = asyncio.create_task(
        server._handle_connection(
            cast(asyncio.StreamReader, reader),
            cast(asyncio.StreamWriter, writer),
        )
    )
    await reader.entered.wait()

    try:
        await server.stop()
        calls_when_stop_returned = len(broadcaster.unsubscribed)
    finally:
        reader.eof.set()
        await connection_task

    assert calls_when_stop_returned == 1
    assert writer.close_calls == 1


# 功能：验证 writer failure 即使 reader 仍阻塞也会从统一 connection finalizer 删除订阅
# 设计：首行触发 subscribe-like handler，response drain 固定失败，第二次 readline 保持阻塞以排除 finally 代偿
async def test_writer_failure_removes_subscription_before_reader_loop_exits() -> None:
    class ScriptedReader:
        # 初始化首个请求与第二次读取 gate
        def __init__(self) -> None:
            self.calls = 0
            self.second_read_entered = asyncio.Event()
            self.eof = asyncio.Event()

        # 首次返回请求，随后阻塞到显式 EOF
        async def readline(self) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "writer-failure",
                        "method": "test.subscribe",
                        "params": {},
                    }
                ).encode() + b"\n"
            self.second_read_entered.set()
            await self.eof.wait()
            return b""

    class FailingWriter:
        # 初始化 connection terminal 信号
        def __init__(self) -> None:
            self.closed = asyncio.Event()

        # 接受完整 response frame
        def write(self, data: bytes) -> None:
            return None

        # 固定制造包含不可信文本的 delivery failure
        async def drain(self) -> None:
            raise RuntimeError("writer-secret")

        # 记录同步 close
        def close(self) -> None:
            return None

        # 标记 wait_closed terminal
        async def wait_closed(self) -> None:
            self.closed.set()

        # 返回稳定 peername
        def get_extra_info(self, name: str, default: object = None) -> object:
            if name == "peername":
                return ("127.0.0.1", 7003)
            return default

    class OwnedSubscriptionBroadcaster:
        # 初始化 active ownership 与 cleanup 信号
        def __init__(self) -> None:
            self.active: set[object] = set()
            self.cleaned = asyncio.Event()

        # 注册测试 connection ownership
        def subscribe(self, context: object) -> None:
            self.active.add(context)

        # 幂等删除 connection ownership
        def unsubscribe_all(self, context: object) -> None:
            self.active.discard(context)
            self.cleaned.set()

    reader = ScriptedReader()
    writer = FailingWriter()
    broadcaster = OwnedSubscriptionBroadcaster()
    subscribed = asyncio.Event()

    # 在真实 handler ContextVar 中登记当前 connection
    async def subscribe_handler(params: dict[str, Any]) -> dict[str, bool]:
        broadcaster.subscribe(get_connection_context())
        subscribed.set()
        return {"subscribed": True}

    server = SocketServer(
        "127.0.0.1",
        0,
        broadcaster=cast(Any, broadcaster),
    )
    server.register("test.subscribe", subscribe_handler)
    connection_task = asyncio.create_task(
        server._handle_connection(
            cast(asyncio.StreamReader, reader),
            cast(asyncio.StreamWriter, writer),
        )
    )

    try:
        await subscribed.wait()
        await reader.second_read_entered.wait()
        await writer.closed.wait()

        assert broadcaster.cleaned.is_set()
        assert broadcaster.active == set()
        assert not connection_task.done()
    finally:
        reader.eof.set()
        await connection_task


# 功能：验证未知 handler 异常返回安全内部错误且日志不泄漏路径或 secret
# 设计：通过真实 TCP request 抛出含敏感文本的 PermissionError，同时检查 response 和 caplog
async def test_unknown_handler_exception_is_redacted_from_log_and_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "startup-secret"
    private_path = "/private/journal"
    failure = PermissionError(f"{private_path} token={secret}")

    # 抛出固定异常对象以命中 SocketServer 的未知 handler 边界
    async def failing_handler(params: dict[str, Any]) -> dict[str, bool]:
        raise failure

    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    server.register("test.failure", failing_handler)
    await server.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    caplog.set_level(logging.ERROR, logger="kama_claude.core.transport.socket_server")

    try:
        await _send_request(writer, "test.failure")
        response = json.loads(await reader.readline())
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()

    assert response["error"]["message"] == "Internal error"
    assert secret not in json.dumps(response)
    assert private_path not in json.dumps(response)
    assert secret not in caplog.text
    assert private_path not in caplog.text
    assert "handler failed method=test.failure role=handler" in caplog.text
