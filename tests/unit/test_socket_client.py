from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from kama_claude.core.transport.socket_client import EventDelivery, IpcError, SocketClient


async def _start_mock_server(
    handler: Any,
) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port: int = server.sockets[0].getsockname()[1]
    return server, port


# 功能：验证 send_command 向 mock server 发送 JSON-RPC 请求并正确解析响应 result
# 设计：用 asyncio.start_server + port 0 启动内存中的 mock server，避免依赖真实 daemon；
#       loop_task 并发运行 run_event_loop，使 send_command 的 future 能被 _dispatch 解析
async def test_send_command_returns_result() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"pong": True}}
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        result = await asyncio.wait_for(
            client.send_command("core.ping", {"client": "test"}),
            timeout=2.0,
        )
        assert result == {"pong": True}

        await loop_task
        await client.close()


# 功能：验证 server 返回 JSON-RPC error 时 send_command 抛出 IpcError 并携带正确错误码
# 设计：mock server 返回 error 对象（code=-32601），断言异常类型和 code 属性，确认客户端的错误路径处理
async def test_send_command_raises_ipc_error() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "error": {"code": -32601, "message": "Method not found"},
        }
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        with pytest.raises(IpcError) as exc_info:
            await asyncio.wait_for(
                client.send_command("core.nope", {}),
                timeout=2.0,
            )
        assert exc_info.value.code == -32601

        await loop_task
        await client.close()


# 功能：验证 server 推送 kind=event 的消息时，on_event 注册的 handler 能收到 event 字典
# 设计：server 先返回 RPC 响应（解除 send_command 的等待），再推送事件；用 asyncio.Event 等待 handler 被调用，
#       避免 sleep 轮询；断言 event 内容中的 type 字段是否正确
async def test_event_push_routed_to_handler() -> None:
    received_events: list[dict[str, Any]] = []
    push_done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"subscription_id": "sub-1"}}
        writer.write(json.dumps(resp).encode() + b"\n")
        push = {"kind": "event", "event": {"type": "run.started", "run_id": "r1"}}
        writer.write(json.dumps(push).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)

        async def collect(event_data: dict[str, Any]) -> None:
            received_events.append(event_data)
            push_done.set()

        client.on_event(collect)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        await asyncio.wait_for(
            client.send_command("event.subscribe", {"topics": ["run.*"]}),
            timeout=2.0,
        )
        await asyncio.wait_for(push_done.wait(), timeout=2.0)

        assert len(received_events) == 1
        assert received_events[0]["type"] == "run.started"

        await loop_task
        await client.close()


# 功能：验证 on_delivery 交付完整元数据且 on_event 仍按原顺序保真交付裸事件
# 设计：mock server 连续发送两个完整 envelope，分别断言类型字段、嵌套值与双回调顺序
async def test_full_delivery_api_preserves_metadata_and_raw_event_order() -> None:
    raw_events: list[dict[str, Any]] = []
    deliveries: list[EventDelivery] = []
    callback_order: list[tuple[str, int]] = []
    delivered = asyncio.Event()
    event_payloads = [
        {"type": "llm.token", "run_id": "r1", "token": "0", "meta": {"ok": False}},
        {"type": "llm.token", "run_id": "r1", "token": "", "meta": {"ok": True}},
    ]

    # 模拟服务端依次发送响应与两条完整 event delivery envelope
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        response = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "result": {
                "subscription_id": "sub-1",
                "daemon_instance_id": "daemon-1",
            },
        }
        writer.write(json.dumps(response).encode() + b"\n")
        for seq, event_payload in enumerate(event_payloads, start=1):
            push = {
                "kind": "event",
                "subscription_id": "sub-1",
                "delivery": "replay" if seq == 1 else "live",
                "event_id": f"evt-{seq}",
                "stream_id": "run:r1",
                "seq": seq,
                "daemon_instance_id": "daemon-1",
                "event": event_payload,
            }
            writer.write(json.dumps(push).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)

        # 收集兼容层裸事件并记录与完整 delivery 回调的交错顺序
        async def collect_raw(event_data: dict[str, Any]) -> None:
            raw_events.append(event_data)
            callback_order.append(("raw", len(raw_events)))
            if len(raw_events) == 2 and len(deliveries) == 2:
                delivered.set()

        # 收集完整 delivery metadata 并记录与裸事件回调的交错顺序
        async def collect_delivery(delivery: EventDelivery) -> None:
            deliveries.append(delivery)
            callback_order.append(("delivery", len(deliveries)))
            if len(raw_events) == 2 and len(deliveries) == 2:
                delivered.set()

        assert hasattr(client, "on_delivery")
        client.on_event(collect_raw)
        client.on_delivery(collect_delivery)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        await asyncio.wait_for(
            client.send_command("event.subscribe", {"topics": ["llm.token"]}),
            timeout=2.0,
        )
        await asyncio.wait_for(delivered.wait(), timeout=2.0)

        assert raw_events == event_payloads
        assert callback_order == [
            ("raw", 1),
            ("delivery", 1),
            ("raw", 2),
            ("delivery", 2),
        ]
        assert [delivery.subscription_id for delivery in deliveries] == ["sub-1", "sub-1"]
        assert [delivery.delivery for delivery in deliveries] == ["replay", "live"]
        assert [delivery.event_id for delivery in deliveries] == ["evt-1", "evt-2"]
        assert [delivery.stream_id for delivery in deliveries] == ["run:r1", "run:r1"]
        assert [delivery.seq for delivery in deliveries] == [1, 2]
        assert [delivery.daemon_instance_id for delivery in deliveries] == [
            "daemon-1",
            "daemon-1",
        ]
        assert [delivery.event for delivery in deliveries] == event_payloads

        await loop_task
        await client.close()


# 功能：验证 server 关闭连接后 run_event_loop 正常退出（不挂起）
# 设计：mock server 立即关闭写流，client readline 收到空字节后退出循环；
#       用 asyncio.wait_for 设置超时，防止 loop 意外挂起导致测试卡死
async def test_run_event_loop_exits_on_server_close() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        await asyncio.wait_for(client.run_event_loop(), timeout=2.0)
        await client.close()
