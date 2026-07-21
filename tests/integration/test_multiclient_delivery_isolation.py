from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from kama_claude.core.transport.socket_client import SocketClient


# 功能：验证 event.unsubscribe 的 ownership 不允许 A 删除 B 且断开 A 后 B 继续收事件
# 设计：使用两个真实 TCP SocketClient 和两个连续 run.started，覆盖协议布尔语义与连接清理隔离
async def test_subscription_ownership_and_disconnect_isolation(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    client_a = SocketClient("127.0.0.1", free_port)
    client_b = SocketClient("127.0.0.1", free_port)
    await client_a.connect()
    await client_b.connect()
    received_b: list[str] = []
    two_events = asyncio.Event()

    # 收集 B 的两个不同 run.started，证明 A 断开不影响 B
    async def on_b(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            received_b.append(str(event["run_id"]))
            if len(received_b) == 2:
                two_events.set()

    client_b.on_event(on_b)
    loop_a = asyncio.create_task(client_a.run_event_loop())
    loop_b = asyncio.create_task(client_b.run_event_loop())

    try:
        result_a = await client_a.send_command(
            "event.subscribe",
            {"topics": ["run.*"], "scope": "global"},
        )
        result_b = await client_b.send_command(
            "event.subscribe",
            {"topics": ["run.*"], "scope": "global"},
        )

        foreign = await client_a.send_command(
            "event.unsubscribe",
            {"subscription_id": result_b["subscription_id"]},
        )
        own_first = await client_a.send_command(
            "event.unsubscribe",
            {"subscription_id": result_a["subscription_id"]},
        )
        own_second = await client_a.send_command(
            "event.unsubscribe",
            {"subscription_id": result_a["subscription_id"]},
        )
        assert foreign == {"removed": False}
        assert own_first == {"removed": True}
        assert own_second == {"removed": False}

        await client_b.send_command(
            "agent.run",
            {"goal": "before A disconnect", "workspace_root": str(tmp_path.resolve())},
        )
        await client_a.close()
        await client_b.send_command(
            "agent.run",
            {"goal": "after A disconnect", "workspace_root": str(tmp_path.resolve())},
        )

        await asyncio.wait_for(two_events.wait(), timeout=5.0)
        assert received_b[0] != received_b[1]
    finally:
        loop_a.cancel()
        loop_b.cancel()
        await asyncio.gather(loop_a, loop_b, return_exceptions=True)
        await client_a.close()
        await client_b.close()


# 功能：验证 7B replay event frames 严格晚于包含 high watermark 的 subscribe success response
# 设计：run 创建后立即订阅其 durable stream，让 high=0 的 live 与 high>0 的 replay 分支共用响应屏障
async def test_durable_replay_starts_after_subscribe_response(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    producer = SocketClient("127.0.0.1", free_port)
    await producer.connect()
    producer_loop = asyncio.create_task(producer.run_event_loop())
    try:
        result = await producer.send_command(
            "agent.run",
            {"goal": "durable response ordering", "workspace_root": str(tmp_path.resolve())},
        )
    finally:
        producer_loop.cancel()
        await asyncio.gather(producer_loop, return_exceptions=True)
        await producer.close()

    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    request_id = "legacy-order"
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "event.subscribe",
        "params": {
            "topics": ["run.*"],
            "scope": f"run:{result['run_id']}",
            "after_seq": 0,
        },
    }
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()
    try:
        response_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert response_line
        response = json.loads(response_line)
        frame = json.loads(await asyncio.wait_for(reader.readline(), timeout=5.0))
    finally:
        writer.close()
        await writer.wait_closed()

    assert response["id"] == request_id
    assert response["result"]["stream_id"] == f"run:{result['run_id']}"
    high_watermark = response["result"]["high_watermark_seq"]
    assert response["result"]["replayed_count"] == 0
    assert frame.get("kind") == "event"
    assert frame.get("stream_id") == f"run:{result['run_id']}"
    assert frame.get("seq") == 1
    assert frame.get("event", {}).get("type") == "run.started"
    assert frame.get("delivery") == ("replay" if high_watermark else "live")
