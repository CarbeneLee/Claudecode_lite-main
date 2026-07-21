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


# 功能：验证 7A legacy replay event frames 仍严格早于 subscribe success response
# 设计：先用真实 run 生成完整 journal，再用 raw TCP 按 NDJSON 顺序读取直到对应 response
async def test_legacy_replay_remains_before_subscribe_response(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    producer = SocketClient("127.0.0.1", free_port)
    await producer.connect()
    run_started = asyncio.Event()
    run_id_holder: list[str] = []

    # 等待 LLM 调用前同步 flush 的 started event，避免把外部 provider 延迟引入交付测试
    async def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            run_id_holder.append(str(event["run_id"]))
            run_started.set()

    producer.on_event(on_event)
    producer_loop = asyncio.create_task(producer.run_event_loop())
    try:
        await producer.send_command(
            "event.subscribe",
            {"topics": ["run.*"], "scope": "global"},
        )
        result = await producer.send_command(
            "agent.run",
            {"goal": "legacy ordering", "workspace_root": str(tmp_path.resolve())},
        )
        await asyncio.wait_for(run_started.wait(), timeout=5.0)
        assert result["run_id"] in run_id_holder
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
            "scope": "global",
            "replay_from_run": result["run_id"],
        },
    }
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()
    frames: list[dict[str, Any]] = []

    try:
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            assert line
            frame = json.loads(line)
            frames.append(frame)
            if frame.get("id") == request_id:
                break
    finally:
        writer.close()
        await writer.wait_closed()

    assert frames[-1]["id"] == request_id
    assert frames[-1]["result"]["replayed_count"] == len(frames) - 1
    assert all(frame.get("kind") == "event" for frame in frames[:-1])
    assert frames[:-1]
