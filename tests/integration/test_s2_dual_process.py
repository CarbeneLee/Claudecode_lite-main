from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from kama_claude.core.transport.socket_client import SocketClient


# 功能：验证 agent.run 命令返回非空 run_id，且 daemon 随即广播 run.started 事件
# 设计：用 SocketClient 封装 IPC 层，asyncio.Event 等待事件而非轮询，
#       timeout=5s 防测试挂起；run.started 在 LLM 调用前触发，无需真实 API Key
async def test_agent_run_returns_run_id_and_emits_started(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()

    started_event: asyncio.Event = asyncio.Event()
    received: dict[str, Any] = {}

    async def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            received.update(event)
            started_event.set()

    client.on_event(on_event)
    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        await client.send_command("event.subscribe", {"topics": ["run.*"], "scope": "global"})
        result = await client.send_command(
            "agent.run",
            {"goal": "hello", "workspace_root": str(tmp_path.resolve())},
        )

        assert result.get("run_id"), "run_id must be non-empty"
        returned_run_id: str = result["run_id"]

        await asyncio.wait_for(started_event.wait(), timeout=5.0)
        assert received.get("run_id") == returned_run_id
        assert received.get("goal") == "hello"
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
        await client.close()


# 功能：验证两个独立客户端同时订阅后，其中一个触发 agent.run，两个都能收到 run.started 广播
# 设计：两个 SocketClient 并行等待事件（asyncio.gather），确认 IpcEventBroadcaster 的扇出语义；
#       不需要两个客户端都发命令，只验证广播覆盖所有订阅者
async def test_two_clients_both_receive_broadcast(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    client1 = SocketClient("127.0.0.1", free_port)
    client2 = SocketClient("127.0.0.1", free_port)
    await client1.connect()
    await client2.connect()

    event1: asyncio.Event = asyncio.Event()
    event2: asyncio.Event = asyncio.Event()

    async def on_event1(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            event1.set()

    async def on_event2(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            event2.set()

    client1.on_event(on_event1)
    client2.on_event(on_event2)

    loop1 = asyncio.create_task(client1.run_event_loop())
    loop2 = asyncio.create_task(client2.run_event_loop())

    try:
        await client1.send_command("event.subscribe", {"topics": ["run.*"], "scope": "global"})
        await client2.send_command("event.subscribe", {"topics": ["run.*"], "scope": "global"})
        await client1.send_command(
            "agent.run",
            {"goal": "broadcast test", "workspace_root": str(tmp_path.resolve())},
        )

        await asyncio.wait_for(
            asyncio.gather(event1.wait(), event2.wait()),
            timeout=5.0,
        )
    finally:
        loop1.cancel()
        loop2.cancel()
        await asyncio.gather(loop1, loop2, return_exceptions=True)
        await client1.close()
        await client2.close()


# 功能：验证客户端断开后按 run 重连，先收到 watermark 响应再收到真实回放事件
# 设计：两个客户分别触发和回放，用 Event 等待目标 run 帧而不使用固定延时或计划数量
async def test_disconnect_and_replay_from_run(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    client1 = SocketClient("127.0.0.1", free_port)
    await client1.connect()

    started_event: asyncio.Event = asyncio.Event()
    run_id = ""

    # 只接受目标 durable run stream 的 run.started 帧
    async def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started" and event.get("run_id") == run_id:
            started_event.set()

    client1.on_event(on_event)
    loop1 = asyncio.create_task(client1.run_event_loop())

    try:
        run_result = await client1.send_command(
            "agent.run",
            {"goal": "replay test", "workspace_root": str(tmp_path.resolve())},
        )
        run_id = run_result["run_id"]
        await client1.send_command(
            "event.subscribe",
            {
                "topics": ["run.*"],
                "scope": f"run:{run_id}",
                "after_seq": 0,
            },
        )
        await asyncio.wait_for(started_event.wait(), timeout=5.0)
    finally:
        loop1.cancel()
        await asyncio.gather(loop1, return_exceptions=True)
        await client1.close()

    client2 = SocketClient("127.0.0.1", free_port)
    await client2.connect()
    replayed_event = asyncio.Event()
    replayed: list[dict[str, Any]] = []

    # 记录与目标 run 匹配的真实回放帧
    async def on_replayed_event(event: dict[str, Any]) -> None:
        if event.get("run_id") == run_id and event.get("type", "").startswith("run."):
            replayed.append(event)
            replayed_event.set()

    client2.on_event(on_replayed_event)
    loop2 = asyncio.create_task(client2.run_event_loop())

    try:
        result = await client2.send_command(
            "event.subscribe",
            {
                "topics": ["run.*"],
                "scope": "global",
                "replay_from_run": run_id,
            },
        )
        assert result["replayed_count"] == 0
        assert result["stream_id"] == f"run:{run_id}"
        assert result["accepted_after_seq"] == 0
        assert result["high_watermark_seq"] >= 1
        await asyncio.wait_for(replayed_event.wait(), timeout=5.0)
        assert replayed
    finally:
        loop2.cancel()
        await asyncio.gather(loop2, return_exceptions=True)
        await client2.close()
