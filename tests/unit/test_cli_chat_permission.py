from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kama_claude.cli.commands import chat as chat_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.transport.socket_client import EventDelivery

type DeliveryHandler = Callable[[EventDelivery], Any]


class _FakeClient:
    # 可控 chat 连接：记录命令、允许测试主动投递事件
    def __init__(self, *, daemon_id: str = "daemon-a") -> None:
        self.daemon_id = daemon_id
        self.handler: DeliveryHandler | None = None
        self.subscribed = asyncio.Event()
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def connect(self) -> None:
        return None

    def on_event(self, handler: Any) -> None:
        raise AssertionError("chat must use full delivery API")

    def on_delivery(self, handler: DeliveryHandler) -> None:
        self.handler = handler

    async def run_event_loop(self) -> None:
        await self.subscribed.wait()
        await asyncio.Event().wait()

    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.commands.append((method, params))
        if method == "session.create":
            return {"session_id": "sess-old", "status": "active"}
        if method == "event.subscribe":
            self.subscribed.set()
            return {
                "subscription_id": "sub-1",
                "stream_id": params["scope"],
                "accepted_after_seq": params["after_seq"],
                "high_watermark_seq": params["after_seq"],
                "daemon_instance_id": self.daemon_id,
            }
        if method == "session.send_message":
            return {"run_id": "run-test"}
        return {}

    async def close(self) -> None:
        self.closed = True

    async def push(self, event_type: str, **extra: Any) -> None:
        assert self.handler is not None
        await self.handler(
            EventDelivery(
                subscription_id="sub-1",
                delivery="live",
                event_id=f"event-{event_type}",
                stream_id="session:sess-old",
                seq=0,
                daemon_instance_id=self.daemon_id,
                event={"type": event_type, **extra},
            )
        )


# 功能：验证 run 激活期间输入的 y（权限提示尚未出现）被排队，并在下一条权限请求时自动生效
# 设计：readline 第二次调用返回 y 后主循环缓冲（active_run 为真、无 pending）；
#       第三次 readline 阻塞证明 y 已消费；此时才投递 permission.requested →
#       自动 respond 发出，y 不会作为消息发送
async def test_y_during_run_buffered_and_applied_to_next_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    lines = ["把任务做完", "y"]
    third_readline_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_readline(prompt: str) -> str:
        await client.subscribed.wait()
        if not lines:
            third_readline_started.set()
            await release.wait()
            raise EOFError
        return lines.pop(0)

    def make_client(host: str, port: int) -> _FakeClient:
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    chat_task = asyncio.create_task(chat_module._chat_async(KamaConfig()))

    # 等首条消息发出
    while not any(m == "session.send_message" for m, _ in client.commands):
        await asyncio.sleep(0.005)

    # run 开始（此时 y 尚未到达主循环 → 无 pending 权限 → 应缓冲而非当消息发送）
    await client.push("run.started", run_id="run-test")

    # 第三次 readline 开始 = y 已被消费并缓冲，主循环回到输入等待
    await asyncio.wait_for(third_readline_started.wait(), timeout=2)

    # 权限请求在 y 之后到达 → 排队决策自动批准，无需用户再输入
    await client.push(
        "permission.requested",
        run_id="run-test",
        tool_use_id="call-1",
        tool_name="bash",
        params={"command": "hostname"},
        param_preview="command='hostname'",
    )

    deadline = asyncio.get_running_loop().time() + 2
    while not any(m == "permission.respond" for m, _ in client.commands):
        assert asyncio.get_running_loop().time() < deadline, "auto respond not sent"
        await asyncio.sleep(0.005)

    release.set()
    await asyncio.wait_for(chat_task, timeout=5)
    assert chat_task.result() == 0

    # y 被排队，没有作为第二条消息发出
    sends = [p for m, p in client.commands if m == "session.send_message"]
    assert len(sends) == 1
    respond = next(
        (p for m, p in client.commands if m == "permission.respond"),
        None,
    )
    assert respond == {"tool_use_id": "call-1", "decision": "allow_once"}


# 功能：验证无激活 run 时 y 仍按普通消息发送（排队仅限 run 激活期间）
# 设计：订阅后立即输入 y，无 run.started 事件 → session.send_message content=y
async def test_y_without_active_run_sent_as_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    lines = ["y"]

    async def fake_readline(prompt: str) -> str:
        await client.subscribed.wait()
        if not lines:
            raise EOFError
        return lines.pop(0)

    def make_client(host: str, port: int) -> _FakeClient:
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    exit_code = await asyncio.wait_for(
        chat_module._chat_async(KamaConfig()), timeout=5
    )

    assert exit_code == 0
    sends = [p for m, p in client.commands if m == "session.send_message"]
    assert sends == [{"session_id": "sess-old", "content": "y"}]


class _HangingClient:
    # send_command 永不返回：用于验证 RPC 超时兜底
    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {}


# 功能：验证 RPC 响应在超时后按 disconnected 处理，主循环不会被无限冻结
# 设计：monkeypatch 超时为 0.05s，永不响应的 client → _send_rpc 在超时内返回 disconnected
async def test_send_rpc_timeout_returns_disconnected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_module, "_RPC_TIMEOUT_S", 0.05)

    result = await asyncio.wait_for(
        chat_module._send_rpc(_HangingClient(), "session.send_message", {}),
        timeout=2.0,
    )

    assert result.outcome == "disconnected"
