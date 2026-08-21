from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from kama_claude.cli.commands import chat as chat_module
from kama_claude.core.config import KamaConfig


# 功能：验证 kama chat 创建 Session 时发送 cwd，后续消息不重复 workspace_root
# 设计：用一条用户输入驱动真实 chat 循环，捕获 session.create 与 send_message 的边界差异
async def test_chat_sends_workspace_only_when_creating_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, dict[str, Any]]] = []
    inputs = iter(["hello"])

    class _Client:
        # 保留与真实 SocketClient 相同的构造界面
        def __init__(self, host: str, port: int) -> None:
            return None

        # 模拟成功建立 IPC 连接
        async def connect(self) -> None:
            return None

        # chat 本测试不需要消费 daemon 事件
        def on_event(self, handler: Any) -> None:
            return None

        # chat reconnect 后改用 full delivery API
        def on_delivery(self, handler: Any) -> None:
            return None

        # 保持事件循环挂起直到 chat 清理任务
        async def run_event_loop(self) -> None:
            await asyncio.Event().wait()

        # 记录 chat 发出的所有 IPC 命令
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            commands.append((method, params))
            if method == "session.create":
                return {"session_id": "sess-test", "status": "active"}
            if method == "event.subscribe":
                return {
                    "subscription_id": "sub-test",
                    "stream_id": "session:sess-test",
                    "accepted_after_seq": 0,
                    "high_watermark_seq": 0,
                    "daemon_instance_id": "daemon-a",
                }
            if method == "session.get_agent_mode":
                return {"agent_mode": "direct", "revision": 0}
            if method == "session.send_message":
                return {"run_id": "run-test"}
            return {}

        # 模拟关闭 IPC 连接
        async def close(self) -> None:
            return None

    # 返回一条消息后用 EOF 结束 chat 循环
    async def fake_readline(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", _Client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    create_params = next(
        params for method, params in commands if method == "session.create"
    )
    message_params = next(
        params for method, params in commands if method == "session.send_message"
    )
    assert create_params == {
        "mode": "chat",
        "workspace_root": str(workspace.resolve()),
    }
    assert message_params == {"session_id": "sess-test", "content": "hello"}
    assert "workspace_root" not in message_params
    assert any(
        method == "event.subscribe"
        and params["scope"] == "session:sess-test"
        and params["after_seq"] == 0
        for method, params in commands
    )
