from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from kama_claude.cli.commands import chat as chat_module
from kama_claude.cli.commands import run as run_module
from kama_claude.core.config import KamaConfig

type EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


# 功能：验证 kama run 在 agent.run 中发送客户端 canonical cwd
# 设计：替换真实 SocketClient 并主动发出 run.finished，捕获完整公开 IPC 调用参数
async def test_run_sends_canonical_client_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        # 初始化 fake client 的事件回调引用
        def __init__(self, host: str, port: int) -> None:
            self.handler: EventHandler | None = None

        # 模拟成功建立 IPC 连接
        async def connect(self) -> None:
            return None

        # 记录 CLI 注册的事件回调
        def on_event(self, handler: EventHandler) -> None:
            self.handler = handler

        # 保持事件循环挂起直到被 CLI 取消
        async def run_event_loop(self) -> None:
            await asyncio.Event().wait()

        # 记录命令并在 agent.run 后发出完成事件
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            commands.append((method, params))
            if method == "agent.run":
                assert self.handler is not None
                await self.handler(
                    {"type": "run.finished", "status": "success", "steps": 1}
                )
                return {"run_id": "run-test"}
            return {"subscription_id": "sub-test"}

        # 模拟关闭 IPC 连接
        async def close(self) -> None:
            return None

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", _Client)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 0
    agent_params = next(params for method, params in commands if method == "agent.run")
    assert agent_params == {
        "goal": "inspect",
        "workspace_root": str(workspace.resolve()),
    }


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
        def on_event(self, handler: EventHandler) -> None:
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
