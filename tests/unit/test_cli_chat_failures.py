from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import pytest

from kama_claude.cli.commands import chat as chat_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.transport.socket_client import EventDelivery, IpcError

type DeliveryHandler = Callable[[EventDelivery], Awaitable[None]]
type FailureKind = Literal["cancelled", "reset"]

_TRANSPORT_SECRET = "secret-transport-detail"


# 模拟 SocketClient 在 EOF 时取消 pending RPC future 或 write/drain reset
async def _raise_transport_failure(kind: FailureKind) -> dict[str, Any]:
    if kind == "reset":
        raise ConnectionResetError(_TRANSPORT_SECRET)
    pending: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    pending.cancel()
    return await pending


class _BaseClient:
    # 初始化可控 chat client 的 delivery 与命令状态
    def __init__(self, index: int, daemon_id: str = "daemon-a") -> None:
        self.index = index
        self.daemon_id = daemon_id
        self.handler: DeliveryHandler | None = None
        self.subscribed = asyncio.Event()
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    # 模拟连接成功
    async def connect(self) -> None:
        return None

    # 保存 full delivery handler
    def on_delivery(self, handler: DeliveryHandler) -> None:
        self.handler = handler

    # 默认保持读循环存活直到 cleanup
    async def run_event_loop(self) -> None:
        await self.subscribed.wait()
        await asyncio.Event().wait()

    # 返回完整 create/subscribe 响应并记录交互命令
    async def send_command(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.commands.append((method, dict(params)))
        if method == "session.create":
            return {"session_id": "sess-new", "status": "active"}
        if method == "event.subscribe":
            self.subscribed.set()
            return {
                "subscription_id": f"sub-{self.index}",
                "stream_id": params["scope"],
                "accepted_after_seq": params["after_seq"],
                "high_watermark_seq": params["after_seq"],
                "daemon_instance_id": self.daemon_id,
            }
        if method == "session.send_message":
            return {"run_id": "run-test"}
        return {}

    # 标记连接已关闭
    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("operation", ["create", "subscribe"])
@pytest.mark.parametrize("failure_kind", ["cancelled", "reset"])
# 功能：验证 initial create/subscribe 的 EOF 与 write reset 均进入有界安全恢复
# 设计：首 client 在指定 pending RPC 失败，第二 client 记录恢复首命令以区分 create no-retry 与订阅保守 fallback
async def test_chat_initial_handshake_transport_failure_recovers_safely(
    operation: Literal["create", "subscribe"],
    failure_kind: FailureKind,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_BaseClient] = []
    failure_started = asyncio.Event()
    loop_exited = asyncio.Event()
    second_subscribed = asyncio.Event()
    create_count = 0

    class _HandshakeClient(_BaseClient):
        # 首连接等目标 RPC 启动后 EOF，恢复连接保持存活
        async def run_event_loop(self) -> None:
            if self.index > 0:
                await self.subscribed.wait()
                await asyncio.Event().wait()
                return
            await failure_started.wait()
            loop_exited.set()

        # 在指定 initial phase 注入 transport failure
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_count
            self.commands.append((method, dict(params)))
            if method == "session.create":
                create_count += 1
                if self.index == 0 and operation == "create":
                    failure_started.set()
                    await loop_exited.wait()
                    return await _raise_transport_failure(failure_kind)
                session_id = "sess-old" if create_count == 1 else "sess-new"
                return {"session_id": session_id, "status": "active"}
            if method == "event.subscribe":
                if self.index == 0 and operation == "subscribe":
                    failure_started.set()
                    await loop_exited.wait()
                    return await _raise_transport_failure(failure_kind)
                self.subscribed.set()
                if self.index > 0:
                    second_subscribed.set()
                return {
                    "subscription_id": f"sub-{self.index}",
                    "stream_id": params["scope"],
                    "accepted_after_seq": params["after_seq"],
                    "high_watermark_seq": params["after_seq"],
                    "daemon_instance_id": "daemon-a",
                }
            return {}

    # 创建握手失败前后的 client
    def make_client(host: str, port: int) -> _BaseClient:
        client = _HandshakeClient(len(clients))
        clients.append(client)
        return client

    # 恢复订阅完成后结束交互
    async def fake_readline(prompt: str) -> str:
        await second_subscribed.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    captured = capsys.readouterr()
    assert _TRANSPORT_SECRET not in captured.out
    assert _TRANSPORT_SECRET not in captured.err
    if operation == "create":
        assert exit_code == 1
        assert len(clients) == 1
        assert sum(
            method == "session.create"
            for client in clients
            for method, _params in client.commands
        ) == 1
        assert clients[0].closed is True
        assert "session creation outcome unknown" in captured.err
    else:
        assert exit_code == 0
        assert len(clients) == 2
        assert clients[1].commands[0][0] == "session.create"


@pytest.mark.parametrize("operation", ["create", "subscribe"])
# 功能：验证 daemon fallback 的 create/subscribe EOF 会在新连接恢复正确 phase
# 设计：daemon-b fallback 中注入 pending cancellation，最终 client 首命令证明新建视图或恢复新 session stream
async def test_chat_fallback_handshake_eof_resumes_correct_phase(
    operation: Literal["create", "subscribe"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_BaseClient] = []
    first_disconnected = asyncio.Event()
    fallback_failure_started = asyncio.Event()
    fallback_loop_exited = asyncio.Event()
    final_subscribed = asyncio.Event()
    create_count = 0

    class _FallbackClient(_BaseClient):
        # 首连接握手后 EOF，fallback 连接在目标 RPC 时 EOF，最终连接保持存活
        async def run_event_loop(self) -> None:
            if self.index == 0:
                await self.subscribed.wait()
                first_disconnected.set()
                return
            if self.index == 1:
                await fallback_failure_started.wait()
                fallback_loop_exited.set()
                return
            await self.subscribed.wait()
            await asyncio.Event().wait()

        # 构造 daemon change 并在 fallback 指定 phase 取消 pending RPC
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_count
            self.commands.append((method, dict(params)))
            if method == "session.create":
                create_count += 1
                if self.index == 1 and operation == "create":
                    fallback_failure_started.set()
                    await fallback_loop_exited.wait()
                    return await _raise_transport_failure("cancelled")
                session_id = "sess-old" if create_count == 1 else "sess-new"
                return {"session_id": session_id, "status": "active"}
            if method == "event.subscribe":
                prior_subscriptions = sum(
                    command == "event.subscribe"
                    for command, _command_params in self.commands[:-1]
                )
                if self.index == 1 and operation == "subscribe" and prior_subscriptions == 1:
                    fallback_failure_started.set()
                    await fallback_loop_exited.wait()
                    return await _raise_transport_failure("cancelled")
                self.subscribed.set()
                if self.index == 2:
                    final_subscribed.set()
                daemon_id = "daemon-a" if self.index == 0 else "daemon-b"
                return {
                    "subscription_id": f"sub-{self.index}-{prior_subscriptions}",
                    "stream_id": params["scope"],
                    "accepted_after_seq": params["after_seq"],
                    "high_watermark_seq": params["after_seq"],
                    "daemon_instance_id": daemon_id,
                }
            return {}

    # 创建 initial、fallback-failure 与最终恢复 client
    def make_client(host: str, port: int) -> _BaseClient:
        daemon_id = "daemon-a" if not clients else "daemon-b"
        client = _FallbackClient(len(clients), daemon_id)
        clients.append(client)
        return client

    # 最终恢复订阅后结束交互
    async def fake_readline(prompt: str) -> str:
        await final_subscribed.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert first_disconnected.is_set()
    if operation == "create":
        assert exit_code == 1
        assert len(clients) == 2
        assert sum(
            method == "session.create"
            for client in clients
            for method, _params in client.commands
        ) == 2
        assert clients[1].closed is True
        assert "session creation outcome unknown" in capsys.readouterr().err
    else:
        assert exit_code == 0
        assert len(clients) == 3
        assert clients[2].commands[0][0] == "event.subscribe"
        assert clients[2].commands[0][1]["scope"] == "session:sess-new"


# 功能：验证 fresh-session subscribe 持续 IpcError 时只允许一次 fallback
# 设计：第三次 create 主动暴露无界循环，断言真实状态机在两次 create/subscribe 后稳定非零
async def test_chat_persistent_subscribe_ipc_error_has_fallback_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    create_calls = 0
    subscribe_calls = 0

    class _PersistentUnavailableClient(_BaseClient):
        # 每次 create 返回不同 session，第三次表明 fallback 已越界
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_calls, subscribe_calls
            self.commands.append((method, dict(params)))
            if method == "session.create":
                create_calls += 1
                if create_calls > 2:
                    raise AssertionError("unbounded session fallback")
                return {"session_id": f"sess-{create_calls}", "status": "active"}
            if method == "event.subscribe":
                subscribe_calls += 1
                raise IpcError(-32000, "secret-persistent-unavailable")
            return {}

    client = _PersistentUnavailableClient(0)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", lambda host, port: client)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 1
    assert create_calls == 2
    assert subscribe_calls == 2
    captured = capsys.readouterr()
    assert "session unavailable after fallback" in captured.err
    assert "secret-persistent-unavailable" not in captured.out
    assert "secret-persistent-unavailable" not in captured.err


@pytest.mark.parametrize("operation", ["message", "permission"])
@pytest.mark.parametrize("failure_kind", ["cancelled", "reset"])
# 功能：验证 message/permission transport failure 安全重连且绝不自动重发
# 设计：目标 RPC 模拟 pending cancel 或 write reset，最终命令计数直接证明 at-most-once 客户端提交
async def test_chat_interactive_transport_failure_is_not_retried(
    operation: Literal["message", "permission"],
    failure_kind: FailureKind,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients: list[_BaseClient] = []
    permission_delivered = asyncio.Event()
    write_failed = asyncio.Event()
    second_subscribed = asyncio.Event()
    input_count = 0

    class _InteractiveResetClient(_BaseClient):
        # 首连接按场景投递 permission 并在 write failure 后 EOF
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index > 0:
                second_subscribed.set()
                await asyncio.Event().wait()
                return
            if operation == "permission":
                assert self.handler is not None
                await self.handler(
                    EventDelivery(
                        subscription_id="sub-0",
                        delivery="live",
                        event_id="event-1",
                        stream_id="session:sess-new",
                        seq=1,
                        daemon_instance_id="daemon-a",
                        event={
                            "type": "permission.requested",
                            "run_id": "run-test",
                            "tool_use_id": "tool-old",
                            "tool_name": "bash",
                            "param_preview": "safe",
                        },
                    )
                )
                permission_delivered.set()
            await write_failed.wait()

        # 指定交互 RPC 在 write/drain 阶段 reset
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            target = "session.send_message" if operation == "message" else "permission.respond"
            if self.index == 0 and method == target:
                self.commands.append((method, dict(params)))
                write_failed.set()
                return await _raise_transport_failure(failure_kind)
            return await super().send_command(method, params)

    # 创建 write reset 前后的 client
    def make_client(host: str, port: int) -> _BaseClient:
        client = _InteractiveResetClient(len(clients))
        clients.append(client)
        return client

    # 返回消息或 permission 决策，恢复订阅后 EOF
    async def fake_readline(prompt: str) -> str:
        nonlocal input_count
        input_count += 1
        if input_count == 1:
            await clients[0].subscribed.wait()
            if operation == "permission":
                await permission_delivered.wait()
                return "y"
            return "hello"
        await second_subscribed.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)
    caplog.set_level("WARNING", logger=chat_module.__name__)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    target = "session.send_message" if operation == "message" else "permission.respond"
    target_commands = [
        params
        for client in clients
        for method, params in client.commands
        if method == target
    ]
    assert len(target_commands) == 1
    captured = capsys.readouterr()
    assert _TRANSPORT_SECRET not in captured.out
    assert _TRANSPORT_SECRET not in captured.err
    assert _TRANSPORT_SECRET not in caplog.text


# 功能：验证 session.close write reset 只产生稳定退出且不泄漏 transport 文本
# 设计：正常 EOF 后仅 close RPC 抛 reset，退出码与 captured output 直接约束 CLI 边界
async def test_chat_close_write_reset_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _CloseResetClient(_BaseClient):
        # close RPC 注入含 secret 的 reset
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "session.close":
                raise ConnectionResetError("secret-close-detail")
            return await super().send_command(method, params)

    client = _CloseResetClient(0)

    # 初始订阅后 EOF
    async def fake_readline(prompt: str) -> str:
        await client.subscribed.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", lambda host, port: client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "secret-close-detail" not in captured.out
    assert "secret-close-detail" not in captured.err


# 功能：验证 buffered gate.open handler failure 安全映射且 cursor 不推进
# 设计：response 前缓冲事件，printer 抛 secret；同时直接检查 delivery state cursor 的失败原子性
async def test_chat_buffered_handler_failure_is_redacted_and_cursor_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    delivery_sent = asyncio.Event()

    class _BufferedClient(_BaseClient):
        # subscribe response 前投递会触发 printer failure 的 delivery
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-0",
                    delivery="replay",
                    event_id="event-1",
                    stream_id="session:sess-new",
                    seq=1,
                    daemon_instance_id="daemon-a",
                    event={"type": "llm.token", "run_id": "run-test", "token": "X"},
                )
            )
            delivery_sent.set()

        # 等 callback 已缓冲后才返回订阅 response
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            result = await super().send_command(method, params)
            if method == "event.subscribe":
                await delivery_sent.wait()
            return result

    # printer 固定抛出含 secret 的渲染失败
    async def fail_handler(
        self: chat_module.ChatPrinter,
        event: dict[str, Any],
    ) -> None:
        raise RuntimeError("secret-buffered-handler")

    client = _BufferedClient(0)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", lambda host, port: client)
    monkeypatch.setattr(chat_module.ChatPrinter, "handle", fail_handler)
    caplog.set_level("WARNING", logger=chat_module.__name__)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "event delivery failed" in captured.err
    assert "secret-buffered-handler" not in captured.out
    assert "secret-buffered-handler" not in captured.err
    assert "secret-buffered-handler" not in caplog.text

    state = chat_module._ChatDeliveryState(
        printer=chat_module.ChatPrinter(),
        stream_id="session:sess-new",
        daemon_instance_id="daemon-a",
        daemon_changed=asyncio.Event(),
        permission_arrived=asyncio.Event(),
    )
    with pytest.raises(RuntimeError):
        await state.handle(
            EventDelivery(
                subscription_id="sub-direct",
                delivery="live",
                event_id="event-direct",
                stream_id="session:sess-new",
                seq=9,
                daemon_instance_id="daemon-a",
                event={"type": "llm.token", "token": "Y"},
            )
        )
    assert state.cursor == 0


# 功能：验证 post-open handler failure 使 loop task 安全非零而非伪装 EOF 重连
# 设计：gate 已打开后真实 loop task 传播 secret RuntimeError，断言稳定分类且只建一条连接
async def test_chat_post_open_handler_failure_is_safe_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients: list[_BaseClient] = []

    class _PostOpenClient(_BaseClient):
        # 订阅 response 后投递事件，让异常成为 loop task 的真实 terminal exception
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            await asyncio.Event().wait()

    # printer 固定抛出 secret，测试会在订阅后直接调用真实注册 handler
    async def fail_handler(
        self: chat_module.ChatPrinter,
        event: dict[str, Any],
    ) -> None:
        raise RuntimeError("secret-post-open-handler")

    delivery_started = asyncio.Event()

    class _DeliveringClient(_PostOpenClient):
        # 让 subscribe response 先返回，再在下一调度点投递事件
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            assert self.handler is not None
            delivery_started.set()
            await self.handler(
                EventDelivery(
                    subscription_id="sub-0",
                    delivery="live",
                    event_id="event-1",
                    stream_id="session:sess-new",
                    seq=1,
                    daemon_instance_id="daemon-a",
                    event={"type": "llm.token", "token": "X"},
                )
            )

    # 为断言连接数创建 client
    def make_client(host: str, port: int) -> _BaseClient:
        client = _DeliveringClient(len(clients))
        clients.append(client)
        return client

    # 保持 input pending，使 loop exception 成为唯一 outcome
    async def fake_readline(prompt: str) -> str:
        await delivery_started.wait()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module.ChatPrinter, "handle", fail_handler)
    caplog.set_level("WARNING", logger=chat_module.__name__)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 1
    assert len(clients) == 1
    captured = capsys.readouterr()
    assert "event delivery failed" in captured.err
    assert "secret-post-open-handler" not in captured.out
    assert "secret-post-open-handler" not in captured.err
    assert "secret-post-open-handler" not in caplog.text


# 功能：验证资源已 terminal 后的重复 cancellation 不触发二次 cleanup且保留首次对象
# 设计：close 先标记 terminal 再停在 return gate，重复 cancel 后断言 close_calls=1 与原异常 identity
async def test_chat_post_cleanup_repeated_cancellation_has_single_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subscribed = asyncio.Event()
    loop_cancelled = asyncio.Event()
    release_loop = asyncio.Event()
    resource_terminal = asyncio.Event()
    release_close_return = asyncio.Event()
    captured: list[asyncio.CancelledError] = []

    class _CleanupClient(_BaseClient):
        # 初始化 close 调用计数
        def __init__(self) -> None:
            super().__init__(0)
            self.close_calls = 0

        # cancellation 后等待 gate 再进入 terminal
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            subscribed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop_cancelled.set()
                await release_loop.wait()
                raise

        # 先完成资源终态，再保留一个可注入重复 cancellation 的返回边界
        async def close(self) -> None:
            self.close_calls += 1
            self.closed = True
            resource_terminal.set()
            await release_close_return.wait()

    client = _CleanupClient()

    # 模拟不可立即终止的 stdin 等待；取消 task 只停止消费其结果
    async def fake_readline(prompt: str) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    original_wait = chat_module._wait_for_chat_activity

    # 捕获 supervisor 首次收到的 cancellation identity
    async def capture_wait(*args: Any, **kwargs: Any) -> str:
        try:
            return await original_wait(*args, **kwargs)
        except asyncio.CancelledError as exc:
            captured.append(exc)
            raise

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", lambda host, port: client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_wait_for_chat_activity", capture_wait)

    task = asyncio.create_task(chat_module._chat_async(KamaConfig()))
    await subscribed.wait()
    task.cancel("first")
    await loop_cancelled.wait()
    release_loop.set()
    await resource_terminal.wait()
    task.cancel("after-resource-terminal")
    release_close_return.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert captured
    assert raised.value is captured[0]
    assert client.close_calls == 1
    assert client.closed is True
