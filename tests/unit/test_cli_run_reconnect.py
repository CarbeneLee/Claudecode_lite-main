from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

import kama_claude.cli.main as cli_main_module
from kama_claude.cli.commands import run as run_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.transport.socket_client import EventDelivery

# CLI run reconnect 行为集中在本模块。

type DeliveryHandler = Callable[[EventDelivery], Awaitable[None]]


# 功能：验证 CLI main 的 --mode plan 会把模式传入 run command，而不是改写 goal
# 设计：替换 cmd_run 和 daemon 配置边界，直接断言 argparse 到 command 的 typed 参数映射
def test_cli_main_passes_plan_mode_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str]] = []

    monkeypatch.setattr(cli_main_module, "get_config", lambda: KamaConfig())
    monkeypatch.setattr(cli_main_module, "setup_logging", lambda _config: None)
    monkeypatch.setattr(
        cli_main_module,
        "cmd_run",
        lambda goal, _config, *, agent_mode="direct": captured.append((goal, agent_mode)),
    )
    monkeypatch.setattr(
        cli_main_module.sys,
        "argv",
        ["kama", "run", "--mode", "plan", "--goal", "/Users/project goal"],
    )

    cli_main_module.main()

    assert captured == [("/Users/project goal", "plan")]


class _ReconnectRunClient:
    # 初始化一条可控 run 连接及其 daemon identity
    def __init__(self, index: int, *, daemon_id: str = "daemon-a") -> None:
        self.index = index
        self.daemon_id = daemon_id
        self.handler: DeliveryHandler | None = None
        self.subscribed = asyncio.Event()
        self.subscriptions: list[int] = []
        self.agent_run_calls = 0

    # 模拟成功建立连接
    async def connect(self) -> None:
        return None

    # 保存 full-delivery handler
    def on_delivery(self, handler: DeliveryHandler) -> None:
        self.handler = handler

    # 首连接投递起始事件后 EOF，重连投递目标 terminal
    async def run_event_loop(self) -> None:
        await self.subscribed.wait()
        assert self.handler is not None
        if self.index == 0:
            event = {"type": "run.started", "run_id": "run-test"}
            seq = 1
        else:
            event = {
                "type": "run.finished",
                "run_id": "run-test",
                "status": "success",
                "steps": 1,
            }
            seq = 2
        await self.handler(
            EventDelivery(
                subscription_id=f"sub-{self.index}",
                delivery="replay",
                event_id=f"event-{seq}",
                stream_id="run:run-test",
                seq=seq,
                daemon_instance_id=self.daemon_id,
                event=event,
            )
        )
        if self.index > 0:
            await asyncio.Event().wait()

    # 返回固定 run_id 并记录每次订阅的 after_seq
    async def send_command(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if method == "agent.run":
            self.agent_run_calls += 1
            return {"run_id": "run-test"}
        assert method == "event.subscribe"
        self.subscriptions.append(params["after_seq"])
        self.subscribed.set()
        return {
            "subscription_id": f"sub-{self.index}",
            "stream_id": "run:run-test",
            "accepted_after_seq": params["after_seq"],
            "high_watermark_seq": params["after_seq"],
            "daemon_instance_id": self.daemon_id,
        }

    # 模拟关闭 IPC 连接
    async def close(self) -> None:
        return None


# 功能：验证 kama run 先发送 canonical cwd 启动 run，再从该 run 的零游标订阅
# 设计：用受控 full-delivery fake 记录命令顺序，先投递其他 run terminal 再投递目标 terminal
async def test_run_sends_canonical_client_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        # 初始化 fake client 的事件回调引用
        def __init__(self, host: str, port: int) -> None:
            self.handler: DeliveryHandler | None = None
            self.subscribed = asyncio.Event()

        # 模拟成功建立 IPC 连接
        async def connect(self) -> None:
            return None

        # 记录 CLI 注册的完整 delivery 回调
        def on_delivery(self, handler: DeliveryHandler) -> None:
            self.handler = handler

        # 订阅响应后依次投递非目标与目标 terminal
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            assert self.handler is not None
            for seq, run_id in ((1, "run-other"), (2, "run-test")):
                await self.handler(
                    EventDelivery(
                        subscription_id="sub-test",
                        delivery="replay",
                        event_id=f"event-{seq}",
                        stream_id="run:run-test",
                        seq=seq,
                        daemon_instance_id="daemon-a",
                        event={
                            "type": "run.finished",
                            "run_id": run_id,
                            "status": "success",
                            "steps": 1,
                        },
                    )
                )
            await asyncio.Event().wait()

        # 记录命令并在 run-scoped 订阅就绪后释放投递
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            commands.append((method, params))
            if method == "agent.run":
                return {"run_id": "run-test"}
            self.subscribed.set()
            return {
                "subscription_id": "sub-test",
                "stream_id": "run:run-test",
                "accepted_after_seq": 0,
                "high_watermark_seq": 0,
                "daemon_instance_id": "daemon-a",
            }

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
    assert commands == [
        (
            "agent.run",
            {
                "goal": "inspect",
                "workspace_root": str(workspace.resolve()),
            },
        ),
        (
            "event.subscribe",
            {
                    "topics": [
                        "run.*",
                        "step.*",
                        "tool.*",
                        "llm.token",
                        "llm.usage",
                        "planner.*",
                    ],
                "scope": "run:run-test",
                "after_seq": 0,
            },
        ),
    ]


# 功能：验证 transient EOF 后在同一 daemon 上用最后成功处理的 seq 续传
# 设计：两个受控 client 分别投递 seq=1 和 terminal seq=2，首连接主动 EOF 且无 sleep
async def test_run_reconnects_same_daemon_from_last_processed_seq(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ReconnectRunClient] = []

    # 为每次连接创建按顺序发送事件的 fake client
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        client = _ReconnectRunClient(len(clients))
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 0
    assert len(clients) == 2
    assert clients[0].subscriptions == [0]
    assert clients[1].subscriptions == [1]
    assert clients[0].agent_run_calls == 1
    assert clients[1].agent_run_calls == 0


# 功能：验证 run 重连次数耗尽后有界失败而不无限等待 terminal
# 设计：首连接 EOF 后使所有后续 connect 立即失败，用构造数直接证明上限
async def test_run_reconnect_exhaustion_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_ReconnectRunClient] = []

    class _FailingReconnectClient(_ReconnectRunClient):
        # 除首个 client 外的连接均模拟 daemon 暂时不可达
        async def connect(self) -> None:
            if self.index > 0:
                raise ConnectionRefusedError

    # 记录构造次数以证明重连上限
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        client = _FailingReconnectClient(len(clients))
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 1
    assert len(clients) == run_module._MAX_RECONNECT_ATTEMPTS + 1
    assert "reconnect attempts exhausted" in capsys.readouterr().err


# 功能：验证 daemon identity 变化时 CLI 不声称 active run 已恢复
# 设计：第二条连接返回新 identity，断言立即非零退出且错误文案显式说明边界
async def test_run_daemon_change_fails_without_active_run_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_ReconnectRunClient] = []

    # 为第二条连接分配不同 daemon identity
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        daemon_id = "daemon-a" if not clients else "daemon-b"
        client = _ReconnectRunClient(len(clients), daemon_id=daemon_id)
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 1
    assert len(clients) == 2
    error = capsys.readouterr().err
    assert "daemon restarted" in error
    assert "historical state unavailable" in error
    assert clients[1].subscriptions == [1]


# 功能：验证 subscribe RPC 的 pending future 因 EOF 取消时会按零游标重连
# 设计：用 loop_exited gate 区分 EOF cancellation 与 caller cancellation，不使用 sleep
async def test_run_subscribe_eof_reconnects_without_treating_it_as_caller_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ReconnectRunClient] = []

    class _SubscribeEofClient(_ReconnectRunClient):
        # 初始化首连接 EOF 同步门闩
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.loop_exited = asyncio.Event()

        # 首连接在订阅开始后终止读循环，后续连接投递 terminal
        async def run_event_loop(self) -> None:
            if self.index > 0:
                await super().run_event_loop()
                return
            await self.subscribed.wait()
            self.loop_exited.set()

        # 首次 subscribe 模拟 SocketClient 在 EOF 时取消 pending RPC future
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if self.index == 0 and method == "event.subscribe":
                self.subscriptions.append(params["after_seq"])
                self.subscribed.set()
                await self.loop_exited.wait()
                pending: asyncio.Future[dict[str, Any]] = (
                    asyncio.get_running_loop().create_future()
                )
                pending.cancel()
                return await pending
            return await super().send_command(method, params)

    # 为 EOF 后的每次连接创建独立 fake client
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        client = _SubscribeEofClient(len(clients))
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 0
    assert [client.subscriptions for client in clients] == [[0], [0]]


# 功能：验证 delivery handler 失败时 cursor 不前移，重连仍从旧游标回放
# 设计：首 client 捕获固定 handler 异常后 EOF，第二 client 的订阅参数直接暴露 cursor
async def test_run_handler_failure_does_not_advance_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ReconnectRunClient] = []
    original_handle = run_module.StdoutPrinter.handle
    calls = 0

    # 首次渲染抛出固定异常，后续 terminal 使用真实 printer
    async def fail_once(
        self: run_module.StdoutPrinter,
        event: dict[str, Any],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("render failed")
        await original_handle(self, event)

    class _HandlerFailureClient(_ReconnectRunClient):
        # 首连接捕获 handler 失败并模拟随后 EOF
        async def run_event_loop(self) -> None:
            if self.index > 0:
                await super().run_event_loop()
                return
            await self.subscribed.wait()
            assert self.handler is not None
            try:
                await self.handler(
                    EventDelivery(
                        subscription_id="sub-0",
                        delivery="replay",
                        event_id="event-1",
                        stream_id="run:run-test",
                        seq=1,
                        daemon_instance_id="daemon-a",
                        event={"type": "run.started", "run_id": "run-test"},
                    )
                )
            except RuntimeError:
                return

    # 为 handler 失败前后创建两条受控连接
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        client = _HandlerFailureClient(len(clients))
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module.StdoutPrinter, "handle", fail_once)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 0
    assert [client.subscriptions for client in clients] == [[0], [0]]


# 功能：验证 caller repeated cancellation 不打断读循环与 socket 清理且保留首次异常对象
# 设计：分别 gate 住 loop cancellation 与 close，用任务竞速证明第二次取消不能提前结束调用
async def test_run_repeated_cancellation_preserves_identity_and_reaps_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_cancelled = asyncio.Event()
    release_loop = asyncio.Event()
    loop_terminal = asyncio.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    subscribed = asyncio.Event()
    captured: list[asyncio.CancelledError] = []
    clients: list[Any] = []

    class _CancellationClient:
        # 初始化可观测的连接终态
        def __init__(self, host: str, port: int) -> None:
            self.handler: DeliveryHandler | None = None
            self.closed = False
            clients.append(self)

        # 模拟连接成功
        async def connect(self) -> None:
            return None

        # 保存 delivery handler
        def on_delivery(self, handler: DeliveryHandler) -> None:
            self.handler = handler

        # cancellation 后停在 gate，供测试发送重复取消
        async def run_event_loop(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                loop_cancelled.set()
                await release_loop.wait()
                raise
            finally:
                loop_terminal.set()

        # 返回 run 与 subscription 响应并暴露等待阶段
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "agent.run":
                return {"run_id": "run-test"}
            subscribed.set()
            return {
                "subscription_id": "sub-test",
                "stream_id": "run:run-test",
                "accepted_after_seq": 0,
                "high_watermark_seq": 0,
                "daemon_instance_id": "daemon-a",
            }

        # 在 close gate 放行后标记 socket terminal
        async def close(self) -> None:
            close_entered.set()
            await release_close.wait()
            self.closed = True

    original_wait = run_module._wait_for_run_outcome

    # 捕获首次从 wait 阶段传播的 CancelledError 对象
    async def capture_wait(
        loop_task: asyncio.Task[None],
        finished: asyncio.Event,
        daemon_changed: asyncio.Event,
    ) -> str:
        try:
            return await original_wait(loop_task, finished, daemon_changed)
        except asyncio.CancelledError as exc:
            captured.append(exc)
            raise

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", _CancellationClient)
    monkeypatch.setattr(run_module, "_wait_for_run_outcome", capture_wait)

    task = asyncio.create_task(run_module._run_async("inspect", KamaConfig()))
    await subscribed.wait()
    task.cancel("first")
    await loop_cancelled.wait()
    task.cancel("repeat-loop")
    release_loop.set()

    close_waiter = asyncio.create_task(close_entered.wait())
    done, _pending = await asyncio.wait(
        {task, close_waiter},
        return_when=asyncio.FIRST_COMPLETED,
    )
    close_started_before_task_exit = close_waiter in done
    if close_started_before_task_exit:
        task.cancel("repeat-close")
    release_close.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await task
    close_waiter.cancel()
    await asyncio.gather(close_waiter, return_exceptions=True)

    assert close_started_before_task_exit
    assert captured
    assert raised.value is captured[0]
    assert loop_terminal.is_set()
    assert clients[0].closed is True


# 功能：验证 connect 阶段 caller cancellation 仍关闭刚创建的 client 并保留异常对象
# 设计：用 connect gate 捕获 cancellation identity，再断言 close 被执行，无需启动读循环
async def test_run_connect_cancellation_closes_new_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_entered = asyncio.Event()
    clients: list[Any] = []
    captured: list[asyncio.CancelledError] = []

    class _ConnectClient:
        # 记录新建但尚未完成连接的 client
        def __init__(self, host: str, port: int) -> None:
            self.closed = False
            clients.append(self)

        # 在连接等待点捕获 caller cancellation
        async def connect(self) -> None:
            connect_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                captured.append(exc)
                raise

        # 本测试不会进入 delivery 注册
        def on_delivery(self, handler: DeliveryHandler) -> None:
            return None

        # 本测试不会启动读循环
        async def run_event_loop(self) -> None:
            raise AssertionError("unreachable")

        # 本测试不会发送命令
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            raise AssertionError("unreachable")

        # 标记未完成连接的 client 已关闭
        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(run_module, "SocketClient", _ConnectClient)
    task = asyncio.create_task(run_module._run_async("inspect", KamaConfig()))
    await connect_entered.wait()
    task.cancel("connect-cancel")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value is captured[0]
    assert clients[0].closed is True


# 功能：验证 event loop 未知异常不会把原始 secret 输出到 CLI
# 设计：订阅响应后让真实等待分支观察 loop exception，只断言稳定通用错误文案
async def test_run_loop_exception_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _SecretLoopClient(_ReconnectRunClient):
        # 订阅就绪后抛出包含 secret 的异常
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            raise RuntimeError("secret-loop-token")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        run_module,
        "SocketClient",
        lambda host, port: _SecretLoopClient(0),
    )

    exit_code = await run_module._run_async("inspect", KamaConfig())
    error = capsys.readouterr().err

    assert exit_code == 1
    assert "event delivery failed" in error
    assert "secret-loop-token" not in error


# 功能：验证 daemon IpcError 文本不会未经处理输出到 CLI
# 设计：agent.run 返回含 secret 的固定协议错误，只断言稳定分类而不依赖错误码
async def test_run_ipc_error_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _SecretIpcClient(_ReconnectRunClient):
        # agent.run 阶段抛出包含 secret 的协议错误
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "agent.run":
                raise run_module.IpcError(-1, "secret-ipc-token")
            return await super().send_command(method, params)

        # 保持读循环存活直到 CLI 清理
        async def run_event_loop(self) -> None:
            await asyncio.Event().wait()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        run_module,
        "SocketClient",
        lambda host, port: _SecretIpcClient(0),
    )

    exit_code = await run_module._run_async("inspect", KamaConfig())
    error = capsys.readouterr().err

    assert exit_code == 1
    assert "IPC command failed" in error
    assert "secret-ipc-token" not in error


# 功能：验证 terminal delivery 与 EOF 同时发生时目标 terminal 语义优先
# 设计：handler 返回后立即结束 loop，使 finished 和 loop task 同轮 ready 并断言成功退出
async def test_run_terminal_wins_over_simultaneous_disconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TerminalEofClient(_ReconnectRunClient):
        # 投递目标 terminal 后立即 EOF
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-test",
                    delivery="replay",
                    event_id="event-1",
                    stream_id="run:run-test",
                    seq=1,
                    daemon_instance_id="daemon-a",
                    event={
                        "type": "run.finished",
                        "run_id": "run-test",
                        "status": "success",
                        "steps": 1,
                    },
                )
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        run_module,
        "SocketClient",
        lambda host, port: _TerminalEofClient(0),
    )

    exit_code = await run_module._run_async("inspect", KamaConfig())

    assert exit_code == 0


# 功能：验证 subscribe response 身份确认前到达的 replay callback 必须先缓冲
# 设计：在 response 前投递另一 daemon 的 terminal，只有 gate 能阻止其被误判为成功终态
async def test_run_buffers_delivery_until_subscribe_identity_is_known(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    delivery_sent = asyncio.Event()

    class _PreResponseDeliveryClient(_ReconnectRunClient):
        # response 返回前投递 daemon-b 的目标 terminal
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-test",
                    delivery="replay",
                    event_id="event-1",
                    stream_id="run:run-test",
                    seq=1,
                    daemon_instance_id="daemon-b",
                    event={
                        "type": "run.finished",
                        "run_id": "run-test",
                        "status": "success",
                        "steps": 1,
                    },
                )
            )
            delivery_sent.set()
            await asyncio.Event().wait()

        # 等待 callback 完成后才返回 daemon-a 的订阅身份
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method == "agent.run":
                return {"run_id": "run-test"}
            self.subscriptions.append(params["after_seq"])
            self.subscribed.set()
            await delivery_sent.wait()
            return {
                "subscription_id": "sub-test",
                "stream_id": "run:run-test",
                "accepted_after_seq": 0,
                "high_watermark_seq": 1,
                "daemon_instance_id": "daemon-a",
            }

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        run_module,
        "SocketClient",
        lambda host, port: _PreResponseDeliveryClient(0),
    )

    exit_code = await run_module._run_async("inspect", KamaConfig())
    error = capsys.readouterr().err

    assert exit_code == 1
    assert "daemon restarted" in error
    assert "historical state unavailable" in error


# 功能：验证 close 阶段 connection reset 不阻止 cursor 重连或 terminal 成功返回
# 设计：两条连接的 close 都抛含 secret 的 reset，断言业务结果与日志同时保持安全稳定
async def test_run_close_reset_is_secondary_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    clients: list[_ReconnectRunClient] = []

    class _CloseResetClient(_ReconnectRunClient):
        # 模拟 socket 已被对端 reset 的关闭阶段异常
        async def close(self) -> None:
            raise ConnectionResetError("secret-close-token")

    # 为 EOF 与 terminal 两阶段创建都会在 close reset 的 client
    def make_client(host: str, port: int) -> _ReconnectRunClient:
        client = _CloseResetClient(len(clients))
        clients.append(client)
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_module, "SocketClient", make_client)
    monkeypatch.setattr(run_module, "_RECONNECT_DELAY_S", 0)
    caplog.set_level("WARNING", logger=run_module.__name__)

    exit_code = await run_module._run_async("inspect", KamaConfig())
    captured = capsys.readouterr()

    assert exit_code == 0
    assert [client.subscriptions for client in clients] == [[0], [1]]
    assert "secret-close-token" not in captured.out
    assert "secret-close-token" not in captured.err
    assert "secret-close-token" not in caplog.text
    assert "role=run_connection_close" in caplog.text
