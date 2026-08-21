from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from kama_claude.cli.commands import chat as chat_module
from kama_claude.core.config import KamaConfig
from kama_claude.core.plan_view import PlanViewV1, projection_digest
from kama_claude.core.transport.socket_client import EventDelivery, IpcError

type DeliveryHandler = Callable[[EventDelivery], Awaitable[None]]


# 构造最小合法 V1 PlanReady payload，供 CLI authority wiring 测试复用
def _plan_event() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision_key": "decision-1:v1",
        "projection_key": "pv1:run-1:decision-1:v1",
        "decision_id": "decision-1",
        "decision_version": 1,
        "decision_content_digest": "decision-digest",
        "architecture_slice_id": "slice-1",
        "architecture_slice_version": 1,
        "architecture_slice_content_digest": "slice-digest",
        "snapshot_digest": "snapshot-digest",
        "goal": "change behavior",
        "architecture_mode": "preserve",
        "selected_approach": "edit existing module",
        "projection_digest": "placeholder",
    }
    plan = PlanViewV1.model_validate(payload)
    full_payload = plan.model_dump(mode="json")
    full_payload["projection_digest"] = projection_digest(full_payload)
    plan = PlanViewV1.model_validate(full_payload)
    return {
        "type": "planner.decision_ready",
        "event_id": "plan-ready:pv1:run-1:decision-1:v1",
        "run_id": "run-1",
        "planner_run_id": "planner-1",
        "session_id": "sess-1",
        "plan": plan.model_dump(mode="json"),
        "ts": "t",
    }


class _ChatClient:
    # 初始化一条可控 chat 连接及其命令记录
    def __init__(self, index: int, *, daemon_id: str = "daemon-a") -> None:
        self.index = index
        self.daemon_id = daemon_id
        self.handler: DeliveryHandler | None = None
        self.subscribed = asyncio.Event()
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.subscriptions: list[dict[str, Any]] = []
        self.closed = False

    # 模拟连接成功
    async def connect(self) -> None:
        return None

    # 禁止 chat 回退到丢失 envelope metadata 的 raw event API
    def on_event(self, handler: Any) -> None:
        raise AssertionError("chat must use full delivery API")

    # 保存 full delivery handler
    def on_delivery(self, handler: DeliveryHandler) -> None:
        self.handler = handler

    # 默认保持读循环存活直到 CLI 清理
    async def run_event_loop(self) -> None:
        await self.subscribed.wait()
        await asyncio.Event().wait()

    # 返回固定 session 与 subscription 响应并记录命令
    async def send_command(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.commands.append((method, params))
        if method == "session.create":
            return {"session_id": "sess-old", "status": "active"}
        if method == "event.subscribe":
            self.subscriptions.append(dict(params))
            self.subscribed.set()
            return {
                "subscription_id": f"sub-{self.index}-{len(self.subscriptions)}",
                "stream_id": params["scope"],
                "accepted_after_seq": params["after_seq"],
                "high_watermark_seq": params["after_seq"],
                "daemon_instance_id": self.daemon_id,
            }
        if method == "session.get_agent_mode":
            return {"agent_mode": "direct", "revision": 0}
        if method == "session.send_message":
            return {"run_id": "run-test"}
        return {}

    # 标记连接已关闭
    async def close(self) -> None:
        self.closed = True


# 功能：验证 chat 创建 session 后才从该 session stream 的零游标订阅
# 设计：readline 等待订阅完成再 EOF，命令记录可同时证明顺序、scope 与 full delivery API
async def test_chat_creates_session_before_session_scoped_subscription(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ChatClient] = []

    # 保存唯一 fake client 供断言
    def make_client(host: str, port: int) -> _ChatClient:
        client = _ChatClient(len(clients))
        clients.append(client)
        return client

    # 订阅建立后结束交互
    async def fake_readline(prompt: str) -> str:
        await clients[0].subscribed.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    methods = [method for method, _params in clients[0].commands]
    assert methods[:3] == [
        "session.create",
        "event.subscribe",
        "session.get_agent_mode",
    ]
    assert clients[0].subscriptions == [
        {
            "topics": [
                "session.*",
                "run.*",
                "tool.*",
                "llm.token",
                    "permission.*",
                    "planner.*",
                    "plan.*",
            ],
            "scope": "session:sess-old",
            "after_seq": 0,
        }
    ]


# 功能：验证 equal-revision mode conflict 只设置 deferred refresh，不在 delivery callback 发 RPC
# 设计：直接运行真实 ChatDeliveryState handler，检查 callback 返回后的 signal 而非伪造 SocketClient 调用
async def test_chat_mode_conflict_is_deferred_without_rpc() -> None:
    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
        mode_snapshot=chat_module.AgentModeSnapshot("direct", 5),
    )

    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="live",
            event_id="mode-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={
                "type": "session.agent_mode_changed",
                "previous_mode": "direct",
                "agent_mode": "plan",
                "revision": 5,
                "ts": "t",
            },
        )
    )

    assert state.mode_refresh_required.is_set()
    assert state.printer.agent_mode == "direct"


# 功能：验证 CLI 只从 committed PlanView 建立 exact approval target 并通过 authority GET 获得 receipt
# 设计：真实 ChatDeliveryState 处理 PlanReady，再走 authority apply 和 command 参数生成，排除 raw 文本猜测路径
async def test_chat_committed_plan_wires_exact_approval_authority() -> None:
    class _Client:
        # 返回与 committed target 完全一致的 authority snapshot
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "plan.get_approval"
            assert params == {
                "session_id": "sess-1",
                "projection_key": "pv1:run-1:decision-1:v1",
            }
            return {
                "session_id": "sess-1",
                "projection_key": "pv1:run-1:decision-1:v1",
                "status": "pending",
                "decision_id": "decision-1",
                "decision_version": 1,
                "content_digest": "decision-digest",
                "commit_receipt_digest": "receipt-digest",
            }

    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="plan-1",
            stream_id="session:sess-1",
            seq=1,
            daemon_instance_id="daemon-a",
            event=_plan_event(),
        )
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="run-finished-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={"type": "run.finished", "run_id": "run-1", "status": "success"},
        )
    )

    projection_key = "pv1:run-1:decision-1:v1"
    assert state.approval_targets[projection_key].decision_id == "decision-1"
    assert state.approval_states[projection_key].snapshot.status == "pending"
    await chat_module._refresh_chat_approvals(_Client(), state)  # type: ignore[arg-type]
    command = state.approval_command_params(projection_key, "approve")
    assert command is not None
    method, params = command
    assert method == "plan.approve"
    assert params["content_digest"] == "decision-digest"
    assert params["commit_receipt_digest"] == "receipt-digest"


# 功能：验证 CLI approval notification 冲突只进入 unknown 并设置 deferred refresh
# 设计：先注入 authority approved，再投递冲突 rejected event，确认 delivery callback 不会直接伪造终态
async def test_chat_approval_conflict_sets_deferred_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
    )
    rendered: list[str] = []
    monkeypatch.setattr(
        state.printer,
        "show_approval",
        lambda snapshot: rendered.append(snapshot.status),
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="plan-1",
            stream_id="session:sess-1",
            seq=1,
            daemon_instance_id="daemon-a",
            event=_plan_event(),
        )
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="run-finished-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={"type": "run.finished", "run_id": "run-1", "status": "success"},
        )
    )
    projection_key = "pv1:run-1:decision-1:v1"
    state.apply_approval_authority(
        projection_key,
        {
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "approved",
            "decision_id": "decision-1",
            "decision_version": 1,
            "content_digest": "decision-digest",
            "commit_receipt_digest": "receipt-digest",
            "action": "approve",
            "record_digest": "record-a",
        },
        epoch=None,
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="live",
            event_id="approval-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={
                "type": "plan.approval_changed",
                "session_id": "sess-1",
                "projection_key": projection_key,
                "status": "rejected",
                "action": "reject",
                "record_digest": "record-b",
                "commit_receipt_digest": "receipt-digest",
                "ts": "t",
            },
        )
    )

    assert state.approval_states[projection_key].snapshot.status == "conflicted/unknown"
    assert state.approval_refresh_required.is_set()
    assert rendered == ["approved", "conflicted/unknown"]
    assert "rejected" not in rendered


# 功能：验证 CLI approve/reject RPC 冲突只渲染本地 conflicted/unknown 状态
# 设计：先注入 approved authority，再合并 disputed rejected response，确保 UI 不把争议输入当作事实
async def test_chat_approval_response_conflict_renders_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
    )
    rendered: list[str] = []
    monkeypatch.setattr(
        state.printer,
        "show_approval",
        lambda snapshot: rendered.append(snapshot.status),
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="plan-1",
            stream_id="session:sess-1",
            seq=1,
            daemon_instance_id="daemon-a",
            event=_plan_event(),
        )
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="run-finished-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={"type": "run.finished", "run_id": "run-1", "status": "success"},
        )
    )
    projection_key = "pv1:run-1:decision-1:v1"
    state.apply_approval_authority(
        projection_key,
        {
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "approved",
            "decision_id": "decision-1",
            "decision_version": 1,
            "content_digest": "decision-digest",
            "commit_receipt_digest": "receipt-digest",
            "action": "approve",
            "record_digest": "record-a",
        },
        epoch=None,
    )

    relation = state.merge_approval_response(
        {
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "rejected",
            "decision_id": "decision-1",
            "decision_version": 1,
            "content_digest": "decision-digest",
            "commit_receipt_digest": "receipt-digest",
            "action": "reject",
            "record_digest": "record-b",
        }
    )

    assert relation is chat_module.ApprovalSnapshotRelation.CONFLICT
    assert state.approval_states[projection_key].snapshot.status == "conflicted/unknown"
    assert rendered == ["approved", "conflicted/unknown"]
    assert "rejected" not in rendered


# 功能：验证 CLI refresh response 返回到新 client/view 后不会覆盖旧 approval state
# 设计：阻塞旧 client 的 authority GET，替换 state.approval_client 后放行，覆盖 await 后 owner/state identity 检查
async def test_chat_stale_approval_refresh_owner_is_discarded() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _Client:
        # 让旧 authority response 跨越 client replacement
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "plan.get_approval"
            started.set()
            await release.wait()
            return {
                "session_id": "sess-1",
                "projection_key": "pv1:run-1:decision-1:v1",
                "status": "approved",
                "decision_id": "decision-1",
                "decision_version": 1,
                "content_digest": "decision-digest",
                "commit_receipt_digest": "receipt-digest",
                "action": "approve",
                "record_digest": "record-old",
            }

    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="plan-1",
            stream_id="session:sess-1",
            seq=1,
            daemon_instance_id="daemon-a",
            event=_plan_event(),
        )
    )
    await state.handle(
        EventDelivery(
            subscription_id="sub",
            delivery="replay",
            event_id="run-finished-1",
            stream_id="session:sess-1",
            seq=2,
            daemon_instance_id="daemon-a",
            event={"type": "run.finished", "run_id": "run-1", "status": "success"},
        )
    )
    old_client = _Client()
    state.approval_client = old_client  # type: ignore[assignment]
    state.approval_refresh_required.set()
    task = asyncio.create_task(chat_module._refresh_chat_approvals(old_client, state))  # type: ignore[arg-type]
    await started.wait()
    state.approval_client = object()  # type: ignore[assignment]
    release.set()
    await task

    assert state.approval_states["pv1:run-1:decision-1:v1"].snapshot.status == "pending"


# 功能：验证 CLI 放弃旧 view 时清除遗留 mode refresh signal
# 设计：直接调用 fresh-view mode reset 边界，防止新 session 继承旧 session 的 deferred 状态
def test_chat_fresh_view_clears_abandoned_mode_refresh_signal() -> None:
    state = chat_module._ChatDeliveryState(
        chat_module.ChatPrinter(),
        "session:sess-1",
        "daemon-a",
        asyncio.Event(),
        asyncio.Event(),
        mode_snapshot=chat_module.AgentModeSnapshot("plan", 20),
    )
    state.mode_refresh_required.set()
    state.printer.agent_mode = "plan"
    state.printer.agent_mode_revision = 20

    chat_module._reset_chat_mode_view(state, state.printer)

    assert not state.mode_refresh_required.is_set()
    assert state.mode_snapshot is None
    assert state.printer.agent_mode == "direct"
    assert state.printer.agent_mode_revision == 0


# 功能：验证 transient EOF 后沿用同一 session 与 handler-success cursor 重连
# 设计：首连接投递 seq=1 后 EOF，readline 同轮 EOF，断线优先级迫使第二连接从 seq=1 续传
async def test_chat_reconnects_same_daemon_with_session_and_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ChatClient] = []
    first_loop_exited = asyncio.Event()

    class _ReconnectClient(_ChatClient):
        # 首连接投递事件后 EOF，第二连接保持存活
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index > 0:
                await asyncio.Event().wait()
                return
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-0",
                    delivery="live",
                    event_id="event-1",
                    stream_id="session:sess-old",
                    seq=1,
                    daemon_instance_id="daemon-a",
                    event={"type": "llm.token", "run_id": "run-test", "token": "A"},
                )
            )
            first_loop_exited.set()

    # 为断线前后创建独立连接
    def make_client(host: str, port: int) -> _ChatClient:
        client = _ReconnectClient(len(clients))
        clients.append(client)
        return client

    # 与首 EOF 同轮结束输入，验证 supervisor 优先执行重连
    async def fake_readline(prompt: str) -> str:
        await first_loop_exited.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    assert len(clients) == 2
    assert clients[0].subscriptions[0]["after_seq"] == 0
    assert clients[1].subscriptions[0] == {
        "topics": [
            "session.*",
            "run.*",
            "tool.*",
            "llm.token",
                "permission.*",
                "planner.*",
                "plan.*",
        ],
        "scope": "session:sess-old",
        "after_seq": 1,
    }
    assert sum(
        method == "session.create"
        for client in clients
        for method, _params in client.commands
    ) == 1


# 功能：验证 initial subscribe EOF fallback 后首条新 session 消息不会被当作旧输入丢弃
# 设计：首订阅 pending future 被 EOF 取消，恢复 session 的第一行发送 hello、第二行 EOF
async def test_chat_initial_subscribe_eof_preserves_first_fresh_session_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ChatClient] = []
    subscribe_started = asyncio.Event()
    first_loop_exited = asyncio.Event()
    second_subscribed = asyncio.Event()
    create_count = 0
    input_count = 0

    class _InitialSubscribeEofClient(_ChatClient):
        # 首连接在 subscribe pending 时 EOF，第二连接保持存活
        async def run_event_loop(self) -> None:
            if self.index > 0:
                await self.subscribed.wait()
                await asyncio.Event().wait()
                return
            await subscribe_started.wait()
            first_loop_exited.set()

        # 首次 subscribe 返回内部 CancelledError，恢复连接正常创建和订阅
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_count
            if method == "session.create":
                self.commands.append((method, dict(params)))
                create_count += 1
                session_id = "sess-old" if create_count == 1 else "sess-new"
                return {"session_id": session_id, "status": "active"}
            if method == "event.subscribe" and self.index == 0:
                self.commands.append((method, dict(params)))
                self.subscriptions.append(dict(params))
                subscribe_started.set()
                await first_loop_exited.wait()
                pending: asyncio.Future[dict[str, Any]] = (
                    asyncio.get_running_loop().create_future()
                )
                pending.cancel()
                return await pending
            result = await super().send_command(method, params)
            if method == "event.subscribe":
                second_subscribed.set()
            return result

    # 创建 initial 与 fallback client
    def make_client(host: str, port: int) -> _ChatClient:
        client = _InitialSubscribeEofClient(len(clients))
        clients.append(client)
        return client

    # fallback 订阅后先发送正常消息，再 EOF
    async def fake_readline(prompt: str) -> str:
        nonlocal input_count
        input_count += 1
        await second_subscribed.wait()
        if input_count == 1:
            return "hello"
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    sent = [
        params
        for client in clients
        for method, params in client.commands
        if method == "session.send_message"
    ]
    assert sent == [{"session_id": "sess-new", "content": "hello"}]


# 功能：验证 daemon identity 改变时明确结束旧视图并新建 session
# 设计：重连订阅旧 stream 返回 daemon-b，断言同连接创建新 session 且游标归零、不混合旧 history
async def test_chat_daemon_change_creates_fresh_session_without_history_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_ChatClient] = []
    first_loop_exited = asyncio.Event()
    create_count = 0

    class _DaemonChangeClient(_ChatClient):
        # 首连接立刻 EOF，重连保持存活
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index == 0:
                first_loop_exited.set()
                return
            await asyncio.Event().wait()

        # 第二次 session.create 返回新 session id
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_count
            if method == "session.create":
                self.commands.append((method, params))
                create_count += 1
                session_id = "sess-old" if create_count == 1 else "sess-new"
                return {"session_id": session_id, "status": "active"}
            return await super().send_command(method, params)

    # 第二条连接代表重启后的 daemon
    def make_client(host: str, port: int) -> _ChatClient:
        daemon_id = "daemon-a" if not clients else "daemon-b"
        client = _DaemonChangeClient(len(clients), daemon_id=daemon_id)
        clients.append(client)
        return client

    # 首断线后结束输入，仍要求 fallback 先完成
    async def fake_readline(prompt: str) -> str:
        await first_loop_exited.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    assert create_count == 2
    assert [sub["scope"] for sub in clients[1].subscriptions] == [
        "session:sess-old",
        "session:sess-new",
    ]
    assert clients[1].subscriptions[-1]["after_seq"] == 0
    output = capsys.readouterr().out
    assert "daemon restarted" in output
    assert "old session view ended" in output


# 功能：验证同 daemon 上旧 session unavailable 时也显式创建新 session
# 设计：第二连接首次 subscribe 抛固定 IpcError，再允许新 session 订阅并断言旧游标不被继承
async def test_chat_session_unavailable_creates_fresh_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_ChatClient] = []
    first_loop_exited = asyncio.Event()
    create_count = 0

    class _UnavailableClient(_ChatClient):
        # 首连接 EOF，第二连接在新订阅后保持存活
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index == 0:
                first_loop_exited.set()
                return
            await asyncio.Event().wait()

        # 重连旧 stream 时报告不可用，其后创建的新 stream 正常订阅
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal create_count
            if method == "session.create":
                self.commands.append((method, params))
                create_count += 1
                session_id = "sess-old" if create_count == 1 else "sess-new"
                return {"session_id": session_id, "status": "active"}
            if method == "event.subscribe" and self.index == 1 and not self.subscriptions:
                self.commands.append((method, params))
                self.subscriptions.append(dict(params))
                raise IpcError(-32000, "secret-session-detail")
            return await super().send_command(method, params)

    # 创建断线前后的 fake client
    def make_client(host: str, port: int) -> _ChatClient:
        client = _UnavailableClient(len(clients))
        clients.append(client)
        return client

    # 首断线后结束输入
    async def fake_readline(prompt: str) -> str:
        await first_loop_exited.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    assert create_count == 2
    assert clients[1].subscriptions[-1]["scope"] == "session:sess-new"
    assert clients[1].subscriptions[-1]["after_seq"] == 0
    captured = capsys.readouterr()
    assert "session unavailable" in captured.out
    assert "secret-session-detail" not in captured.out
    assert "secret-session-detail" not in captured.err


# 功能：验证 permission pending 期间断线后绝不发送旧 tool_use_id
# 设计：首连接先投递 permission 再 EOF，输入 y 与 EOF 同轮完成，重连优先清空 pending 决策
async def test_chat_disconnect_clears_stale_permission_before_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ChatClient] = []
    permission_delivered = asyncio.Event()
    input_count = 0

    class _PermissionClient(_ChatClient):
        # 首连接投递 permission 后断线，第二连接保持存活
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index > 0:
                await asyncio.Event().wait()
                return
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-0",
                    delivery="live",
                    event_id="event-1",
                    stream_id="session:sess-old",
                    seq=1,
                    daemon_instance_id="daemon-a",
                    event={
                        "type": "permission.requested",
                        "run_id": "run-test",
                        "tool_use_id": "old-tool-use",
                        "tool_name": "bash",
                        "param_preview": "safe",
                    },
                )
            )
            permission_delivered.set()

    # 为 permission 断线前后创建连接
    def make_client(host: str, port: int) -> _ChatClient:
        client = _PermissionClient(len(clients))
        clients.append(client)
        return client

    # 首次返回旧 permission 决策，第二次结束交互
    async def fake_readline(prompt: str) -> str:
        nonlocal input_count
        input_count += 1
        if input_count == 1:
            await permission_delivered.wait()
            return "y"
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    assert len(clients) == 2
    permission_commands = [
        params
        for client in clients
        for method, params in client.commands
        if method == "permission.respond"
    ]
    assert permission_commands == []
    assert all("old-tool-use" not in str(client.commands) for client in clients[1:])


# 功能：验证 chat 重连次数耗尽后有界非零退出
# 设计：首连接 EOF 后所有 connect 立即失败，以 client 构造数证明不会无限重试
async def test_chat_reconnect_exhaustion_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients: list[_ChatClient] = []
    first_loop_exited = asyncio.Event()

    class _FailingClient(_ChatClient):
        # 重连 client 连接失败
        async def connect(self) -> None:
            if self.index > 0:
                raise ConnectionRefusedError

        # 首连接订阅后 EOF
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            first_loop_exited.set()

    # 创建每次重连的独立 client
    def make_client(host: str, port: int) -> _ChatClient:
        client = _FailingClient(len(clients))
        clients.append(client)
        return client

    # 与断线同轮 EOF，supervisor 仍必须先耗尽有界重连
    async def fake_readline(prompt: str) -> str:
        await first_loop_exited.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 1
    assert len(clients) == chat_module._MAX_RECONNECT_ATTEMPTS + 1
    assert "reconnect attempts exhausted" in capsys.readouterr().err


# 功能：验证 subscribe response 前的 delivery 不会绕过 daemon identity gate
# 设计：首连接先回调 daemon-b 事件再返回 daemon-a response，重连游标必须仍为零
async def test_chat_buffers_delivery_until_subscribe_identity_is_known(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_ChatClient] = []
    delivery_sent = asyncio.Event()

    class _PreResponseClient(_ChatClient):
        # 首连接在 response 前投递另一 daemon 的事件，第二连接保持存活
        async def run_event_loop(self) -> None:
            await self.subscribed.wait()
            if self.index > 0:
                await asyncio.Event().wait()
                return
            assert self.handler is not None
            await self.handler(
                EventDelivery(
                    subscription_id="sub-0",
                    delivery="replay",
                    event_id="event-9",
                    stream_id="session:sess-old",
                    seq=9,
                    daemon_instance_id="daemon-b",
                    event={"type": "llm.token", "run_id": "run-test", "token": "X"},
                )
            )
            delivery_sent.set()

        # 首次订阅等待 pre-response callback 完成后再返回 daemon-a identity
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
        ) -> dict[str, Any]:
            if method != "event.subscribe" or self.index > 0:
                return await super().send_command(method, params)
            self.commands.append((method, params))
            self.subscriptions.append(dict(params))
            self.subscribed.set()
            await delivery_sent.wait()
            return {
                "subscription_id": "sub-0",
                "stream_id": params["scope"],
                "accepted_after_seq": params["after_seq"],
                "high_watermark_seq": 9,
                "daemon_instance_id": "daemon-a",
            }

    # 为 identity mismatch 前后创建独立连接
    def make_client(host: str, port: int) -> _ChatClient:
        client = _PreResponseClient(len(clients))
        clients.append(client)
        return client

    # 与首连接回调同轮 EOF，supervisor 仍须先处理 daemon mismatch
    async def fake_readline(prompt: str) -> str:
        await delivery_sent.wait()
        raise EOFError

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)
    monkeypatch.setattr(chat_module, "_RECONNECT_DELAY_S", 0)

    exit_code = await chat_module._chat_async(KamaConfig())

    assert exit_code == 0
    assert len(clients) == 2
    assert clients[1].subscriptions[0]["after_seq"] == 0
