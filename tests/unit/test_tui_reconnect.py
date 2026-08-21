from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kama_claude.tui.app as tui_app_module
from kama_claude.core.approval import ApprovalSnapshotState
from kama_claude.core.plan_view import PlanViewV1, projection_digest
from kama_claude.core.session.model import AgentModeSnapshot
from kama_claude.core.transport.socket_client import EventDelivery
from kama_claude.tui.app import (
    KamaTuiApp,
    _ConnectionLost,
    _SessionCreateOutcomeUnknown,
    _TuiReconnectState,
)


# 构造最小合法 PlanReady projection，覆盖 TUI committed approval wiring
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
    payload = PlanViewV1.model_validate(payload).model_dump(mode="json")
    payload["projection_digest"] = projection_digest(payload)
    return {
        "type": "planner.decision_ready",
        "event_id": "plan-ready:pv1:run-1:decision-1:v1",
        "run_id": "run-1",
        "planner_run_id": "planner-1",
        "session_id": "sess-1",
        "plan": PlanViewV1.model_validate(payload).model_dump(mode="json"),
        "ts": "t",
    }


# 构造指定持久元数据的 delivery，便于验证失败重放边界
def _delivery(*, event_id: str, stream_id: str, seq: int) -> EventDelivery:
    return EventDelivery(
        subscription_id="sub-test",
        delivery="replay",
        event_id=event_id,
        stream_id=stream_id,
        seq=seq,
        daemon_instance_id="daemon-a",
        event={"type": "llm.token", "token": "hello", "run_id": "run-1", "ts": "t"},
    )


class _ScriptedClient:
    # 初始化可由 Event 门控的单次 TUI 连接脚本
    def __init__(
        self,
        *,
        daemon_instance_id: str,
        session_id: str = "sess-new",
        delivery: EventDelivery | None = None,
        fail_handshake: bool = False,
        fail_create_response: bool = False,
        durable_error: Exception | None = None,
    ) -> None:
        self.daemon_instance_id = daemon_instance_id
        self.session_id = session_id
        self.delivery = delivery
        self.fail_handshake = fail_handshake
        self.fail_create_response = fail_create_response
        self.durable_error = durable_error
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.subscribed = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self._delivery_handler: Any = None

    # 模拟成功建立 TCP 连接
    async def connect(self) -> None:
        return None

    # 保持读循环至测试显式放行
    async def run_event_loop(self) -> None:
        await self.release.wait()

    # 记录完整 delivery handler 以模拟 response-first 边界
    def on_delivery(self, handler: Any) -> None:
        self._delivery_handler = handler

    # 按方法名返回受控响应并记录真实订阅参数
    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.commands.append((method, params))
        if method == "event.subscribe" and params.get("scope") == "global":
            if self.fail_handshake:
                raise asyncio.CancelledError
            return {
                "subscription_id": "sub-handshake",
                "daemon_instance_id": self.daemon_instance_id,
            }
        if method == "event.unsubscribe":
            return {"removed": True}
        if method == "session.get_history":
            return {"messages": []}
        if method == "session.get_agent_mode":
            return {"agent_mode": "direct", "revision": 0}
        if method == "session.create":
            if self.fail_create_response:
                raise asyncio.CancelledError
            return {"session_id": self.session_id, "status": "active"}
        if method == "event.subscribe":
            if self.durable_error is not None:
                raise self.durable_error
            if self.delivery is not None:
                await self._delivery_handler(self.delivery)
            self.subscribed.set()
            return {
                "subscription_id": "sub-durable",
                "daemon_instance_id": self.daemon_instance_id,
                "stream_id": params["scope"],
                "accepted_after_seq": params["after_seq"],
                "high_watermark_seq": params["after_seq"],
            }
        raise AssertionError(f"unexpected method {method}")

    # 记录连接已被 cleanup 关闭
    async def close(self) -> None:
        self.closed = True


# 功能：验证同 daemon 重连保留 session、busy、pending 标识与 per-stream cursor
# 设计：纯状态对象避开 Textual DOM，直接锁定重连不应破坏的客户端事实
def test_same_daemon_reattach_preserves_active_state() -> None:
    state = _TuiReconnectState(daemon_instance_id="daemon-a")
    state.session_id = "sess-1"
    state.busy = True
    state.pending_tool_ids.add("tool-1")
    state.pending_permission_ids.add("perm-1")
    state.stream_cursors["session:sess-1"] = 7

    transition = state.accept_daemon("daemon-a")

    assert transition == "same"
    assert state.session_id == "sess-1"
    assert state.busy
    assert state.pending_tool_ids == {"tool-1"}
    assert state.pending_permission_ids == {"perm-1"}
    assert state.stream_cursors == {"session:sess-1": 7}


# 功能：验证 daemon identity 变化会原子失效旧 session 与所有未完成 UI 状态
# 设计：预先填充所有敏感状态，一次 transition 后断言全清空，防止新 session 混入旧历史
def test_daemon_change_invalidates_old_session_and_pending_state() -> None:
    state = _TuiReconnectState(daemon_instance_id="daemon-a")
    state.session_id = "sess-old"
    state.busy = True
    state.pending_tool_ids.add("tool-old")
    state.pending_permission_ids.add("perm-old")
    state.stream_cursors["session:sess-old"] = 9
    state.rendered_event_ids.add("evt-old")

    transition = state.accept_daemon("daemon-b")

    assert transition == "changed"
    assert state.daemon_instance_id == "daemon-b"
    assert state.session_id is None
    assert not state.busy
    assert state.pending_tool_ids == set()
    assert state.pending_permission_ids == set()
    assert state.stream_cursors == {}
    assert state.rendered_event_ids == set()


# 功能：验证 --replay 每次重连都从 run stream 的已提交 cursor 继续
# 设计：先推进指定 run cursor，再生成订阅参数，防止重连退化为每次从零全量回放
def test_replay_subscription_uses_run_stream_cursor() -> None:
    state = _TuiReconnectState(daemon_instance_id="daemon-a", replay_run_id="run-7")
    state.stream_cursors["run:run-7"] = 12

    assert state.subscription_target() == ("run:run-7", 12)


# 功能：验证 TUI authority GET 的 stale response 不会降低本地 mode revision
# 设计：直接调用纯 authority 应用边界，隔离 socket 重连，锁定 daemon response 的四分支语义
def test_tui_stale_authority_response_cannot_downgrade_revision() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    app._mode_snapshot = AgentModeSnapshot("plan", 5)
    app._agent_mode = "plan"
    app._agent_mode_revision = 5

    app._apply_mode_authority(AgentModeSnapshot("direct", 4))

    assert app._agent_mode == "plan"
    assert app._agent_mode_revision == 5


# 功能：验证 equal-revision conflict 只设置刷新标记且 delivery callback 不同步发 RPC
# 设计：fake client 记录调用，先 await delivery callback 再检查调用列表，随后清理 deferred worker
async def test_tui_mode_conflict_defers_and_coalesces_refresh() -> None:
    class _Client:
        # 记录 deferred worker 的 authority 查询次数
        def __init__(self) -> None:
            self.calls = 0

        # 独立 worker 调用的最小 mode response
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "session.get_agent_mode"
            self.calls += 1
            return {"agent_mode": "plan", "revision": 5}

    app = KamaTuiApp("127.0.0.1", 9999)
    client = _Client()
    app._client = client  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._mode_snapshot = AgentModeSnapshot("direct", 5)
    app._agent_mode = "direct"
    app._agent_mode_revision = 5

    delivery = EventDelivery(
        subscription_id="sub",
        delivery="live",
        event_id="mode-conflict",
        stream_id="session:sess-1",
        seq=4,
        daemon_instance_id="daemon-a",
        event={
            "type": "session.agent_mode_changed",
            "previous_mode": "direct",
            "agent_mode": "plan",
            "revision": 5,
            "ts": "t",
        },
    )
    await app._handle_delivery(delivery)
    assert client.calls == 0

    app._schedule_mode_refresh()
    app._schedule_mode_refresh()
    assert app._mode_refresh_task is not None
    await app._mode_refresh_task

    assert client.calls == 1
    assert app._agent_mode == "plan"


# 功能：验证 deferred mode response 不会污染 session replacement 后的新视图
# 设计：用两个 session owner 和 response gate 模拟旧 RPC 跨 await 返回，直接断言新视图保持 authority
async def test_tui_deferred_mode_refresh_discards_stale_session_response() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _Client:
        # 阻塞旧 session 的 mode 查询直到测试完成 view replacement
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "session.get_agent_mode"
            assert params["session_id"] == "sess-1"
            started.set()
            await release.wait()
            return {"agent_mode": "plan", "revision": 20}

    app = KamaTuiApp("127.0.0.1", 9999)
    client = _Client()
    app._client = client  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._mode_snapshot = AgentModeSnapshot("plan", 20)
    app._agent_mode = "plan"
    app._agent_mode_revision = 20

    app._schedule_mode_refresh()
    await started.wait()

    app._client = object()  # type: ignore[assignment]
    app._session_id = "sess-2"
    app._reconnect_state.daemon_instance_id = "daemon-b"
    app._mode_snapshot = AgentModeSnapshot("direct", 0)
    app._agent_mode = "direct"
    app._agent_mode_revision = 0
    release.set()

    assert app._mode_refresh_task is not None
    await app._mode_refresh_task

    assert app._agent_mode == "direct"
    assert app._agent_mode_revision == 0


# 功能：验证旧 client/daemon 的 deferred response 被丢弃而不依赖 revision 比较
# 设计：保持 session id 相同但替换 client 与 daemon identity，覆盖不同连接 lineage 的 owner 校验
async def test_tui_deferred_mode_refresh_discards_stale_client_response() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _OldClient:
        # 阻塞旧 client 的 mode 查询
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "session.get_agent_mode"
            started.set()
            await release.wait()
            return {"agent_mode": "plan", "revision": 20}

    app = KamaTuiApp("127.0.0.1", 9999)
    old_client = _OldClient()
    app._client = old_client  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._mode_snapshot = AgentModeSnapshot("direct", 0)

    app._schedule_mode_refresh()
    await started.wait()

    app._client = object()  # type: ignore[assignment]
    app._reconnect_state.daemon_instance_id = "daemon-b"
    release.set()
    assert app._mode_refresh_task is not None
    await app._mode_refresh_task

    assert app._agent_mode == "direct"
    assert app._agent_mode_revision == 0


# 功能：验证 TUI PlanReady 绑定 exact approval target，并在独立 worker 中查询 authority
# 设计：fake client 记录 get_approval 调用，先处理真实 event 再等待 refresh task，覆盖 callback 不同步 RPC 的边界
async def test_tui_committed_plan_queries_exact_approval_authority() -> None:
    class _Client:
        # 返回与 committed PlanView 完全一致的 pending authority
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "plan.get_approval"
            assert params["projection_key"] == "pv1:run-1:decision-1:v1"
            return {
                "session_id": "sess-1",
                "projection_key": "pv1:run-1:decision-1:v1",
                "status": "pending",
                "decision_id": "decision-1",
                "decision_version": 1,
                "content_digest": "decision-digest",
                "commit_receipt_digest": "receipt-digest",
            }

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._append = lambda _widget: None  # type: ignore[method-assign]

    app._handle_event(_plan_event())
    app._handle_event({"type": "run.finished", "run_id": "run-1", "status": "success"})
    projection_key = "pv1:run-1:decision-1:v1"
    app._schedule_approval_refresh(projection_key)
    assert app._approval_refresh_tasks[projection_key] is not None
    await app._approval_refresh_tasks[projection_key]

    state = app._approval_states[projection_key]
    assert state.snapshot.status == "pending"
    assert state.snapshot.commit_receipt_digest == "receipt-digest"


# 功能：验证 TUI approve control 使用 authority receipt 和 exact decision identity
# 设计：同一个 fake client 先响应 get 再响应 approve，直接运行 worker 方法而非只检查 slash 文案
async def test_tui_approval_control_merges_rpc_result() -> None:
    class _Client:
        # 返回 pending authority 或 approved command result
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            if method == "plan.get_approval":
                return {
                    "session_id": "sess-1",
                    "projection_key": "pv1:run-1:decision-1:v1",
                    "status": "pending",
                    "decision_id": "decision-1",
                    "decision_version": 1,
                    "content_digest": "decision-digest",
                    "commit_receipt_digest": "receipt-digest",
                }
            assert method == "plan.approve"
            assert params["content_digest"] == "decision-digest"
            assert params["commit_receipt_digest"] == "receipt-digest"
            return {
                "session_id": "sess-1",
                "projection_key": "pv1:run-1:decision-1:v1",
                "status": "approved",
                "decision_id": "decision-1",
                "decision_version": 1,
                "content_digest": "decision-digest",
                "commit_receipt_digest": "receipt-digest",
                "action": "approve",
                "record_digest": "record-a",
            }

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._append = lambda _widget: None  # type: ignore[method-assign]
    app._handle_event(_plan_event())
    app._handle_event({"type": "run.finished", "run_id": "run-1", "status": "success"})

    await app._do_approval_command("/approve", "approve")

    assert app._approval_states["pv1:run-1:decision-1:v1"].snapshot.status == "approved"


# 功能：验证 TUI approval event 冲突只渲染本地 conflicted/unknown 状态
# 设计：先应用 approved，再投递 disputed rejected event，确保 UI 不把 incoming terminal 当作 authority
def test_tui_approval_event_conflict_renders_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = object()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    rendered: list[str] = []
    monkeypatch.setattr(
        app,
        "_show_approval_snapshot",
        lambda snapshot: rendered.append(snapshot.status),
    )
    app._append = lambda _widget: None  # type: ignore[method-assign]
    app._handle_event(_plan_event())
    app._handle_event({"type": "run.finished", "run_id": "run-1", "status": "success"})
    projection_key = "pv1:run-1:decision-1:v1"

    app._handle_approval_event(
        {
            "type": "plan.approval_changed",
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "approved",
            "action": "approve",
            "record_digest": "record-a",
            "commit_receipt_digest": "receipt-digest",
        }
    )
    app._handle_approval_event(
        {
            "type": "plan.approval_changed",
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "rejected",
            "action": "reject",
            "record_digest": "record-b",
            "commit_receipt_digest": "receipt-digest",
        }
    )

    assert app._approval_states[projection_key].snapshot.status == "conflicted/unknown"
    assert rendered == ["approved", "conflicted/unknown"]
    assert "rejected" not in rendered


# 功能：验证 TUI approve/reject RPC 冲突只渲染本地 conflicted/unknown 状态
# 设计：先合并 approved authority，再合并 disputed rejected response，覆盖非 notification 的同一 UI 边界
def test_tui_approval_response_conflict_renders_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = object()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    rendered: list[str] = []
    monkeypatch.setattr(
        app,
        "_show_approval_snapshot",
        lambda snapshot: rendered.append(snapshot.status),
    )
    app._append = lambda _widget: None  # type: ignore[method-assign]
    app._handle_event(_plan_event())
    app._handle_event({"type": "run.finished", "run_id": "run-1", "status": "success"})
    projection_key = "pv1:run-1:decision-1:v1"

    app._handle_approval_event(
        {
            "type": "plan.approval_changed",
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "approved",
            "action": "approve",
            "record_digest": "record-a",
            "commit_receipt_digest": "receipt-digest",
        }
    )
    merged = app._merge_approval_response(
        {
            "session_id": "sess-1",
            "projection_key": projection_key,
            "status": "rejected",
            "action": "reject",
            "record_digest": "record-b",
            "commit_receipt_digest": "receipt-digest",
            "decision_id": "decision-1",
            "decision_version": 1,
            "content_digest": "decision-digest",
        }
    )

    assert merged is True
    assert app._approval_states[projection_key].snapshot.status == "conflicted/unknown"
    assert rendered == ["approved", "conflicted/unknown"]
    assert "rejected" not in rendered


# 功能：验证 TUI approval refresh 返回到被替换的 projection state 后不会更新新 view
# 设计：阻塞旧 GET，替换同 key 的 state object 后放行，覆盖 await 后 owner/state identity 检查
async def test_tui_stale_approval_refresh_state_is_discarded() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _Client:
        # 让旧 view 的 authority response 跨过 state replacement
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

    app = KamaTuiApp("127.0.0.1", 9999)
    client = _Client()
    app._client = client  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._append = lambda _widget: None  # type: ignore[method-assign]
    app._handle_event(_plan_event())
    app._handle_event({"type": "run.finished", "run_id": "run-1", "status": "success"})
    projection_key = "pv1:run-1:decision-1:v1"
    old_state = app._approval_states[projection_key]
    app._schedule_approval_refresh(projection_key)
    await started.wait()
    app._approval_states[projection_key] = ApprovalSnapshotState(old_state.owner)
    release.set()
    task = app._approval_refresh_tasks.get(projection_key)
    assert task is not None
    await task

    assert app._approval_states[projection_key].snapshot.status == "pending"


# 功能：验证 /plan、/direct、/mode worker 返回后若 view owner 已替换，不渲染旧结果
# 设计：参数化三条 mode command，共用 response gate 覆盖 command 与 query 两种 RPC 路径
@pytest.mark.parametrize("content", ["/plan", "/direct", "/mode"])
async def test_tui_mode_command_discards_stale_owner_response(content: str) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    rendered: list[str] = []

    class _Client:
        # 阻塞旧 owner 的 set mode 命令
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            expected_method = (
                "session.get_agent_mode"
                if content == "/mode"
                else "session.set_agent_mode"
            )
            assert method == expected_method
            assert params["session_id"] == "sess-1"
            started.set()
            await release.wait()
            return {"agent_mode": "plan", "revision": 20}

    app = KamaTuiApp("127.0.0.1", 9999)
    old_client = _Client()
    app._client = old_client  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._append = lambda widget: rendered.append(str(widget))  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]

    command_task = asyncio.create_task(app._do_mode_command(content))
    await started.wait()
    app._client = object()  # type: ignore[assignment]
    app._session_id = "sess-2"
    app._reconnect_state.daemon_instance_id = "daemon-b"
    app._mode_snapshot = AgentModeSnapshot("direct", 0)
    release.set()
    await command_task

    assert app._agent_mode == "direct"
    assert app._agent_mode_revision == 0
    assert rendered == []


# 功能：验证当前 owner 的 mode command 仍应用 authority 并渲染成功提示
# 设计：不替换 owner，使用最小 fake client 证明正常路径没有被 stale guard 误拒绝
async def test_tui_mode_command_applies_current_owner_response() -> None:
    rendered: list[str] = []

    class _Client:
        # 返回当前 owner 的完整 mode response
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "session.set_agent_mode"
            assert params["session_id"] == "sess-1"
            return {"agent_mode": "plan", "revision": 1}

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._append = lambda widget: rendered.append(str(widget))  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]

    await app._do_mode_command("/plan")

    assert app._agent_mode == "plan"
    assert app._agent_mode_revision == 1
    assert len(rendered) == 1


# 功能：验证 mode RPC 缺少任一必需字段时 fail closed 而不合成默认 snapshot
# 设计：参数化缺失 mode/revision 两种 wire contract 破坏，区分新 RPC 严格性与 legacy event 兼容
@pytest.mark.parametrize("payload", [{"agent_mode": "plan"}, {"revision": 5}])
async def test_tui_mode_rpc_missing_fields_fails_closed(
    payload: dict[str, object],
) -> None:
    class _Client:
        # 返回缺少一个必需字段的非法新 mode response
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            return payload

    app = KamaTuiApp("127.0.0.1", 9999)

    assert await app._refresh_agent_mode(_Client(), "sess-1") is None  # type: ignore[arg-type]


# 功能：验证 handler 失败不提交 cursor，同 daemon 重连后仍从旧 seq 重放
# 设计：用 Event 门控首次 handler 异常与第二次成功，不依赖 sleep 或 task 私有字段
async def test_handler_failure_keeps_delivery_retryable_after_same_daemon_reconnect() -> None:
    state = _TuiReconnectState(daemon_instance_id="daemon-a")
    delivery = _delivery(event_id="evt-1", stream_id="run:run-1", seq=4)
    entered = asyncio.Event()
    attempts = 0

    async def handler(_: dict[str, Any]) -> None:
        nonlocal attempts
        attempts += 1
        entered.set()
        if attempts == 1:
            raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        await state.process_delivery(delivery, handler)
    await entered.wait()

    assert state.stream_cursors == {}
    assert state.accept_daemon("daemon-a") == "same"
    assert state.subscription_target(stream_id="run:run-1") == ("run:run-1", 0)

    await state.process_delivery(delivery, handler)

    assert attempts == 2
    assert state.stream_cursors == {"run:run-1": 4}
    assert state.rendered_event_ids == {"evt-1"}


# 功能：验证首次连接先握手取得 daemon identity，再创建 session 并订阅 session stream
# 设计：在 subscribe response 返回前投递真实 delivery，用 cursor 证明 gate 直到响应后才放行
async def test_initial_connection_uses_response_gate_then_session_stream() -> None:
    delivery = _delivery(event_id="evt-1", stream_id="session:sess-new", seq=1)
    client = _ScriptedClient(daemon_instance_id="daemon-a", delivery=delivery)
    app = KamaTuiApp("127.0.0.1", 9999)
    rendered: list[dict[str, Any]] = []
    app._handle_event_inner = rendered.append  # type: ignore[method-assign]
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await client.subscribed.wait()
    for _ in range(20):
        if rendered:
            break
        await asyncio.sleep(0)

    assert app._session_id == "sess-new"
    assert app._reconnect_state.daemon_instance_id == "daemon-a"
    assert rendered == [delivery.event]
    assert app._stream_cursors == {"session:sess-new": 1}
    assert client.commands[:3] == [
        ("event.subscribe", {"topics": [], "scope": "global"}),
        ("event.unsubscribe", {"subscription_id": "sub-handshake"}),
        (
            "session.create",
            {"mode": "chat", "workspace_root": str(Path.cwd().resolve())},
        ),
    ]
    assert client.commands[3][1]["scope"] == "session:sess-new"
    assert client.commands[3][1]["after_seq"] == 0
    assert client.commands[4][0] == "session.get_agent_mode"

    client.release.set()
    await task
    assert client.closed


@pytest.mark.parametrize(
    ("busy", "pending_permission"),
    [(False, False), (True, False), (True, True)],
)
# 功能：验证 idle、running 和 permission 三种断线在同 daemon 上均原位续接
# 设计：参数化三个状态快照，只放行一次重连订阅并检查无 session.create
async def test_same_daemon_connection_preserves_idle_running_and_permission(
    busy: bool,
    pending_permission: bool,
) -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-a")
    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._busy = busy
    app._stream_cursors["session:sess-1"] = 8
    if pending_permission:
        app._reconnect_state.pending_permission_ids.add("perm-1")
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await client.subscribed.wait()

    assert app._session_id == "sess-1"
    assert app._busy is busy
    assert app._reconnect_state.pending_permission_ids == ({"perm-1"} if pending_permission else set())
    assert not any(method == "session.create" for method, _ in client.commands)
    durable = [params for method, params in client.commands if method == "event.subscribe"][-1]
    assert durable["scope"] == "session:sess-1"
    assert durable["after_seq"] == 8

    client.release.set()
    await task


# 功能：验证握手阶段 EOF 被归类为 transient disconnect，不创建 session
# 设计：fake RPC future 以 CancelledError 表示 SocketClient EOF 取消，与外层 task cancellation 分开
async def test_handshake_disconnect_is_retryable_and_does_not_create_session() -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-a", fail_handshake=True)
    app = KamaTuiApp("127.0.0.1", 9999)

    with pytest.raises(_ConnectionLost):
        await app._run_connection(client)  # type: ignore[arg-type]

    assert app._session_id is None
    assert [method for method, _ in client.commands] == ["event.subscribe"]
    assert client.closed


# 功能：验证 session.create 响应丢失后不在同 daemon 上重发非幂等创建
# 设计：在已知 daemon identity 后取消 create future，断言记录 unknown 并抛出明确终态
async def test_session_create_response_loss_is_explicitly_unknown() -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-a", fail_create_response=True)
    app = KamaTuiApp("127.0.0.1", 9999)

    with pytest.raises(_SessionCreateOutcomeUnknown):
        await app._run_connection(client)  # type: ignore[arg-type]

    assert app._reconnect_state.session_create_unknown_daemon_id == "daemon-a"
    assert app._session_id is None
    assert [method for method, _ in client.commands].count("session.create") == 1

    retry = _ScriptedClient(daemon_instance_id="daemon-a")
    with pytest.raises(_SessionCreateOutcomeUnknown):
        await app._run_connection(retry)  # type: ignore[arg-type]
    assert not any(method == "session.create" for method, _ in retry.commands)


# 功能：验证 create 响应丢失后同 daemon 不重发，但新 daemon 会清 marker 并恢复一次创建
# 设计：串联 A-丢响应、A-重握手、B-新实例三条连接，用 ready Event 精确终止
async def test_unknown_create_retries_handshake_until_new_daemon_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_a = _ScriptedClient(
        daemon_instance_id="daemon-a",
        fail_create_response=True,
    )
    second_a = _ScriptedClient(daemon_instance_id="daemon-a")
    daemon_b = _ScriptedClient(daemon_instance_id="daemon-b", session_id="sess-b")
    clients = [first_a, second_a, daemon_b]
    ready = asyncio.Event()

    # 依次交付两个旧 daemon 连接和一个新 daemon 连接
    def client_factory(host: str, port: int) -> _ScriptedClient:
        return clients.pop(0)

    # 跳过退避墙钟时间但保留重连调度点
    async def no_wait() -> None:
        return None

    monkeypatch.setattr(tui_app_module, "SocketClient", client_factory)
    app = KamaTuiApp("127.0.0.1", 9999)
    app._append = lambda widget: None  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._wait_before_reconnect = no_wait  # type: ignore[method-assign]
    app._set_connected_ui = ready.set  # type: ignore[method-assign]

    socket_task = asyncio.create_task(app._socket_loop())
    await ready.wait()

    assert [method for method, _ in first_a.commands].count("session.create") == 1
    assert [method for method, _ in second_a.commands].count("session.create") == 0
    assert [method for method, _ in daemon_b.commands].count("session.create") == 1
    assert app._session_id == "sess-b"

    socket_task.cancel()
    await asyncio.gather(socket_task, return_exceptions=True)


# 功能：验证真实 --replay 连接不创建 chat session，且按 run cursor 续传
# 设计：运行完整单次连接脚本，直接审计发往 event.subscribe 的 scope/after_seq
async def test_replay_connection_subscribes_run_cursor_without_session_create() -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-a")
    app = KamaTuiApp("127.0.0.1", 9999, replay_run_id="run-7")
    app._stream_cursors["run:run-7"] = 12
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await client.subscribed.wait()

    assert not any(method == "session.create" for method, _ in client.commands)
    durable = [params for method, params in client.commands if method == "event.subscribe"][-1]
    assert durable["scope"] == "run:run-7"
    assert durable["after_seq"] == 12

    client.release.set()
    await task


# 功能：验证 daemon 变化后历史 replay 仍从已提交 run cursor 继续而不混入 chat session
# 设计：先记录旧 daemon 的 replay cursor，再用新 identity 握手并审计 durable 订阅
async def test_replay_cursor_survives_daemon_change_for_historical_stream() -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-b")
    app = KamaTuiApp("127.0.0.1", 9999, replay_run_id="run-7")
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._stream_cursors["run:run-7"] = 12
    app._rendered_event_ids.add("evt-old")
    notices: list[str] = []
    ready = asyncio.Event()
    app._append = lambda widget: notices.append(str(widget.content))  # type: ignore[method-assign]
    app._set_connected_ui = ready.set  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await ready.wait()

    durable = [params for method, params in client.commands if method == "event.subscribe"][-1]
    assert durable["scope"] == "run:run-7"
    assert durable["after_seq"] == 12
    assert app._rendered_event_ids == {"evt-old"}
    assert any("historical replay continuing" in notice for notice in notices)

    client.release.set()
    await task


# 功能：验证新 daemon 没有旧 run stream owner 时 replay 明确不可用且稳定停止
# 设计：在 durable subscribe 注入含 secret 的 IpcError，断言不提前宣称 continuing、不重试、不泄漏
async def test_replay_new_daemon_unknown_stream_stops_without_false_continuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kama_claude.core.transport.socket_client import IpcError

    client = _ScriptedClient(
        daemon_instance_id="daemon-b",
        durable_error=IpcError(-32603, "secret-old-run-owner"),
    )

    # 只允许一次新 daemon 连接，任何 transient 重试都会立即暴露
    def client_factory(host: str, port: int) -> _ScriptedClient:
        return client

    # 跳过退避墙钟时间，使错误 transient 重试能立即被命令计数捕获
    async def no_wait() -> None:
        return None

    monkeypatch.setattr(tui_app_module, "SocketClient", client_factory)
    app = KamaTuiApp("127.0.0.1", 9999, replay_run_id="run-old")
    app._reconnect_state.daemon_instance_id = "daemon-a"
    notices: list[str] = []
    app._append = lambda widget: notices.append(str(widget.content))  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._wait_before_reconnect = no_wait  # type: ignore[method-assign]

    await app._socket_loop()

    rendered = "\n".join(notices)
    assert "active execution is not recoverable" in rendered
    assert "historical replay unavailable" in rendered
    assert "historical replay continuing" not in rendered
    assert "secret-old-run-owner" not in rendered
    assert [method for method, _ in client.commands].count("event.subscribe") == 2


# 功能：验证 daemon 变化在创建新 session 前清理真实 pending 容器并显示恢复边界
# 设计：同时填充状态对象和 App widget 索引，通过完整握手证明两层都被清理
async def test_daemon_change_clears_widget_indexes_before_new_session() -> None:
    client = _ScriptedClient(daemon_instance_id="daemon-b", session_id="sess-new")
    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-old"
    app._busy = True
    app._pending_tool_blocks["tool-old"] = object()  # type: ignore[assignment]
    app._pending_permission_blocks["perm-old"] = object()  # type: ignore[assignment]
    app._permission_owners["perm-old"] = ("daemon-a", "sess-old")
    app._reconnect_state.pending_tool_ids.add("tool-old")
    app._reconnect_state.pending_permission_ids.add("perm-old")
    notices: list[str] = []
    app._append = lambda widget: notices.append(str(widget.content))  # type: ignore[method-assign]
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await client.subscribed.wait()

    assert app._session_id == "sess-new"
    assert not app._busy
    assert app._pending_tool_blocks == {}
    assert app._pending_permission_blocks == {}
    assert app._permission_owners == {}
    assert any("daemon restarted" in notice for notice in notices)

    client.release.set()
    await task


# 功能：验证 daemon 变化后旧权限控件消息不会发往新 session
# 设计：保留旧 owner 快照却切换当前 daemon/session，断言 permission.respond 零调用
async def test_stale_permission_decision_is_not_sent_to_new_session() -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        # 记录任何意外的权限响应
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            sent.append((method, params))
            return {"ok": True}

    class _Widget:
        removed = False

        # 模拟移除已失效的权限选择控件
        def remove(self) -> None:
            self.removed = True

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._reconnect_state.daemon_instance_id = "daemon-b"
    app._session_id = "sess-new"
    app._permission_owners["perm-old"] = ("daemon-a", "sess-old")
    app._reconnect_state.pending_permission_ids.add("perm-old")
    widget = _Widget()
    message = SimpleNamespace(
        tool_use_id="perm-old",
        decision="allow_once",
        widget=widget,
    )

    await app.on_permission_select_decided(message)  # type: ignore[arg-type]

    assert sent == []
    assert widget.removed
    assert "perm-old" not in app._reconnect_state.pending_permission_ids


# 功能：验证同 daemon 断线期间的权限点击不会吞掉 pending 决策
# 设计：保留有效 owner 但将 client 置空，断言控件和 owner 继续等待同 daemon 重连
async def test_permission_decision_while_disconnected_remains_pending() -> None:
    class _Widget:
        removed = False

        # 记录控件是否被过早移除
        def remove(self) -> None:
            self.removed = True

    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._permission_owners["perm-1"] = ("daemon-a", "sess-1")
    app._reconnect_state.pending_permission_ids.add("perm-1")
    widget = _Widget()
    message = SimpleNamespace(
        tool_use_id="perm-1",
        decision="allow_once",
        widget=widget,
    )

    await app.on_permission_select_decided(message)  # type: ignore[arg-type]

    assert not widget.removed
    assert app._permission_owners == {"perm-1": ("daemon-a", "sess-1")}
    assert app._reconnect_state.pending_permission_ids == {"perm-1"}


# 功能：验证 permission.respond 在 RPC 成功前不提交任何 UI/pending 状态
# 设计：用 entered/release Event 将 RPC 悬停在响应前，分别断言响应前保留、响应后清理
async def test_permission_state_commits_only_after_rpc_success() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Client:
        # 在权限 RPC 返回前挂起，暴露真正 commit boundary
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            entered.set()
            await release.wait()
            return {"ok": True}

    class _Widget:
        removed = False

        # 记录权限控件只在成功后被移除
        def remove(self) -> None:
            self.removed = True

    class _Block:
        resolved: str | None = None

        # 记录权限摘要只在 RPC 成功后收缩
        def _resolve(self, decision: str) -> None:
            self.resolved = decision

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._permission_owners["perm-1"] = ("daemon-a", "sess-1")
    app._reconnect_state.pending_permission_ids.add("perm-1")
    block = _Block()
    app._pending_permission_blocks["perm-1"] = block  # type: ignore[assignment]
    widget = _Widget()
    app._permission_selects["perm-1"] = widget  # type: ignore[assignment]
    message = SimpleNamespace(tool_use_id="perm-1", decision="allow_once", widget=widget)

    task = asyncio.create_task(
        app.on_permission_select_decided(message)  # type: ignore[arg-type]
    )
    await entered.wait()

    assert not widget.removed
    assert block.resolved is None
    assert app._permission_owners == {"perm-1": ("daemon-a", "sess-1")}
    assert app._reconnect_state.pending_permission_ids == {"perm-1"}

    release.set()
    await task

    assert widget.removed
    assert block.resolved == "allow_once"
    assert app._permission_owners == {}
    assert app._reconnect_state.pending_permission_ids == set()


@pytest.mark.parametrize("failure_kind", ["ipc", "eof"])
# 功能：验证 permission.respond transport/IpcError 保留可重试状态且不泄漏异常文本
# 设计：分别注入含 secret 的 IpcError 与 EOF cancellation，观测完整 pending/UI 快照
async def test_permission_rpc_failure_preserves_retry_state_without_secret(
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
) -> None:
    class _Client:
        # 模拟服务端权限响应失败并携带不得泄漏的文本
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            from kama_claude.core.transport.socket_client import IpcError

            if failure_kind == "eof":
                raise asyncio.CancelledError
            raise IpcError(-32000, "secret-permission-token")

    class _Widget:
        removed = False

        # 记录失败时控件仍留在 UI 中
        def remove(self) -> None:
            self.removed = True

    class _Prompt:
        disabled = True
        read_only = False
        border_title = "permission required"
        focused = False

        # 记录失败时不应恢复输入焦点
        def focus(self) -> None:
            self.focused = True

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._permission_owners["perm-1"] = ("daemon-a", "sess-1")
    app._reconnect_state.pending_permission_ids.add("perm-1")
    block = object()
    app._pending_permission_blocks["perm-1"] = block  # type: ignore[assignment]
    prompt = _Prompt()
    app._prompt = lambda: prompt  # type: ignore[method-assign]
    widget = _Widget()
    caplog.set_level(logging.WARNING)

    await app.on_permission_select_decided(  # type: ignore[arg-type]
        SimpleNamespace(tool_use_id="perm-1", decision="allow_once", widget=widget)
    )

    assert not widget.removed
    assert app._pending_permission_blocks == {"perm-1": block}
    assert app._permission_owners == {"perm-1": ("daemon-a", "sess-1")}
    assert app._reconnect_state.pending_permission_ids == {"perm-1"}
    assert prompt.disabled
    assert not prompt.focused
    assert "secret-permission-token" not in caplog.text
    assert "role=permission_respond" in caplog.text


# 功能：验证 permission.respond 响应丢失后，同 daemon 重放 granted 会终结旧 pending 且不重发决策
# 设计：首次 RPC 用 internal CancelledError 模拟 EOF，再投递真实 durable granted delivery 验证单一终态路径
async def test_replayed_permission_granted_finishes_lost_response_without_resend() -> None:
    sent: list[tuple[str, dict[str, Any]]] = []

    class _Client:
        # 记录唯一一次权限 RPC 后以 EOF cancellation 丢失响应
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            sent.append((method, params))
            raise asyncio.CancelledError

    class _Widget:
        removed = False

        # 记录 replay terminal event 是否移除旧权限控件
        def remove(self) -> None:
            self.removed = True

    class _Block:
        resolved: str | None = None

        # 记录 granted 事件传递的最终 decision
        def _resolve(self, decision: str) -> None:
            self.resolved = decision

    class _Prompt:
        disabled = True
        read_only = False
        border_title = "permission required"
        focused = False

        # 记录终态处理后的焦点恢复
        def focus(self) -> None:
            self.focused = True

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._busy = False
    app._permission_owners["perm-1"] = ("daemon-a", "sess-1")
    app._reconnect_state.pending_permission_ids.add("perm-1")
    block = _Block()
    widget = _Widget()
    prompt = _Prompt()
    app._pending_permission_blocks["perm-1"] = block  # type: ignore[assignment]
    app._permission_selects["perm-1"] = widget  # type: ignore[attr-defined]
    app._prompt = lambda: prompt  # type: ignore[method-assign]

    await app.on_permission_select_decided(  # type: ignore[arg-type]
        SimpleNamespace(tool_use_id="perm-1", decision="allow_once", widget=widget)
    )
    assert app._reconnect_state.pending_permission_ids == {"perm-1"}

    app._client = None
    granted = EventDelivery(
        subscription_id="sub-session",
        delivery="replay",
        event_id="evt-granted",
        stream_id="session:sess-1",
        seq=9,
        daemon_instance_id="daemon-a",
        event={
            "type": "permission.granted",
            "run_id": "run-1",
            "tool_use_id": "perm-1",
            "decision": "allow_once",
            "ts": "t",
        },
    )
    await app._handle_delivery(granted)

    assert len(sent) == 1
    assert app._pending_permission_blocks == {}
    assert app._permission_owners == {}
    assert app._reconnect_state.pending_permission_ids == set()
    assert block.resolved == "allow_once"
    assert widget.removed
    assert not prompt.disabled
    assert prompt.focused


@pytest.mark.parametrize(
    ("event_type", "decision", "busy", "prompt_disabled"),
    [
        ("permission.denied", "deny_once", False, False),
        ("permission.granted", "auto_allow", True, True),
    ],
)
# 功能：验证 granted/denied 共用终态清理，且仅在非 busy 时恢复 prompt
# 设计：参数化旧 denied 语义与 busy granted 边界，观测 block、owner、widget 和 prompt
async def test_permission_terminal_events_share_cleanup_without_busy_prompt_regression(
    event_type: str,
    decision: str,
    busy: bool,
    prompt_disabled: bool,
) -> None:
    class _Widget:
        removed = False

        # 记录 terminal event 移除对应 tool_use_id 控件
        def remove(self) -> None:
            self.removed = True

    class _Block:
        resolved: str | None = None

        # 记录 granted/denied 共用路径的 decision
        def _resolve(self, value: str) -> None:
            self.resolved = value

    class _Prompt:
        disabled = True
        read_only = False
        border_title = "permission required"

        # 非 busy 终态才恢复焦点
        def focus(self) -> None:
            return None

    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._busy = busy
    app._permission_owners["perm-1"] = ("daemon-a", "sess-1")
    app._reconnect_state.pending_permission_ids.add("perm-1")
    block = _Block()
    widget = _Widget()
    prompt = _Prompt()
    app._pending_permission_blocks["perm-1"] = block  # type: ignore[assignment]
    app._permission_selects["perm-1"] = widget  # type: ignore[attr-defined]
    app._prompt = lambda: prompt  # type: ignore[method-assign]

    app._handle_event_inner(
        {
            "type": event_type,
            "run_id": "run-1",
            "tool_use_id": "perm-1",
            "decision": decision,
            "ts": "t",
        }
    )

    assert app._pending_permission_blocks == {}
    assert app._permission_owners == {}
    assert app._reconnect_state.pending_permission_ids == set()
    assert block.resolved == decision
    assert widget.removed
    assert prompt.disabled is prompt_disabled


# 功能：验证 delivery handler 异常会驱动 socket loop 换连接并从旧 cursor 自动重放
# 设计：两个真实 fake client 依次投递同一 delivery，用 Event 确认第二次成功而非 sleep
async def test_handler_failure_drives_automatic_same_daemon_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = _delivery(event_id="evt-retry", stream_id="session:sess-1", seq=5)
    first = _ScriptedClient(daemon_instance_id="daemon-a", delivery=delivery)
    second = _ScriptedClient(daemon_instance_id="daemon-a", delivery=delivery)
    clients = [first, second]

    # 按顺序为 socket loop 提供两个独立连接
    def client_factory(host: str, port: int) -> _ScriptedClient:
        return clients.pop(0)

    monkeypatch.setattr(tui_app_module, "SocketClient", client_factory)
    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    rendered = asyncio.Event()
    attempts = 0

    # 首次渲染失败，第二次成功并通知测试结束
    def render(_: dict[str, Any]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("render failed")
        rendered.set()

    # 跳过生产退避时间，保留重连调度点
    async def no_wait() -> None:
        return None

    app._handle_event_inner = render  # type: ignore[method-assign]
    app._wait_before_reconnect = no_wait  # type: ignore[method-assign]
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]

    socket_task = asyncio.create_task(app._socket_loop())
    await rendered.wait()

    assert attempts == 2
    assert app._stream_cursors == {"session:sess-1": 5}
    assert first.closed

    socket_task.cancel()
    await asyncio.gather(socket_task, return_exceptions=True)
    assert second.closed


# 功能：验证重连尝试在连续握手断线后有界结束并显示固定提示
# 设计：五个 Event 驱动 fake 都在 handshake EOF，将退避替换为可计数协程而非 sleep
async def test_reconnect_attempts_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = [
        _ScriptedClient(daemon_instance_id="daemon-a", fail_handshake=True)
        for _ in range(5)
    ]
    waits = 0

    # 按顺序返回每次都在握手阶段断开的客户端
    def client_factory(host: str, port: int) -> _ScriptedClient:
        return clients.pop(0)

    # 记录退避调度点而不消耗墙钟时间
    async def no_wait() -> None:
        nonlocal waits
        waits += 1

    monkeypatch.setattr(tui_app_module, "SocketClient", client_factory)
    app = KamaTuiApp("127.0.0.1", 9999)
    notices: list[str] = []
    app._append = lambda widget: notices.append(str(widget.content))  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._wait_before_reconnect = no_wait  # type: ignore[method-assign]

    await app._socket_loop()

    assert clients == []
    assert waits == 4
    assert any("reconnect exhausted" in notice for notice in notices)


# 功能：验证每次完成 durable subscribe+gate.open 都会重置连续失败预算
# 设计：五次成功恢复后主动 EOF，第六次保持在线；与 exhausted Event 竞速避免 timeout/sleep
async def test_successful_reattach_resets_consecutive_failure_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovered_clients = [
        _ScriptedClient(daemon_instance_id="daemon-a") for _ in range(6)
    ]
    for client in recovered_clients[:5]:
        client.release.set()
    sixth = recovered_clients[5]
    clients = list(recovered_clients)
    exhausted = asyncio.Event()

    # 顺序交付六次都完成 durable 订阅的连接
    def client_factory(host: str, port: int) -> _ScriptedClient:
        return clients.pop(0)

    # 跳过退避墙钟时间但保留尝试边界
    async def no_wait() -> None:
        return None

    # 若错误耗尽预算，通过提示文本触发竞速失败分支
    def append_notice(widget: Any) -> None:
        if "reconnect exhausted" in str(widget.content):
            exhausted.set()

    monkeypatch.setattr(tui_app_module, "SocketClient", client_factory)
    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._append = append_notice  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]
    app._wait_before_reconnect = no_wait  # type: ignore[method-assign]

    socket_task = asyncio.create_task(app._socket_loop())
    sixth_wait = asyncio.create_task(sixth.subscribed.wait())
    exhausted_wait = asyncio.create_task(exhausted.wait())
    done, pending = await asyncio.wait(
        {sixth_wait, exhausted_wait},
        return_when=asyncio.FIRST_COMPLETED,
    )

    assert sixth_wait in done
    assert not exhausted.is_set()

    for waiter in pending:
        waiter.cancel()
    socket_task.cancel()
    await asyncio.gather(socket_task, *pending, return_exceptions=True)


# 功能：验证 socket worker 取消时即使 cleanup 期间再次取消也会关闭连接并保留首次语义
# 设计：用 close entered/release 两个 Event 将第二次 cancel 精确注入 cleanup，不读 task 私有字段
async def test_repeated_cancellation_waits_for_connection_cleanup() -> None:
    close_entered = asyncio.Event()
    close_release = asyncio.Event()

    class _CleanupClient(_ScriptedClient):
        # 在关闭边界挂起，便于注入第二次 cancellation
        async def close(self) -> None:
            close_entered.set()
            await close_release.wait()
            self.closed = True

    client = _CleanupClient(daemon_instance_id="daemon-a")
    app = KamaTuiApp("127.0.0.1", 9999)
    app._reconnect_state.daemon_instance_id = "daemon-a"
    app._session_id = "sess-1"
    app._set_connected_ui = lambda: None  # type: ignore[method-assign]

    task = asyncio.create_task(app._run_connection(client))  # type: ignore[arg-type]
    await client.subscribed.wait()
    task.cancel("outer-cancel")
    await close_entered.wait()
    task.cancel("cleanup-cancel")
    close_release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await task

    assert caught.value.args == ("outer-cancel",)
    assert client.closed
