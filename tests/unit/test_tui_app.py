from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from rich.markdown import Markdown
from textual.widget import Widget

import kama_claude.tui.app as tui_app_module
from kama_claude.core.transport.socket_client import EventDelivery
from kama_claude.tui.app import (
    KamaTuiApp,
    LLMStreamBlock,
    ToolCallBlock,
    _param_summary,
    _preview,
)


# 构造带指定持久元数据的真实 delivery 供 TUI cursor 测试使用
def _delivery(
    *,
    event: dict[str, Any],
    event_id: str | None,
    stream_id: str | None,
    seq: int | None,
) -> EventDelivery:
    return EventDelivery(
        subscription_id="sub-test",
        delivery="replay",
        event_id=event_id,
        stream_id=stream_id,
        seq=seq,
        daemon_instance_id="daemon-test",
        event=event,
    )


# 功能：验证 _preview 超出长度时截断并追加省略号
# 设计：不依赖任何 TUI 组件，纯函数测试
def test_preview_truncates() -> None:
    assert _preview("abcde", 3) == "abc…"
    assert _preview("ab", 5) == "ab"


# 功能：验证工具参数摘要优先展示工具最关键字段
# 设计：覆盖 read_file/bash/note_save 三类常见工具，避免工具块摘要退化成整段 JSON
def test_param_summary_prefers_key_fields() -> None:
    assert _param_summary("read_file", {"path": "README.md"}) == "path='README.md'"
    assert _param_summary("bash", {"command": "echo hi", "timeout": 1}) == "command='echo hi'"
    assert _param_summary("note_save", {"content": "Python 3.12"}) == "content='Python 3.12'"


# 功能：验证 llm.token 事件累积到 LLMStreamBlock，不连续 token 各自新开一块
# 设计：monkey-patch _append 收集追加的 widgets，断言 token 追加到同一块；
#       发送非 token 事件后新 block 被重置，下一个 token 开启新块
def test_llm_tokens_accumulate_in_block() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "Hello", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "llm.token", "token": " world", "run_id": "r", "ts": "t"})

    assert len(appended) == 1  # same block reused
    assert isinstance(appended[0], LLMStreamBlock)
    assert appended[0]._text == "Hello world"  # type: ignore[attr-defined]


# 功能：验证 LLMStreamBlock 结束时会把累积文本渲染为 Rich Markdown
# 设计：直接调用 finalize_markdown，断言 renderable 类型，覆盖 Markdown polish 的核心行为
def test_llm_block_finalize_renders_markdown() -> None:
    block = LLMStreamBlock()
    block.append_token("## Title\n\n- one\n\n```python\nprint('hi')\n```")
    block.finalize_markdown()
    assert isinstance(block.content, Markdown)


# 功能：验证非 token 事件后 _current_llm 被重置，下一个 token 开启新块
# 设计：插入 step.started 中断流，验证之前的 block 被 finalize，之后的 llm.token 创建新 LLMStreamBlock
def test_llm_block_resets_after_non_token_event() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "A", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "step.started", "run_id": "r", "step": 2, "ts": "t"})
    app._handle_event({"type": "llm.token", "token": "B", "run_id": "r", "ts": "t"})

    llm_blocks = [w for w in appended if isinstance(w, LLMStreamBlock)]
    assert len(llm_blocks) == 2
    assert llm_blocks[0]._finalized  # type: ignore[attr-defined]


# 功能：验证 run.started 事件追加 Static widget 且包含 run_id 和 goal
# 设计：monkey-patch _append，断言追加的 widget 的 renderable 包含关键字段
def test_run_started_appends_widget_with_content() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.started", "run_id": "run-abc", "goal": "do the thing", "ts": "t"
    })

    assert len(appended) == 1
    rendered = appended[0].content
    assert "run-abc" in rendered
    assert "do the thing" in rendered


# 功能：验证 run.finished success 追加包含 "completed" 的 widget
# 设计：monkey-patch _append，检查 rendered 内容包含 completed 和 green
def test_run_finished_success_shows_completed() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "success", "steps": 3, "ts": "t"
    })

    rendered = appended[0].content
    assert "completed" in rendered
    assert "green" in rendered


# 功能：验证 run.finished failed 追加包含 "failed" 和 red 的 widget
# 设计：与 success 对称，检查颜色标记差异
def test_run_finished_failed_shows_red() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "failed",
        "steps": 1, "reason": "llm_error", "ts": "t"
    })

    rendered = appended[0].content
    assert "failed" in rendered
    assert "red" in rendered


# 功能：验证 tool.call_started 追加 ToolCallBlock，call_finished 更新其结果
# 设计：直接调用 _handle_event 两次，通过 _pending_tool_blocks 验证状态流转
def test_tool_call_started_and_finished() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "tool.call_started",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "params": {"command": "echo hi"},
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" in app._pending_tool_blocks  # type: ignore[attr-defined]

    app._handle_event({
        "type": "tool.call_finished",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "elapsed_ms": 42,
        "output": "hi",
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" not in app._pending_tool_blocks  # type: ignore[attr-defined]
    block = appended[0]
    assert isinstance(block, ToolCallBlock)
    assert block._finished  # type: ignore[attr-defined]
    assert block._output == "hi"  # type: ignore[attr-defined]


# 功能：验证 note_save 成功完成时工具块摘要显示 remembered
# 设计：直接操作 ToolCallBlock，覆盖 note_save 的特殊低噪声展示策略
def test_note_save_tool_block_shows_remembered() -> None:
    block = ToolCallBlock("note_save", {"content": "Python 3.12"})
    block.set_result("saved", 3)
    assert "remembered" in block._summary()  # type: ignore[attr-defined]


# 功能：验证提交用户输入时会追加 user turn，并进入 busy 状态
# 设计：用 fake client 替代 SocketClient，直接调用 on_chat_text_area_submitted，
#       覆盖 TextArea 清空内容 + 设置 busy 占位符的核心状态迁移
async def test_input_submit_appends_user_turn_and_disables_prompt() -> None:
    class _FakeArea:
        def __init__(self) -> None:
            self.disabled = False
            self.border_title = ""
            self.text = "hello"

    class _FakeEvent:
        def __init__(self, area: _FakeArea) -> None:
            self.value = area.text
            self.text_area = area

    class _FakeClient:
        async def send_command(self, method: str, params: dict) -> dict:
            return {"run_id": "run-1"}

    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._client = _FakeClient()  # type: ignore[assignment]
    app._session_id = "sess-1"

    area = _FakeArea()
    event = _FakeEvent(area)
    await app.on_chat_text_area_submitted(event)  # type: ignore[arg-type]

    assert app._busy  # type: ignore[attr-defined]
    assert area.disabled
    assert area.text == ""
    assert "agent is working" in area.border_title.lower()
    assert appended[0].content == "[bold]>[/bold] hello"


# 功能：验证未知事件类型不抛异常也不追加任何 widget
# 设计：发送 type 为 unknown 的事件，断言 appended 为空
def test_unknown_event_silently_ignored() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "some.unknown.type", "run_id": "r", "ts": "t"})
    assert appended == []


# 功能：验证同一事件经 run/session 重叠订阅到达时两条 stream cursor 均推进但只渲染一次
# 设计：使用共享 event_id 的两个真实 EventDelivery，以可观测 render 计数区分 cursor 确认与 UI 去重
async def test_delivery_advances_overlapping_stream_cursors_and_renders_once() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    rendered: list[dict[str, Any]] = []
    app._handle_event_inner = rendered.append  # type: ignore[method-assign]
    event = {
        "type": "run.finished",
        "run_id": "run-1",
        "status": "success",
        "steps": 2,
        "ts": "t",
    }

    await app._handle_delivery(
        _delivery(event=event, event_id="evt-1", stream_id="run:run-1", seq=7)
    )
    await app._handle_delivery(
        _delivery(event=event, event_id="evt-1", stream_id="session:sess-1", seq=11)
    )

    assert rendered == [event]
    assert app._stream_cursors == {"run:run-1": 7, "session:sess-1": 11}


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "tool.call_started",
            "tool_use_id": "tool-1",
            "tool_name": "bash",
            "params": {"command": "true"},
            "run_id": "run-1",
            "ts": "t",
        },
        {"type": "llm.token", "token": "hello", "run_id": "run-1", "ts": "t"},
        {
            "type": "run.finished",
            "run_id": "run-1",
            "status": "success",
            "steps": 1,
            "ts": "t",
        },
    ],
)
# 功能：验证重连重放不会重复渲染工具块、LLM token 或 run terminal 块
# 设计：对三种有状态 UI 事件重放同一 event_id/seq，直接观测渲染调用次数和 cursor
async def test_replayed_delivery_does_not_repeat_stateful_render(
    event: dict[str, Any],
) -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    rendered: list[dict[str, Any]] = []
    app._handle_event_inner = rendered.append  # type: ignore[method-assign]
    delivery = _delivery(
        event=event,
        event_id="evt-stateful",
        stream_id="run:run-1",
        seq=3,
    )

    await app._handle_delivery(delivery)
    await app._handle_delivery(delivery)

    assert rendered == [event]
    assert app._stream_cursors == {"run:run-1": 3}


@pytest.mark.parametrize(
    ("event_id", "stream_id", "seq"),
    [
        (None, "run:run-1", 1),
        ("evt-live", None, 1),
        ("evt-live", "run:run-1", None),
    ],
)
# 功能：验证缺失任一持久元数据的 live-only 事件仍渲染且不污染 cursor/dedup 状态
# 设计：逐一缺失 event_id、stream_id、seq，重复投递以证明兼容路径没有持久去重副作用
async def test_live_only_delivery_does_not_change_durable_tracking(
    event_id: str | None,
    stream_id: str | None,
    seq: int | None,
) -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    rendered: list[dict[str, Any]] = []
    app._handle_event_inner = rendered.append  # type: ignore[method-assign]
    event = {"type": "llm.token", "token": "live", "run_id": "run-1", "ts": "t"}
    delivery = _delivery(
        event=event,
        event_id=event_id,
        stream_id=stream_id,
        seq=seq,
    )

    await app._handle_delivery(delivery)
    await app._handle_delivery(delivery)

    assert rendered == [event, event]
    assert app._stream_cursors == {}
    assert app._rendered_event_ids == set()


# 功能：验证事件 handler 失败时 cursor/dedup 不前移，后续重放仍可成功处理
# 设计：首次处理抛出固定异常，然后替换为可观测 handler 重放同一 delivery，断言提交发生在成功之后
async def test_delivery_handler_failure_leaves_cursor_retryable() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    event = {"type": "run.started", "run_id": "run-1", "goal": "goal", "ts": "t"}
    delivery = _delivery(
        event=event,
        event_id="evt-retry",
        stream_id="run:run-1",
        seq=5,
    )

    # 模拟首次 UI 渲染在提交 cursor 前失败
    def fail(_: dict[str, Any]) -> None:
        raise RuntimeError("render failed")

    app._handle_event_inner = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="render failed"):
        await app._handle_delivery(delivery)

    assert app._stream_cursors == {}
    assert app._rendered_event_ids == set()

    rendered: list[dict[str, Any]] = []
    app._handle_event_inner = rendered.append  # type: ignore[method-assign]
    await app._handle_delivery(delivery)

    assert rendered == [event]
    assert app._stream_cursors == {"run:run-1": 5}


# 功能：验证真实 tool.call_started handler 在重放时只创建一个状态块
# 设计：保留真实 TUI 路由与 ToolCallBlock，只替换 DOM mount 边界，避免 list.append handler 假阳性
async def test_real_tool_handler_deduplicates_stateful_replay() -> None:
    app = KamaTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = appended.append  # type: ignore[method-assign]
    event = {
        "type": "tool.call_started",
        "tool_use_id": "tool-replay",
        "tool_name": "bash",
        "params": {"command": "true"},
        "run_id": "run-1",
        "ts": "t",
    }
    delivery = _delivery(
        event=event,
        event_id="evt-tool-replay",
        stream_id="run:run-1",
        seq=6,
    )

    await app._handle_delivery(delivery)
    await app._handle_delivery(delivery)

    assert len(appended) == 1
    assert isinstance(appended[0], ToolCallBlock)
    assert list(app._pending_tool_blocks) == ["tool-replay"]
    assert app._stream_cursors == {"run:run-1": 6}


# 功能：验证 compact/send_message 错误不会把底层异常 secret 渲染到 TUI
# 设计：两种命令分别注入含 secret 的 IpcError/RuntimeError，只允许固定安全文案
async def test_tui_command_errors_use_fixed_safe_messages() -> None:
    from kama_claude.core.transport.socket_client import IpcError

    class _Client:
        # 按命令类型抛出两种含 secret 的底层失败
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            if method == "session.compact":
                raise IpcError(-32000, "secret-compact-token")
            raise RuntimeError("secret-send-token")

    class _Prompt:
        disabled = True
        read_only = False
        border_title = ""

    app = KamaTuiApp("127.0.0.1", 9999)
    app._client = _Client()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._busy = True
    prompt = _Prompt()
    appended: list[str] = []
    app._prompt = lambda: prompt  # type: ignore[method-assign]
    app._append = lambda widget: appended.append(str(widget.content))  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]

    await app._do_compact()
    await app._do_send_message("hello")

    rendered = "\n".join(appended)
    assert "secret-compact-token" not in rendered
    assert "secret-send-token" not in rendered
    assert "compact failed" in rendered
    assert "send failed" in rendered


# 功能：验证 TUI 的 session.create 发送当前进程的 canonical cwd
# 设计：直接运行 socket loop 并替换 SocketClient/DOM 边界，捕获真实 send_command payload
async def test_tui_session_create_sends_canonical_client_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, dict[str, Any]]] = []
    created = asyncio.Event()

    class _Client:
        # 保留与真实 SocketClient 相同的构造界面
        def __init__(self, host: str, port: int) -> None:
            return None

        # 模拟成功建立 IPC 连接
        async def connect(self) -> None:
            return None

        # 保持事件循环挂起，由测试取消 socket loop
        async def run_event_loop(self) -> None:
            await asyncio.Event().wait()

        # TUI workspace 测试不需要处理完整 delivery
        def on_delivery(self, handler: object) -> None:
            return None

        # 记录 IPC payload，在 session.create 后通知测试停止循环
        async def send_command(
            self,
            method: str,
            params: dict[str, Any],
            ) -> dict[str, Any]:
                commands.append((method, params))
                if method == "event.subscribe" and params.get("scope") == "global":
                    return {
                        "subscription_id": "sub-handshake",
                        "daemon_instance_id": "daemon-test",
                    }
                if method == "event.unsubscribe":
                    return {"removed": True}
                if method == "session.create":
                    created.set()
                    return {"session_id": "sess-test", "status": "active"}
                if method == "event.subscribe":
                    return {
                        "subscription_id": "sub-test",
                        "daemon_instance_id": "daemon-test",
                        "stream_id": params["scope"],
                        "accepted_after_seq": params["after_seq"],
                        "high_watermark_seq": params["after_seq"],
                    }
                raise AssertionError(f"unexpected method {method}")

        # 模拟关闭 IPC 连接
        async def close(self) -> None:
            return None

    class _Header:
        # 模拟 Label.update 以隔离 Textual DOM
        def update(self, content: str) -> None:
            return None

    class _Prompt:
        disabled = True
        read_only = False
        border_title = ""

        # 模拟连接完成后输入框获得焦点
        def focus(self) -> None:
            return None

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(tui_app_module, "SocketClient", _Client)
    app = KamaTuiApp("127.0.0.1", 9999)
    prompt = _Prompt()
    app.query_one = lambda *args, **kwargs: _Header()  # type: ignore[method-assign]
    app._prompt = lambda: prompt  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._break_llm = lambda: None  # type: ignore[method-assign]

    socket_task = asyncio.create_task(app._socket_loop())
    await asyncio.wait_for(created.wait(), timeout=1.0)
    socket_task.cancel()
    await asyncio.gather(socket_task, return_exceptions=True)

    create_params = next(
        params for method, params in commands if method == "session.create"
    )
    assert create_params == {
        "mode": "chat",
        "workspace_root": str(workspace.resolve()),
    }
