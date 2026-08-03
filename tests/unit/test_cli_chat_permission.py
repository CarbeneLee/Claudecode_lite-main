from __future__ import annotations

import asyncio
import io
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


# 功能：验证 stdin 读取线程是 daemon 线程——退出时解释器不会 join 阻塞的 input()
#       线程而挂死（修复前 run_in_executor 的默认线程池是非 daemon，Ctrl+C 后进程无法退出）
# 设计：patch sys.stdin 让 input() 立即返回，断言线程 daemon 标记并正常完成
def test_stdin_reader_thread_is_daemon() -> None:
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    try:
        with patch("sys.stdin", io.StringIO("x\n")):
            thread = chat_module._spawn_stdin_reader("> ", future)
            assert thread.daemon is True
            thread.join(timeout=2)
            assert not thread.is_alive()
        assert loop.run_until_complete(asyncio.wait_for(future, timeout=2)) == "x"
    finally:
        loop.close()


# 功能：验证 _readline 从 stdin 读取一行（回归保护：readline 改为 daemon 线程实现后行为不变）
async def test_readline_returns_line_from_stdin() -> None:
    with patch("sys.stdin", io.StringIO("hello world\n")):
        value = await asyncio.wait_for(chat_module._readline("> "), timeout=5)
    assert value == "hello world"


# 功能：回归保护——事件循环空闲阻塞（select 无事件、无定时器）时，stdin 线程
#       晚到的输入必须唤醒循环。直接跨线程 set_result 只入就绪队列不唤醒 select，
#       输入会被永久吞掉（真实故障：chat 输入任务后毫无响应、无 send_message）。
# 设计：子进程内用无数据的管道 fd 让 select() 阻塞，父进程延迟 1s 写入 stdin；
#       修复前 _readline 永远不返回（超时强杀），修复后 ~1s 内返回
def test_readline_wakes_idle_event_loop_from_stdin_thread() -> None:
    import subprocess
    import sys
    import textwrap

    child = textwrap.dedent(
        r"""
        import asyncio
        import os
        import time

        from kama_claude.cli.commands.chat import _readline

        async def main() -> None:
            loop = asyncio.get_running_loop()
            rr, _ = os.pipe()  # 让 select() 有 fd 可阻塞但永远无事件
            loop.add_reader(rr, lambda: None)
            print("READY", flush=True)
            t0 = time.monotonic()
            value = await _readline("> ")
            print(f"GOT {value} {time.monotonic() - t0:.2f}s", flush=True)

        asyncio.run(main())
        """
    )
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=read_fd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(read_fd)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    time.sleep(1.0)  # 等子进程进入空闲 select()
    os.write(write_fd, b"hello task\n")
    os.close(write_fd)
    try:
        out, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"空闲事件循环未收到 stdin 输入（跨线程唤醒失效）: {out}")
    assert "GOT hello task" in out


# 功能：验证 Ctrl+C（CancelledError 取消 _chat_async）时向 daemon 发送 session.close——
#       让 daemon 取消后台 run 并关闭会话，而不是仅断开 socket 留下僵尸 run
# 设计：订阅完成后阻塞 readline（模拟用户未输入时按 Ctrl+C），取消 chat task，
#       断言 session.close RPC 已发出且连接已关闭
async def test_ctrl_c_cancellation_sends_session_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    readline_started = asyncio.Event()

    async def fake_readline(prompt: str) -> str:
        await client.subscribed.wait()
        readline_started.set()
        await asyncio.Event().wait()

    def make_client(host: str, port: int) -> _FakeClient:
        return client

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(chat_module, "SocketClient", make_client)
    monkeypatch.setattr(chat_module, "_readline", fake_readline)

    chat_task = asyncio.create_task(chat_module._chat_async(KamaConfig()))
    await readline_started.wait()
    await asyncio.sleep(0.05)  # 主循环已进入 interactive 等待

    chat_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(chat_task, timeout=5)

    close_commands = [p for m, p in client.commands if m == "session.close"]
    assert close_commands == [{"session_id": "sess-old"}]
    assert client.closed


# 功能：验证 run 失败（status=failed）会向用户显示失败原因——
#       LLM 失败（如 key 错误）不能被静默吞掉，用户必须能看到 run 没成功
# 设计：发送消息后推送 run.finished(status=failed, reason=llm_error)，
#       断言终端输出包含 [run failed: llm_error]
async def test_run_failure_is_displayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _FakeClient()
    lines = ["把任务做完"]
    second_readline_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_readline(prompt: str) -> str:
        await client.subscribed.wait()
        if not lines:
            second_readline_started.set()
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

    while not any(m == "session.send_message" for m, _ in client.commands):
        await asyncio.sleep(0.005)
    await client.push("run.started", run_id="run-test")
    await client.push(
        "run.finished",
        run_id="run-test",
        status="failed",
        reason="llm_error",
        steps=1,
    )

    # 输出已渲染后才让 chat 退出（轮询收集，避免与 chat_task 取消竞争）
    deadline = asyncio.get_running_loop().time() + 2
    captured = ""
    while "[run failed: llm_error]" not in captured:
        assert asyncio.get_running_loop().time() < deadline, "failure not displayed"
        captured += capsys.readouterr().out
        await asyncio.sleep(0.005)

    release.set()
    await asyncio.wait_for(chat_task, timeout=5)
    assert chat_task.result() == 0


# 功能：验证 run 被取消时显示 [run cancelled]（区别于失败，文案友好）
async def test_printer_shows_run_cancelled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    printer = chat_module.ChatPrinter()
    await printer.handle(
        {"type": "run.finished", "status": "cancelled", "reason": "cancelled", "steps": 1}
    )
    out = capsys.readouterr().out
    assert "[run cancelled]" in out
