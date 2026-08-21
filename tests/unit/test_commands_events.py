from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from kama_claude.core.bus import commands as commands_module
from kama_claude.core.bus.commands import (
    AgentRunCommand,
    Command,
    EchoCommand,
    EventSubscribeCommand,
    EventSubscribeResult,
    PingCommand,
    PongResult,
    SessionCreateCommand,
    SessionGetAgentModeResult,
    SessionSetAgentModeResult,
)
from kama_claude.core.bus.events import CoreStartedEvent, SessionAgentModeChangedEvent


# 功能：验证 PingCommand 序列化后再反序列化，client 和 type 字段完整保留
# 设计：JSON 往返测试确认 wire 协议的序列化正确性，type 字段是 discriminated union 的判别键
def test_ping_command_roundtrip() -> None:
    cmd = PingCommand(client="cli/0.0.1")
    cmd2 = PingCommand.model_validate_json(cmd.model_dump_json())
    assert cmd2.client == "cli/0.0.1"
    assert cmd2.type == "core.ping"


# 功能：验证 EchoCommand 序列化为 JSON 后再反序列化，message 和 type 字段完整保留
# 设计：直接使用 EchoCommand 做 JSON 往返，覆盖命令模型自身的字段约束和默认 type 判别键
def test_echo_command_roundtrip() -> None:
    cmd = EchoCommand(message="hello echo")
    cmd2 = EchoCommand.model_validate_json(cmd.model_dump_json())
    assert cmd2.message == "hello echo"
    assert cmd2.type == "core.echo"


# 功能：验证 EchoCommand 的 JSON 可以通过 Command 判别联合反序列化回 EchoCommand
# 设计：用 TypeAdapter(Command) 模拟协议层按 type 分发，防止新增命令模型后忘记接入 Command union
def test_echo_command_roundtrip_through_command_union() -> None:
    cmd = EchoCommand(message="你好，echo")
    cmd2 = TypeAdapter(Command).validate_json(cmd.model_dump_json())
    assert isinstance(cmd2, EchoCommand)
    assert cmd2.message == "你好，echo"
    assert cmd2.type == "core.echo"


# 功能：验证 AgentRunCommand 序列化往返后保留 workspace_root
# 设计：使用绝对路径字符串做 JSON 往返，锁定客户端到 daemon 的 wire 字段
def test_agent_run_command_roundtrip_preserves_workspace_root() -> None:
    cmd = AgentRunCommand(goal="inspect", workspace_root="/tmp/project")

    restored = AgentRunCommand.model_validate_json(cmd.model_dump_json())

    assert restored.workspace_root == "/tmp/project"


# 功能：验证 SessionCreateCommand 序列化往返后保留 workspace_root
# 设计：同时保留 mode 和 workspace_root，覆盖 chat session 创建命令的完整输入
def test_session_create_command_roundtrip_preserves_workspace_root() -> None:
    cmd = SessionCreateCommand(mode="chat", workspace_root="/tmp/project")

    restored = SessionCreateCommand.model_validate_json(cmd.model_dump_json())

    assert restored.mode == "chat"
    assert restored.workspace_root == "/tmp/project"


# 功能：验证 AgentRunCommand 缺少 workspace_root 时拒绝校验
# 设计：直接校验最小原始 params，确认不能用默认 cwd 隐式补全
def test_agent_run_command_missing_workspace_root_raises() -> None:
    with pytest.raises(ValidationError):
        AgentRunCommand.model_validate({"goal": "inspect"})


# 功能：验证 SessionCreateCommand 缺少 workspace_root 时拒绝校验
# 设计：不传 workspace_root 构造命令，防止旧客户端静默绑定 daemon cwd
def test_session_create_command_missing_workspace_root_raises() -> None:
    with pytest.raises(ValidationError):
        SessionCreateCommand.model_validate({"mode": "chat"})


# 功能：验证 AgentRunCommand 的 workspace_root 类型错误仍由 Pydantic 拒绝
# 设计：传入整数而非字符串，锁定类型错误不进入 workspace domain 校验器
def test_agent_run_command_invalid_workspace_root_type_raises() -> None:
    with pytest.raises(ValidationError):
        AgentRunCommand.model_validate({"goal": "inspect", "workspace_root": 123})


# 功能：验证 SessionCreateCommand 的 workspace_root 类型错误仍由 Pydantic 拒绝
# 设计：使用 list 触发字符串字段校验，确认 handler 不会改写 Invalid params 语义
def test_session_create_command_invalid_workspace_root_type_raises() -> None:
    with pytest.raises(ValidationError):
        SessionCreateCommand.model_validate(
            {"mode": "chat", "workspace_root": ["project"]}
        )


# 功能：验证 PingCommand 的 type 字段默认值为 "core.ping"
# 设计：Literal 默认值测试，type 是 Command union 的判别键，必须与 union 定义完全一致，否则反序列化时会路由到错误类型
def test_ping_command_default_type() -> None:
    cmd = PingCommand(client="x")
    assert cmd.type == "core.ping"


# 功能：验证缺少必填 client 字段时 pydantic 校验失败
# 设计：传入空 dict 触发校验，确认 client 是必填字段，防止 daemon 收到不完整的 ping 命令进入 handler
def test_ping_command_missing_client_raises() -> None:
    with pytest.raises(ValidationError):
        PingCommand.model_validate({})


# 功能：验证 PongResult 序列化往返后所有字段完整保留
# 设计：与 PingCommand 对称，测试命令-响应对的两端序列化，确认 int 和 str 字段类型在往返中不变
def test_pong_result_roundtrip() -> None:
    pong = PongResult(server_version="0.0.1", uptime_ms=42, received_at="2026-05-11T00:00:00Z")
    pong2 = PongResult.model_validate(pong.model_dump())
    assert pong2.server_version == "0.0.1"
    assert pong2.uptime_ms == 42


# 功能：验证 CoreStartedEvent 序列化往返后 listen_addr 和 type 字段正确保留
# 设计：CoreStartedEvent 是 daemon 启动通知，往返测试确认 type 的 Literal 约束在反序列化后保持（不被字段名覆盖）
def test_core_started_event_roundtrip() -> None:
    evt = CoreStartedEvent(listen_addr="127.0.0.1:7437", version="0.0.1")
    evt2 = CoreStartedEvent.model_validate_json(evt.model_dump_json())
    assert evt2.listen_addr == "127.0.0.1:7437"
    assert evt2.type == "core.started"


# 功能：验证 event.unsubscribe 命令可通过 Command 判别联合往返
# 设计：使用真实 subscription_id 序列化，防止模型存在但遗漏 union 接线
def test_event_unsubscribe_roundtrip_through_command_union() -> None:
    model = getattr(commands_module, "EventUnsubscribeCommand", None)
    assert model is not None
    command = model(subscription_id="sub-12345678")

    restored = TypeAdapter(Command).validate_json(command.model_dump_json())

    assert isinstance(restored, model)
    assert restored.subscription_id == "sub-12345678"


# 功能：验证 event.unsubscribe result 明确区分已删除与不可见订阅
# 设计：对 true/false 两种结果做 JSON 往返，锁定不泄漏 ownership 之外信息的单一布尔字段
@pytest.mark.parametrize("removed", [True, False])
def test_event_unsubscribe_result_roundtrip(removed: bool) -> None:
    model = getattr(commands_module, "EventUnsubscribeResult", None)
    assert model is not None
    result = model(removed=removed)

    restored = model.model_validate_json(result.model_dump_json())

    assert restored.removed is removed


# 功能：验证 event.subscribe 可携带非负 after_seq 且保留 stream scope
# 设计：通过 Command 判别联合往返，锁定 7B cursor 输入但不加入 7C daemon identity
def test_event_subscribe_after_seq_roundtrip() -> None:
    command = EventSubscribeCommand(
        topics=["run.*"],
        scope="run:run-1",
        after_seq=42,
    )

    restored = TypeAdapter(Command).validate_json(command.model_dump_json())

    assert isinstance(restored, EventSubscribeCommand)
    assert restored.after_seq == 42
    assert restored.scope == "run:run-1"


# 功能：验证 subscribe result 返回 daemon 身份及 response-time cursor metadata
# 设计：执行 JSON 往返并保留必填身份和 cursor 字段，锁定 7C 重连握手所需的完整事实
def test_event_subscribe_result_carries_response_time_cursor_metadata() -> None:
    result = EventSubscribeResult(
        subscription_id="sub-12345678",
        daemon_instance_id="daemon-1",
        replayed_count=0,
        stream_id="run:run-1",
        accepted_after_seq=3,
        high_watermark_seq=9,
    )

    restored = EventSubscribeResult.model_validate_json(result.model_dump_json())

    assert restored.daemon_instance_id == "daemon-1"
    assert restored.stream_id == "run:run-1"
    assert restored.accepted_after_seq == 3
    assert restored.high_watermark_seq == 9


# 功能：验证 mode result 与 changed event 在 wire 往返中携带 revision
# 设计：同时覆盖 set/get response 和 event union 使用的 Pydantic 边界，防止只修改 daemon 内部模型
def test_agent_mode_protocol_roundtrip_carries_revision() -> None:
    set_result = SessionSetAgentModeResult(agent_mode="plan", revision=3)
    get_result = SessionGetAgentModeResult(agent_mode="plan", revision=3)
    event = SessionAgentModeChangedEvent(
        session_id="sess-1",
        previous_mode="direct",
        agent_mode="plan",
        revision=3,
        ts="t",
    )

    assert SessionSetAgentModeResult.model_validate_json(set_result.model_dump_json()).revision == 3
    assert SessionGetAgentModeResult.model_validate_json(get_result.model_dump_json()).revision == 3
    assert SessionAgentModeChangedEvent.model_validate_json(event.model_dump_json()).revision == 3


# 功能：验证旧 mode changed event 缺少 revision 时兼容为零且非法 mode 仍被拒绝
# 设计：legacy payload 走默认值，伪造 executor 走 Literal 校验，覆盖 replay 兼容与 fail-closed 边界
def test_legacy_mode_event_defaults_revision_and_rejects_invalid_mode() -> None:
    legacy = SessionAgentModeChangedEvent.model_validate(
        {
            "session_id": "sess-1",
            "previous_mode": "direct",
            "agent_mode": "plan",
            "ts": "t",
        }
    )

    assert legacy.revision == 0
    with pytest.raises(ValidationError):
        SessionAgentModeChangedEvent.model_validate(
            {
                "session_id": "sess-1",
                "previous_mode": "direct",
                "agent_mode": "executor",
                "revision": 1,
                "ts": "t",
            }
        )
    with pytest.raises(ValidationError):
        SessionAgentModeChangedEvent.model_validate(
            {
                "session_id": "sess-1",
                "previous_mode": "direct",
                "agent_mode": "plan",
                "revision": True,
                "ts": "t",
            }
        )
