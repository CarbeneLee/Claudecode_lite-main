from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, StrictInt

from kama_claude.core.execution import ExecutionStatus
from kama_claude.core.session.model import (
    MAX_AGENT_MODE_REVISION,
    AgentMode,
    SessionMode,
    SessionStatus,
)


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str


class PongResult(BaseModel):
    server_version: str
    uptime_ms: int
    received_at: str  # ISO 8601


class EchoCommand(BaseModel):
    type: Literal["core.echo"] = "core.echo"
    message: str


class EchoResult(BaseModel):
    server_version: str
    received_at: str  # ISO 8601
    message: str


class AgentRunCommand(BaseModel):
    type: Literal["agent.run"] = "agent.run"
    goal: str
    workspace_root: str
    agent_mode: AgentMode = "direct"


class AgentRunResult(BaseModel):
    run_id: str


class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str]          # fnmatch 模式，如 ["step.*", "tool.*"]
    scope: str = "global"      # "global" | "run:<run_id>"
    replay_from_run: str | None = None  # 设置则先从 events.jsonl 回放历史再接实时流
    after_seq: int | None = Field(default=None, ge=0)


class EventSubscribeResult(BaseModel):
    subscription_id: str
    daemon_instance_id: str
    replayed_count: int = 0
    stream_id: str | None = None
    accepted_after_seq: int | None = None
    high_watermark_seq: int | None = None


class EventUnsubscribeCommand(BaseModel):
    type: Literal["event.unsubscribe"] = "event.unsubscribe"
    subscription_id: str


class EventUnsubscribeResult(BaseModel):
    removed: bool


class SessionCreateCommand(BaseModel):
    type: Literal["session.create"] = "session.create"
    mode: SessionMode = "chat"
    agent_mode: AgentMode = "direct"
    title: str = ""
    workspace_root: str


class SessionCreateResult(BaseModel):
    session_id: str
    status: SessionStatus


class SessionSendMessageCommand(BaseModel):
    type: Literal["session.send_message"] = "session.send_message"
    session_id: str
    content: str


class SessionSendMessageResult(BaseModel):
    run_id: str


class SessionGetHistoryCommand(BaseModel):
    type: Literal["session.get_history"] = "session.get_history"
    session_id: str
    include_projection_metadata: bool = False


class SessionGetHistoryResult(BaseModel):
    messages: list[dict[str, Any]]


class SessionSetAgentModeCommand(BaseModel):
    type: Literal["session.set_agent_mode"] = "session.set_agent_mode"
    session_id: str
    agent_mode: AgentMode


class SessionSetAgentModeResult(BaseModel):
    agent_mode: AgentMode
    revision: StrictInt = Field(ge=0, le=MAX_AGENT_MODE_REVISION)


class SessionGetAgentModeCommand(BaseModel):
    type: Literal["session.get_agent_mode"] = "session.get_agent_mode"
    session_id: str


class SessionGetAgentModeResult(BaseModel):
    agent_mode: AgentMode
    revision: StrictInt = Field(ge=0, le=MAX_AGENT_MODE_REVISION)


class SessionCloseCommand(BaseModel):
    type: Literal["session.close"] = "session.close"
    session_id: str


class SessionCloseResult(BaseModel):
    status: SessionStatus


class PermissionRespondCommand(BaseModel):
    type: Literal["permission.respond"] = "permission.respond"
    tool_use_id: str
    # "allow_once" | "always_allow" | "deny_once" | "always_deny"
    decision: str


class PermissionRespondResult(BaseModel):
    ok: bool = True


class SessionCompactCommand(BaseModel):
    type: Literal["session.compact"] = "session.compact"
    session_id: str
    focus: str = ""


class SessionCompactResult(BaseModel):
    summary_tokens: int
    saved_tokens: int


class PlanGetApprovalCommand(BaseModel):
    type: Literal["plan.get_approval"] = "plan.get_approval"
    session_id: str
    projection_key: str


class PlanApprovalResult(BaseModel):
    session_id: str
    projection_key: str
    status: Literal["pending", "approved", "rejected", "conflicted/unknown"]
    decision_id: str | None = None
    decision_version: int | None = None
    content_digest: str | None = None
    commit_receipt_digest: str | None = None
    action: Literal["approve", "reject"] | None = None
    record_digest: str | None = None


class PlanGetApprovalResult(PlanApprovalResult):
    pass


class PlanApproveResult(PlanApprovalResult):
    pass


class PlanRejectResult(PlanApprovalResult):
    pass


class PlanApproveCommand(BaseModel):
    type: Literal["plan.approve"] = "plan.approve"
    session_id: str
    projection_key: str
    decision_id: str
    decision_version: int
    content_digest: str
    commit_receipt_digest: str


class PlanRejectCommand(BaseModel):
    type: Literal["plan.reject"] = "plan.reject"
    session_id: str
    projection_key: str
    decision_id: str
    decision_version: int
    content_digest: str
    commit_receipt_digest: str


class PlanExecuteCommand(BaseModel):
    type: Literal["plan.execute"] = "plan.execute"
    session_id: str
    projection_key: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


class PlanExecuteResult(BaseModel):
    session_id: str
    request_id: str
    execution_id: str
    run_id: str
    projection_key: str
    status: ExecutionStatus
    status_revision: StrictInt = Field(default=0, ge=0)
    status_digest: str | None = None
    reason: str | None = None


class PlanGetExecutionCommand(BaseModel):
    type: Literal["plan.get_execution"] = "plan.get_execution"
    session_id: str
    request_id: str = Field(min_length=1)


class PlanGetExecutionResult(BaseModel):
    session_id: str
    request_id: str
    execution_id: str
    run_id: str
    projection_key: str
    status: ExecutionStatus
    status_revision: StrictInt = Field(default=0, ge=0)
    status_digest: str | None = None
    reason: str | None = None


# 根据 type 字段决定命令类型的判别联合
Command = Annotated[
    PingCommand
    | EchoCommand
    | AgentRunCommand
    | EventSubscribeCommand
    | EventUnsubscribeCommand
    | SessionCreateCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionSetAgentModeCommand
    | SessionGetAgentModeCommand
    | SessionCloseCommand
    | PermissionRespondCommand
    | SessionCompactCommand
    | PlanGetApprovalCommand
    | PlanApproveCommand
    | PlanRejectCommand
    | PlanExecuteCommand
    | PlanGetExecutionCommand,
    Discriminator("type"),
]
