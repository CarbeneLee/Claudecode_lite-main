from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, StrictInt, model_validator

from kama_claude.core.plan_view import (
    LegacyPlanViewV0,
    PlanViewEventValue,
    PlanViewV1,
    decode_plan_view_record,
)
from kama_claude.core.session.model import MAX_AGENT_MODE_REVISION, AgentMode

ExecutionStatusValue = Literal[
    "admitted",
    "running",
    "completed_unverified",
    "failed",
    "cancelled",
    "scope_denied",
    "inconclusive",
    "interrupted",
]


class CoreStartedEvent(BaseModel):
    type: Literal["core.started"] = "core.started"
    listen_addr: str  # e.g. "127.0.0.1:7437"
    version: str


class RunStartedEvent(BaseModel):
    type: Literal["run.started"] = "run.started"
    run_id: str
    goal: str
    ts: str  # ISO 8601
    execution_id: str | None = None
    execution_status: Literal["admitted", "running"] | None = None


class RunFinishedEvent(BaseModel):
    type: Literal["run.finished"] = "run.finished"
    run_id: str
    status: str  # "success" | "failed"
    reason: str | None = None  # "exceeded_max_steps" | "cancelled" | "llm_error" | ...
    steps: int
    ts: str
    execution_id: str | None = None
    execution_status: ExecutionStatusValue | None = None


class StepStartedEvent(BaseModel):
    type: Literal["step.started"] = "step.started"
    run_id: str
    step: int
    ts: str


class StepFinishedEvent(BaseModel):
    type: Literal["step.finished"] = "step.finished"
    run_id: str
    step: int
    ts: str


class ToolCallStartedEvent(BaseModel):
    type: Literal["tool.call_started"] = "tool.call_started"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    ts: str


class ToolCallFinishedEvent(BaseModel):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    run_id: str
    tool_use_id: str
    tool_name: str
    elapsed_ms: int
    output: str = ""  # tool result content, for TUI display
    ts: str


class ToolCallFailedEvent(BaseModel):
    type: Literal["tool.call_failed"] = "tool.call_failed"
    run_id: str
    tool_use_id: str
    tool_name: str
    # 与 tools.errors 的稳定 error_type taxonomy 一致；字段保持 str 以兼容 trace/replay
    error_class: str
    error_message: str
    elapsed_ms: int
    attempt: int = 1  # 1=first attempt, 2=first retry, 3=second retry
    ts: str


class LlmTokenEvent(BaseModel):
    type: Literal["llm.token"] = "llm.token"
    run_id: str
    token: str
    ts: str


class LlmUsageEvent(BaseModel):
    type: Literal["llm.usage"] = "llm.usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    context_pct: float = 0.0
    ts: str


class LlmModelSelectedEvent(BaseModel):
    type: Literal["llm.model_selected"] = "llm.model_selected"
    run_id: str
    model: str
    strategy: str  # "static" | "rule_based" | "cost_budget"
    ts: str


class LogLineEvent(BaseModel):
    type: Literal["log.line"] = "log.line"
    run_id: str
    level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    source: str
    message: str
    ts: str


class SessionCreatedEvent(BaseModel):
    type: Literal["session.created"] = "session.created"
    session_id: str
    mode: str
    ts: str


class SessionMessageReceivedEvent(BaseModel):
    type: Literal["session.message_received"] = "session.message_received"
    session_id: str
    content: str
    ts: str


class SessionWaitingForInputEvent(BaseModel):
    type: Literal["session.waiting_for_input"] = "session.waiting_for_input"
    session_id: str
    last_run_id: str
    ts: str


class SessionResumedEvent(BaseModel):
    type: Literal["session.resumed"] = "session.resumed"
    session_id: str
    ts: str


class SessionAgentModeChangedEvent(BaseModel):
    type: Literal["session.agent_mode_changed"] = "session.agent_mode_changed"
    session_id: str
    previous_mode: AgentMode
    agent_mode: AgentMode
    revision: StrictInt = Field(default=0, ge=0, le=MAX_AGENT_MODE_REVISION)
    ts: str


class SessionClosedEvent(BaseModel):
    type: Literal["session.closed"] = "session.closed"
    session_id: str
    ts: str


class ContextCompactedEvent(BaseModel):
    type: Literal["context.compacted"] = "context.compacted"
    session_id: str
    run_id: str
    original_tokens: int
    summary_tokens: int
    ts: str


class GitRunDiffEvent(BaseModel):
    type: Literal["git.run_diff"] = "git.run_diff"
    run_id: str
    stat: str  # diff stat 摘要（截断保护）
    ts: str


class PermissionRequestedEvent(BaseModel):
    type: Literal["permission.requested"] = "permission.requested"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    param_preview: str
    session_id: str
    ts: str


class PermissionGrantedEvent(BaseModel):
    type: Literal["permission.granted"] = "permission.granted"
    run_id: str
    tool_use_id: str
    # "allow_once" | "always_allow" | "auto_allow"
    decision: str
    ts: str


class PermissionDeniedEvent(BaseModel):
    type: Literal["permission.denied"] = "permission.denied"
    run_id: str
    tool_use_id: str
    # "deny_once" | "always_deny" | "auto_deny"
    decision: str
    ts: str


class SubagentStartedEvent(BaseModel):
    type: Literal["subagent.started"] = "subagent.started"
    run_id: str          # 子 agent run_id
    parent_run_id: str
    description: str
    ts: str


class SubagentFinishedEvent(BaseModel):
    type: Literal["subagent.finished"] = "subagent.finished"
    run_id: str
    parent_run_id: str
    status: str          # "success" | "failed"
    ts: str


class SkillInvokedEvent(BaseModel):
    type: Literal["skill.invoked"] = "skill.invoked"
    skill_name: str
    arguments: str
    run_id: str
    ts: str


class PlannerDecisionReadyEvent(BaseModel):
    type: Literal["planner.decision_ready"] = "planner.decision_ready"
    event_id: str
    run_id: str
    planner_run_id: str
    session_id: str
    plan: PlanViewEventValue
    # 旧客户端兼容 alias；V1 event 中由 projection_key 派生，不是第二事实源
    plan_key: str = ""
    decision_key: str = ""
    projection_key: str = ""
    decision_id: str = ""
    decision_version: int = 0
    ts: str
    snapshot_digest: str = ""
    content_digest: str = ""  # legacy alias for decision_content_digest
    decision_content_digest: str = ""
    projection_digest: str = ""

    # 在 wire boundary 强制 outer identity 由 PlanViewV1 派生并保持一致
    @model_validator(mode="after")
    def _derive_identity_from_plan(self) -> PlannerDecisionReadyEvent:
        if isinstance(self.plan, PlanViewV1):
            # 直接传入模型实例时也必须重算 projection digest，不能只验证 outer alias
            decode_plan_view_record(self.plan)
            expected_projection_key = f"pv1:{self.run_id}:{self.plan.decision_key}"
            if self.plan.projection_key != expected_projection_key:
                raise ValueError(
                    "PlannerDecisionReadyEvent plan projection does not match run"
                )
            expected = {
                "decision_key": self.plan.decision_key,
                "projection_key": self.plan.projection_key,
                "decision_id": self.plan.decision_id,
                "decision_version": self.plan.decision_version,
                "snapshot_digest": self.plan.snapshot_digest,
                "decision_content_digest": self.plan.decision_content_digest,
                "projection_digest": self.plan.projection_digest,
            }
            for field, value in expected.items():
                actual = getattr(self, field)
                if actual not in ("", 0) and actual != value:
                    raise ValueError(f"PlannerDecisionReadyEvent {field} does not match PlanView")
                setattr(self, field, value)
            if self.plan_key not in ("", self.projection_key):
                raise ValueError("PlannerDecisionReadyEvent plan_key does not match projection")
            if self.content_digest not in ("", self.decision_content_digest):
                raise ValueError("PlannerDecisionReadyEvent content_digest does not match decision")
            self.plan_key = self.projection_key
            self.content_digest = self.decision_content_digest
            expected_event_id = f"plan-ready:{self.projection_key}"
            if self.event_id != expected_event_id:
                raise ValueError("PlannerDecisionReadyEvent event_id does not match projection")
        elif isinstance(self.plan, LegacyPlanViewV0):
            if self.plan_key and self.plan_key != self.plan.plan_key:
                raise ValueError("legacy PlannerDecisionReadyEvent plan_key conflict")
            if self.content_digest and self.plan.content_digest:
                if self.content_digest != self.plan.content_digest:
                    raise ValueError("legacy PlannerDecisionReadyEvent content_digest conflict")
            self.plan_key = self.plan.plan_key
            if not self.content_digest:
                self.content_digest = self.plan.content_digest
        return self


class PlanApprovalChangedEvent(BaseModel):
    type: Literal["plan.approval_changed"] = "plan.approval_changed"
    event_id: str
    session_id: str
    projection_key: str
    status: Literal["approved", "rejected"]
    action: Literal["approve", "reject"]
    record_digest: str
    commit_receipt_digest: str
    ts: str


# 根据 type 字段决定事件类型的判别联合
Event = Annotated[
    CoreStartedEvent
    | RunStartedEvent
    | RunFinishedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | ToolCallFailedEvent
    | LlmTokenEvent
    | LlmUsageEvent
    | LlmModelSelectedEvent
    | LogLineEvent
    | SessionCreatedEvent
    | SessionMessageReceivedEvent
    | SessionWaitingForInputEvent
    | SessionResumedEvent
    | SessionAgentModeChangedEvent
    | SessionClosedEvent
    | ContextCompactedEvent
    | GitRunDiffEvent
    | PermissionRequestedEvent
    | PermissionGrantedEvent
    | PermissionDeniedEvent
    | SubagentStartedEvent
    | SubagentFinishedEvent
    | SkillInvokedEvent
    | PlannerDecisionReadyEvent
    | PlanApprovalChangedEvent,
    Discriminator("type"),
]
