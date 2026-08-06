from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.eval.failure import FailureCategory


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TokenUsage(_StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_tokens: int = Field(ge=0)


class BasicMetrics(_StrictModel):
    task_success: bool
    runtime_success: bool
    step_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    wall_latency_ms: float = Field(ge=0)
    token_usage: TokenUsage
    failure_category: FailureCategory


# 从已经通过 sanity 的事件机械计算 Phase 8A basic metrics
def compute_basic_metrics(
    events: list[dict[str, Any]],
    *,
    task_success: bool,
    runtime_success: bool,
    wall_latency_ms: float,
    failure_category: FailureCategory,
) -> BasicMetrics:
    step_count = sum(event.get("type") == "step.started" for event in events)
    tool_ids = {
        str(event["tool_use_id"])
        for event in events
        if event.get("type") == "tool.call_started"
    }
    failed_attempts: dict[str, int] = {}
    finished_tools: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    cache_tokens = 0
    for event in events:
        event_type = event.get("type")
        if event_type == "tool.call_failed":
            tool_id = str(event["tool_use_id"])
            failed_attempts[tool_id] = max(
                failed_attempts.get(tool_id, 0),
                int(event.get("attempt", 1)),
            )
        elif event_type == "tool.call_finished":
            finished_tools.add(str(event["tool_use_id"]))
        elif event_type == "llm.usage":
            input_tokens += int(event.get("input_tokens", 0))
            output_tokens += int(event.get("output_tokens", 0))
            cache_tokens += int(event.get("cache_read_input_tokens", 0))
            cache_tokens += int(event.get("cache_creation_input_tokens", 0))
    retry_count = sum(
        max(0, attempts - 1) + int(tool_id in finished_tools and attempts > 0)
        for tool_id, attempts in failed_attempts.items()
    )
    return BasicMetrics(
        task_success=task_success,
        runtime_success=runtime_success,
        step_count=step_count,
        tool_count=len(tool_ids),
        retry_count=retry_count,
        wall_latency_ms=wall_latency_ms,
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
        ),
        failure_category=failure_category,
    )
