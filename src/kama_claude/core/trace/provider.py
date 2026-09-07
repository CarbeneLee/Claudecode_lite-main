from __future__ import annotations

import dataclasses
import inspect
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.types import LlmResponse
from kama_claude.core.trace.record import TraceRecord
from kama_claude.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TracingProvider:
    # 包裹真实 LLMProvider，在每次 chat() 调用前后向 TraceWriter 写入完整 API I/O 记录
    def __init__(
        self,
        inner: LLMProvider,
        trace: TraceWriter,
        *,
        include_payload: bool = True,
    ) -> None:
        self._inner = inner
        self._trace = trace
        self._include_payload = include_payload

    # 转发 inner provider 的 route capability，确保 tracing 不改变 gateway 选择
    @property
    def continuation_policy(self) -> object:
        return getattr(self._inner, "continuation_policy", None)

    # 转发 inner provider 的 wire usage schema
    @property
    def usage_schema(self) -> str:
        return str(getattr(self._inner, "usage_schema", "native_anthropic_v1"))

    # 转发 inner provider 的 route identity
    @property
    def route_identity(self) -> str:
        return str(getattr(self._inner, "route_identity", "anthropic"))

    # 转发 inner provider 的 wire protocol，确保 usage/continuation identity 不丢失
    @property
    def protocol(self) -> str:
        value = getattr(self._inner, "protocol", "anthropic")
        return str(value) if isinstance(value, str) else "anthropic"

    # 转发 inner provider 的 model 名称
    @property
    def model(self) -> str:
        value = getattr(self._inner, "model", getattr(self._inner, "_model", ""))
        return str(value) if isinstance(value, str) else ""

    # 转发 provider context capacity，避免 trace wrapper 把 DeepSeek 路由降级为默认窗口
    @property
    def context_window(self) -> int | None:
        value = getattr(self._inner, "context_window", None)
        return value if isinstance(value, int) and value > 0 else None

    # 转发 provider 的最大生成预算，供 gateway 计算 output reserve
    @property
    def max_output_tokens(self) -> int:
        value = getattr(self._inner, "max_output_tokens", 8192)
        return value if isinstance(value, int) and value > 0 else 8192

    # 转发 provider 的 output budget 语义，保持 trace wrapper 与裸 adapter 一致
    @property
    def output_budget_semantics(self) -> str:
        return str(getattr(self._inner, "output_budget_semantics", "inclusive_total"))

    # 记录 CORE→LLM 请求，调用真实 provider，记录 LLM→CORE 响应（含延迟）
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
        max_output_tokens: int | None = None,
        immutable_system: str | None = None,
        semi_stable_context: str = "",
    ) -> LlmResponse:
        call_data: dict[str, Any]
        if self._include_payload:
            call_data = {"messages": messages, "tool_schemas": tool_schemas, "system": system}
        else:
            call_data = {
                "message_count": len(messages),
                "tool_count": len(tool_schemas),
            }

        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→LLM",
                layer="llm",
                kind="api_call",
                run_id=run_id,
                step=step,
                data=call_data,
            )
        )

        t0 = time.monotonic()
        # 旧 provider 可能尚未声明 A/B system slots；沿用 gateway 的 capability
        # 检查，避免 tracing wrapper 让兼容 provider 因新 kwargs 失效。
        parameters: Mapping[str, inspect.Parameter] = {}
        try:
            parameters = inspect.signature(self._inner.chat).parameters
        except (TypeError, ValueError):
            parameters = {}
        supports_immutable = "immutable_system" in parameters
        supports_semi_stable = "semi_stable_context" in parameters
        supports_layout = supports_immutable or supports_semi_stable
        inner_system = system
        if not supports_layout:
            # The gateway may have split Layer A/B for this tracing wrapper,
            # while a legacy inner adapter only accepts one complete system
            # slot. Reassemble the logical prompt before forwarding it.
            inner_system = "\n\n".join(
                part
                for part in (immutable_system, semi_stable_context, system)
                if part
            ) or None
        inner_kwargs: dict[str, object] = {
            "step": step,
            "system": inner_system,
        }
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "max_output_tokens" in parameters:
            inner_kwargs["max_output_tokens"] = max_output_tokens
        if supports_immutable:
            inner_kwargs["immutable_system"] = immutable_system
        if supports_semi_stable:
            inner_kwargs["semi_stable_context"] = semi_stable_context
        inner_chat = cast(Any, self._inner).chat
        result = cast(LlmResponse, await inner_chat(
            messages,
            tool_schemas,
            bus,
            run_id,
            **inner_kwargs,
        ))
        latency_ms = int((time.monotonic() - t0) * 1000)

        resp_data: dict[str, Any]
        if self._include_payload:
            resp_data = {
                "stop_reason": result.stop_reason,
                "text": result.text,
                "tool_calls": [dataclasses.asdict(tc) for tc in result.tool_calls],
                "continuation": (
                    result.continuation_state.identity_digest()
                    if result.continuation_state is not None
                    else None
                ),
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }
        else:
            resp_data = {
                "stop_reason": result.stop_reason,
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }

        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="LLM→CORE",
                layer="llm",
                kind="api_response",
                run_id=run_id,
                step=step,
                data=resp_data,
            )
        )

        return result
