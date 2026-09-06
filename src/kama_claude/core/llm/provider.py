from __future__ import annotations

import asyncio
import copy
import logging
import os
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx

from kama_claude.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import (
    DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
    NATIVE_ANTHROPIC_CONTINUATION_POLICY,
    LlmResponse,
    ProviderContinuationPolicy,
    ProviderContinuationState,
    ToolCallBlock,
    UsageStats,
)
from kama_claude.core.llm.usage import ProviderUsageNormalizer, RawUsageEnvelope

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-7": 200_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
}

_MAX_STREAM_RETRIES = 3
_RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
_CONTINUATION_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "redacted_thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_item",
    }
)

log = logging.getLogger(__name__)


# 返回指定模型的最大 context window token 数
def _context_window(model: str) -> int | None:
    lowered = model.lower()
    if lowered.startswith("deepseek-v4"):
        return 1_000_000
    return _MODEL_CONTEXT_WINDOWS.get(model)


_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnthropicProvider:
    # 初始化 Anthropic 客户端；client 可在测试时注入以跳过 API key 检查
    def __init__(self, model: str, client: Any = None, *, max_output_tokens: int = 8192) -> None:
        self._model = model
        self._base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        self._is_deepseek_route = (
            "deepseek" in self._base_url.lower() or "deepseek" in model.lower()
        )
        self._usage_schema = (
            "deepseek_anthropic_messages_v1"
            if self._is_deepseek_route
            else "native_anthropic_v1"
        )
        self._continuation_policy = (
            DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY
            if self._is_deepseek_route
            else NATIVE_ANTHROPIC_CONTINUATION_POLICY
        )
        self._max_output_tokens = max(1, max_output_tokens)
        if client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise SystemExit("ANTHROPIC_API_KEY not set")
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client: Any = anthropic.AsyncAnthropic(**client_kwargs)
        else:
            self._client = client

    # 返回当前 adapter 的 continuation policy，供 gateway 与 renderer 复用
    @property
    def continuation_policy(self) -> ProviderContinuationPolicy:
        return self._continuation_policy

    # 返回当前 adapter 使用的模型标识，供 gateway、trace 与 subagent receipt 复用
    @property
    def model(self) -> str:
        return self._model

    # 返回当前 adapter 的 wire protocol 标识，供 continuation 与 usage identity 复用
    @property
    def protocol(self) -> str:
        return "anthropic"

    # 返回当前 provider 的 wire usage schema 标识
    @property
    def usage_schema(self) -> str:
        return self._usage_schema

    # 返回当前 provider route 的稳定身份
    @property
    def route_identity(self) -> str:
        route = self._base_url or "anthropic"
        return f"{self._model}:{route}"

    # 返回当前 route 的 provider context capacity
    @property
    def context_window(self) -> int | None:
        return 1_000_000 if self._is_deepseek_route else _context_window(self._model)

    # 返回 provider-normalized 最大生成预算（inclusive semantics）
    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    # 返回 provider 的总生成预算语义，避免 gateway 重复计算 reasoning tokens
    @property
    def output_budget_semantics(self) -> str:
        return self._continuation_policy.output_budget_semantics

    # 流式调用 Anthropic API，逐 token 发布事件并返回 LlmResponse；网络中断时自动重试
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
        await bus.publish(
            LlmModelSelectedEvent(run_id=run_id, model=self._model, strategy="static", ts=_now())
        )

        if immutable_system is not None:
            # A/B are serialized as independent blocks so Anthropic's explicit
            # cache breakpoint remains before dynamic contract/checkpoint text.
            system_blocks: list[dict[str, object]] = []
            for text, cacheable in (
                (immutable_system, True),
                (semi_stable_context, True),
            ):
                if not text:
                    continue
                block: dict[str, object] = {"type": "text", "text": text}
                if cacheable and not self._is_deepseek_route:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)
            dynamic_text = system or ""
            if dynamic_text:
                system_blocks.append({"type": "text", "text": dynamic_text})
            if not system_blocks:
                system_blocks = [{"type": "text", "text": _SYSTEM_PROMPT}]
        else:
            system_block: dict[str, object] = {
                "type": "text",
                "text": system or _SYSTEM_PROMPT,
            }
            if not self._is_deepseek_route:
                system_block["cache_control"] = {"type": "ephemeral"}
            system_blocks = [system_block]

        tools: list[dict[str, object]] = [copy.deepcopy(schema) for schema in tool_schemas]
        if self._is_deepseek_route:
            # The compatibility endpoint ignores Anthropic cache breakpoints;
            # remove any caller-supplied marker so the wire request cannot
            # accidentally imply that DeepSeek honors that control plane.
            tools = [
                {key: value for key, value in schema.items() if key != "cache_control"}
                for schema in tools
            ]
        if tools and not self._is_deepseek_route:
            last = dict(tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            tools = tools[:-1] + [last]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max(1, max_output_tokens or self._max_output_tokens),
            "system": system_blocks,
            "messages": messages,
        }
        if self._is_deepseek_route:
            # DeepSeek Anthropic compatibility requires thinking blocks on tool follow-ups;
            # budget_tokens is accepted but ignored by the endpoint.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        if tools:
            kwargs["tools"] = tools

        text_parts: list[str] = []
        final_message: Any = None

        for attempt in range(1, _MAX_STREAM_RETRIES + 1):
            text_parts = []
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        # Only publish token events on the first attempt to avoid TUI duplicates
                        if attempt == 1:
                            await bus.publish(LlmTokenEvent(run_id=run_id, token=text, ts=_now()))
                        text_parts.append(text)
                    final_message = await stream.get_final_message()
                break  # success
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                if attempt == _MAX_STREAM_RETRIES:
                    log.error(
                        "stream failed after %d attempts run_id=%s step=%d: %s",
                        _MAX_STREAM_RETRIES, run_id, step, exc,
                    )
                    raise
                delay = _RETRY_BACKOFF_S[attempt - 1]
                log.warning(
                    "stream dropped (attempt %d/%d) run_id=%s step=%d: %s — retrying in %.0fs",
                    attempt, _MAX_STREAM_RETRIES, run_id, step, exc, delay,
                )
                await asyncio.sleep(delay)

        assert final_message is not None

        usage = final_message.usage
        usage_payload = _usage_payload(usage)
        normalized = ProviderUsageNormalizer().normalize(
            RawUsageEnvelope(
                usage_schema=self._usage_schema,
                payload=usage_payload,
                route=self.route_identity,
            )
        )
        cache_read: int = normalized.cache_hit_tokens
        cache_create: int = _usage_integer(
            usage,
            "cache_creation_input_tokens",
            normalized.cache_creation_input_tokens,
        )
        context_window = self.context_window
        context_pct = (
            normalized.total_input_tokens / context_window
            if context_window is not None and context_window > 0
            else 0.0
        )

        await bus.publish(
            LlmUsageEvent(
                run_id=run_id,
                # 保留既有事件字段语义；normalized total 通过 UsageStats 提供 occupancy
                input_tokens=_usage_integer(usage, "input_tokens", normalized.total_input_tokens),
                output_tokens=normalized.output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_create,
                context_pct=context_pct,
                ts=_now(),
            )
        )

        tool_calls: list[ToolCallBlock] = []
        thinking_blocks: list[dict[str, object]] = []
        for block in final_message.content:
            block_type = (
                block.get("type")
                if isinstance(block, dict)
                else getattr(block, "type", "")
            )
            if block_type == "tool_use":
                block_id = (
                    block.get("id", "")
                    if isinstance(block, dict)
                    else getattr(block, "id", "")
                )
                block_name = (
                    block.get("name", "")
                    if isinstance(block, dict)
                    else getattr(block, "name", "")
                )
                block_input = (
                    block.get("input", {})
                    if isinstance(block, dict)
                    else getattr(block, "input", {})
                )
                input_payload = block_input if isinstance(block_input, dict) else {}
                tool_calls.append(
                    ToolCallBlock(
                        id=str(block_id), name=str(block_name), input=dict(input_payload)
                    )
                )
            elif block_type in _CONTINUATION_BLOCK_TYPES:
                # provider continuation blocks must be passed back verbatim in subsequent requests
                thinking_blocks.append(_continuation_block(block))

        continuation_state = ProviderContinuationState.from_blocks(
            thinking_blocks,
            policy=self._continuation_policy,
            route_identity=self.route_identity,
            protocol="anthropic",
        )

        return LlmResponse(
            stop_reason=final_message.stop_reason or "end_turn",
            tool_calls=tool_calls,
            text="".join(text_parts),
            thinking_blocks=thinking_blocks,
            continuation_state=continuation_state,
            usage=UsageStats(
                input_tokens=normalized.total_input_tokens,
                output_tokens=normalized.output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_create,
                context_pct=context_pct,
                cache_hit_tokens=normalized.cache_hit_tokens,
                cache_miss_tokens=normalized.cache_miss_tokens,
                reasoning_output_tokens=normalized.reasoning_output_tokens,
                usage_schema=self._usage_schema,
                measurement_source=normalized.measurement_source,
                route_identity=normalized.route,
            ),
        )


# 将 SDK usage 对象或 dict 转为 schema normalizer 可处理的 payload
def _usage_payload(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        return copy.deepcopy(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "reasoning_tokens",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[key] = value
    details = getattr(usage, "completion_tokens_details", None)
    if isinstance(details, dict):
        payload["completion_tokens_details"] = details
    elif details is not None and hasattr(details, "model_dump"):
        dumped_details = details.model_dump()
        if isinstance(dumped_details, dict):
            payload["completion_tokens_details"] = dumped_details
    return payload


# 从 SDK usage 对象读取数值字段，兼容测试 stub 与真实 response
def _usage_integer(usage: Any, key: str, default: int = 0) -> int:
    value = usage.get(key, default) if isinstance(usage, dict) else getattr(usage, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0, int(value))


# 将 Anthropic SDK content block 转成未编辑的 provider continuation block
def _continuation_block(block: Any) -> dict[str, object]:
    if isinstance(block, dict):
        return copy.deepcopy(block)
    if hasattr(block, "model_dump"):
        dumped = block.model_dump()
        if isinstance(dumped, dict):
            return copy.deepcopy(dumped)
    result: dict[str, object] = {"type": str(getattr(block, "type", "thinking"))}
    for key in (
        "thinking",
        "signature",
        "data",
        "reasoning_content",
        "content",
        "summary",
        "encrypted_content",
        "id",
        "index",
    ):
        value = getattr(block, key, None)
        if isinstance(value, (str, int, float, bool, dict, list, tuple)):
            result[key] = copy.deepcopy(value)
    return result
