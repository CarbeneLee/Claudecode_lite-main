from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

UsageSchema = Literal[
    "deepseek_anthropic_messages_v1",
    "deepseek_chat_completions_v1",
    "deepseek_responses_v1",
    "openai_responses_v1",
    "native_anthropic_v1",
    "openai_chat_completions_v1",
]
MeasurementConfidence = Literal["provider", "derived", "estimated"]
SUPPORTED_USAGE_SCHEMAS = frozenset(
    {
        "deepseek_anthropic_messages_v1",
        "deepseek_chat_completions_v1",
        "deepseek_responses_v1",
        "openai_responses_v1",
        "native_anthropic_v1",
        "openai_chat_completions_v1",
    }
)


@dataclass(frozen=True, slots=True)
class RawUsageEnvelope:
    """Adapter output before schema-specific usage normalization."""

    usage_schema: str
    payload: dict[str, Any]
    route: str = ""
    surface_revision: int | None = None
    log_revision: int | None = None
    request_envelope_id: str = ""


@dataclass(frozen=True, slots=True)
class NormalizedUsage:
    """Provider-neutral context and output usage telemetry."""

    total_input_tokens: int
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    confidence: MeasurementConfidence = "provider"
    usage_schema: str = ""
    measurement_source: str = "provider"
    route: str = ""
    surface_revision: int | None = None
    log_revision: int | None = None
    # Anthropic cache creation is a distinct cost field from uncached input.
    cache_creation_input_tokens: int = 0
    request_envelope_id: str = ""

    # 暴露与 UsageStats 一致的命名，便于调用方不依赖 normalizer 内部字段名
    @property
    def measurement_confidence(self) -> MeasurementConfidence:
        return self.confidence

    # 将归一化 usage 转为旧 LlmResponse UsageStats，保持现有调用方兼容
    def to_usage_stats(self, context_window: int = 0) -> Any:
        from kama_claude.core.llm.types import UsageStats

        context_pct = (
            self.total_input_tokens / context_window if context_window > 0 else 0.0
        )
        return UsageStats(
            input_tokens=self.total_input_tokens,
            output_tokens=self.output_tokens,
            cache_read_input_tokens=self.cache_hit_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            context_pct=context_pct,
            cache_hit_tokens=self.cache_hit_tokens,
            cache_miss_tokens=self.cache_miss_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens,
            measurement_confidence=self.confidence,
            usage_schema=self.usage_schema or "legacy",
            measurement_source=self.measurement_source,
            route_identity=self.route,
            surface_revision=self.surface_revision,
            log_revision=self.log_revision,
            request_envelope_id=self.request_envelope_id,
        )


@dataclass(frozen=True, slots=True)
class UsageBaseline:
    """成功请求 usage 与 surface/route/prefix 身份的绑定。"""

    surface_revision: int
    route_epoch: str
    prefix_epoch: str
    envelope_id: str
    continuation_policy_identity: str
    total_input_tokens: int
    confidence: MeasurementConfidence
    usage_schema: str = ""
    measurement_source: str = "provider"
    log_revision: int | None = None


class ReplayAwareTokenMeter:
    """在 envelope 兼容时用已测量 baseline 加 committed surface delta 估算 occupancy。"""

    # 初始化为空 baseline 的 replay-aware meter
    def __init__(self) -> None:
        self._baseline: UsageBaseline | None = None

    # 记录最近一次 provider 成功 usage baseline
    def record(
        self,
        usage: NormalizedUsage,
        *,
        surface_revision: int,
        route_epoch: str,
        prefix_epoch: str,
        envelope_id: str,
        continuation_policy_identity: str,
        usage_schema: str = "",
        measurement_source: str = "provider",
        log_revision: int | None = None,
    ) -> UsageBaseline:
        self._baseline = UsageBaseline(
            surface_revision=surface_revision,
            route_epoch=route_epoch,
            prefix_epoch=prefix_epoch,
            envelope_id=envelope_id,
            continuation_policy_identity=continuation_policy_identity,
            total_input_tokens=usage.total_input_tokens,
            confidence=usage.confidence,
            usage_schema=usage_schema or usage.usage_schema,
            measurement_source=measurement_source or usage.measurement_source,
            log_revision=log_revision,
        )
        return self._baseline

    # 在同 route/prefix/policy 下按 committed surface delta 估算新的输入 occupancy
    def estimate(
        self,
        *,
        surface_revision: int,
        route_epoch: str,
        prefix_epoch: str,
        envelope_id: str,
        continuation_policy_identity: str,
        signed_delta_tokens: int = 0,
    ) -> tuple[int, MeasurementConfidence] | None:
        baseline = self._baseline
        if baseline is None:
            return None
        if (
            baseline.route_epoch != route_epoch
            or baseline.prefix_epoch != prefix_epoch
            or baseline.continuation_policy_identity != continuation_policy_identity
        ):
            return None
        # A committed surface mutation normally changes the full envelope id;
        # an explicit signed delta is the proof that a caller has accounted for
        # a mutation while intentionally reusing the prior envelope identity.
        if baseline.envelope_id != envelope_id and signed_delta_tokens == 0:
            return None
        if surface_revision != baseline.surface_revision and signed_delta_tokens == 0:
            return None
        revision_delta = surface_revision - baseline.surface_revision
        if revision_delta < 0:
            return None
        return (
            max(0, baseline.total_input_tokens + signed_delta_tokens),
            "derived" if signed_delta_tokens else baseline.confidence,
        )

    # 清除 baseline，强制下一次完整 reprice
    def invalidate(self) -> None:
        self._baseline = None

    # 返回当前 baseline
    @property
    def baseline(self) -> UsageBaseline | None:
        return self._baseline


# 从 wire payload 读取非负整数，拒绝 bool/未知类型以保持 usage 保守
def _integer(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return default


# 从嵌套 usage details 读取非负整数，缺失时返回零
def _nested_integer(payload: dict[str, Any], parent: str, key: str) -> int:
    nested = payload.get(parent)
    if isinstance(nested, dict):
        return _integer(nested, key)
    return 0


class ProviderUsageNormalizer:
    """按 adapter 标注的 wire/API schema 解析 usage。"""

    # 按 wire schema 分派具体 parser，未知 schema fail closed
    def normalize(self, envelope: RawUsageEnvelope) -> NormalizedUsage:
        parsers = {
            "deepseek_anthropic_messages_v1": self._normalize_deepseek_anthropic,
            "native_anthropic_v1": self._normalize_anthropic,
            "deepseek_chat_completions_v1": self._normalize_deepseek_chat,
            "openai_chat_completions_v1": self._normalize_openai_chat,
            "deepseek_responses_v1": self._normalize_responses,
            "openai_responses_v1": self._normalize_responses,
        }
        parser = parsers.get(envelope.usage_schema)
        if parser is None:
            raise ValueError(f"unsupported usage schema: {envelope.usage_schema}")
        # A few compatibility adapters wrap the wire usage object in a top-level
        # ``usage`` member while native SDK responses expose it directly.  The
        # adapter still owns the schema label; unwrapping here only normalizes the
        # envelope shape and never guesses a vendor-specific field layout.
        payload = envelope.payload
        nested_usage = payload.get("usage")
        if isinstance(nested_usage, dict):
            payload = nested_usage
        result = parser(payload)
        return NormalizedUsage(
            total_input_tokens=result.total_input_tokens,
            cache_hit_tokens=result.cache_hit_tokens,
            cache_miss_tokens=result.cache_miss_tokens,
            output_tokens=result.output_tokens,
            reasoning_output_tokens=result.reasoning_output_tokens,
            confidence=result.confidence,
            usage_schema=envelope.usage_schema,
            measurement_source=result.measurement_source,
            route=envelope.route,
            surface_revision=envelope.surface_revision,
            log_revision=envelope.log_revision,
            cache_creation_input_tokens=result.cache_creation_input_tokens,
            request_envelope_id=envelope.request_envelope_id,
        )

    # 解析 DeepSeek Anthropic-compatible endpoint 的两种 usage 变体
    def _normalize_deepseek_anthropic(self, payload: dict[str, Any]) -> NormalizedUsage:
        hit = _integer(payload, "prompt_cache_hit_tokens")
        miss = _integer(payload, "prompt_cache_miss_tokens")
        prompt = _integer(payload, "prompt_tokens")
        input_tokens = _integer(payload, "input_tokens")
        if hit or miss or prompt:
            # DeepSeek-shaped responses define prompt_tokens as hit+miss. A
            # compatibility response can omit it; in that case an
            # input_tokens field is the uncached portion, not a replacement
            # for the cache-hit count.
            if prompt:
                total = prompt
                if miss == 0:
                    miss = max(0, prompt - hit)
            elif hit and miss:
                total = hit + miss
            elif hit:
                total = input_tokens + hit
                miss = input_tokens
            else:
                total = miss
            completion = _integer(payload, "output_tokens") or _integer(
                payload, "completion_tokens"
            )
            return NormalizedUsage(
                total_input_tokens=total,
                cache_hit_tokens=hit,
                cache_miss_tokens=miss,
                output_tokens=completion,
                reasoning_output_tokens=(
                    _integer(payload, "reasoning_tokens")
                    or _nested_integer(payload, "completion_tokens_details", "reasoning_tokens")
                ),
                confidence="provider",
                cache_creation_input_tokens=0,
            )
        return self._normalize_anthropic(payload)

    # 解析 Anthropic messages usage，cache read/create 同样占用 context
    def _normalize_anthropic(self, payload: dict[str, Any]) -> NormalizedUsage:
        input_tokens = _integer(payload, "input_tokens")
        cache_hit = _integer(payload, "cache_read_input_tokens")
        cache_create = _integer(payload, "cache_creation_input_tokens")
        # 某些兼容端会返回 DeepSeek 字段而保留 Anthropic endpoint
        if cache_hit == 0:
            cache_hit = _integer(payload, "prompt_cache_hit_tokens")
        if cache_create == 0:
            cache_create = _integer(payload, "prompt_cache_miss_tokens")
        total = input_tokens + cache_hit + cache_create
        prompt_tokens = _integer(payload, "prompt_tokens")
        if prompt_tokens > 0:
            total = prompt_tokens
        known_cache = cache_hit + cache_create
        # Anthropic 的 input_tokens 表示未命中 cache 的部分，cache creation 也属于 miss
        miss = input_tokens + cache_create
        if prompt_tokens > 0 and known_cache > 0:
            miss = max(0, prompt_tokens - cache_hit)
        elif prompt_tokens > 0 and known_cache == 0:
            miss = prompt_tokens
        elif input_tokens > 0 and cache_create > 0:
            miss = input_tokens + cache_create
        return NormalizedUsage(
            total_input_tokens=total,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=miss,
            output_tokens=_integer(
                payload, "output_tokens", _integer(payload, "completion_tokens")
            ),
            reasoning_output_tokens=_integer(payload, "reasoning_tokens")
            or _nested_integer(payload, "completion_tokens_details", "reasoning_tokens"),
            confidence="provider",
            cache_creation_input_tokens=cache_create,
        )

    # 解析 DeepSeek Chat Completions usage，prompt_tokens 是 hit+miss 总数
    def _normalize_deepseek_chat(self, payload: dict[str, Any]) -> NormalizedUsage:
        prompt = _integer(payload, "prompt_tokens")
        hit = _integer(payload, "prompt_cache_hit_tokens")
        miss = _integer(payload, "prompt_cache_miss_tokens")
        if prompt == 0:
            prompt = hit + miss
        if miss == 0 and prompt >= hit:
            miss = prompt - hit
        completion = _integer(payload, "completion_tokens") or _integer(payload, "output_tokens")
        reasoning = _nested_integer(payload, "completion_tokens_details", "reasoning_tokens")
        if reasoning == 0:
            reasoning = _integer(payload, "reasoning_tokens")
        return NormalizedUsage(prompt, hit, miss, completion, reasoning, "provider")

    # 解析 OpenAI-compatible Chat usage，cached_tokens 是 prompt_tokens 的子集
    def _normalize_openai_chat(self, payload: dict[str, Any]) -> NormalizedUsage:
        prompt = _integer(payload, "prompt_tokens") or _integer(payload, "input_tokens")
        details = payload.get("prompt_tokens_details")
        hit = _integer(details, "cached_tokens") if isinstance(details, dict) else 0
        completion = _integer(payload, "completion_tokens") or _integer(payload, "output_tokens")
        completion_details = payload.get("completion_tokens_details")
        reasoning = (
            _integer(completion_details, "reasoning_tokens")
            if isinstance(completion_details, dict)
            else _integer(payload, "reasoning_tokens")
        )
        return NormalizedUsage(prompt, hit, max(0, prompt - hit), completion, reasoning, "provider")

    # 解析 Responses API 的 input/output details，不假设 vendor 字段
    def _normalize_responses(self, payload: dict[str, Any]) -> NormalizedUsage:
        input_tokens = _integer(payload, "input_tokens")
        output_tokens = _integer(payload, "output_tokens")
        details = payload.get("input_tokens_details")
        hit = _integer(details, "cached_tokens") if isinstance(details, dict) else 0
        output_details = payload.get("output_tokens_details")
        reasoning = (
            _integer(output_details, "reasoning_tokens")
            if isinstance(output_details, dict)
            else _integer(payload, "reasoning_tokens")
        )
        return NormalizedUsage(
            input_tokens,
            hit,
            max(0, input_tokens - hit),
            output_tokens,
            reasoning,
            "provider",
        )


# 从 mapping 或对象读取兼容 request/context 字段
def _value(source: object, key: str, default: object = None) -> object:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


# 计算 provider-normalized 总生成预算，避免 reasoning token 重复计数
def normalize_output_reserve(request: object, context_spec: object) -> int:
    semantics = _value(context_spec, "output_budget_semantics", "inclusive_total")
    if semantics == "inclusive_total":
        value = _first_present(
            request,
            ("max_output_tokens", "max_tokens"),
            default=None,
        )
        if value is None:
            return _as_nonnegative_int(
                _value(context_spec, "certified_worst_case_output", 0)
            )
        return _as_nonnegative_int(value)
    if semantics == "visible_plus_reasoning":
        visible = _first_present(
            request,
            ("visible_output_tokens", "max_output_tokens", "max_tokens"),
            default=None,
        )
        if visible is None:
            return _as_nonnegative_int(
                _value(context_spec, "certified_worst_case_output", 0)
            )
        reasoning = _value(request, "reasoning_budget", 0)
        return _as_nonnegative_int(visible) + _as_nonnegative_int(reasoning)
    certified = _value(context_spec, "certified_worst_case_output", 0)
    return _as_nonnegative_int(certified)


# 按字段顺序读取第一个非 None request 值，兼容对象同时声明空 max_output_tokens 与 max_tokens
def _first_present(source: object, keys: tuple[str, ...], *, default: object) -> object:
    for key in keys:
        value = _value(source, key, None)
        if value is not None:
            return value
    return default


# 将动态 request/context 值安全转换为非负整数，未知类型按零处理
def _as_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


# 用稳定的字符估算构建 admission 前的保守 input token 上界
def estimate_input_tokens(
    messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]],
    system: str | None,
) -> int:
    # Treat every UTF-8 wire byte as a possible token.  This is intentionally
    # conservative: it is an admission upper bound, not a tokenizer estimate.
    payload = {
        "system": system or "",
        "tools": tool_schemas,
        "messages": messages,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return len(encoded)
