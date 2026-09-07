from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.types import (
    LlmResponse,
    ProviderContinuationPolicy,
    ProviderContinuationState,
)
from kama_claude.core.llm.usage import (
    SUPPORTED_USAGE_SCHEMAS,
    NormalizedUsage,
    ReplayAwareTokenMeter,
    estimate_input_tokens,
    normalize_output_reserve,
)


class ContextAdmissionError(RuntimeError):
    """Provider hard-cap admission failure; caller must reduce the surface first."""

    # 保存 admission 失败时的保守 occupancy、输出预留与 provider capacity
    def __init__(
        self,
        message: str,
        *,
        input_tokens: int,
        output_reserve: int,
        capacity: int,
        origin: str = "admission",
    ) -> None:
        super().__init__(message)
        self.error_code = "CONTEXT_WINDOW_EXCEEDED"
        self.input_tokens = input_tokens
        self.output_reserve = output_reserve
        self.capacity = capacity
        self.origin = origin


# 将不同 SDK 的 context overflow 异常归一化到统一的 hard-cap 信号
def _is_provider_context_overflow(exc: BaseException) -> bool:
    code = str(getattr(exc, "error_code", "")).lower()
    status = str(getattr(exc, "code", "")).lower()
    message = str(exc).lower()
    markers = (
        "context_window_exceeded",
        "context length",
        "context_length_exceeded",
        "maximum context",
        "max context",
        "prompt is too long",
        "prompt too long",
        "too many tokens",
        "token limit",
    )
    for value in (code, status, message):
        if any(marker in value for marker in markers):
            return True
    return False


@dataclass(frozen=True, slots=True)
class StablePrefixBoundary:
    """Logical boundary separating immutable/semi-stable prompt bytes from dynamic state."""

    prefix_epoch: str
    immutable_hash: str
    semi_stable_hash: str
    logical_hash: str
    provider_serialized_hash: str
    boundary_position: int
    cache_policy: str


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Prepared request identity used by admission, cache telemetry and CAS."""

    messages: tuple[dict[str, object], ...]
    tool_schemas: tuple[dict[str, object], ...]
    system: str
    route_epoch: str
    prefix_epoch: str
    request_envelope_id: str
    stable_prefix: StablePrefixBoundary
    output_reserve: int
    immutable_system: str = ""
    semi_stable_context: str = ""
    surface_revision: int = 0
    request_option_identity: str = ""


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Hard admission result with normalized occupancy and output reserve."""

    envelope: RequestEnvelope
    estimated_input_tokens: int
    output_reserve: int
    capacity: int
    state: str


@dataclass(frozen=True, slots=True)
class MeterBaseline:
    """Last successful usage baseline tied to exact surface/route/prefix identity."""

    surface_revision: int
    route_epoch: str
    prefix_epoch: str
    envelope_id: str
    continuation_policy_identity: str
    total_input_tokens: int
    confidence: str
    usage_schema: str = ""
    measurement_source: str = "provider"
    log_revision: int | None = None


# 将 request/layout 对象编码为稳定 JSON bytes，供 prefix 和 envelope digest 使用
def _stable_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


# 对工具 schema 做 canonical 排序，确保 stable prefix 不受注册顺序漂移影响
def canonical_tool_schemas(tool_schemas: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        copy.deepcopy(schema)
        for schema in sorted(
            tool_schemas,
            key=lambda schema: (
                str(schema.get("name", "")),
                _stable_json(schema),
            ),
        )
    ]


# 计算 provider-neutral stable prefix boundary，不把 dynamic session state 混入 A/B hash
def build_stable_prefix_boundary(
    *,
    immutable_system: str,
    tool_schemas: list[dict[str, object]],
    semi_stable_context: str = "",
    cache_policy: str = "provider_defined",
    prefix_epoch: str | None = None,
) -> StablePrefixBoundary:
    canonical_tools = canonical_tool_schemas(tool_schemas)
    immutable_payload = {"system": immutable_system, "tools": canonical_tools}
    semi_payload = {"context": semi_stable_context}
    immutable_hash = hashlib.sha256(_stable_json(immutable_payload)).hexdigest()
    semi_hash = hashlib.sha256(_stable_json(semi_payload)).hexdigest()
    logical_hash = hashlib.sha256(
        _stable_json({"immutable": immutable_hash, "semi": semi_hash})
    ).hexdigest()
    provider_serialized = (
        immutable_system
        + "\n"
        + semi_stable_context
        + "\n"
        + _stable_json(canonical_tools).decode()
    )
    provider_hash = hashlib.sha256(provider_serialized.encode("utf-8")).hexdigest()
    epoch = prefix_epoch or f"prefix-{logical_hash[:16]}"
    return StablePrefixBoundary(
        prefix_epoch=epoch,
        immutable_hash=immutable_hash,
        semi_stable_hash=semi_hash,
        logical_hash=logical_hash,
        provider_serialized_hash=provider_hash,
        boundary_position=len(immutable_system) + len(semi_stable_context),
        cache_policy=cache_policy,
    )


# 从完整 system 文本中剥离已单独传输的 A/B 前缀，避免 provider 重复注入
def _dynamic_system_text(
    rendered_system: str,
    immutable_system: str,
    semi_stable_context: str,
) -> str:
    dynamic = rendered_system
    if immutable_system and dynamic.startswith(immutable_system):
        dynamic = dynamic[len(immutable_system) :]
    semi_prefix = "\n\n" + semi_stable_context
    if semi_stable_context and dynamic.startswith(semi_prefix):
        dynamic = dynamic[len(semi_prefix) :]
    elif semi_stable_context:
        # A legacy/custom adapter may receive the complete rendered system
        # without the immutable slot; remove the one exact semi-stable block
        # rather than sending Layer B twice.
        embedded = semi_prefix + "\n\n"
        if embedded in dynamic:
            dynamic = dynamic.replace(embedded, "\n\n", 1)
    return dynamic.lstrip("\n")


class ProviderRequestGateway:
    """所有 normal/maintenance generation 共用的 provider capacity admission facade."""

    # 初始化 gateway，并从 adapter capability 推断 route、usage schema 与 continuation policy
    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        context_window: int | None = None,
        route_identity: str | None = None,
        protocol: str | None = None,
        usage_schema: str | None = None,
        continuation_policy: ProviderContinuationPolicy | None = None,
        output_budget_semantics: str | None = None,
        safety_margin: int = 1024,
        soft_trigger_ratio: float = 0.80,
        target_ratio: float = 0.60,
    ) -> None:
        self._provider = provider
        provider_model = getattr(provider, "model", getattr(provider, "_model", ""))
        self._model = model or (provider_model if isinstance(provider_model, str) else "")
        provider_protocol = getattr(provider, "protocol", None)
        selected_protocol = protocol or (
            provider_protocol if isinstance(provider_protocol, str) else "anthropic"
        )
        self._protocol = selected_protocol.lower().replace("_", "-")
        provider_window = getattr(provider, "context_window", None)
        if context_window is not None:
            self._context_window = context_window
        elif isinstance(provider_window, int):
            self._context_window = provider_window
        elif "deepseek-v4" in self._model.lower():
            self._context_window = 1_000_000
        elif self._model:
            raise ValueError(
                "context capacity is unknown for model; configure context_window"
            )
        else:
            # Untyped legacy test doubles have no route identity from which a
            # real capacity could be inferred.  Keep their compatibility path
            # explicit; every concrete adapter must expose model/capacity.
            self._context_window = 200_000
        if self._context_window <= 0:
            raise ValueError("context_window must be positive")
        provider_route = getattr(provider, "route_identity", None)
        self._route_identity = route_identity or (
            provider_route
            if isinstance(provider_route, str)
            else f"{self._model}:{self._protocol}"
        )
        provider_schema = getattr(provider, "usage_schema", None)
        self._usage_schema = usage_schema or (
            str(provider_schema)
            if isinstance(provider_schema, str)
            else self._default_usage_schema()
        )
        if self._usage_schema not in SUPPORTED_USAGE_SCHEMAS:
            raise ValueError(f"unsupported usage schema: {self._usage_schema}")
        provider_policy = getattr(provider, "continuation_policy", None)
        self._continuation_policy = continuation_policy or (
            provider_policy
            if isinstance(provider_policy, ProviderContinuationPolicy)
            else ProviderContinuationPolicy.for_route(
                vendor=self._route_identity,
                protocol=self._protocol,
                model=self._model,
            )
        )
        provider_budget_semantics = getattr(provider, "output_budget_semantics", None)
        self._output_budget_semantics = output_budget_semantics or (
            str(provider_budget_semantics)
            if isinstance(provider_budget_semantics, str)
            else getattr(self._continuation_policy, "output_budget_semantics", "inclusive_total")
        )
        provider_max_output = getattr(provider, "max_output_tokens", 8192)
        self._max_output_tokens = (
            max(1, provider_max_output)
            if isinstance(provider_max_output, int)
            else 8192
        )
        self._safety_margin = max(0, safety_margin)
        if not 0 < target_ratio < soft_trigger_ratio <= 1:
            raise ValueError("target_ratio must be below soft_trigger_ratio in (0, 1]")
        self._soft_trigger_ratio = soft_trigger_ratio
        self._target_ratio = target_ratio
        self._route_epoch = self._epoch(
            "route",
            f"{self._route_identity}:{self._protocol}:{self._model}:"
            f"{self._usage_schema}:{self._continuation_policy.identity_key()}",
        )
        self._prefix_epoch = "prefix-0"
        self._last_prefix_logical_hash: str | None = None
        self._meter_baseline: MeterBaseline | None = None
        self._replay_meter = ReplayAwareTokenMeter()

    # route/protocol 尚未声明 adapter capability 时选择最保守的 wire usage schema
    def _default_usage_schema(self) -> str:
        route = self._route_identity.lower()
        if self._protocol in {"anthropic", "anthropic-compatible", "anthropic-messages"}:
            return (
                "deepseek_anthropic_messages_v1"
                if "deepseek" in route or "deepseek" in self._model.lower()
                else "native_anthropic_v1"
            )
        if self._protocol in {"responses", "responses-api", "openai-responses"}:
            return (
                "deepseek_responses_v1"
                if "deepseek" in route or "deepseek" in self._model.lower()
                else "openai_responses_v1"
            )
        if self._protocol in {
            "openai",
            "chat-completions",
            "chat_completions",
            "openai-chat-completions",
        }:
            return (
                "deepseek_chat_completions_v1"
                if "deepseek" in route or "deepseek" in self._model.lower()
                else "openai_chat_completions_v1"
            )
        return "native_anthropic_v1"

    # 生成 epoch 的稳定摘要，供 CAS 和 prepared request identity 使用
    @staticmethod
    def _epoch(kind: str, value: str) -> str:
        return f"{kind}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"

    # 返回被 gateway 包装的原始 provider，供 tracing/测试读取 capability
    @property
    def provider(self) -> LLMProvider:
        return self._provider

    # 返回当前 route epoch
    @property
    def route_epoch(self) -> str:
        return self._route_epoch

    # 返回当前 route identity，供 checkpoint envelope 与 replay diagnostics 使用
    @property
    def route_identity(self) -> str:
        return self._route_identity

    # 返回当前 cacheable prefix epoch
    @property
    def prefix_epoch(self) -> str:
        return self._prefix_epoch

    # 返回 provider context capacity
    @property
    def context_window(self) -> int:
        return self._context_window

    # 返回 soft trigger 与 hysteresis target，供 compaction lifecycle 选择释放量
    @property
    def soft_trigger_ratio(self) -> float:
        return self._soft_trigger_ratio

    # 返回 hysteresis target，不把优化目标误判为 hard safety
    @property
    def target_ratio(self) -> float:
        return self._target_ratio

    # 返回 continuation capability
    @property
    def continuation_policy(self) -> ProviderContinuationPolicy:
        return self._continuation_policy

    # 返回 provider-normalized output budget semantics，供 maintenance 与测试复用
    @property
    def output_budget_semantics(self) -> str:
        return str(self._output_budget_semantics)

    # 兼容未声明 capability 的 legacy provider，保留其已注册工具顺序
    def _wire_tool_schemas(
        self,
        envelope: RequestEnvelope,
        original: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if all(
            not hasattr(self._provider, attribute)
            for attribute in ("route_identity", "usage_schema", "continuation_policy")
        ):
            return copy.deepcopy(original)
        return [copy.deepcopy(schema) for schema in envelope.tool_schemas]

    # 返回 provider-neutral cache serialization policy
    @property
    def cache_policy(self) -> str:
        if "deepseek" in self._route_identity.lower():
            return "automatic_exact_prefix"
        if self._protocol in {"anthropic", "anthropic-compatible", "anthropic-messages"}:
            return "explicit_breakpoints"
        return "provider_defined"

    # 记录 route/model/protocol 变化而不增加 surface revision
    def change_route(
        self,
        route_identity: str,
        *,
        protocol: str | None = None,
        model: str | None = None,
        context_window: int | None = None,
        usage_schema: str | None = None,
        continuation_policy: ProviderContinuationPolicy | None = None,
    ) -> None:
        self._route_identity = route_identity
        if protocol is not None:
            self._protocol = protocol.lower().replace("_", "-")
        if model is not None:
            self._model = model
        if context_window is not None:
            if context_window <= 0:
                raise ValueError("context_window must be positive")
            self._context_window = context_window
        elif model is not None:
            if "deepseek-v4" in self._model.lower():
                self._context_window = 1_000_000
            else:
                raise ValueError(
                    "context capacity is unknown for model; configure context_window"
                )
        self._usage_schema = usage_schema or self._default_usage_schema()
        if self._usage_schema not in SUPPORTED_USAGE_SCHEMAS:
            raise ValueError(f"unsupported usage schema: {self._usage_schema}")
        self._continuation_policy = continuation_policy or ProviderContinuationPolicy.for_route(
            vendor=self._route_identity,
            protocol=self._protocol,
            model=self._model,
        )
        self._output_budget_semantics = self._continuation_policy.output_budget_semantics
        self._route_epoch = self._epoch(
            "route",
            f"{self._route_identity}:{self._protocol}:{self._model}:"
            f"{self._usage_schema}:{self._continuation_policy.identity_key()}",
        )
        self._meter_baseline = None
        self._replay_meter.invalidate()

    # 使 A/B prefix 变化并淘汰 meter baseline
    def invalidate_prefix(self, prefix_identity: str) -> None:
        self._prefix_epoch = self._epoch("prefix", prefix_identity)
        self._last_prefix_logical_hash = None
        self._meter_baseline = None
        self._replay_meter.invalidate()

    # 构造逻辑 stable prefix，不包含 TaskContract/checkpoint/recent tail 动态文本
    def stable_prefix_boundary(
        self,
        *,
        immutable_system: str,
        tool_schemas: list[dict[str, object]],
        semi_stable_context: str = "",
    ) -> StablePrefixBoundary:
        boundary = build_stable_prefix_boundary(
            immutable_system=immutable_system,
            tool_schemas=tool_schemas,
            semi_stable_context=semi_stable_context,
            cache_policy=self.cache_policy,
        )
        if self._last_prefix_logical_hash is None:
            self._last_prefix_logical_hash = boundary.logical_hash
        elif self._last_prefix_logical_hash != boundary.logical_hash:
            self._prefix_epoch = self._epoch("prefix", boundary.logical_hash)
            self._last_prefix_logical_hash = boundary.logical_hash
            self._meter_baseline = None
            self._replay_meter.invalidate()
        return replace(boundary, prefix_epoch=self._prefix_epoch)

    # 生成带 dynamic Layer C 的 prepared request envelope
    def prepare_request(
        self,
        *,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        system: str | None,
        immutable_system: str | None = None,
        semi_stable_context: str = "",
        request: object | None = None,
        output_reserve: int | None = None,
        surface_revision: int = 0,
        request_options: Mapping[str, object] | None = None,
    ) -> RequestEnvelope:
        canonical_tools = canonical_tool_schemas(tool_schemas)
        base_system = immutable_system if immutable_system is not None else (system or "")
        boundary = self.stable_prefix_boundary(
            immutable_system=base_system,
            tool_schemas=canonical_tools,
            semi_stable_context=semi_stable_context,
        )
        rendered_system = system or ""
        reserve = output_reserve
        if reserve is None:
            request_obj = request or {"max_tokens": self._max_output_tokens}
            reserve = normalize_output_reserve(
                request_obj,
                {
                    "output_budget_semantics": self._output_budget_semantics,
                    "certified_worst_case_output": self._max_output_tokens,
                },
            )
        option_payload: dict[str, object] = dict(request_options or {})
        # response format, tool choice, thinking mode and similar wire options
        # can invalidate provider prefix compatibility even when conversation
        # bytes are unchanged.  Keep them out of Layer A/B hashes but bind them
        # to the prepared request identity.  Accept both mapping and typed
        # request objects used by provider adapters.
        for key in ("response_format", "tool_choice", "thinking"):
            if key in option_payload:
                continue
            value: object = None
            present = False
            if isinstance(request, Mapping) and key in request:
                value = request[key]
                present = True
            elif request is not None and hasattr(request, key):
                value = getattr(request, key)
                present = value is not None
            if present:
                option_payload[key] = copy.deepcopy(value)
        option_identity = hashlib.sha256(_stable_json(option_payload)).hexdigest()
        message_copy = tuple(copy.deepcopy(messages))
        envelope_id = hashlib.sha256(
            _stable_json(
                {
                    "system": rendered_system,
                    "messages": message_copy,
                    "tools": canonical_tools,
                    "route": self._route_epoch,
                    "prefix": self._prefix_epoch,
                    "continuation": self._continuation_policy.identity_key(),
                    "request_options": option_identity,
                }
            )
        ).hexdigest()
        return RequestEnvelope(
            messages=message_copy,
            tool_schemas=tuple(canonical_tools),
            system=rendered_system,
            route_epoch=self._route_epoch,
            prefix_epoch=self._prefix_epoch,
            request_envelope_id=envelope_id,
            stable_prefix=boundary,
            output_reserve=reserve,
            immutable_system=immutable_system or "",
            semi_stable_context=semi_stable_context,
            surface_revision=surface_revision,
            request_option_identity=option_identity,
        )

    # 在 provider capacity 上执行 hard admission，soft/target 策略由 compaction lifecycle 决定
    def admit(self, envelope: RequestEnvelope) -> AdmissionResult:
        if (
            envelope.route_epoch != self._route_epoch
            or envelope.prefix_epoch != self._prefix_epoch
        ):
            raise RuntimeError("request envelope was invalidated by route or prefix change")
        # Estimate the same logical system bytes that the adapter will put on
        # the wire: A/B are separate slots, while dynamic text is sent once.
        system_for_estimate = envelope.system
        if envelope.immutable_system:
            dynamic = _dynamic_system_text(
                envelope.system,
                envelope.immutable_system,
                envelope.semi_stable_context,
            )
            if envelope.system and not envelope.system.startswith(envelope.immutable_system):
                dynamic = envelope.system
            system_for_estimate = "\n\n".join(
                part
                for part in (
                    envelope.immutable_system,
                    envelope.semi_stable_context,
                    dynamic,
                )
                if part
            )
        elif envelope.semi_stable_context:
            # Adapters that expose only the semi-stable slot still serialize
            # Layer B separately; include it in the local occupancy estimate
            # even when no immutable slot was supplied by the caller.
            dynamic = _dynamic_system_text(
                envelope.system,
                "",
                envelope.semi_stable_context,
            )
            system_for_estimate = "\n\n".join(
                part
                for part in (envelope.semi_stable_context, dynamic)
                if part
            )
        estimate = estimate_input_tokens(
            list(envelope.messages),
            list(envelope.tool_schemas),
            system_for_estimate,
        )
        baseline = self._replay_meter.estimate(
            surface_revision=envelope.surface_revision,
            route_epoch=envelope.route_epoch,
            prefix_epoch=envelope.prefix_epoch,
            envelope_id=envelope.request_envelope_id,
            continuation_policy_identity=self._continuation_policy.identity_key(),
        )
        if baseline is not None:
            estimate = baseline[0]
        required = estimate + envelope.output_reserve + self._safety_margin
        if required > self._context_window:
            raise ContextAdmissionError(
                "request exceeds provider context capacity",
                input_tokens=estimate,
                output_reserve=envelope.output_reserve,
                capacity=self._context_window,
            )
        ratio = required / self._context_window
        state = (
            "SOFT_PRESSURED"
            if ratio >= self._soft_trigger_ratio
            else ("SAFE_ABOVE_TARGET" if ratio > self._target_ratio else "AT_TARGET")
        )
        return AdmissionResult(
            envelope=envelope,
            estimated_input_tokens=estimate,
            output_reserve=envelope.output_reserve,
            capacity=self._context_window,
            state=state,
        )

    # 通过同一 gateway 执行 normal 或 bounded maintenance generation
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
        request: object | None = None,
        output_reserve: int | None = None,
        surface_revision: int = 0,
        maintenance: bool = False,
        immutable_system: str | None = None,
        semi_stable_context: str = "",
        request_options: Mapping[str, object] | None = None,
    ) -> LlmResponse:
        if output_reserve is None and max_output_tokens is not None:
            request_for_reserve: object = request
            if request_for_reserve is None:
                request_for_reserve = {"max_output_tokens": max_output_tokens}
            elif isinstance(request_for_reserve, Mapping):
                request_for_reserve = {
                    **request_for_reserve,
                    "max_output_tokens": max_output_tokens,
                }
            else:
                request_for_reserve = {
                    "max_output_tokens": max_output_tokens,
                    "visible_output_tokens": getattr(
                        request_for_reserve, "visible_output_tokens", max_output_tokens
                    ),
                    "reasoning_budget": getattr(request_for_reserve, "reasoning_budget", 0),
                }
            output_reserve = normalize_output_reserve(
                request_for_reserve,
                {
                    "output_budget_semantics": self._output_budget_semantics,
                    "certified_worst_case_output": self._max_output_tokens,
                },
            )
        envelope = self.prepare_request(
            messages=messages,
            tool_schemas=tool_schemas,
            system=system,
            immutable_system=immutable_system,
            semi_stable_context=semi_stable_context,
            request=request,
            output_reserve=output_reserve,
            surface_revision=surface_revision,
            request_options=request_options,
        )
        admission = self.admit(envelope)
        try:
            parameters: Mapping[str, inspect.Parameter] = inspect.signature(
                self._provider.chat
            ).parameters
        except (TypeError, ValueError):
            parameters = {}
        supports_max_output = "max_output_tokens" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        # A generic ``**kwargs`` is not proof that an adapter understands the
        # layered system contract.  Treating every such callable as layout-aware
        # would strip the complete legacy system prompt before it reaches test
        # doubles and third-party adapters.  Concrete adapters opt in by naming
        # the slots explicitly (AnthropicProvider does); legacy callables keep
        # receiving the original rendered system as one block.
        supports_immutable = "immutable_system" in parameters
        supports_semi_stable = "semi_stable_context" in parameters
        supports_layout = supports_immutable or supports_semi_stable
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        provider_system = envelope.system
        if supports_immutable:
            provider_system = _dynamic_system_text(
                provider_system,
                envelope.immutable_system,
                envelope.semi_stable_context if supports_semi_stable else "",
            )
        elif supports_semi_stable:
            provider_system = _dynamic_system_text(
                provider_system,
                "",
                envelope.semi_stable_context,
            )
        provider_kwargs: dict[str, object] = {}
        if accepts_kwargs or "step" in parameters:
            provider_kwargs["step"] = step
        if accepts_kwargs or "system" in parameters:
            provider_kwargs["system"] = (
                provider_system if supports_layout else envelope.system
            )
        if supports_max_output:
            provider_kwargs["max_output_tokens"] = envelope.output_reserve
        if supports_immutable:
            provider_kwargs["immutable_system"] = envelope.immutable_system
        if supports_semi_stable:
            provider_kwargs["semi_stable_context"] = envelope.semi_stable_context
        provider_chat = cast(Any, self._provider).chat
        try:
            response = cast(LlmResponse, await provider_chat(
                list(envelope.messages),
                self._wire_tool_schemas(envelope, tool_schemas),
                bus,
                run_id,
                **provider_kwargs,
            ))
        except Exception as exc:
            if not _is_provider_context_overflow(exc):
                raise
            raise ContextAdmissionError(
                "provider rejected request because it exceeds context capacity",
                input_tokens=admission.estimated_input_tokens,
                output_reserve=admission.output_reserve,
                capacity=admission.capacity,
                origin="provider",
            ) from exc
        response = self._normalize_continuation(response)
        if response.usage is not None:
            usage = response.usage
            usage.route_identity = self._route_identity
            usage.surface_revision = surface_revision
            usage.request_envelope_id = envelope.request_envelope_id
            self._meter_baseline = MeterBaseline(
                surface_revision=surface_revision,
                route_epoch=envelope.route_epoch,
                prefix_epoch=envelope.prefix_epoch,
                envelope_id=envelope.request_envelope_id,
                continuation_policy_identity=self._continuation_policy.identity_key(),
                total_input_tokens=usage.total_input_tokens,
                confidence=usage.measurement_confidence,
                usage_schema=usage.usage_schema,
                measurement_source=usage.measurement_source,
                log_revision=usage.log_revision,
            )
            normalized_usage = NormalizedUsage(
                total_input_tokens=usage.total_input_tokens,
                cache_hit_tokens=usage.cache_hit_tokens,
                cache_miss_tokens=usage.cache_miss_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_output_tokens=usage.reasoning_output_tokens,
                confidence=usage.measurement_confidence,  # type: ignore[arg-type]
                usage_schema=usage.usage_schema,
                measurement_source=usage.measurement_source,
                route=self._route_identity,
                surface_revision=surface_revision,
                log_revision=usage.log_revision,
            )
            self._replay_meter.record(
                normalized_usage,
                surface_revision=surface_revision,
                route_epoch=envelope.route_epoch,
                prefix_epoch=envelope.prefix_epoch,
                envelope_id=envelope.request_envelope_id,
                continuation_policy_identity=self._continuation_policy.identity_key(),
                usage_schema=usage.usage_schema,
                measurement_source=usage.measurement_source,
                log_revision=usage.log_revision,
            )
        if maintenance and (response.tool_calls or response.stop_reason == "tool_use"):
            raise RuntimeError("maintenance generation returned an unexpected tool call")
        return response

    # 将 legacy provider 的 thinking blocks 绑定到当前 route capability，保持后续 replay policy 一致
    def _normalize_continuation(self, response: LlmResponse) -> LlmResponse:
        state = response.continuation_state
        blocks = response.thinking_blocks
        if not blocks and state is not None:
            blocks = state.as_blocks()
        if not blocks:
            return response
        if state is not None and (
            state.policy.identity_key() == self._continuation_policy.identity_key()
            and state.route_identity == self._route_identity
            and state.protocol == self._protocol
        ):
            # Keep the legacy response field populated even when a capable
            # adapter returned only the canonical continuation state.
            response.thinking_blocks = copy.deepcopy(blocks)
            return response
        response.continuation_state = ProviderContinuationState.from_thinking_blocks(
            blocks,
            policy=self._continuation_policy,
            route_identity=self._route_identity,
            protocol=self._protocol,
        )
        response.thinking_blocks = response.continuation_state.as_blocks()
        return response

    # 读取最近一次 provider usage baseline
    @property
    def meter_baseline(self) -> MeterBaseline | None:
        return self._meter_baseline


# 将任意 legacy provider 提升为统一 gateway，避免生成路径绕过 admission
def ensure_gateway(provider: LLMProvider) -> ProviderRequestGateway:
    if isinstance(provider, ProviderRequestGateway):
        return provider
    return ProviderRequestGateway(provider)


AdmissionGate = ProviderRequestGateway
MaintenanceRequestGate = ProviderRequestGateway


__all__ = [
    "AdmissionResult",
    "ContextAdmissionError",
    "MeterBaseline",
    "AdmissionGate",
    "MaintenanceRequestGate",
    "ProviderRequestGateway",
    "RequestEnvelope",
    "StablePrefixBoundary",
    "canonical_tool_schemas",
    "build_stable_prefix_boundary",
    "ensure_gateway",
]
