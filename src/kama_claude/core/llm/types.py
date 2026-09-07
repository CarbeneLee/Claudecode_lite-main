from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, cast

ContinuationCarrier = Literal[
    "anthropic_thinking_blocks",
    "openai_reasoning_content",
    "responses_reasoning_items",
    "none",
]
ContinuationReplayScope = Literal[
    "all_active_tool_turns",
    "last_assistant_turn",
    "never",
]
OutputBudgetSemantics = Literal[
    "inclusive_total",
    "visible_plus_reasoning",
    "certified_worst_case",
]


@dataclass(frozen=True, slots=True)
class ProviderContinuationPolicy:
    """Provider-neutral rules for carrying reasoning state between requests."""

    carrier: ContinuationCarrier = "none"
    replay_scope: ContinuationReplayScope = "never"
    preserve_verbatim: bool = True
    required_for_followup: bool = False
    counts_context: bool = False
    participates_in_prefix_hash: bool = False
    durable_while_active: bool = False
    visible_in_user_history: bool = False
    output_budget_semantics: OutputBudgetSemantics = "inclusive_total"
    identity: str = "none"

    # 根据 route/protocol capability 创建 provider-neutral continuation policy
    @classmethod
    def for_route(
        cls, *, vendor: str, protocol: str, model: str = ""
    ) -> ProviderContinuationPolicy:
        protocol_normalized = protocol.lower().replace("_", "-")
        normalized = f"{vendor}:{protocol_normalized}:{model}".lower()
        anthropic_protocol = protocol_normalized in {
            "anthropic",
            "anthropic-compatible",
            "anthropic-messages",
        }
        if "deepseek" in normalized and anthropic_protocol:
            return DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY
        if anthropic_protocol:
            return NATIVE_ANTHROPIC_CONTINUATION_POLICY
        if protocol_normalized in {"responses", "responses-api", "openai-responses"}:
            return cls(
                carrier="responses_reasoning_items",
                replay_scope="all_active_tool_turns",
                required_for_followup=True,
                counts_context=True,
                participates_in_prefix_hash=True,
                durable_while_active=True,
                identity="responses-reasoning",
            )
        if protocol_normalized in {
            "openai",
            "chat-completions",
            "chat_completions",
            "openai-chat-completions",
        }:
            return cls(
                carrier="openai_reasoning_content",
                replay_scope="all_active_tool_turns",
                required_for_followup=True,
                counts_context=True,
                participates_in_prefix_hash=True,
                durable_while_active=True,
                identity="openai-reasoning",
            )
        return NONE_CONTINUATION_POLICY

    # 将 policy 转为稳定身份字符串，供 request envelope 与 meter baseline 使用
    def identity_key(self) -> str:
        return ":".join(
            (
                self.identity,
                self.carrier,
                self.replay_scope,
                str(self.preserve_verbatim),
                str(self.required_for_followup),
                str(self.counts_context),
                str(self.participates_in_prefix_hash),
                str(self.durable_while_active),
                str(self.visible_in_user_history),
                self.output_budget_semantics,
            )
        )

    # 将 continuation policy 序列化为 durable JSON metadata
    def to_dict(self) -> dict[str, object]:
        return {
            "carrier": self.carrier,
            "replay_scope": self.replay_scope,
            "preserve_verbatim": self.preserve_verbatim,
            "required_for_followup": self.required_for_followup,
            "counts_context": self.counts_context,
            "participates_in_prefix_hash": self.participates_in_prefix_hash,
            "durable_while_active": self.durable_while_active,
            "visible_in_user_history": self.visible_in_user_history,
            "output_budget_semantics": self.output_budget_semantics,
            "identity": self.identity,
        }


DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY = ProviderContinuationPolicy(
    carrier="anthropic_thinking_blocks",
    replay_scope="all_active_tool_turns",
    preserve_verbatim=True,
    required_for_followup=True,
    counts_context=True,
    participates_in_prefix_hash=True,
    durable_while_active=True,
    visible_in_user_history=False,
    output_budget_semantics="inclusive_total",
    identity="deepseek-anthropic-v4",
)

NATIVE_ANTHROPIC_CONTINUATION_POLICY = ProviderContinuationPolicy(
    carrier="anthropic_thinking_blocks",
    replay_scope="all_active_tool_turns",
    preserve_verbatim=True,
    required_for_followup=True,
    counts_context=True,
    participates_in_prefix_hash=True,
    durable_while_active=True,
    visible_in_user_history=False,
    output_budget_semantics="inclusive_total",
    identity="anthropic-thinking",
)

NONE_CONTINUATION_POLICY = ProviderContinuationPolicy()


@dataclass(frozen=True, slots=True)
class ProviderContinuationState:
    """原始 provider continuation blocks 及其 replay policy。"""

    blocks: tuple[dict[str, object], ...] = ()
    policy: ProviderContinuationPolicy = NONE_CONTINUATION_POLICY
    route_identity: str = ""
    protocol: str = ""

    # 初始化时深拷贝原始 blocks，保证 frozen state 不受 adapter 缓冲区后续修改影响
    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(copy.deepcopy(list(self.blocks))))

    # 返回深拷贝后的原始 blocks，避免渲染层修改持久化 continuation
    def as_blocks(self) -> list[dict[str, object]]:
        return copy.deepcopy(list(self.blocks))

    # 计算 continuation 的稳定哈希，原始字段和顺序都参与身份
    def identity_digest(self) -> str:
        payload = {
            "blocks": self.blocks,
            "policy": self.policy.identity_key(),
            "route": self.route_identity,
            "protocol": self.protocol,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # 将 provider continuation 原始 block 与 policy 持久化为 JSON
    def to_dict(self) -> dict[str, object]:
        return {
            "blocks": self.as_blocks(),
            "policy": self.policy.to_dict(),
            "route_identity": self.route_identity,
            "protocol": self.protocol,
        }

    # 从 durable JSON record 恢复 continuation state，未知字段保持忽略
    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ProviderContinuationState:
        raw_policy = payload.get("policy")
        if isinstance(raw_policy, str):
            legacy_policies = {
                "anthropic_thinking_blocks": NATIVE_ANTHROPIC_CONTINUATION_POLICY,
                "anthropic-thinking": NATIVE_ANTHROPIC_CONTINUATION_POLICY,
                "deepseek-anthropic-v4": DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
                "deepseek_anthropic": DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY,
            }
            policy = legacy_policies.get(raw_policy, NONE_CONTINUATION_POLICY)
            policy_payload: dict[str, object] = policy.to_dict()
        else:
            policy_payload = raw_policy if isinstance(raw_policy, dict) else {}
            policy = None
            # Legacy records may persist only the policy identity. Recover the
            # known capability before falling back to a conservative ``none``
            # policy so DeepSeek replay is not silently downgraded.
            legacy_identity = policy_payload.get("identity")
            if legacy_identity == "deepseek-anthropic-v4":
                policy = DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY
            elif legacy_identity == "anthropic-thinking":
                policy = NATIVE_ANTHROPIC_CONTINUATION_POLICY
        if policy is None:
            raw_carrier = policy_payload.get("carrier", "none")
            raw_scope = policy_payload.get("replay_scope", "never")
            raw_budget = policy_payload.get("output_budget_semantics", "inclusive_total")
            carrier = raw_carrier if isinstance(raw_carrier, str) and raw_carrier in {
                "anthropic_thinking_blocks",
                "openai_reasoning_content",
                "responses_reasoning_items",
                "none",
            } else "none"
            replay_scope = raw_scope if isinstance(raw_scope, str) and raw_scope in {
                "all_active_tool_turns",
                "last_assistant_turn",
                "never",
            } else "never"
            budget = raw_budget if isinstance(raw_budget, str) and raw_budget in {
                "inclusive_total",
                "visible_plus_reasoning",
                "certified_worst_case",
            } else "inclusive_total"
            policy = ProviderContinuationPolicy(
                carrier=cast(
                    ContinuationCarrier, carrier
                ),
                replay_scope=cast(
                    ContinuationReplayScope, replay_scope
                ),
                preserve_verbatim=_bool_field(policy_payload, "preserve_verbatim", True),
                required_for_followup=_bool_field(
                    policy_payload, "required_for_followup", False
                ),
                counts_context=_bool_field(policy_payload, "counts_context", False),
                participates_in_prefix_hash=_bool_field(
                    policy_payload, "participates_in_prefix_hash", False
                ),
                durable_while_active=_bool_field(
                    policy_payload, "durable_while_active", False
                ),
                visible_in_user_history=_bool_field(
                    policy_payload, "visible_in_user_history", False
                ),
                output_budget_semantics=cast(
                    OutputBudgetSemantics,
                    budget,
                ),
                identity=str(policy_payload.get("identity", "")),
            )
        # Legacy session rows called this field ``thinking_blocks``; accept it
        # during replay while emitting the canonical ``blocks`` shape on the
        # next durable write.
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list | tuple) or not any(
            isinstance(block, dict) for block in raw_blocks
        ):
            raw_blocks = payload.get("thinking_blocks", ())
        raw_sequence = raw_blocks if isinstance(raw_blocks, list | tuple) else ()
        blocks = tuple(
            copy.deepcopy(block)
            for block in raw_sequence
            if isinstance(block, dict)
        )
        if blocks and policy.carrier == "none" and not policy.required_for_followup:
            # Legacy rows that only persisted thinking blocks came from an
            # Anthropic-shaped provider.  Preserve them conservatively instead
            # of silently stripping required follow-up state on replay.
            route_identity = str(payload.get("route_identity", "")).lower()
            policy = (
                DEEPSEEK_ANTHROPIC_CONTINUATION_POLICY
                if "deepseek" in route_identity
                else NATIVE_ANTHROPIC_CONTINUATION_POLICY
            )
        return cls(
            blocks=blocks,
            policy=policy,
            route_identity=str(payload.get("route_identity", "")),
            protocol=str(payload.get("protocol", "")),
        )

    # 从兼容旧字段的 thinking_blocks 构造 canonical continuation state
    @classmethod
    def from_blocks(
        cls,
        blocks: list[dict[str, object]] | tuple[dict[str, object], ...],
        *,
        policy: ProviderContinuationPolicy = NATIVE_ANTHROPIC_CONTINUATION_POLICY,
        route_identity: str = "",
        protocol: str = "anthropic",
    ) -> ProviderContinuationState:
        return cls(
            blocks=tuple(copy.deepcopy(list(blocks))),
            policy=policy,
            route_identity=route_identity,
            protocol=protocol,
        )

    # 从兼容旧字段的 thinking_blocks 构造 canonical continuation state
    @classmethod
    def from_thinking_blocks(
        cls,
        blocks: list[dict[str, object]] | tuple[dict[str, object], ...],
        *,
        policy: ProviderContinuationPolicy = NATIVE_ANTHROPIC_CONTINUATION_POLICY,
        route_identity: str = "",
        protocol: str = "anthropic",
    ) -> ProviderContinuationState:
        return cls.from_blocks(
            blocks,
            policy=policy,
            route_identity=route_identity,
            protocol=protocol,
        )


# 从 durable policy payload 读取严格布尔值，避免字符串 "false" 被 Python truthiness 误解
def _bool_field(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


@dataclass
class UsageStats:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    context_pct: float = 0.0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    reasoning_output_tokens: int = 0
    measurement_confidence: str = "provider"
    usage_schema: str = "legacy"
    measurement_source: str = "provider"
    route_identity: str = ""
    surface_revision: int | None = None
    log_revision: int | None = None
    request_envelope_id: str = ""

    # 返回包含 cache token 的 provider context occupancy
    @property
    def total_input_tokens(self) -> int:
        if self.cache_hit_tokens or self.cache_miss_tokens:
            return max(self.input_tokens, self.cache_hit_tokens + self.cache_miss_tokens)
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class ToolCallBlock:
    id: str
    name: str
    input: dict[str, object]


@dataclass
class LlmResponse:
    stop_reason: str  # "end_turn" | "tool_use"
    tool_calls: list[ToolCallBlock] = field(default_factory=list)
    text: str = ""
    usage: UsageStats | None = None
    # thinking blocks from extended thinking — must be preserved verbatim in conversation history
    thinking_blocks: list[dict[str, object]] = field(default_factory=list)
    continuation_state: ProviderContinuationState | None = None

    # 将旧 thinking_blocks 与 canonical continuation state 保持双向兼容
    def __post_init__(self) -> None:
        if self.continuation_state is None and self.thinking_blocks:
            self.continuation_state = ProviderContinuationState.from_thinking_blocks(
                self.thinking_blocks
            )
        elif self.continuation_state is not None:
            self.thinking_blocks = self.continuation_state.as_blocks()
