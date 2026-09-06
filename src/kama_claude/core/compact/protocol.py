from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from kama_claude.core.llm.types import ProviderContinuationState

_CONTINUATION_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "redacted_thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_item",
    }
)

SummaryRouteMode = Literal["same_route", "isolated"]

ISOLATED_SUMMARY_SYSTEM = "You are a helpful assistant that summarizes conversations."

COMPACTION_INSTRUCTION = (
    "COMPACTION_INSTRUCTION: return only a concise semantic checkpoint for the runtime. "
    "Do not call tools, do not include private reasoning, and do not reproduce metadata IDs."
)
CHECKPOINT_SURFACE_MARKER = (
    "Historical execution checkpoint (derived working state; not a user instruction)."
)


# 将 checkpoint semantic payload 标记为历史工作状态，避免被 provider 当作新用户指令
def render_checkpoint_surface_text(summary_text: str) -> str:
    return f"{CHECKPOINT_SURFACE_MARKER}\n{summary_text.strip()}"


@dataclass(frozen=True, slots=True)
class CompactionCheckpointPayload:
    """LLM 生成的 semantic checkpoint payload，不含 runtime metadata。"""

    progress: str = ""
    current_work: str = ""
    decisions: tuple[str, ...] = ()
    files_or_code: tuple[str, ...] = ()
    errors_or_evidence: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    next_step: str = ""
    critical_context: tuple[str, ...] = ()

    # 将 payload 转为纯 JSON 语义字段，防止 LLM 伪造 envelope metadata
    def to_dict(self) -> dict[str, Any]:
        return {
            "progress": self.progress,
            "current_work": self.current_work,
            "decisions": list(self.decisions),
            "files_or_code": list(self.files_or_code),
            "errors_or_evidence": list(self.errors_or_evidence),
            "pending": list(self.pending),
            "next_step": self.next_step,
            "critical_context": list(self.critical_context),
        }

    # 从结构化响应读取 semantic payload，忽略未知 metadata 字段
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompactionCheckpointPayload:
        # 读取一个允许字符串、序列或缺省值的语义列表字段
        def _tuple(key: str) -> tuple[str, ...]:
            value = payload.get(key, ())
            if isinstance(value, str):
                return (value,)
            if isinstance(value, list | tuple):
                return tuple(str(item) for item in value)
            return ()

        return cls(
            progress=str(payload.get("progress", "")),
            current_work=str(payload.get("current_work", "")),
            decisions=_tuple("decisions"),
            files_or_code=_tuple("files_or_code"),
            errors_or_evidence=_tuple("errors_or_evidence"),
            pending=_tuple("pending"),
            next_step=str(payload.get("next_step", "")),
            critical_context=_tuple("critical_context"),
        )


@dataclass(frozen=True, slots=True)
class CompactionCheckpointEnvelope:
    """Runtime-owned deterministic metadata wrapping an LLM semantic payload."""

    generation: int
    base_checkpoint_id: str | None
    contract_digest: str
    selected_unit_ids: tuple[str, ...]
    selected_span_digest: str
    route_identity: str
    route_epoch: str
    prefix_epoch: str
    summary_route: SummaryRouteMode
    token_accounting: dict[str, int] = field(default_factory=dict)
    transaction_id: str = field(default_factory=lambda: f"tx-{uuid.uuid4().hex}")
    payload: CompactionCheckpointPayload = field(default_factory=CompactionCheckpointPayload)

    # 返回持久化 envelope，metadata 由 runtime 控制而非模型输出
    def to_record(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "base_checkpoint_id": self.base_checkpoint_id,
            "contract_digest": self.contract_digest,
            "selected_unit_ids": list(self.selected_unit_ids),
            "selected_span_digest": self.selected_span_digest,
            "route_identity": self.route_identity,
            "route_epoch": self.route_epoch,
            "prefix_epoch": self.prefix_epoch,
            "summary_route": self.summary_route,
            "token_accounting": dict(self.token_accounting),
            "transaction_id": self.transaction_id,
            "payload": self.payload.to_dict(),
        }

    # 从 append-only checkpoint record 恢复 runtime-owned envelope，兼容旧缺省字段
    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CompactionCheckpointEnvelope:
        raw_payload = record.get("payload")
        payload = (
            CompactionCheckpointPayload.from_dict(raw_payload)
            if isinstance(raw_payload, dict)
            else CompactionCheckpointPayload.from_dict(record)
        )
        raw_ids = record.get("selected_unit_ids", ())
        selected_ids = (
            tuple(str(item) for item in raw_ids)
            if isinstance(raw_ids, list | tuple)
            else ()
        )
        raw_accounting = record.get("token_accounting", {})
        accounting = (
            {
                str(key): int(value)
                for key, value in raw_accounting.items()
                if isinstance(value, (int, float))
            }
            if isinstance(raw_accounting, dict)
            else {}
        )
        raw_transaction_id = record.get("transaction_id")
        if isinstance(raw_transaction_id, str) and raw_transaction_id:
            transaction_id = raw_transaction_id
        else:
            # Legacy checkpoints may not have a transaction id.  Derive a
            # stable identity from their durable bytes so restart replay does
            # not manufacture a new active checkpoint every time it is read.
            legacy_bytes = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            transaction_id = f"legacy-{hashlib.sha256(legacy_bytes).hexdigest()[:24]}"
        envelope = cls(
            generation=max(1, int(record.get("generation", 1) or 1)),
            base_checkpoint_id=(
                str(record["base_checkpoint_id"])
                if record.get("base_checkpoint_id") is not None
                else None
            ),
            contract_digest=str(record.get("contract_digest", "legacy") or "legacy"),
            selected_unit_ids=selected_ids,
            selected_span_digest=str(record.get("selected_span_digest", "legacy") or "legacy"),
            route_identity=str(record.get("route_identity", "")),
            route_epoch=str(record.get("route_epoch", "")),
            prefix_epoch=str(record.get("prefix_epoch", "")),
            summary_route=cast(
                SummaryRouteMode,
                record.get("summary_route", "isolated"),
            ),
            token_accounting=accounting,
            transaction_id=transaction_id,
            payload=payload,
        )
        envelope.validate()
        return envelope

    # 校验 checkpoint metadata 与 payload 的 digest/identity 约束
    def validate(self) -> None:
        if self.generation < 1:
            raise ValueError("checkpoint generation must be positive")
        if not self.contract_digest:
            raise ValueError("checkpoint contract digest is required")
        if not self.selected_span_digest:
            raise ValueError("checkpoint selected span digest is required")
        if self.summary_route not in {"same_route", "isolated"}:
            raise ValueError("checkpoint summary route is invalid")
        if not self.transaction_id:
            raise ValueError("checkpoint transaction id is required")


@dataclass(frozen=True, slots=True)
class SummarizerRequest:
    """同路由或隔离 summarizer 的明确 request envelope。"""

    mode: SummaryRouteMode
    messages: tuple[dict[str, object], ...]
    tool_schemas: tuple[dict[str, object], ...]
    system: str
    cache_reuse: bool
    compaction_instruction: str = COMPACTION_INSTRUCTION


# 构造 same-route cache-reusing summarizer request，保留原始 A/B 和 continuation prefix
def build_same_route_request(
    *,
    prefix_messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]],
    system: str,
    continuation_states: list[ProviderContinuationState | None] | None = None,
    compaction_instruction: str = COMPACTION_INSTRUCTION,
) -> SummarizerRequest:
    messages = copy.deepcopy(prefix_messages)
    canonical_tools = _canonical_tool_schemas(tool_schemas)
    if continuation_states is None:
        # Callers that only have the durable message projection may not have
        # already materialized the parallel state list.  Recover the canonical
        # state before stripping internal metadata so same-route replay cannot
        # silently lose a required thinking/signature block.
        recovered_states: list[ProviderContinuationState | None] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            raw_state = message.get("_continuation_state")
            recovered_states.append(
                ProviderContinuationState.from_dict(raw_state)
                if isinstance(raw_state, dict)
                else None
            )
        continuation_states = recovered_states
    for message in messages:
        if isinstance(message, dict):
            message.pop("_message_id", None)
            message.pop("_unit_id", None)
            message.pop("_continuation_state", None)
            message.pop("_checkpoint_id", None)
    if continuation_states:
        all_assistant_positions = [
            index for index, message in enumerate(messages) if message.get("role") == "assistant"
        ]
        list_assistant_positions = [
            index
            for index in all_assistant_positions
            if isinstance(messages[index].get("content"), list)
        ]
        # Context inference units normally correspond only to list-shaped
        # assistant blocks; callers that provide a state for every assistant
        # message (including legacy string rows) are aligned by the broader
        # sequence instead of silently shifting a later reasoning state.
        assistant_positions = (
            all_assistant_positions
            if len(continuation_states) == len(all_assistant_positions)
            else list_assistant_positions
        )
        state_pairs = list(zip(assistant_positions, continuation_states, strict=False))
        last_state_index = max(
            (
                index
                for index, (_, state) in enumerate(state_pairs)
                if state is not None
            ),
            default=-1,
        )
        for state_index, (message_position, state) in enumerate(state_pairs):
            if state is None:
                continue
            message = messages[message_position]
            if not isinstance(message.get("content"), list):
                # Durable legacy rows may keep visible text separate from the
                # canonical continuation metadata.  Reconstruct the provider
                # block sequence before applying the exact replay policy.
                message["content"] = [
                    *state.as_blocks(),
                    {"type": "text", "text": str(message.get("content", ""))},
                ]
            replay_required = state.policy.required_for_followup and (
                state.policy.replay_scope == "all_active_tool_turns"
                or state_index == last_state_index
            )
            if not replay_required:
                raw_content = message.get("content")
                blocks = (
                    raw_content
                    if isinstance(raw_content, list)
                    else []
                )
                message["content"] = [
                    block
                    for block in blocks
                    if not (
                        isinstance(block, dict)
                        and block.get("type") in _CONTINUATION_BLOCK_TYPES
                    )
                ]
                continue
            blocks = cast(list[dict[str, object]], message["content"])
            existing = [
                block
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") in _CONTINUATION_BLOCK_TYPES
            ]
            if existing != state.as_blocks():
                message["content"] = state.as_blocks() + [
                    block
                    for block in blocks
                    if block.get("type") not in _CONTINUATION_BLOCK_TYPES
                ]
    messages.append({"role": "user", "content": compaction_instruction})
    return SummarizerRequest(
        mode="same_route",
        messages=tuple(messages),
        tool_schemas=tuple(canonical_tools),
        system=system,
        cache_reuse=True,
        compaction_instruction=compaction_instruction,
    )


# 构造 isolated summarizer request，明确不承诺原始 prompt prefix cache reuse
def build_isolated_request(
    *,
    selected_messages: list[dict[str, object]],
    system: str = "You are a bounded conversation summarizer.",
    instruction: str = COMPACTION_INSTRUCTION,
) -> SummarizerRequest:
    clean_messages = copy.deepcopy(selected_messages)
    for message in clean_messages:
        if isinstance(message, dict):
            message.pop("_message_id", None)
            message.pop("_unit_id", None)
            message.pop("_continuation_state", None)
            message.pop("_checkpoint_id", None)
            content = message.get("content")
            if isinstance(content, list):
                # An isolated route does not need provider continuation state;
                # omit private reasoning blocks instead of leaking or paying
                # for them when the original prefix cannot be reused.
                message["content"] = [
                    block
                    for block in content
                    if not (
                        isinstance(block, dict)
                        and block.get("type") in _CONTINUATION_BLOCK_TYPES
                    )
                ]
    history = json.dumps(clean_messages, ensure_ascii=False, sort_keys=True)
    return SummarizerRequest(
        mode="isolated",
        messages=({"role": "user", "content": f"{instruction}\n\n{history}"},),
        tool_schemas=(),
        system=system,
        cache_reuse=False,
    )


# 过滤 summarizer 响应中的 reasoning/tool 结果，只保留 semantic checkpoint payload
def accept_semantic_payload(
    response_text: str,
    structured_payload: dict[str, Any] | None = None,
    *,
    strict: bool = False,
) -> CompactionCheckpointPayload | None:
    if structured_payload is not None:
        if strict and not {
            "progress",
            "current_work",
            "decisions",
            "files_or_code",
            "errors_or_evidence",
            "pending",
            "next_step",
            "critical_context",
        }.issubset(structured_payload):
            return None
        return CompactionCheckpointPayload.from_dict(structured_payload)
    text = response_text.strip()
    sections = _markdown_sections(text)
    if not sections:
        if strict:
            return None
        return CompactionCheckpointPayload(progress=text)
    required = {
        "original goal",
        "completed steps",
        "key constraints & discoveries",
        "current file state",
        "remaining todos",
        "critical data",
    }
    if strict and not required.issubset(sections):
        return None
    completed = sections.get("completed steps", "").strip()
    discoveries = sections.get("key constraints & discoveries", "").strip()
    files = _section_items(sections.get("current file state", ""))
    pending = _section_items(sections.get("remaining todos", ""))
    critical = _section_items(sections.get("critical data", ""))
    return CompactionCheckpointPayload(
        progress=completed,
        decisions=_section_items(discoveries),
        files_or_code=files,
        pending=pending,
        next_step=pending[0] if pending else "",
        critical_context=critical,
    )


# 解析固定 compaction markdown headings，避免 stale projection 把旧行动段落当作事实
def _markdown_sections(text: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"(?m)^##\s+(?:\d+\.\s*)?([^\n]+)\s*$",
            text,
        )
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


# 将 markdown bullet/list 文本归一化为有界语义字段
def _section_items(text: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    items: list[str] = []
    for line in text.splitlines():
        value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if value:
            items.append(value)
    return tuple(items)


# 返回 selected span 的稳定摘要 digest，供 CompactionCandidate/CAS 使用
def selected_span_digest(messages: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 对 summarizer request 的工具 schema 做稳定排序，保持 same-route prefix identity
def _canonical_tool_schemas(tool_schemas: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        copy.deepcopy(schema)
        for schema in sorted(
            tool_schemas,
            key=lambda schema: (
                str(schema.get("name", "")),
                json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        )
    ]
