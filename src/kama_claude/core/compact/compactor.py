from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kama_claude.core.bus.events import ContextCompactedEvent
from kama_claude.core.compact.budget import truncate_tool_results
from kama_claude.core.compact.protocol import (
    ISOLATED_SUMMARY_SYSTEM,
    CompactionCheckpointEnvelope,
    CompactionCheckpointPayload,
    accept_semantic_payload,
    build_isolated_request,
    build_same_route_request,
    render_checkpoint_surface_text,
    selected_span_digest,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.gateway import (
    AdmissionResult,
    ContextAdmissionError,
    ProviderRequestGateway,
    ensure_gateway,
)
from kama_claude.core.llm.usage import estimate_input_tokens
from kama_claude.core.session.surface import CompactionCandidate
from kama_claude.core.task_contract import project_checkpoint

if TYPE_CHECKING:
    from kama_claude.core.context import ExecutionContext
    from kama_claude.core.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class CompactionConflict(RuntimeError):
    """异步摘要基于过期 surface/route/prefix 快照，调用方必须重新 admission。"""

    # 提供稳定冲突错误码，供 loop、事件和调用方统一处理
    error_code = "CONFLICT"

_COMPACT_PROMPT = """\
You are compressing an agent conversation into a handoff summary.
Another LLM instance will continue this task from your summary alone — make it complete.

Structure your response with exactly these six sections:

## 1. Original Goal
One sentence describing what the user asked the agent to accomplish.

## 2. Completed Steps
Bullet list of what has been done. Be specific (file paths, commands run, decisions made).

## 3. Key Constraints & Discoveries
Facts learned during the run that affect future decisions \
(e.g., API limitations, file formats, user preferences stated mid-conversation).

## 4. Current File State
For each file that was created or modified: path, a one-line description of its current state.

## 5. Remaining TODOs
Ordered list of what still needs to be done to complete the original goal.

## 6. Critical Data
Any values the next LLM needs verbatim: IDs, tokens, exact error messages, config values \
discovered during the run.

Be concise. Omit reasoning steps and intermediate attempts. Keep conclusions.\
"""


# 返回当前 UTC 时间的简短时间戳字符串（用于文件名）
def _ts_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CompactionResult:
    summary_text: str
    original_token_estimate: int
    summary_tokens: int
    checkpoint_payload: CompactionCheckpointPayload | None = None
    cache_reuse: bool = False
    selected_span_digest: str = ""
    checkpoint_envelope: CompactionCheckpointEnvelope | None = None
    retained_messages: list[dict[str, Any]] = field(default_factory=list)
    route_identity: str = ""
    route_epoch: str = ""
    prefix_epoch: str = ""
    request_envelope_id: str = ""
    selected_unit_ids: tuple[str, ...] = ()


class Compactor:
    # 初始化压缩器，绑定事件总线、session 目录和 session ID
    def __init__(
        self,
        bus: EventBus,
        session_dir: Path,
        session_id: str,
        *,
        summary_max_tokens: int = 4_096,
        tool_result_limit: int = 8_000,
        tool_result_keep: int = 4_000,
        recent_tail_ratio: float = 0.10,
        recent_tail_max_tokens: int = 64 * 1024,
    ) -> None:
        self._bus = bus
        self._session_dir = session_dir
        self._session_id = session_id
        self._summary_max_tokens = max(1, summary_max_tokens)
        self._tool_result_limit = max(1, tool_result_limit)
        self._tool_result_keep = max(1, min(tool_result_keep, self._tool_result_limit))
        self._recent_tail_ratio = min(max(recent_tail_ratio, 0.0), 0.95)
        self._recent_tail_max_tokens = max(0, recent_tail_max_tokens)

    # 压缩 ExecutionContext.messages，就地替换消息列表并写 summary 文件
    async def compact(
        self,
        context: ExecutionContext,
        provider: LLMProvider,
        focus: str = "",
        *,
        same_route: bool = False,
        system: str | None = None,
        immutable_system: str | None = None,
        semi_stable_context: str = "",
        tool_schemas: list[dict[str, object]] | None = None,
    ) -> CompactionResult | None:
        gateway = ensure_gateway(provider)
        # Snapshot must carry the gateway's current route/prefix epochs even
        # when a previous request already established non-default epochs.
        if same_route:
            # Establish the cacheable A/B identity before capturing the snapshot;
            # a first request must not manufacture a prefix change mid-CAS.
            gateway.stable_prefix_boundary(
                immutable_system=immutable_system or "",
                tool_schemas=tool_schemas or [],
                semi_stable_context=semi_stable_context,
            )
        else:
            # Isolated mode has a distinct fixed system envelope. Establish its
            # prefix epoch before the surface snapshot so the intentional route
            # change is not mistaken for an in-flight CAS race.
            gateway.stable_prefix_boundary(
                immutable_system=ISOLATED_SUMMARY_SYSTEM,
                tool_schemas=[],
                semi_stable_context="",
            )
        context.surface_state.synchronize_epochs(
            route_epoch=gateway.route_epoch,
            prefix_epoch=gateway.prefix_epoch,
        )
        snapshot = context.surface_snapshot()
        original_messages = context.provider_messages(include_internal_metadata=True)
        # Align continuation state to the assistant messages that actually
        # remain on the provider surface.  ``context.inference_units`` may
        # still contain a shadowed checkpoint acknowledgement; passing that
        # positional state list directly would shift a later DeepSeek
        # reasoning block onto the wrong assistant turn.
        state_by_unit_id = {
            unit.unit_id: (None if unit.shadowed else unit.continuation_state)
            for unit in context.inference_units
        }
        active_continuation_states = [
            state_by_unit_id.get(str(message.get("_unit_id")))
            for message in original_messages
            if message.get("role") == "assistant"
        ]
        result = await self.compact_messages(
            original_messages,
            gateway,
            focus=focus,
            same_route=same_route,
            source_system=system,
            source_immutable_system=immutable_system,
            source_semi_stable_context=semi_stable_context,
            source_tool_schemas=tool_schemas,
            continuation_states=active_continuation_states,
            surface_revision=snapshot.surface_revision,
            protected_message_ids=(
                context.pending_directives.protected_message_ids()
                | context.uncovered_directive_message_ids()
            ),
            task_contract=context.task_contract,
        )
        if result is None:
            return None

        selected_count = len(original_messages) - len(result.retained_messages)
        selected_ids = tuple(
            str(message["_unit_id"])
            for message in original_messages[:selected_count]
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), list)
            and message.get("_unit_id")
        )
        candidate = CompactionCandidate.from_snapshot(
            snapshot,
            selected_unit_ids=selected_ids,
            selected_span_digest=result.selected_span_digest,
            route_epoch=result.route_epoch,
            prefix_epoch=result.prefix_epoch,
            request_envelope_id=result.request_envelope_id,
        )
        # Build and validate the runtime-owned envelope before mutating the
        # active surface.  A malformed payload must never shadow raw units
        # without a checkpoint that can replace them.
        previous_checkpoint = context.checkpoint_envelope
        checkpoint = self.build_checkpoint_envelope(
            result,
            contract_digest=(
                context.task_contract.digest if context.task_contract else "legacy"
            ),
            generation=(previous_checkpoint.generation + 1 if previous_checkpoint else 1),
            base_checkpoint_id=(
                previous_checkpoint.transaction_id if previous_checkpoint else None
            ),
            selected_unit_ids=selected_ids,
        )
        async with context.surface_state.mutation_lane:
            context.refresh_surface_state()
            current_messages = truncate_tool_results(
                context.provider_messages(include_internal_metadata=True),
                limit=context.tool_result_limit,
                keep=context.tool_result_keep,
            )
            current_digest = selected_span_digest(current_messages[:selected_count])
            current_summary_request = _build_summary_request(
                current_messages[:selected_count],
                same_route=same_route,
                source_system=system or "",
                source_immutable_system=immutable_system or "",
                source_semi_stable_context=semi_stable_context,
                source_tool_schemas=tool_schemas or [],
                continuation_states=[
                    state_by_unit_id.get(str(message.get("_unit_id")))
                    for message in current_messages
                    if message.get("role") == "assistant"
                ],
                instruction=(
                    _COMPACT_PROMPT
                    + (
                        f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"
                        if focus.strip()
                        else ""
                    )
                ),
            )
            current_envelope = gateway.prepare_request(
                messages=[dict(message) for message in current_summary_request.messages],
                tool_schemas=[dict(schema) for schema in current_summary_request.tool_schemas],
                system=current_summary_request.system,
                immutable_system=(immutable_system or None) if same_route else None,
                semi_stable_context=semi_stable_context if same_route else "",
                request={"max_tokens": self._summary_max_tokens},
                output_reserve=self._summary_max_tokens,
            )
            # Do not synchronize the context to epochs observed after the
            # asynchronous call: a route/prefix change must be a CAS conflict,
            # not an implicit acceptance of a stale summary.
            current_route_epoch = gateway.route_epoch
            current_prefix_epoch = gateway.prefix_epoch
            if (
                not context.surface_state.candidate_matches(candidate)
                or current_digest != candidate.selected_span_digest
                or current_route_epoch != candidate.route_epoch
                or current_prefix_epoch != candidate.prefix_epoch
                or current_envelope.request_envelope_id != candidate.request_envelope_id
            ):
                raise CompactionConflict("compaction candidate no longer matches surface")
            if not context.surface_state.commit_candidate(candidate):
                raise CompactionConflict("compaction candidate CAS failed")

            context.shadow_inference_units(set(selected_ids))
            context.compacted = True
            result = replace(
                result,
                selected_span_digest=checkpoint.selected_span_digest,
                checkpoint_envelope=checkpoint,
            )
            context.checkpoint_envelope = checkpoint
            context.checkpoint_contract_digest = checkpoint.contract_digest
            context.checkpoint_projection = project_checkpoint(
                checkpoint.payload.to_dict(),
                checkpoint_contract_digest=checkpoint.contract_digest,
                current_contract_digest=(
                    context.task_contract.digest if context.task_contract else "legacy"
                ),
            )
            context.messages = [
                {
                    "role": "user",
                    "content": render_checkpoint_surface_text(result.summary_text),
                    "_checkpoint_id": checkpoint.transaction_id,
                },
                {
                    "role": "assistant",
                    "content": "Understood, I'll continue from this summary.",
                    "_checkpoint_id": checkpoint.transaction_id,
                },
                *copy.deepcopy(result.retained_messages),
            ]
            # Rebuild the unit index from the retained surface.  Visible-only
            # legacy assistant rows do not carry an ID in their public shape,
            # so filtering the old positional list would misalign a retained
            # continuation unit after compaction.
            context.shadowed_unit_ids.update(selected_ids)
            context._rebuild_inference_units()
            for unit in context.inference_units:
                unit.shadowed = unit.unit_id in context.shadowed_unit_ids
            context.refresh_surface_state(increment=False)
        self._write_summary(result.summary_text)
        await self._bus.publish(
            ContextCompactedEvent(
                session_id=self._session_id,
                run_id=context.run_id,
                original_tokens=result.original_token_estimate,
                summary_tokens=result.summary_tokens,
                ts=_now(),
            )
        )
        logger.info(
            "context compacted session=%s run=%s original≈%d summary=%d tokens",
            self._session_id, context.run_id,
            result.original_token_estimate, result.summary_tokens,
        )
        return result

    # 从 compaction result 组装 runtime-owned checkpoint envelope，不让 LLM 伪造 metadata
    def build_checkpoint_envelope(
        self,
        result: CompactionResult,
        *,
        contract_digest: str,
        generation: int = 1,
        base_checkpoint_id: str | None = None,
        selected_unit_ids: tuple[str, ...] | None = None,
    ) -> CompactionCheckpointEnvelope:
        selected = (
            result.selected_unit_ids
            if selected_unit_ids is None
            else tuple(selected_unit_ids)
        )
        return CompactionCheckpointEnvelope(
            generation=generation,
            base_checkpoint_id=base_checkpoint_id,
            contract_digest=contract_digest or "legacy",
            selected_unit_ids=selected,
            selected_span_digest=result.selected_span_digest,
            route_identity=result.route_identity,
            route_epoch=result.route_epoch,
            prefix_epoch=result.prefix_epoch,
            summary_route="same_route" if result.cache_reuse else "isolated",
            token_accounting={
                "original_input_estimate": result.original_token_estimate,
                "summary_output_tokens": result.summary_tokens,
            },
            payload=(
                result.checkpoint_payload
                or CompactionCheckpointPayload(progress=result.summary_text)
            ),
        )

    # 纯函数式压缩：接收消息列表，返回 CompactionResult；失败时返回 None
    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        focus: str = "",
        *,
        same_route: bool = False,
        source_system: str | None = None,
        source_immutable_system: str | None = None,
        source_semi_stable_context: str = "",
        source_tool_schemas: list[dict[str, object]] | None = None,
        continuation_states: list[Any] | None = None,
        protected_message_ids: frozenset[str] = frozenset(),
        max_output_tokens: int | None = None,
        surface_revision: int = 0,
        task_contract: Any | None = None,
    ) -> CompactionResult | None:
        # deterministic prune happens before selecting/sending the summary range
        selected_messages = truncate_tool_results(
            messages,
            limit=self._tool_result_limit,
            keep=self._tool_result_keep,
        )
        gateway = ensure_gateway(provider)
        summary_budget = max_output_tokens or self._summary_max_tokens

        prompt = _COMPACT_PROMPT
        if focus.strip():
            prompt += f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"

        selection = _select_admissible_prefix(
            selected_messages,
            gateway=gateway,
            same_route=same_route,
            source_system=source_system or "",
            source_immutable_system=source_immutable_system or "",
            source_semi_stable_context=source_semi_stable_context,
            source_tool_schemas=source_tool_schemas or [],
            continuation_states=continuation_states,
            protected_message_ids=protected_message_ids,
            instruction=prompt,
            max_output_tokens=summary_budget,
            surface_revision=surface_revision,
            recent_tail_ratio=self._recent_tail_ratio,
            recent_tail_max_tokens=self._recent_tail_max_tokens,
        )
        if selection is None:
            logger.warning("compactor: no admissible useful compactable range")
            return None
        compactable_messages, retained_messages, admission = selection
        summary_request = _build_summary_request(
            compactable_messages,
            same_route=same_route,
            source_system=source_system or "",
            source_immutable_system=source_immutable_system or "",
            source_semi_stable_context=source_semi_stable_context,
            source_tool_schemas=source_tool_schemas or [],
            continuation_states=continuation_states,
            instruction=prompt,
        )
        original_estimate = estimate_input_tokens(
            compactable_messages,
            list(summary_request.tool_schemas),
            summary_request.system,
        )

        try:
            response = await gateway.chat(
                messages=[dict(message) for message in summary_request.messages],
                tool_schemas=[dict(schema) for schema in summary_request.tool_schemas],
                bus=EventBus(),
                run_id="compact",
                step=0,
                system=summary_request.system,
                request={"max_tokens": summary_budget},
                output_reserve=summary_budget,
                maintenance=True,
                surface_revision=surface_revision,
                immutable_system=(source_immutable_system or None) if same_route else None,
                semi_stable_context=source_semi_stable_context if same_route else "",
            )
        except Exception:
            logger.exception("compactor: LLM call failed, skipping compaction")
            return None

        if response.tool_calls:
            logger.warning("compactor: maintenance response returned tool call")
            return None
        summary_text = response.text.strip()
        if not summary_text:
            logger.warning("compactor: LLM returned empty summary, skipping compaction")
            return None

        summary_tokens = (
            response.usage.output_tokens
            if response.usage
            else estimate_input_tokens(
                [{"role": "assistant", "content": summary_text}],
                [],
                "",
            )
        )
        checkpoint_payload = accept_semantic_payload(summary_text, strict=True)
        if checkpoint_payload is None:
            logger.warning("compactor: summary structure is invalid, skipping compaction")
            return None
        if summary_tokens >= original_estimate:
            logger.warning("compactor: summary is not smaller than selected span")
            return None
        if task_contract is not None and _summary_violates_contract(
            checkpoint_payload,
            task_contract,
        ):
            logger.warning("compactor: summary contains action fields forbidden by contract")
            return None

        return CompactionResult(
            summary_text=summary_text,
            original_token_estimate=original_estimate,
            summary_tokens=summary_tokens,
            checkpoint_payload=checkpoint_payload,
            cache_reuse=same_route,
            selected_span_digest=selected_span_digest(compactable_messages),
            retained_messages=copy.deepcopy(retained_messages),
            route_identity=gateway.route_identity,
            route_epoch=admission.envelope.route_epoch,
            prefix_epoch=admission.envelope.prefix_epoch,
            request_envelope_id=admission.envelope.request_envelope_id,
            selected_unit_ids=tuple(
                str(message["_unit_id"])
                for message in compactable_messages
                if message.get("role") == "assistant" and message.get("_unit_id")
            ),
        )

    # 将摘要文本写入 session 目录的 summary_<ts>.md
    def _write_summary(self, text: str) -> None:
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_dir / f"summary_{_ts_compact()}.md"
            path.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("compactor: failed to write summary file")


# 判断 checkpoint 的行动字段是否重新激活当前 contract 已禁止的动作
def _summary_violates_contract(
    payload: CompactionCheckpointPayload,
    task_contract: Any,
) -> bool:
    prohibitions = getattr(task_contract, "prohibitions", ())
    if not isinstance(prohibitions, tuple | list):
        return False
    action_text = " ".join(
        [
            payload.current_work,
            payload.next_step,
            *payload.pending,
        ]
    ).lower()
    if not action_text.strip():
        return False
    for prohibition in prohibitions:
        raw = str(prohibition).lower().strip()
        remainder = re.sub(
            r"\b(?:do\s+not|don't|must\s+not|never|no)\b",
            " ",
            raw,
        )
        tokens = re.findall(r"[a-z0-9_./-]+", remainder)
        if tokens and all(token in action_text for token in tokens):
            if not re.search(
                r"\b(?:do\s+not|don't|must\s+not|never|no)\b",
                action_text,
            ):
                return True
    return False


# 构造同路由或隔离路由的 maintenance summary request
def _build_summary_request(
    messages: list[dict[str, Any]],
    *,
    same_route: bool,
    source_system: str,
    source_immutable_system: str,
    source_semi_stable_context: str,
    source_tool_schemas: list[dict[str, object]],
    continuation_states: list[Any] | None,
    instruction: str,
) -> Any:
    if same_route:
        return build_same_route_request(
            prefix_messages=messages,
            tool_schemas=source_tool_schemas,
            system=source_system,
            continuation_states=continuation_states,
            compaction_instruction=instruction,
        )
    return build_isolated_request(
        selected_messages=messages,
        system=ISOLATED_SUMMARY_SYSTEM,
        instruction=instruction,
    )


# 检查候选 summary prefix 是否保持 tool_use/tool_result 配对
def _is_balanced_prefix(messages: list[dict[str, Any]]) -> bool:
    pending: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if message.get("role") == "assistant":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    pending.add(str(block.get("id", "")))
        elif message.get("role") == "user":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    pending.discard(str(block.get("tool_use_id", "")))
    return not pending


# 在 maintenance request admission 前选择最大且自身可容纳的 balanced compactable prefix
def _select_admissible_prefix(
    messages: list[dict[str, Any]],
    *,
    gateway: ProviderRequestGateway,
    same_route: bool,
    source_system: str,
    source_immutable_system: str,
    source_semi_stable_context: str,
    source_tool_schemas: list[dict[str, object]],
    continuation_states: list[Any] | None,
    protected_message_ids: frozenset[str],
    instruction: str,
    max_output_tokens: int,
    surface_revision: int = 0,
    recent_tail_ratio: float = 0.10,
    recent_tail_max_tokens: int = 64 * 1024,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], AdmissionResult] | None:
    if not messages:
        return None

    total_estimate = estimate_input_tokens(messages, source_tool_schemas, source_system)
    requested_tail = min(
        max(0, recent_tail_max_tokens),
        max(0, int(gateway.context_window * max(0.0, recent_tail_ratio))),
    )
    # Tail is a soft hysteresis preference.  When the whole input is smaller
    # than the requested floor, leave at least one token/message available.
    tail_floor = (
        min(requested_tail, max(0, total_estimate - 1))
        if total_estimate >= int(gateway.context_window * gateway.target_ratio)
        else 0
    )

    # 试探一个 prefix 终点，确保 tool 配对、directive 保护和 summary request 可 admission
    def try_end(end: int) -> AdmissionResult | None:
        candidate = messages[:end]
        retained = messages[end:]
        if not candidate or not _is_balanced_prefix(candidate):
            return None
        if tail_floor and estimate_input_tokens(retained, [], "") < tail_floor:
            return None
        if any(
            str(message.get("_message_id", "")) in protected_message_ids
            for message in candidate
        ):
            return None
        request = _build_summary_request(
            candidate,
            same_route=same_route,
            source_system=source_system,
            source_immutable_system=source_immutable_system,
            source_semi_stable_context=source_semi_stable_context,
            source_tool_schemas=source_tool_schemas,
            continuation_states=continuation_states,
            instruction=instruction,
        )
        envelope = gateway.prepare_request(
            messages=[dict(message) for message in request.messages],
            tool_schemas=[dict(schema) for schema in request.tool_schemas],
            system=request.system,
            immutable_system=(source_immutable_system or None) if same_route else None,
            semi_stable_context=source_semi_stable_context if same_route else "",
            request={"max_tokens": max_output_tokens},
            output_reserve=max_output_tokens,
            surface_revision=surface_revision,
        )
        try:
            return gateway.admit(envelope)
        except ContextAdmissionError:
            return None

    low, high = 1, len(messages)
    best_end = 0
    best_admission: AdmissionResult | None = None
    for _ in range(max(1, len(messages).bit_length() + 1)):
        if low > high:
            break
        mid = (low + high) // 2
        admission = try_end(mid)
        if admission is None:
            high = mid - 1
        else:
            best_end = mid
            best_admission = admission
            low = mid + 1

    # Tool boundaries are not monotonic; close the binary-search gap with a bounded scan.
    scan_start = min(len(messages), max(best_end, high + 1, 64))
    for end in range(scan_start, max(0, best_end), -1):
        admission = try_end(end)
        if admission is not None:
            best_end = end
            best_admission = admission
            break
    if best_admission is None or best_end <= 0:
        return None
    return messages[:best_end], messages[best_end:], best_admission


# 将消息列表序列化为可供 LLM 阅读的纯文本
def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            blocks: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    blocks.append(block.get("text", ""))
                elif btype == "tool_use":
                    blocks.append(
                        f"<tool_call name={block.get('name')} id={block.get('id')}>\n"
                        f"{block.get('input', {})}\n</tool_call>"
                    )
                elif btype == "tool_result":
                    blocks.append(
                        f"<tool_result id={block.get('tool_use_id')}>\n"
                        f"{block.get('content', '')}\n</tool_result>"
                    )
            parts.append(f"[{role}]\n" + "\n".join(blocks))
    return "\n\n".join(parts)
