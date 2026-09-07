from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from kama_claude.core.compact.budget import truncate_tool_results
from kama_claude.core.compact.protocol import CompactionCheckpointEnvelope
from kama_claude.core.llm.types import ProviderContinuationState
from kama_claude.core.session.surface import SurfaceSnapshot, SurfaceState
from kama_claude.core.task_contract import (
    CheckpointProjection,
    PendingDirectiveSet,
    TaskContractRecord,
    directive_text_digest,
    project_checkpoint,
    project_stale_checkpoint,
    render_authority_framing,
)

_CONTINUATION_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "redacted_thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_item",
    }
)

REPOSITORY_CHANGE_DISCIPLINE = """## Repository Change Discipline
Prefer editing existing files to creating new ones
Don't add features, refactor, or introduce abstractions beyond what the task requires
Don't design for hypothetical future requirements
A bug fix doesn't need surrounding cleanup"""


@dataclass
class InferenceUnit:
    """闭合 provider inference 的可重放状态单元。"""

    unit_id: str
    assistant_visible_content: list[dict[str, Any]] = field(default_factory=list)
    continuation_state: ProviderContinuationState | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    shadowed: bool = False

    # 返回 provider 需要的原始 continuation blocks
    def provider_continuation_blocks(self) -> list[dict[str, object]]:
        if self.continuation_state is None:
            return []
        return self.continuation_state.as_blocks()

    # 返回默认隐藏 continuation 的用户历史 projection
    def user_history_content(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(block)
            for block in self.assistant_visible_content
            if block.get("type") not in _CONTINUATION_BLOCK_TYPES
        ]


@dataclass # 唯一的、贯穿整个 run 生命周期的有状态对象（状态机、消息容器和系统提示词构建器）
class ExecutionContext:
    run_id: str # 本次运行的唯一标识
    goal: str # 用户的初始目标文本
    max_steps: int # 最大执行步数
    prefill_messages: list[dict[str, Any]] = field(default_factory=list) # session 回放的完整历史
    session_notes: str = "" # 持久化笔记
    global_context: str = "" # ~/.kama/context.md 内容
    project_context: str = "" # .kama/context.md 内容
    repository_instructions: str = "" # workspace 显式仓库规则及来源标识
    messages: list[dict[str, Any]] = field(default_factory=list) # ← 核心：完整的对话历史
    # 只收集本次 run 产生的原始 assistant/tool rows，供终局 durable append 使用
    raw_run_messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    step: int = 0 # 当前步数计数器
    status: str = "running"  # "running" | "success" | "failed" 
    reason: str | None = None # 失败原因（成功时为 None）
    result: str = "" # 最终文本输出
    compacted: bool = False
    task_contract: TaskContractRecord | None = None
    pending_directives: PendingDirectiveSet = field(default_factory=PendingDirectiveSet)
    checkpoint_projection: CheckpointProjection | None = None
    checkpoint_envelope: CompactionCheckpointEnvelope | None = None
    checkpoint_contract_digest: str = ""
    directive_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    inference_units: list[InferenceUnit] = field(default_factory=list)
    shadowed_unit_ids: set[str] = field(default_factory=set)
    provider_name: str = ""
    provider_model: str = ""
    provider_attempt_count: int = 1
    tool_result_limit: int = 8_000
    tool_result_keep: int = 4_000
    surface_state: SurfaceState = field(default_factory=SurfaceState, repr=False)
    # skill 或 subagent 角色可覆盖默认 system prompt
    system_prompt_override: str | None = None #system_prompt_override 存在时，base prompt 被完全跳过
    # 每个上下文层都通过 .strip() 检查是否为空。
    # 空文件不会产生空的 ## Global Context\n 标题，保持最终 prompt 的整洁
    # 初始化消息历史，优先使用 session 完整回放内容
    def __post_init__(self) -> None:
        self.tool_result_limit = max(1, self.tool_result_limit)
        self.tool_result_keep = max(1, min(self.tool_result_keep, self.tool_result_limit))
        if self.prefill_messages: # prefill_message非空
            # 防御性拷贝，避免外部修改 prefill_messages 导致跨 run 数据污染
            self.messages = [copy.deepcopy(m) for m in self.prefill_messages]
        # 如果 prefill_messages 为空且 messages 不为空，则将 goal 作为用户消息追加
        elif not self.messages:
            self.messages.append({"role": "user", "content": self.goal})
        self._rebuild_inference_units()
        self.refresh_surface_state(increment=False)

    # 计算当前消息投影的稳定 head identity，避免把内部元数据暴露给 provider
    def _surface_head_id(self) -> str | None:
        if not self.messages:
            return None
        canonical_messages: list[dict[str, Any]] = []
        for message in self.messages:
            canonical = copy.deepcopy(message)
            raw_continuation = canonical.get("_continuation_state")
            if (
                canonical.get("role") == "assistant"
                and isinstance(raw_continuation, dict)
            ):
                # Continuation may be stored separately from visible content;
                # include the provider blocks in the surface identity so a
                # reasoning-only change cannot evade the CAS snapshot.
                state = ProviderContinuationState.from_dict(raw_continuation)
                content = canonical.get("content")
                if isinstance(content, list):
                    canonical["content"] = state.as_blocks() + [
                        block
                        for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") in _CONTINUATION_BLOCK_TYPES
                        )
                    ]
                else:
                    canonical["content"] = [
                        *state.as_blocks(),
                        {"type": "text", "text": str(content or "")},
                    ]
            for key in (
                "_message_id",
                "_unit_id",
                "_checkpoint_id",
                "_continuation_state",
            ):
                canonical.pop(key, None)
            canonical_messages.append(canonical)
        encoded = json.dumps(
            canonical_messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # 将 context 的 contract、pending、checkpoint 和消息 head 同步到 CAS surface state
    def refresh_surface_state(self, *, increment: bool = True) -> None:
        self._refresh_checkpoint_projection()
        head_id = self._surface_head_id()
        contract_version = self.task_contract.version if self.task_contract is not None else 0
        contract_digest = self.task_contract.digest if self.task_contract is not None else ""
        pending_watermark = self.pending_directives.reconciliation_watermark
        pending_digest = self.pending_directives.digest()
        checkpoint_id = (
            self.checkpoint_envelope.transaction_id
            if self.checkpoint_envelope is not None
            else None
        )
        checkpoint_status = (
            self.checkpoint_projection.status
            if self.checkpoint_projection is not None
            else "NONE"
        )
        changed = (
            self.surface_state.active_head_id != head_id
            or self.surface_state.contract_version != contract_version
            or self.surface_state.contract_digest != contract_digest
            or self.surface_state.pending_reconciliation_watermark != pending_watermark
            or self.surface_state.pending_digest != pending_digest
            or self.surface_state.active_checkpoint_id != checkpoint_id
            or self.surface_state.active_checkpoint_status != checkpoint_status
        )
        if increment and changed:
            self.surface_state.note_surface_mutation(
                head_id,
                contract_version=contract_version,
                contract_digest=contract_digest,
                pending_watermark=pending_watermark,
                pending_digest=pending_digest,
                checkpoint_id=checkpoint_id,
                checkpoint_status=checkpoint_status,
            )
            # Keep explicit clears (no checkpoint/empty pending state) visible to CAS.
        self.surface_state.active_head_id = head_id
        self.surface_state.contract_version = contract_version
        self.surface_state.contract_digest = contract_digest
        self.surface_state.pending_reconciliation_watermark = pending_watermark
        self.surface_state.pending_digest = pending_digest
        self.surface_state.active_checkpoint_id = checkpoint_id
        self.surface_state.active_checkpoint_status = checkpoint_status

    # 在 contract 或 pending directive 变化后自动降级 checkpoint，避免旧行动字段继续生效
    def _refresh_checkpoint_projection(self) -> None:
        projection = self.checkpoint_projection
        if projection is None:
            if self.checkpoint_envelope is None:
                return
            current_digest = (
                self.task_contract.digest if self.task_contract is not None else "legacy"
            )
            self.checkpoint_projection = project_checkpoint(
                self.checkpoint_envelope.payload.to_dict(),
                checkpoint_contract_digest=self.checkpoint_envelope.contract_digest,
                current_contract_digest=current_digest,
            )
            projection = self.checkpoint_projection
        checkpoint_digest = self.checkpoint_contract_digest
        if self.checkpoint_envelope is not None:
            checkpoint_digest = self.checkpoint_envelope.contract_digest
        current_digest = self.task_contract.digest if self.task_contract is not None else "legacy"
        stale = bool(self.pending_directives.unresolved()) or (
            (checkpoint_digest or "legacy") != (current_digest or "legacy")
        )
        if stale and projection.status != "HISTORICAL_BACKGROUND":
            facts = projection.facts
            self.checkpoint_projection = project_stale_checkpoint(
                facts,
                reason=(
                    "unresolved user directive"
                    if self.pending_directives.unresolved()
                    else "contract digest changed"
                ),
            )
        elif (
            not stale
            and self.checkpoint_envelope is not None
            and projection.status == "HISTORICAL_BACKGROUND"
        ):
            self.checkpoint_projection = project_checkpoint(
                self.checkpoint_envelope.payload.to_dict(),
                checkpoint_contract_digest=self.checkpoint_envelope.contract_digest,
                current_contract_digest=current_digest,
            )

    # 捕获 compaction/CAS 使用的 semantic surface snapshot
    def surface_snapshot(self) -> SurfaceSnapshot:
        self.refresh_surface_state()
        return self.surface_state.snapshot()

    # 从 prefill history 重建 assistant inference units，支持 daemon restart 后 continuation replay
    def _rebuild_inference_units(self) -> None:
        self.inference_units.clear()
        message_units: list[tuple[int, InferenceUnit]] = []
        for message_index, message in enumerate(self.messages):
            if message.get("role") != "assistant":
                continue
            # Keep internal replay metadata on the canonical message object.
            # Provider/UI projections strip it at their boundaries, while
            # compaction persistence must retain exact continuation policy and
            # route identity for active DeepSeek units.
            raw_unit_id = message.get("_unit_id")
            raw_continuation = message.get("_continuation_state")
            if isinstance(raw_continuation, dict):
                state = ProviderContinuationState.from_dict(raw_continuation)
                if isinstance(message.get("content"), list):
                    existing_blocks = [
                        block
                        for block in message["content"]
                        if isinstance(block, dict)
                    ]
                    message["content"] = state.as_blocks() + [
                        block
                        for block in existing_blocks
                        if block.get("type") not in _CONTINUATION_BLOCK_TYPES
                    ]
                else:
                    text = str(message.get("content", ""))
                    message["content"] = [
                        *state.as_blocks(),
                        {"type": "text", "text": text},
                    ]
            if not isinstance(message.get("content"), list):
                continue
            blocks = [block for block in message["content"] if isinstance(block, dict)]
            thinking = [
                block
                for block in blocks
                if block.get("type") in _CONTINUATION_BLOCK_TYPES
            ]
            continuation = None
            if isinstance(raw_continuation, dict):
                continuation = ProviderContinuationState.from_dict(raw_continuation)
            elif thinking:
                continuation = ProviderContinuationState.from_thinking_blocks(thinking)
            if continuation is not None and not isinstance(raw_continuation, dict):
                message["_continuation_state"] = continuation.to_dict()
            if raw_unit_id in (None, ""):
                message["_unit_id"] = (
                    "prefill-"
                    + hashlib.sha256(
                        json.dumps(
                            {
                                "index": message_index,
                                "content": message.get("content", []),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                )
            visible = [
                block
                for block in blocks
                if block.get("type") not in _CONTINUATION_BLOCK_TYPES | {"tool_use"}
            ]
            calls = [block for block in blocks if block.get("type") == "tool_use"]
            unit = InferenceUnit(
                unit_id=(
                    str(message.get("_unit_id", raw_unit_id))
                ),
                assistant_visible_content=copy.deepcopy(visible),
                continuation_state=continuation,
                tool_calls=copy.deepcopy(calls),
            )
            self.inference_units.append(unit)
            message_units.append((message_index, unit))

        # 依据消息顺序把 durable tool_result blocks 归回对应 closed inference unit
        current_unit: InferenceUnit | None = None
        unit_by_index = {index: unit for index, unit in message_units}
        for message_index, message in enumerate(self.messages):
            if message_index in unit_by_index:
                current_unit = unit_by_index[message_index]
                continue
            if current_unit is None or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            results = [
                copy.deepcopy(block)
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            current_unit.tool_results.extend(results)

    # 返回当前 run 的可信 policy 与上下文组合；override 只能替换 role/base 槽位
    def system_prompt(self, base: str) -> str:
        self._refresh_checkpoint_projection()
        # base 硬编码在 AgentLoop.run() 中
        parts = [self.system_prompt_override if self.system_prompt_override else base]
        parts.append("\n\n" + REPOSITORY_CHANGE_DISCIPLINE)
        if self.repository_instructions.strip():
            parts.append(
                "\n\n## Repository Instructions\n" + self.repository_instructions
            )
        if self.global_context.strip(): #~/.kama/context.md,跨项目记录用户偏好和全局规则
            parts.append("\n\n## Global Context\n" + self.global_context.strip())
        # .kama/context.md 作用于当前项目，包含项目目标、约束和已知事实
        if self.project_context.strip():
            parts.append("\n\n## Project Context\n" + self.project_context.strip())
        if self.session_notes.strip(): #SessionStore 持久化，跨多轮对话持续记忆的事实
            # 运行时元指令提示 LLM 可以使用持久化记忆工具
            parts.append(
                "\n\n## Session Notes\n"
                + self.session_notes.strip()
                + "\n\nRemember important durable facts by calling note_save."
            )
        if (
            self.task_contract is not None
            or self.pending_directives.unresolved()
            or self.checkpoint_projection
        ):
            parts.append(
                "\n\n"
                + render_authority_framing(
                    self.task_contract,
                    self.pending_directives,
                    self.checkpoint_projection,
                )
            )
        return "".join(parts)#因为每个部分都以 \n\n 开头，用空字符串 join 比 "\n\n".join() 更精确

    # 返回 Layer A 的 immutable harness instructions，不包含 session dynamic state
    def immutable_system_prompt(self, base: str) -> str:
        base_text = self.system_prompt_override if self.system_prompt_override else base
        return base_text + "\n\n" + REPOSITORY_CHANGE_DISCIPLINE

    # 返回 Layer B 的 workspace/policy context，来源变化时由 gateway 更新 PrefixEpoch
    def semi_stable_context(self) -> str:
        parts: list[str] = []
        if self.repository_instructions.strip():
            parts.append("## Repository Instructions\n" + self.repository_instructions.strip())
        if self.global_context.strip():
            parts.append("## Global Context\n" + self.global_context.strip())
        if self.project_context.strip():
            parts.append("## Project Context\n" + self.project_context.strip())
        if self.session_notes.strip():
            parts.append("## Session Notes\n" + self.session_notes.strip())
        return "\n\n".join(parts)

    # 将一条本 run 原始消息复制到终局持久化缓冲区，不受 active surface 替换影响
    def _record_raw_run_message(
        self,
        *,
        role: str,
        content: Any,
        unit_id: str | None = None,
        continuation_state: ProviderContinuationState | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "message_id": f"{self.run_id}-raw-{len(self.raw_run_messages):04d}",
            "role": role,
            "content": copy.deepcopy(content),
        }
        if unit_id is not None:
            message["_unit_id"] = unit_id
        if continuation_state is not None:
            message["_continuation_state"] = continuation_state.to_dict()
        self.raw_run_messages.append(message)
        return message

    # 将 LLM 响应的 content blocks 追加为 assistant 消息
    def add_assistant_message(
        self,
        content: list[Any],
        continuation_state: ProviderContinuationState | None = None,
        unit_id: str | None = None,
    ) -> None:
        blocks = copy.deepcopy(content)
        resolved_unit_id = unit_id or (
            "unit-"
            + hashlib.sha256(
                json.dumps(
                    {
                        "index": len(self.messages),
                        "content": blocks,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": blocks,
        }
        # Preserve the legacy message shape for ordinary visible-only turns;
        # durable continuation turns carry their stable unit identity explicitly.
        if unit_id is not None:
            message["_unit_id"] = resolved_unit_id
        self.messages.append(message)
        normalized_blocks = [block for block in blocks if isinstance(block, dict)]
        visible = [
            block
            for block in normalized_blocks
            if block.get("type") not in _CONTINUATION_BLOCK_TYPES | {"tool_use"}
        ]
        tool_calls = [block for block in normalized_blocks if block.get("type") == "tool_use"]
        if continuation_state is None:
            thinking = [
                block
                for block in normalized_blocks
                if block.get("type") in _CONTINUATION_BLOCK_TYPES
            ]
            if thinking:
                continuation_state = ProviderContinuationState.from_thinking_blocks(thinking)
        if continuation_state is not None:
            message["_unit_id"] = resolved_unit_id
            message["_continuation_state"] = continuation_state.to_dict()
        self._record_raw_run_message(
            role="assistant",
            content=blocks,
            unit_id=resolved_unit_id,
            continuation_state=continuation_state,
        )
        self.inference_units.append(
            InferenceUnit(
                unit_id=resolved_unit_id,
                assistant_visible_content=visible,
                continuation_state=continuation_state,
                tool_calls=copy.deepcopy(tool_calls),
            )
        )
        self.refresh_surface_state()

    # 将工具调用结果追加为 user 消息并保留 bounded evidence metadata
    def add_tool_result(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        *,
        evidence_ref: str | None = None,
        raw_truncated: bool = False,
        original_size: int | None = None,
        captured_size: int | None = None,
    ) -> None:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            # ToolResult.content 写回 ExecutionContext；错误会带 is_error=True
            block["is_error"] = True
        if evidence_ref:
            block["evidence_ref"] = evidence_ref
        if raw_truncated:
            block["raw_truncated"] = True
        if original_size is not None:
            block["original_size"] = original_size
        if captured_size is not None:
            block["captured_size"] = captured_size

        last = self.messages[-1] if self.messages else None
        # 五个条件全部满足时追加到上一条 user 消息，否则创建新消息
        if (
            last is not None #	messages 列表为空
            and last["role"] == "user" #上一条是 assistant（需要在新的 user 消息中追加）
            and isinstance(last["content"], list) #user content 是纯文本字符串（如初始 "请帮我..."）
            and last["content"] #content 列表为空（防御性检查）
            # content 中有 text 类型的 block（混合内容）
            and all(b.get("type") == "tool_result" for b in last["content"])
        ):
            last["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})
        matching_unit = next(
            (
                unit
                for unit in reversed(self.inference_units)
                if any(str(call.get("id", "")) == tool_use_id for call in unit.tool_calls)
            ),
            self.inference_units[-1] if self.inference_units else None,
        )
        if matching_unit is not None:
            matching_unit.tool_results.append(copy.deepcopy(block))
        matching_unit_id = matching_unit.unit_id if matching_unit is not None else None
        last_raw = self.raw_run_messages[-1] if self.raw_run_messages else None
        if (
            last_raw is not None
            and last_raw.get("role") == "user"
            and isinstance(last_raw.get("content"), list)
            and all(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in last_raw["content"]
            )
            and last_raw.get("_unit_id") == matching_unit_id
        ):
            last_raw["content"].append(copy.deepcopy(block))
        else:
            self._record_raw_run_message(
                role="user",
                content=[block],
                unit_id=matching_unit_id,
            )
        self.refresh_surface_state()

    # 将指定 inference unit 标记为已被 checkpoint shadow，停止 provider request replay
    def shadow_inference_units(self, unit_ids: set[str]) -> None:
        self.shadowed_unit_ids.update(unit_ids)
        for unit in self.inference_units:
            if unit.unit_id in unit_ids:
                unit.shadowed = True

    # 返回仍需进入 provider request 的 inference units
    def active_inference_units(self) -> list[InferenceUnit]:
        return [unit for unit in self.inference_units if not unit.shadowed]

    # 返回未被 durable semantic coverage 覆盖的用户消息，供 compaction 保守保护
    def uncovered_directive_message_ids(self) -> frozenset[str]:
        protected: set[str] = set()
        for message in self.messages:
            if message.get("role") != "user":
                continue
            message_id = str(message.get("_message_id", ""))
            if not message_id:
                continue
            content = message.get("content")
            if isinstance(content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                continue
            record = self.directive_coverage.get(message_id)
            raw_text = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            coverage_matches = (
                isinstance(record, dict)
                and record.get("exact_text_digest") == directive_text_digest(raw_text)
            )
            if (
                record is None
                or record.get("coverage_status") == "unresolved"
                or not coverage_matches
            ):
                protected.add(message_id)
        return frozenset(protected)

    # 返回 provider request 专用消息，补齐 active units 的原始 continuation blocks
    def provider_messages(self, *, include_internal_metadata: bool = False) -> list[dict[str, Any]]:
        self._refresh_checkpoint_projection()
        rendered: list[dict[str, Any]] = []
        unit_index = 0
        active_units = self.active_inference_units()
        last_active_unit_id = active_units[-1].unit_id if active_units else None
        for message in self.messages:
            copied = copy.deepcopy(message)
            if (
                self.checkpoint_projection is not None
                and self.checkpoint_projection.status == "HISTORICAL_BACKGROUND"
                and copied.get("_checkpoint_id")
            ):
                # stale checkpoint summary/ack 已由 fact-only projection 取代
                # 不能再次作为 user instruction 回放
                if message.get("role") == "assistant" and isinstance(
                    message.get("content"), list
                ):
                    unit_index += 1
                continue
            if not include_internal_metadata:
                copied.pop("_unit_id", None)
                copied.pop("_continuation_state", None)
                copied.pop("_message_id", None)
                copied.pop("_checkpoint_id", None)
            if copied.get("role") == "assistant" and isinstance(copied.get("content"), list):
                unit = (
                    self.inference_units[unit_index]
                    if unit_index < len(self.inference_units)
                    else None
                )
                unit_index += 1
                blocks = copied["content"]
                if include_internal_metadata and unit is not None:
                    copied["_unit_id"] = unit.unit_id
                    if unit.continuation_state is not None:
                        copied["_continuation_state"] = unit.continuation_state.to_dict()
                replay_required = bool(
                    unit is not None
                    and unit.continuation_state is not None
                    and unit.continuation_state.policy.required_for_followup
                    and unit.continuation_state.policy.replay_scope != "never"
                    and (
                        unit.continuation_state.policy.replay_scope == "all_active_tool_turns"
                        or unit.unit_id == last_active_unit_id
                    )
                )
                if unit is not None and (unit.shadowed or not replay_required):
                    blocks[:] = [
                        block
                        for block in blocks
                        if block.get("type") not in _CONTINUATION_BLOCK_TYPES
                    ]
                elif unit is not None and unit.continuation_state is not None:
                    original = unit.provider_continuation_blocks()
                    existing = [
                        block
                        for block in blocks
                        if block.get("type") in _CONTINUATION_BLOCK_TYPES
                    ]
                    if existing != original:
                        blocks[:] = original + [
                            block
                            for block in blocks
                            if block.get("type") not in _CONTINUATION_BLOCK_TYPES
                        ]
            rendered.append(copied)
        # 统一在 provider ingress 施加 model-visible tool receipt 上限；durable
        # session/tool artifacts 仍由各 producer 的绝对 cap 负责保存。
        return truncate_tool_results(
            rendered,
            limit=self.tool_result_limit,
            keep=self.tool_result_keep,
        )

    # 生成隐藏 reasoning 的 UI history projection，不改变 provider messages
    def user_history_messages(self) -> list[dict[str, Any]]:
        self._refresh_checkpoint_projection()
        projected: list[dict[str, Any]] = []
        for message in self.messages:
            content = message.get("content")
            if message.get("role") == "assistant" and isinstance(content, list):
                content = [
                    copy.deepcopy(block)
                    for block in content
                    if block.get("type") not in _CONTINUATION_BLOCK_TYPES
                ]
            projected.append({"role": message.get("role"), "content": content})
        return projected

    # 返回 True 表示 loop 应停止（状态不再是 running）
    def is_done(self) -> bool:
        return self.status != "running"

    # 将 run 标记为成功
    def mark_success(self) -> None:
        self.status = "success"

    # 将 run 标记为失败并记录原因
    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason
'''
Anthropic API 要求同一步的多个 tool_result 必须放在同一条 role: user 消息的 content 数组中：


# ✅ 正确格式
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "tool_001", "content": "..."},
    {"type": "tool_result", "tool_use_id": "tool_002", "content": "..."},
]}

# ❌ 错误格式（会导致 API 拒绝）
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_001", ...}]}
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_002", ...}]}
'''

'''
当前 messages 末尾                     调用 add_tool_result 后
────────────────────────────────────    ──────────────────────────
[..., {role:assistant, content:[...tool_use...]}]
                                     → [..., {role:assistant, ...},
                                          {role:user, content:[tool_result_A]}]  ← 新建 user 消息

[..., {role:user, content:[tool_result_A]}]
                                     → [..., {role:user, content:[tool_result_A,
                                                                   tool_result_B]}]  ← 追加到同一条

[..., {role:user, content:"纯文本消息"}]
                                     → [..., {role:user, content:"纯文本"},
                                          {role:user, content:[tool_result_A]}]  ← 新建 user 消息
'''

'''
AgentRunner.run_and_capture()
  │  创建 ExecutionContext  ──────────────────────┐
  │  读取 context.status / context.result          │
  │                                                │
AgentLoop.run(context)                             │
  │  轮询 context.is_done()         ◄──────────────┤
  │  调用 context.system_prompt()                  │
  │  调用 context.add_assistant_message()          │
  │  调用 context.add_tool_result()                │
  │  设置 context.step += 1                        │
  │  设置 context.result / mark_success / mark_failed
  │                                                │
Compactor.compact(context, provider)               │
  │  读取 context.messages                         │
  │  就地替换 context.messages = [...]  ◄──────────┤
  │                                                │
EventWriter                                        │
  │  读取 context.run_id（事件关联）                │
  └────────────────────────────────────────────────┘
'''
