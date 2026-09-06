from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kama_claude.core.execution import (
    TERMINAL_EXECUTION_STATUSES,
    ApprovedExecutionBinding,
    ExecutionStatus,
    ExecutionStatusProjection,
)
from kama_claude.core.session.model import Session
from kama_claude.core.task_contract import directive_text_digest

logger = logging.getLogger(__name__)

MessageContent = str | list[dict[str, Any]]
_CONTINUATION_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "redacted_thinking",
        "reasoning",
        "reasoning_content",
        "reasoning_item",
    }
)


# 使用稳定 JSON 编码计算 planning payload 摘要
def _planning_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 按 verification status 校验 observed image identity 的强弱要求
def _verification_image_identity_matches(
    expected_image_id: str,
    observed_image_id: str | None,
    status: str,
) -> bool:
    if observed_image_id is not None and observed_image_id != expected_image_id:
        return False
    if status in {"verification_passed", "verification_failed"}:
        return observed_image_id == expected_image_id
    return True


class SessionStore:
    # 初始化 session 文件存储根目录
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    # 返回指定 session 的目录路径
    def session_dir(self, sid: str) -> Path:
        return self._root / sid

    # 列出具有 meta.json 的持久化 session，供 daemon 重启 reconciliation 使用
    def list_session_ids(self) -> list[str]:
        return sorted(
            path.name
            for path in self._root.iterdir()
            if path.is_dir() and (path / "meta.json").is_file()
        )

    # 返回指定 session 下的 runs 目录路径
    def runs_dir(self, sid: str) -> Path:
        return self.session_dir(sid) / "runs"

    # 创建并返回指定 session 目录，供 journal owner 在 meta 前注册
    def ensure_session_dir(self, sid: str) -> Path:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # 将 session meta 写入 meta.json
    def write_meta(self, session: Session) -> None:
        path = self.session_dir(session.id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # 从 meta.json 读取 session meta
    def read_meta(self, sid: str) -> Session:
        data = json.loads((self.session_dir(sid) / "meta.json").read_text(encoding="utf-8"))
        return Session.from_dict(data)

    # 追加一条 Anthropic API 消息到 thread.jsonl
    def append_message(
        self,
        sid: str,
        role: str,
        content: MessageContent,
        run_id: str | None = None,
        projection_metadata: dict[str, Any] | None = None,
        *,
        message_id: str | None = None,
        unit_id: str | None = None,
        continuation_state: dict[str, Any] | None = None,
        record_type: str = "message",
    ) -> str:
        row: dict[str, Any] = {
            "ts": _now(),
            "record_type": record_type,
            "role": role,
            "content": content,
            "message_id": message_id or f"msg-{uuid.uuid4().hex}",
        }
        if run_id is not None:
            row["run_id"] = run_id
        if unit_id is not None:
            row["unit_id"] = unit_id
        if continuation_state is not None:
            row["continuation_state"] = copy.deepcopy(continuation_state)
        if projection_metadata:
            row["projection_metadata"] = dict(projection_metadata)
        self._append_row(sid, row)
        return str(row["message_id"])

    # 将任意 typed append-only record fsync 到 session thread
    def append_record(self, sid: str, record_type: str, payload: dict[str, Any]) -> None:
        row = {"ts": _now(), "record_type": record_type, **dict(payload)}
        self._append_row(sid, row)

    # 将单条 row 追加到 thread.jsonl 并刷新到磁盘
    def _append_row(self, sid: str, row: dict[str, Any]) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "thread.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # 追加 durable provider continuation state，不把 reasoning 当作普通用户文本
    def append_continuation_message(
        self,
        sid: str,
        *,
        role: str,
        content: MessageContent,
        continuation_state: dict[str, Any],
        run_id: str | None = None,
        unit_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self.append_message(
            sid,
            role,
            content,
            run_id=run_id,
            message_id=message_id,
            unit_id=unit_id,
            continuation_state=continuation_state,
            record_type="step_commit",
        )

    # 追加独立 directive coverage record
    def append_directive_coverage(self, sid: str, record: Any) -> None:
        payload = {
            "message_id": record.message_id,
            "semantic_contract_digest": record.semantic_contract_digest,
            "classification": record.classification,
            "exact_text_digest": record.exact_text_digest,
            "coverage_status": record.coverage_status,
            "covered_at": record.covered_at,
        }
        self.append_record(sid, "directive_coverage", payload)

    # 读取当前 session 的全部 directive coverage records
    def read_directive_coverage(self, sid: str) -> list[dict[str, Any]]:
        return [
            {key: row.get(key) for key in (
                "message_id",
                "semantic_contract_digest",
                "classification",
                "exact_text_digest",
                "coverage_status",
                "covered_at",
            )}
            for row in self._read_thread_rows(sid)
            if row.get("record_type") == "directive_coverage"
        ]

    # 按 message_id 折叠 coverage 更新，供 compaction 判断消息是否已安全覆盖
    def read_directive_coverage_state(self, sid: str) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.read_directive_coverage(sid):
            message_id = str(row.get("message_id", ""))
            if message_id:
                latest[message_id] = row
        return latest

    # 返回 active surface 中尚未 durable 覆盖的 task-shaping user message IDs
    def read_uncovered_directive_message_ids(self, sid: str) -> frozenset[str]:
        coverage = self.read_directive_coverage_state(sid)
        result: set[str] = set()
        # Reduce the append-only log to the active projection first.  Looking
        # only at raw message rows would miss user messages retained inside a
        # surface_replace replacement and would incorrectly treat them as
        # covered on the next compaction.
        for message in self.read_messages_with_metadata(sid):
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
            record = coverage.get(message_id)
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
                result.add(message_id)
        return frozenset(result)

    # 追加 ordered pending directive overlay record
    def append_pending_directive(self, sid: str, overlay: Any) -> None:
        self.append_record(
            sid,
            "pending_directive",
            {
                "message_id": overlay.message_id,
                "original_order": overlay.original_order,
                "raw_text": overlay.raw_text,
                "classification": overlay.classification,
                "status": overlay.status,
            },
        )

    # 追加 pending directive 的 covered/superseded 状态更新，不改写旧记录
    def resolve_pending_directive(
        self,
        sid: str,
        message_id: str,
        *,
        status: str = "covered",
        watermark: str = "",
    ) -> None:
        self.append_record(
            sid,
            "pending_directive",
            {
                "message_id": message_id,
                "status": status,
                "reconciliation_watermark": watermark,
            },
        )

    # 读取仍处于 unresolved 状态的 ordered pending directive overlays
    def read_pending_directives(self, sid: str) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        watermark = ""
        for row in self._read_thread_rows(sid):
            if row.get("record_type") != "pending_directive":
                continue
            if row.get("reconciliation_watermark") not in (None, ""):
                watermark = str(row["reconciliation_watermark"])
            message_id = str(row.get("message_id", ""))
            if message_id:
                previous = latest.get(message_id, {"message_id": message_id})
                for key in (
                    "original_order",
                    "raw_text",
                    "classification",
                    "status",
                ):
                    value = row.get(key)
                    if value not in (None, ""):
                        previous[key] = value
                latest[message_id] = previous
        result = sorted(
            (item for item in latest.values() if item.get("status", "unresolved") == "unresolved"),
            key=lambda item: int(item.get("original_order") or 0),
        )
        if watermark:
            for item in result:
                item["reconciliation_watermark"] = watermark
        return result

    # 返回当前 session 已分配过的最大 pending 顺序，解析已覆盖项以保持全局用户顺序单调
    def next_pending_order(self, sid: str) -> int:
        maximum = -1
        for row in self._read_thread_rows(sid):
            if row.get("record_type") != "pending_directive":
                continue
            value = row.get("original_order")
            try:
                if isinstance(value, (int, str)):
                    maximum = max(maximum, int(value))
            except (TypeError, ValueError):
                continue
        return maximum + 1

    # 将 durable pending rows 恢复为 ordered PendingDirectiveSet
    def read_pending_directive_set(self, sid: str) -> Any:
        from kama_claude.core.task_contract import PendingDirectiveSet

        pending = PendingDirectiveSet()
        items = self.read_pending_directives(sid)
        for item in items:
            pending.add(
                str(item.get("message_id", "")),
                str(item.get("raw_text", "")),
                str(item.get("classification", "unknown")),  # type: ignore[arg-type]
                original_order=int(item.get("original_order") or 0),
            )
        watermark = ""
        for row in self._read_thread_rows(sid):
            if row.get("record_type") != "pending_directive":
                continue
            value = row.get("reconciliation_watermark")
            if value not in (None, ""):
                watermark = str(value)
        pending.reconciliation_watermark = watermark
        return pending

    # 追加 checkpoint envelope；metadata 与 semantic payload 分开持久化
    def append_checkpoint(
        self,
        sid: str,
        envelope: dict[str, Any],
        *,
        atomic_surface_replace: bool = False,
    ) -> None:
        from kama_claude.core.compact.protocol import CompactionCheckpointEnvelope

        # Validate runtime-owned metadata before it becomes durable; legacy
        # records are accepted through the envelope compatibility defaults.
        CompactionCheckpointEnvelope.from_record(envelope)
        payload = dict(envelope)
        if atomic_surface_replace:
            if not envelope.get("transaction_id"):
                raise ValueError("atomic checkpoint requires transaction_id")
            # Mark new two-record compactions so a crash after this envelope
            # but before surface_replace cannot make an orphan checkpoint look
            # active.  Legacy callers keep the historical standalone behavior.
            payload["atomic_surface_replace"] = True
        self.append_record(sid, "checkpoint", payload)

    # 追加 durable TaskContractRecord 的 persistence metadata 与语义字段
    def append_task_contract(self, sid: str, record: Any) -> None:
        self.append_record(
            sid,
            "task_contract",
            {
                "version": record.version,
                "digest": record.digest,
                "goal": record.goal,
                "requirements": list(record.requirements),
                "constraints": list(record.constraints),
                "prohibitions": list(record.prohibitions),
                "acceptance_criteria": list(record.acceptance_criteria),
                "authorization": list(record.authorization),
                "source_message_ids": list(record.source_message_ids),
                "created_at": record.created_at,
                "transaction_id": record.transaction_id,
            },
        )

    # 读取按 version 排序的 TaskContract persistence records
    def read_task_contracts(self, sid: str) -> list[dict[str, Any]]:
        records = [
            dict(row)
            for row in self._read_thread_rows(sid)
            if row.get("record_type") == "task_contract"
        ]
        return sorted(records, key=lambda row: int(row.get("version") or 0))

    # 读取当前最高版本 TaskContractRecord，缺失时返回 None
    def read_latest_task_contract(self, sid: str) -> Any:
        records = self.read_task_contracts(sid)
        if not records:
            return None
        from kama_claude.core.task_contract import TaskContractRecord

        return TaskContractRecord.from_dict(records[-1])

    # 读取 durable checkpoint envelopes，保留 deterministic metadata 与 semantic payload
    def read_checkpoints(self, sid: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._read_thread_rows(sid)
            if row.get("record_type") == "checkpoint"
        ]

    # 读取当前原子 surface 已提交的 checkpoint，忽略 crash 后未被 replacement 引用的孤儿 envelope
    def read_committed_checkpoints(self, sid: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._committed_checkpoint_rows(self._read_thread_rows(sid))
        ]

    # 根据当前 contract digest 将最新 checkpoint 恢复为 active 或 factual projection
    def read_checkpoint_projection(
        self,
        sid: str,
        *,
        current_contract_digest: str = "",
        pending_unresolved: bool = False,
    ) -> Any:
        checkpoints = self._committed_checkpoint_rows(self._read_thread_rows(sid))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        payload = self._checkpoint_payload(latest)
        from kama_claude.core.task_contract import project_checkpoint, project_stale_checkpoint

        normalized_current_digest = current_contract_digest or "legacy"
        normalized_checkpoint_digest = str(latest.get("contract_digest", "") or "legacy")
        if pending_unresolved:
            return project_stale_checkpoint(payload, reason="unresolved user directive")

        return project_checkpoint(
            payload,
            checkpoint_contract_digest=normalized_checkpoint_digest,
            current_contract_digest=normalized_current_digest,
        )

    # 读取最新 checkpoint envelope，供 runner 重建 active/stale projection 与 CAS identity
    def read_latest_checkpoint_envelope(self, sid: str) -> Any | None:
        checkpoints = self._committed_checkpoint_rows(self._read_thread_rows(sid))
        if not checkpoints:
            return None
        from kama_claude.core.compact.protocol import CompactionCheckpointEnvelope

        return CompactionCheckpointEnvelope.from_record(checkpoints[-1])

    # 追加 surface replacement transaction，旧 message 只被 shadow 而非删除
    def append_surface_replace(
        self,
        sid: str,
        *,
        shadowed_message_ids: list[str],
        replacement_messages: list[dict[str, Any]],
        replacement_contract_digest: str = "",
        base_surface_revision: int,
        surface_revision: int,
        transaction_id: str,
        checkpoint_required: bool = False,
    ) -> None:
        self.append_record(
            sid,
            "surface_replace",
            {
                "shadowed_message_ids": list(shadowed_message_ids),
                "replacement_messages": replacement_messages,
                "replacement_contract_digest": replacement_contract_digest,
                "base_surface_revision": base_surface_revision,
                "surface_revision": surface_revision,
                "transaction_id": transaction_id,
                "checkpoint_required": checkpoint_required,
            },
        )

    # 以 projection_key+projection_digest 幂等追加已 committed PlanView thread projection
    def append_plan_projection(
        self,
        sid: str,
        content: MessageContent,
        *,
        run_id: str,
        projection_key: str,
        projection_digest: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        for row in self._read_thread_rows(sid):
            existing = row.get("projection_metadata")
            if not isinstance(existing, dict):
                continue
            if existing.get("projection_key") != projection_key:
                continue
            if existing.get("projection_digest") != projection_digest:
                logger.warning(
                    "plan projection integrity conflict sid=%s projection_key=%s",
                    sid,
                    projection_key,
                )
            return False
        projection_metadata = dict(metadata or {})
        projection_metadata.update(
            {
                "projection_kind": "plan",
                "projection_key": projection_key,
                "projection_digest": projection_digest,
            }
        )
        self.append_message(
            sid,
            "assistant",
            content,
            run_id=run_id,
            projection_metadata=projection_metadata,
        )
        return True

    # 批量追加一次 run 新产生的消息到 thread.jsonl
    def append_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        run_id: str,
    ) -> None:
        for msg in messages:
            continuation_state = msg.get("continuation_state", msg.get("_continuation_state"))
            if continuation_state is None:
                content = msg.get("content")
                if str(msg.get("role")) == "assistant" and isinstance(content, list):
                    continuation_blocks = [
                        block
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") in _CONTINUATION_BLOCK_TYPES
                    ]
                    if continuation_blocks:
                        from kama_claude.core.llm.types import (
                            NATIVE_ANTHROPIC_CONTINUATION_POLICY,
                            ProviderContinuationState,
                        )

                        continuation_state = ProviderContinuationState.from_thinking_blocks(
                            continuation_blocks,
                            policy=NATIVE_ANTHROPIC_CONTINUATION_POLICY,
                            protocol="anthropic",
                        ).to_dict()
            self.append_message(
                sid,
                role=str(msg["role"]),
                content=msg["content"],
                run_id=run_id,
                message_id=(str(msg["message_id"]) if msg.get("message_id") else None),
                unit_id=(
                    str(msg.get("unit_id", msg.get("_unit_id")))
                    if msg.get("unit_id", msg.get("_unit_id"))
                    else None
                ),
                continuation_state=(
                    dict(continuation_state)
                    if isinstance(continuation_state, dict)
                    else None
                ),
                record_type="step_commit",
            )

    # 读取完整 thread 并返回可直接传给 Anthropic 的 messages
    def read_messages(
        self,
        sid: str,
        *,
        include_internal_metadata: bool = False,
        tool_result_limit: int | None = 8_000,
        tool_result_keep: int = 4_000,
    ) -> list[dict[str, Any]]:
        rows = self._read_thread_rows(sid)
        contract_rows = [
            row for row in rows if row.get("record_type") == "task_contract"
        ]
        current_contract_digest = (
            str(contract_rows[-1].get("digest", "")) if contract_rows else ""
        )
        pending_unresolved = bool(self.read_pending_directives(sid))
        stale_background: list[dict[str, Any]] = []
        checkpoint_rows = self._committed_checkpoint_rows(rows)
        if checkpoint_rows:
            row = checkpoint_rows[-1]
            checkpoint_digest = str(row.get("contract_digest", ""))
            payload = self._checkpoint_payload(row)
            stale = pending_unresolved or (
                (checkpoint_digest or "legacy") != (current_contract_digest or "legacy")
            )
            if isinstance(payload, dict) and stale:
                from kama_claude.core.task_contract import project_stale_checkpoint

                projection = project_stale_checkpoint(
                    payload,
                    reason=(
                        "unresolved user directive"
                        if pending_unresolved
                        else "contract digest changed"
                    ),
                )
                stale_background.append(
                    {
                        "role": "assistant",
                        "content": (
                            projection.framing
                            + "\n"
                            + json.dumps(projection.facts, ensure_ascii=False, sort_keys=True)
                        ),
                    }
                )
        ordered: list[tuple[str, dict[str, Any]]] = []
        available_checkpoint_ids: set[str] = set()
        for row in rows:
            record_type = row.get("record_type")
            if record_type == "checkpoint":
                transaction_id = row.get("transaction_id")
                if transaction_id:
                    available_checkpoint_ids.add(str(transaction_id))
                continue
            if record_type in {
                "directive_coverage",
                "pending_directive",
                "task_contract",
                "checkpoint",
                "surface_state",
            }:
                continue
            if record_type == "surface_replace":
                replacement_messages = row.get("replacement_messages", [])
                # A checkpoint-marked replacement is a two-record commit.  If
                # the process crashed after the replacement row but before its
                # envelope, leave the raw surface active and let the next run
                # retry compaction instead of replaying a partial replacement.
                if not self._replacement_is_committed(
                    row,
                    available_checkpoint_ids,
                ):
                    continue
                shadowed = {str(item) for item in row.get("shadowed_message_ids", [])}
                first_index = next(
                    (
                        index
                        for index, (message_id, _) in enumerate(ordered)
                        if message_id in shadowed
                    ),
                    len(ordered),
                )
                ordered = [
                    item for item in ordered if item[0] not in shadowed
                ]
                replacements = replacement_messages
                # Any stale background (contract mismatch or unresolved user directive)
                # suppresses the replacement's action-oriented summary. Facts are
                # projected separately above, so old next_step/pending text cannot
                # remain active merely because its digest matches the current contract.
                replacement_is_stale = bool(stale_background)
                if isinstance(replacements, list):
                    replacement_rows: list[tuple[str, dict[str, Any]]] = []
                    for replacement_index, message in enumerate(replacements):
                        if not isinstance(message, dict) or message.get("role") not in {
                            "user",
                            "assistant",
                        }:
                            continue
                        if replacement_is_stale and (
                            message.get("checkpoint_id")
                            or message.get("checkpoint_message")
                            or str(message.get("message_id", "")).startswith(
                                "checkpoint-"
                            )
                        ):
                            continue
                        # Legacy replacement rows may lack message_id.  Derive a
                        # deterministic fallback from their transaction/index so
                        # replay does not change active_head identity on every read.
                        raw_replacement_id = message.get("message_id")
                        replacement_id = (
                            str(raw_replacement_id)
                            if raw_replacement_id
                            else "replacement-"
                            f"{row.get('transaction_id', 'legacy')}-"
                            f"{replacement_index}"
                        )
                        replacement_rows.append(
                            (
                                replacement_id,
                                self._provider_message_from_row(
                                    message,
                                    include_internal_metadata=include_internal_metadata,
                                ),
                            )
                        )
                    ordered[first_index:first_index] = replacement_rows
                continue
            message_id = str(row.get("message_id", f"legacy-{len(ordered)}"))
            ordered.append(
                (
                    message_id,
                    self._provider_message_from_row(
                        row,
                        include_internal_metadata=include_internal_metadata,
                    ),
                )
            )
        # 历史 facts 放在当前 raw surface 之前，避免它看起来像最新用户指令
        messages = stale_background + [message for _, message in ordered]

        messages = self._trim_orphan_tool_use(messages)
        if tool_result_limit is None:
            return messages
        from kama_claude.core.compact.budget import truncate_tool_results
        return truncate_tool_results(
            messages,
            limit=max(1, tool_result_limit),
            keep=max(1, min(tool_result_keep, tool_result_limit)),
        )

    # 为 runner 提供带 continuation/unit metadata 的内部回放消息，公共 history 默认不暴露这些字段
    def read_messages_with_metadata(self, sid: str) -> list[dict[str, Any]]:
        # Runner owns the model ingress budget; preserve durable raw receipts here.
        return self.read_messages(
            sid,
            include_internal_metadata=True,
            tool_result_limit=None,
        )

    # 将 durable message row 投影为 provider 消息，并按需携带内部 continuation metadata
    @staticmethod
    def _provider_message_from_row(
        row: dict[str, Any],
        *,
        include_internal_metadata: bool,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": row["role"],
            "content": row.get("content", ""),
        }
        if include_internal_metadata:
            if row.get("message_id"):
                message["_message_id"] = str(row["message_id"])
            if row.get("unit_id"):
                message["_unit_id"] = str(row["unit_id"])
            continuation_state = row.get("continuation_state")
            if isinstance(continuation_state, dict):
                message["_continuation_state"] = copy.deepcopy(continuation_state)
            if row.get("checkpoint_id"):
                message["_checkpoint_id"] = str(row["checkpoint_id"])
        return message

    # 读取供 TUI 使用的 projection metadata；不作为 provider message 输入
    def read_history_projection(self, sid: str) -> list[dict[str, Any]]:
        rows = self._read_thread_rows(sid)
        contract_rows = [row for row in rows if row.get("record_type") == "task_contract"]
        current_contract_digest = (
            str(contract_rows[-1].get("digest", "")) if contract_rows else ""
        )
        pending_unresolved = bool(self.read_pending_directives(sid))
        stale_checkpoint = False
        historical_projection: list[dict[str, Any]] = []
        if rows:
            checkpoint_rows = self._committed_checkpoint_rows(rows)
            if checkpoint_rows:
                latest_checkpoint = checkpoint_rows[-1]
                stale_checkpoint = pending_unresolved or (
                    (
                        str(latest_checkpoint.get("contract_digest", ""))
                        or "legacy"
                    )
                    != (current_contract_digest or "legacy")
                )
                payload = self._checkpoint_payload(latest_checkpoint)
                if stale_checkpoint and isinstance(payload, dict):
                    from kama_claude.core.task_contract import project_stale_checkpoint

                    projection = project_stale_checkpoint(
                        payload,
                        reason=(
                            "unresolved user directive"
                            if pending_unresolved
                            else "contract digest changed"
                        ),
                    )
                    historical_projection.append(
                        {
                            "role": "assistant",
                            "content": (
                                projection.framing
                                + "\n"
                                + json.dumps(projection.facts, ensure_ascii=False, sort_keys=True)
                            ),
                            "projection_metadata": {"projection_kind": "historical_checkpoint"},
                        }
                    )
        ordered: list[tuple[str, dict[str, Any]]] = []
        available_checkpoint_ids: set[str] = set()
        for row in rows:
            record_type = row.get("record_type")
            if record_type == "checkpoint":
                transaction_id = row.get("transaction_id")
                if transaction_id:
                    available_checkpoint_ids.add(str(transaction_id))
                continue
            if record_type in {
                "directive_coverage",
                "pending_directive",
                "task_contract",
                "checkpoint",
                "surface_state",
            }:
                continue
            if record_type == "surface_replace":
                replacement_messages = row.get("replacement_messages", [])
                if not self._replacement_is_committed(
                    row,
                    available_checkpoint_ids,
                ):
                    continue
                shadowed = {str(item) for item in row.get("shadowed_message_ids", [])}
                first_index = next(
                    (
                        index
                        for index, (message_id, _) in enumerate(ordered)
                        if message_id in shadowed
                    ),
                    len(ordered),
                )
                ordered = [item for item in ordered if item[0] not in shadowed]
                replacements = replacement_messages
                if isinstance(replacements, list):
                    replacement_rows: list[tuple[str, dict[str, Any]]] = []
                    for replacement_index, message in enumerate(replacements):
                        if not isinstance(message, dict) or message.get("role") not in {
                            "user",
                            "assistant",
                        }:
                            continue
                        if stale_checkpoint and (
                            message.get("checkpoint_id")
                            or message.get("checkpoint_message")
                            or str(message.get("message_id", "")).startswith(
                                "checkpoint-"
                            )
                        ):
                            continue
                        raw_replacement_id = message.get("message_id")
                        replacement_id = (
                            str(raw_replacement_id)
                            if raw_replacement_id
                            else "replacement-"
                            f"{row.get('transaction_id', 'legacy')}-"
                            f"{replacement_index}"
                        )
                        replacement_rows.append(
                            (
                                replacement_id,
                                {
                                    "role": message["role"],
                                    "content": self._visible_history_content(
                                        message.get("role", ""),
                                        message.get("content", ""),
                                    ),
                                },
                            )
                        )
                    ordered[first_index:first_index] = replacement_rows
                continue
            message = {"role": row["role"], "content": row.get("content", "")}
            if message["role"] == "assistant" and isinstance(message["content"], list):
                message["content"] = [
                    block
                    for block in message["content"]
                    if isinstance(block, dict)
                    and block.get("type") not in _CONTINUATION_BLOCK_TYPES
                ]
            metadata = row.get("projection_metadata")
            if isinstance(metadata, dict):
                message["projection_metadata"] = {
                    key: metadata[key]
                    for key in (
                        "projection_kind",
                        "plan_key",
                        "decision_key",
                        "projection_key",
                        "run_id",
                        "planner_run_id",
                        "decision_id",
                        "decision_version",
                        "content_digest",
                        "decision_content_digest",
                        "projection_digest",
                    )
                    if key in metadata
                }
            ordered.append((str(row.get("message_id", f"legacy-{len(ordered)}")), message))
        return historical_projection + [message for _, message in ordered]

    # 从 assistant history projection 隐去 provider reasoning，保留 user/tool 可见块
    @staticmethod
    def _visible_history_content(role: object, content: object) -> object:
        if role != "assistant" or not isinstance(content, list):
            return copy.deepcopy(content)
        return [
            copy.deepcopy(block)
            for block in content
            if isinstance(block, dict)
            and block.get("type") not in _CONTINUATION_BLOCK_TYPES
        ]

    # 读取并校验 thread 原始行，统一过滤损坏行和未知 role
    def _read_thread_rows(self, sid: str) -> list[dict[str, Any]]:
        path = self.session_dir(sid) / "thread.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip broken thread row sid=%s line=%s", sid, line_no)
                continue
            role = row.get("role")
            record_type = row.get("record_type", "message")
            if record_type in {
                "directive_coverage",
                "pending_directive",
                "task_contract",
                "checkpoint",
                "surface_replace",
                "surface_state",
            }:
                rows.append(row)
                continue
            if role not in ("user", "assistant"):
                logger.warning(
                    "skip unknown thread role sid=%s line=%s role=%s",
                    sid,
                    line_no,
                    role,
                )
                continue
            if not row.get("message_id"):
                row["message_id"] = f"legacy-{sid}-{line_no}"
            rows.append(row)
        return rows

    # 返回 replacement row 中引用的 checkpoint transaction IDs
    @staticmethod
    def _replacement_checkpoint_ids(row: dict[str, Any]) -> frozenset[str]:
        replacements = row.get("replacement_messages", [])
        if not isinstance(replacements, list):
            return frozenset()
        return frozenset(
            str(message.get("checkpoint_id"))
            for message in replacements
            if isinstance(message, dict) and message.get("checkpoint_id")
        )

    # 从 envelope 或 legacy 顶层字段提取 checkpoint semantic payload
    @staticmethod
    def _checkpoint_payload(row: dict[str, Any]) -> dict[str, Any]:
        payload = row.get("payload")
        if isinstance(payload, dict):
            return dict(payload)
        return {
            key: row.get(key)
            for key in (
                "progress",
                "current_work",
                "decisions",
                "files_or_code",
                "errors_or_evidence",
                "pending",
                "next_step",
                "critical_context",
            )
            if key in row
        }

    # 只接受已经看到 checkpoint envelope 的原子 replacement，崩溃半提交时 fail closed
    @classmethod
    def _replacement_is_committed(
        cls,
        row: dict[str, Any],
        available_checkpoint_ids: set[str],
    ) -> bool:
        if row.get("checkpoint_required") is not True:
            return True
        checkpoint_ids = cls._replacement_checkpoint_ids(row)
        return bool(checkpoint_ids) and checkpoint_ids <= available_checkpoint_ids

    # 过滤尚未被原子 surface replacement 引用的 checkpoint；无 atomic 标记时兼容 legacy rows
    @classmethod
    def _committed_checkpoint_rows(
        cls,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        checkpoint_rows = [
            row for row in rows if row.get("record_type") == "checkpoint"
        ]
        if not checkpoint_rows:
            return []
        has_atomic_replacement = any(
            (
                row.get("record_type") == "surface_replace"
                and row.get("checkpoint_required") is True
            )
            or (
                row.get("record_type") == "checkpoint"
                and row.get("atomic_surface_replace") is True
            )
            for row in rows
        )
        if not has_atomic_replacement:
            # Old sessions wrote checkpoint metadata without a two-record commit.
            # Preserve that replay behavior until a new atomic replacement exists.
            return checkpoint_rows
        available_checkpoint_ids: set[str] = set()
        committed_ids: set[str] = set()
        for row in rows:
            record_type = row.get("record_type")
            if record_type == "checkpoint":
                transaction_id = row.get("transaction_id")
                if transaction_id:
                    available_checkpoint_ids.add(str(transaction_id))
            elif record_type == "surface_replace" and cls._replacement_is_committed(
                row,
                available_checkpoint_ids,
            ):
                committed_ids.update(cls._replacement_checkpoint_ids(row))
        return [
            row
            for row in checkpoint_rows
            if str(row.get("transaction_id", "")) in committed_ids
            or row.get("atomic_surface_replace") is not True
        ]

    # 裁掉尾部未配对 tool_use 以及其后的消息，避免 Anthropic messages.invalid
    def _trim_orphan_tool_use(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending: set[str] = set()
        last_balanced = 0
        for idx, msg in enumerate(messages, start=1):
            content = msg.get("content")
            if isinstance(content, list):
                if msg.get("role") == "assistant":
                    for block in content:
                        if block.get("type") == "tool_use":
                            pending.add(str(block.get("id", "")))
                elif msg.get("role") == "user":
                    for block in content:
                        if block.get("type") == "tool_result":
                            pending.discard(str(block.get("tool_use_id", "")))
            if not pending:
                last_balanced = idx
        if pending:
            logger.warning("trim orphan tool_use blocks from thread")
            return messages[:last_balanced]
        return messages

    # 读取 append-only surface replacement 的最高 revision，缺失时从零开始
    def read_surface_revision(self, sid: str) -> int:
        rows = self._read_thread_rows(sid)
        revisions: list[int] = []
        available_checkpoint_ids: set[str] = set()
        for row in rows:
            record_type = row.get("record_type")
            if record_type == "checkpoint":
                transaction_id = row.get("transaction_id")
                if transaction_id:
                    available_checkpoint_ids.add(str(transaction_id))
                continue
            if record_type == "surface_replace" and not self._replacement_is_committed(
                row,
                available_checkpoint_ids,
            ):
                continue
            if record_type not in {"surface_replace", "surface_state"}:
                continue
            value = row.get("surface_revision")
            if isinstance(value, (int, str)):
                try:
                    revisions.append(int(value))
                except ValueError:
                    continue
        return max(revisions, default=0)

    # 追加当前 semantic surface revision，供 daemon 重启后恢复 CAS 身份
    def append_surface_state(
        self,
        sid: str,
        *,
        surface_revision: int,
        active_head_id: str | None,
        contract_version: int = 0,
        contract_digest: str = "",
        pending_watermark: str = "",
        pending_digest: str = "",
        active_checkpoint_id: str | None = None,
        active_checkpoint_status: str = "NONE",
    ) -> None:
        self.append_record(
            sid,
            "surface_state",
            {
                "surface_revision": surface_revision,
                "active_head_id": active_head_id,
                "contract_version": contract_version,
                "contract_digest": contract_digest,
                "pending_watermark": pending_watermark,
                "pending_digest": pending_digest,
                "active_checkpoint_id": active_checkpoint_id,
                "active_checkpoint_status": active_checkpoint_status,
            },
        )

    # 读取最近一次 durable surface metadata；不存在时返回空值
    def read_surface_state(self, sid: str) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        for row in self._read_thread_rows(sid):
            if row.get("record_type") == "surface_state":
                latest = dict(row)
        return latest

    # 计算当前 provider message surface 的稳定 head，排除内部身份元数据
    def surface_head_id(self, sid: str) -> str | None:
        messages = self.read_messages_with_metadata(sid)
        if not messages:
            return None
        canonical: list[dict[str, Any]] = []
        for message in messages:
            item = copy.deepcopy(message)
            raw_continuation = item.get("_continuation_state")
            if (
                item.get("role") == "assistant"
                and isinstance(raw_continuation, dict)
                and not isinstance(item.get("content"), list)
            ):
                # Match ExecutionContext's restart canonicalization: a legacy
                # text row plus durable continuation blocks is one provider
                # assistant content sequence for surface identity purposes.
                from kama_claude.core.llm.types import ProviderContinuationState

                state = ProviderContinuationState.from_dict(raw_continuation)
                item["content"] = [
                    *state.as_blocks(),
                    {"type": "text", "text": str(item.get("content", ""))},
                ]
            elif item.get("role") == "assistant" and isinstance(raw_continuation, dict):
                # A durable state is authoritative over any legacy visible
                # thinking blocks embedded in the content list.
                from kama_claude.core.llm.types import ProviderContinuationState

                state = ProviderContinuationState.from_dict(raw_continuation)
                blocks = item["content"]
                if isinstance(blocks, list):
                    item["content"] = [
                        *state.as_blocks(),
                        *[
                            block
                            for block in blocks
                            if not (
                                isinstance(block, dict)
                                and block.get("type") in _CONTINUATION_BLOCK_TYPES
                            )
                        ],
                    ]
            for key in ("_message_id", "_unit_id", "_checkpoint_id"):
                item.pop(key, None)
            item.pop("_continuation_state", None)
            canonical.append(item)
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    # 以 append-only surface_replace 记录覆盖当前 provider surface，不删除旧 transcript
    def write_compacted(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        *,
        contract_digest: str = "",
        base_surface_revision: int | None = None,
        surface_revision: int | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        rows = self._read_thread_rows(sid)
        current_surface_revision = self.read_surface_revision(sid)
        expected_base = (
            current_surface_revision
            if base_surface_revision is None
            else base_surface_revision
        )
        if expected_base != current_surface_revision:
            raise ValueError("surface revision conflict")
        next_surface_revision = (
            expected_base + 1 if surface_revision is None else surface_revision
        )
        if next_surface_revision <= expected_base:
            raise ValueError("surface revision must advance")
        existing_ids = [
            str(row.get("message_id"))
            for row in rows
            if row.get("role") in {"user", "assistant"} and row.get("message_id")
        ]
        for row in rows:
            if row.get("record_type") != "surface_replace":
                continue
            replacements = row.get("replacement_messages", [])
            if isinstance(replacements, list):
                existing_ids.extend(
                    str(message["message_id"])
                    for message in replacements
                    if isinstance(message, dict) and message.get("message_id")
                )
        replacement: list[dict[str, Any]] = []
        transaction_id = f"compact-{uuid.uuid4().hex}"
        for msg in messages:
            # Retained raw units keep their durable IDs; only newly-created
            # checkpoint summary/ack rows receive synthetic checkpoint IDs.
            retained_id = msg.get("message_id", msg.get("_message_id"))
            message_id = (
                str(retained_id)
                if retained_id not in (None, "")
                else f"checkpoint-{uuid.uuid4().hex}"
            )
            checkpoint_marker = (
                str(msg["_checkpoint_id"])
                if msg.get("_checkpoint_id")
                else (
                    checkpoint_id
                    if checkpoint_id is not None and retained_id in (None, "")
                    else None
                )
            )
            replacement.append(
                {
                    "message_id": message_id,
                    "role": msg["role"],
                    "content": msg["content"],
                    **(
                        {"checkpoint_message": True}
                        if retained_id in (None, "")
                        else {}
                    ),
                    **(
                        {"checkpoint_id": checkpoint_marker}
                        if checkpoint_marker is not None
                        else {}
                    ),
                **(
                    {"unit_id": str(msg["_unit_id"])}
                    if msg.get("_unit_id")
                    else ({"unit_id": str(msg["unit_id"])} if msg.get("unit_id") else {})
                ),
                **(
                    {"continuation_state": copy.deepcopy(msg["_continuation_state"])}
                    if isinstance(msg.get("_continuation_state"), dict)
                    else (
                        {"continuation_state": copy.deepcopy(msg["continuation_state"])}
                        if isinstance(msg.get("continuation_state"), dict)
                        else {}
                    )
                ),
                }
            )
        self.append_surface_replace(
            sid,
            shadowed_message_ids=existing_ids,
            replacement_messages=replacement,
            replacement_contract_digest=contract_digest,
            base_surface_revision=expected_base,
            surface_revision=next_surface_revision,
            transaction_id=transaction_id,
            checkpoint_required=bool(
                checkpoint_id
                or any(
                    isinstance(msg, dict) and msg.get("_checkpoint_id")
                    for msg in messages
                )
            ),
        )

    # 读取 notes.md 全文，文件不存在时返回空字符串
    def read_notes(self, sid: str) -> str:
        path = self.session_dir(sid) / "notes.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # 将一条主动笔记追加到 notes.md
    def append_note(self, sid: str, content: str, run_id: str) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "notes.md").open("a", encoding="utf-8") as f:
            f.write(f"## Note ({_now()}, {run_id})\n{content}\n\n")

    # 以 digest envelope 和 atomic replace 写入 grounding artifact
    def write_grounding(self, sid: str, payload: dict[str, Any]) -> None:
        planning = self.session_dir(sid) / "planning"
        planning.mkdir(parents=True, exist_ok=True)
        target = planning / "grounding.json"
        temporary = planning / "grounding.json.tmp"
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        temporary.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    # 读取并校验 grounding artifact；缺失返回 None，损坏则 fail closed
    def read_grounding(self, sid: str) -> dict[str, Any] | None:
        path = self.session_dir(sid) / "planning" / "grounding.json"
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("planning artifact is corrupt") from exc
        if not isinstance(payload, dict) or not isinstance(digest, str):
            raise ValueError("planning artifact is corrupt")
        if _planning_digest(payload) != digest:
            raise ValueError("planning artifact digest mismatch")
        return payload

    # 以 create-once 原子语义写入一个 immutable PlannerDecision 版本
    def write_decision(
        self,
        sid: str,
        decision_id: str,
        version: int,
        payload: dict[str, Any],
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", decision_id) or version < 1:
            raise ValueError("invalid decision identity")
        decisions = self.session_dir(sid) / "planning" / "decisions"
        decisions.mkdir(parents=True, exist_ok=True)
        target = decisions / f"{decision_id}-v{version}.json"
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = (
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        if target.exists():
            if self.read_decision(sid, decision_id, version) == payload:
                return
            raise ValueError("immutable decision conflict")
        temporary = decisions / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.link(temporary, target)
        except FileExistsError:
            if self.read_decision(sid, decision_id, version) != payload:
                raise ValueError("immutable decision conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 读取并校验指定 immutable PlannerDecision 版本
    def read_decision(
        self,
        sid: str,
        decision_id: str,
        version: int,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", decision_id) or version < 1:
            raise ValueError("invalid decision identity")
        path = (
            self.session_dir(sid)
            / "planning"
            / "decisions"
            / f"{decision_id}-v{version}.json"
        )
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("planning artifact is corrupt") from exc
        if not isinstance(payload, dict) or not isinstance(digest, str):
            raise ValueError("planning artifact is corrupt")
        if _planning_digest(payload) != digest:
            raise ValueError("planning artifact digest mismatch")
        content_digest = payload.get("content_digest")
        if content_digest is not None:
            content_payload = dict(payload)
            del content_payload["content_digest"]
            if (
                not isinstance(content_digest, str)
                or _planning_digest(content_payload) != content_digest
            ):
                raise ValueError("planning artifact content digest mismatch")
        return payload

    # 按 identity 和 version 排序列出全部已持久化 PlannerDecision payload
    def list_decisions(self, sid: str) -> list[dict[str, Any]]:
        decisions = self.session_dir(sid) / "planning" / "decisions"
        if not decisions.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(decisions.glob("*.json")):
            match = re.fullmatch(r"(.+)-v([1-9][0-9]*)\.json", path.name)
            if match is None:
                raise ValueError("planning artifact is corrupt")
            decision_id, raw_version = match.groups()
            payload = self.read_decision(sid, decision_id, int(raw_version))
            if (
                payload.get("decision_id") != decision_id
                or payload.get("version") != int(raw_version)
            ):
                raise ValueError("planning artifact identity mismatch")
            items.append(payload)
        return sorted(
            items,
            key=lambda item: (str(item["decision_id"]), int(item["version"])),
        )

    # 返回 derived committed receipt 的稳定哈希文件路径
    def committed_plan_receipt_path(self, sid: str, projection_key: str) -> Path:
        key_hash = hashlib.sha256(projection_key.encode("utf-8")).hexdigest()
        return self.session_dir(sid) / "planning" / "committed" / f"{key_hash}.json"

    # 读取并校验 derived committed receipt envelope；缺失返回 None
    def read_committed_plan_receipt(
        self,
        sid: str,
        projection_key: str,
    ) -> dict[str, Any] | None:
        path = self.committed_plan_receipt_path(sid, projection_key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("committed receipt corrupt") from exc
        if not isinstance(payload, dict) or not isinstance(digest, str):
            raise ValueError("committed receipt corrupt")
        if _planning_digest(payload) != digest:
            raise ValueError("committed receipt digest mismatch")
        return payload

    # 创建或显式修复 derived receipt；修复只由上层独立证据验证后请求
    def write_committed_plan_receipt(
        self,
        sid: str,
        projection_key: str,
        payload: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        path = self.committed_plan_receipt_path(sid, projection_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists() and not replace:
            current = self.read_committed_plan_receipt(sid, projection_key)
            if current == payload:
                return
            raise ValueError("committed receipt immutable conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    # 返回 execution output snapshot 的外部 artifact 根目录
    def verification_root(self, sid: str) -> Path:
        return self.session_dir(sid) / "planning" / "verification"

    # 返回 execution output snapshot 的固定子目录
    def verification_snapshot_root(self, sid: str) -> Path:
        return self.verification_root(sid) / "snapshots"

    # 返回按 manifest digest 固定的 sealed snapshot artifact 路径
    def verification_snapshot_path(self, sid: str, manifest_digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
            raise ValueError("invalid verification snapshot digest")
        return self.verification_snapshot_root(sid) / manifest_digest

    # 读取并校验 sealed snapshot artifact，缺失或损坏时 fail closed
    def read_execution_output_snapshot(
        self,
        sid: str,
        manifest_digest: str,
    ) -> Any:
        from kama_claude.core.verification import _read_snapshot_artifact

        artifact = _read_snapshot_artifact(
            self.verification_snapshot_path(sid, manifest_digest)
        )
        if artifact.manifest.manifest_digest != manifest_digest:
            raise ValueError("verification snapshot identity mismatch")
        return artifact

    # 按 binding identity 查找 sealed snapshot，拒绝从 live workspace 重建
    def find_execution_output_snapshot(
        self,
        sid: str,
        *,
        request_id: str,
        execution_id: str,
        execution_run_id: str,
        projection_key: str,
    ) -> Any:
        from kama_claude.core.verification import _read_snapshot_artifact

        root = self.verification_snapshot_root(sid)
        if not root.exists():
            raise ValueError("verification snapshot is missing")
        matches: list[Any] = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            artifact = _read_snapshot_artifact(directory)
            manifest = artifact.manifest
            if (
                manifest.session_id == sid
                and manifest.request_id == request_id
                and manifest.execution_id == execution_id
                and manifest.execution_run_id == execution_run_id
                and manifest.projection_key == projection_key
            ):
                matches.append(artifact)
        if len(matches) != 1:
            raise ValueError("verification snapshot identity is missing or ambiguous")
        return matches[0]

    # 返回按 request identity 固定的 completion receipt 路径
    def execution_completion_receipt_path(self, sid: str, request_id: str) -> Path:
        key_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return (
            self.session_dir(sid)
            / "planning"
            / "verification"
            / "completions"
            / f"{key_hash}.json"
        )

    # 读取并校验 execution completion receipt，缺失时返回 None
    def read_execution_completion_receipt(
        self,
        sid: str,
        request_id: str,
    ) -> Any | None:
        from kama_claude.core.verification import ExecutionCompletionReceipt

        path = self.execution_completion_receipt_path(sid, request_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
            if not isinstance(payload, dict) or not isinstance(digest, str):
                raise ValueError("completion receipt corrupt")
            if _planning_digest(payload) != digest:
                raise ValueError("completion receipt envelope digest mismatch")
            receipt = ExecutionCompletionReceipt.model_validate(payload)
            receipt.verify_digest()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("completion receipt corrupt") from exc
        if receipt.session_id != sid or receipt.request_id != request_id:
            raise ValueError("completion receipt identity mismatch")
        return receipt

    # 以 create-once 语义写入 execution completion receipt，禁止冲突覆盖
    def write_execution_completion_receipt(
        self,
        receipt: Any,
        *,
        replace: bool = False,
    ) -> None:
        from kama_claude.core.verification import ExecutionCompletionReceipt

        if not isinstance(receipt, ExecutionCompletionReceipt):
            raise TypeError("invalid completion receipt")
        receipt.verify_digest()
        path = self.execution_completion_receipt_path(
            receipt.session_id,
            receipt.request_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = receipt.model_dump(mode="json")
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists() and not replace:
            current = self.read_execution_completion_receipt(
                receipt.session_id,
                receipt.request_id,
            )
            if current == receipt:
                return
            raise ValueError("completion receipt conflict")
        if path.exists() and replace:
            try:
                current = self.read_execution_completion_receipt(
                    receipt.session_id,
                    receipt.request_id,
                )
            except ValueError:
                current = None
            if current is not None:
                if current == receipt:
                    return
                raise ValueError("completion receipt conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            if replace:
                temporary.replace(path)
                return
            os.link(temporary, path)
        except FileExistsError:
            current = self.read_execution_completion_receipt(
                receipt.session_id,
                receipt.request_id,
            )
            if current != receipt:
                raise ValueError("completion receipt conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 返回按 verification request identity 固定的 immutable binding 路径
    def verification_binding_path(self, sid: str, request_id: str) -> Path:
        key_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return (
            self.session_dir(sid)
            / "planning"
            / "verification"
            / "bindings"
            / f"{key_hash}.json"
        )

    # 读取并校验 verification admission binding，缺失时返回 None
    def read_verification_binding(
        self,
        sid: str,
        request_id: str,
    ) -> Any | None:
        from kama_claude.core.verification import VerificationBinding

        path = self.verification_binding_path(sid, request_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
            if not isinstance(payload, dict) or not isinstance(digest, str):
                raise ValueError("verification binding corrupt")
            if _planning_digest(payload) != digest:
                raise ValueError("verification binding envelope digest mismatch")
            binding = VerificationBinding.model_validate(payload)
            binding.verify_digest()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("verification binding corrupt") from exc
        if binding.session_id != sid or binding.verification_request_id != request_id:
            raise ValueError("verification binding identity mismatch")
        return binding

    # 以 create-once 语义写入 verification admission binding
    def write_verification_binding(self, binding: Any) -> None:
        from kama_claude.core.verification import VerificationBinding

        if not isinstance(binding, VerificationBinding):
            raise TypeError("invalid verification binding")
        binding.verify_digest()
        path = self.verification_binding_path(
            binding.session_id,
            binding.verification_request_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = binding.model_dump(mode="json")
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            current = self.read_verification_binding(
                binding.session_id,
                binding.verification_request_id,
            )
            if current == binding:
                return
            raise ValueError("verification binding conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.link(temporary, path)
        except FileExistsError:
            current = self.read_verification_binding(
                binding.session_id,
                binding.verification_request_id,
            )
            if current != binding:
                raise ValueError("verification binding conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 返回按 verification identity 固定的 terminal result 路径
    def verification_result_path(self, sid: str, verification_id: str) -> Path:
        key_hash = hashlib.sha256(verification_id.encode("utf-8")).hexdigest()
        return (
            self.session_dir(sid)
            / "planning"
            / "verification"
            / "results"
            / f"{key_hash}.json"
        )

    # 读取并校验 verification terminal result，缺失时返回 None
    def read_verification_result(
        self,
        sid: str,
        verification_id: str,
    ) -> Any | None:
        from kama_claude.core.verification import VerificationResult

        path = self.verification_result_path(sid, verification_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
            if not isinstance(payload, dict) or not isinstance(digest, str):
                raise ValueError("verification result corrupt")
            if _planning_digest(payload) != digest:
                raise ValueError("verification result envelope digest mismatch")
            result = VerificationResult.model_validate(payload)
            result.verify_digest()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("verification result corrupt") from exc
        if result.verification_id != verification_id:
            raise ValueError("verification result identity mismatch")
        binding = self.read_verification_binding(sid, result.verification_request_id)
        if binding is None or (
            binding.verification_id != result.verification_id
            or binding.execution_id != result.execution_id
            or binding.input_digest != result.input_digest
            or binding.spec_digest != result.spec_digest
            or binding.runtime_profile_digest != result.runtime_profile_digest
            or binding.expected_image_id != result.expected_image_id
            or not _verification_image_identity_matches(
                binding.expected_image_id,
                result.observed_container_image_id,
                result.status,
            )
        ):
            raise ValueError("verification result identity mismatch")
        return result

    # 以 create-once 语义写入 verification terminal result
    def write_verification_result(self, sid: str, result: Any) -> None:
        from kama_claude.core.verification import VerificationResult

        if not isinstance(result, VerificationResult):
            raise TypeError("invalid verification result")
        result.verify_digest()
        binding = self.read_verification_binding(sid, result.verification_request_id)
        if binding is None:
            raise ValueError("verification binding is missing")
        if (
            binding.verification_id != result.verification_id
            or binding.execution_id != result.execution_id
            or binding.input_digest != result.input_digest
            or binding.spec_digest != result.spec_digest
            or binding.runtime_profile_digest != result.runtime_profile_digest
            or binding.expected_image_id != result.expected_image_id
            or not _verification_image_identity_matches(
                binding.expected_image_id,
                result.observed_container_image_id,
                result.status,
            )
        ):
            raise ValueError("verification result identity mismatch")
        path = self.verification_result_path(sid, result.verification_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            current = self.read_verification_result(sid, result.verification_id)
            if current == result:
                return
            raise ValueError("verification result conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.link(temporary, path)
        except FileExistsError:
            current = self.read_verification_result(sid, result.verification_id)
            if current != result:
                raise ValueError("verification result conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 返回 immutable user approval record 的稳定哈希文件路径
    def approval_record_path(self, sid: str, projection_key: str) -> Path:
        key_hash = hashlib.sha256(projection_key.encode("utf-8")).hexdigest()
        return self.session_dir(sid) / "planning" / "approvals" / f"{key_hash}.json"

    # 读取并校验 immutable approval record；缺失返回 None，损坏直接失败
    def read_approval_record(
        self,
        sid: str,
        projection_key: str,
    ) -> dict[str, Any] | None:
        path = self.approval_record_path(sid, projection_key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("approval record corrupt") from exc
        if not isinstance(payload, dict) or not isinstance(digest, str):
            raise ValueError("approval record corrupt")
        if _planning_digest(payload) != digest:
            raise ValueError("approval record digest mismatch")
        return payload

    # create-once 写入 immutable user authority，已有不同 bytes 时拒绝覆盖
    def write_approval_record(
        self,
        sid: str,
        projection_key: str,
        payload: dict[str, Any],
    ) -> None:
        path = self.approval_record_path(sid, projection_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            current = self.read_approval_record(sid, projection_key)
            if current == payload:
                return
            raise ValueError("approval record immutable conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.link(temporary, path)
        except FileExistsError:
            current = self.read_approval_record(sid, projection_key)
            if current != payload:
                raise ValueError("approval record immutable conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 返回 approved execution binding 的稳定 request namespace 路径
    def approved_execution_binding_path(self, sid: str, request_id: str) -> Path:
        key_hash = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return self.session_dir(sid) / "planning" / "executions" / f"{key_hash}.json"

    # 读取并校验 immutable execution binding 与 monotonic status projection
    def read_approved_execution_binding(
        self,
        sid: str,
        request_id: str,
    ) -> ApprovedExecutionBinding | None:
        path = self.approved_execution_binding_path(sid, request_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
            if not isinstance(payload, dict) or not isinstance(digest, str):
                raise ValueError("approved execution binding corrupt")
            if _planning_digest(payload) != digest:
                raise ValueError("approved execution binding envelope digest mismatch")
            binding = ApprovedExecutionBinding.model_validate(payload["binding"])
            binding.verify_digest()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("approved execution binding corrupt") from exc
        if binding.session_id != sid or binding.request_id != request_id:
            raise ValueError("approved execution binding identity mismatch")
        return binding

    # 兼容 execution record 命名，仍只返回 immutable binding authority
    def read_approved_execution_record(
        self,
        sid: str,
        request_id: str,
    ) -> ApprovedExecutionBinding | None:
        return self.read_approved_execution_binding(sid, request_id)

    # 列出 session 下所有 immutable execution binding，供 daemon restart reconcile
    def list_approved_execution_bindings(
        self,
        sid: str,
    ) -> list[ApprovedExecutionBinding]:
        directory = self.session_dir(sid) / "planning" / "executions"
        if not directory.exists():
            return []
        bindings: list[ApprovedExecutionBinding] = []
        for path in sorted(directory.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                payload = envelope["payload"]
                digest = envelope["digest"]
                if not isinstance(payload, dict) or not isinstance(digest, str):
                    raise ValueError("approved execution binding corrupt")
                if _planning_digest(payload) != digest:
                    raise ValueError("approved execution binding envelope digest mismatch")
                binding = ApprovedExecutionBinding.model_validate(payload["binding"])
                binding.verify_digest()
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError("approved execution binding corrupt") from exc
            if binding.session_id != sid:
                raise ValueError("approved execution binding identity mismatch")
            bindings.append(binding)
        return bindings

    # 读取 binding 对应的 monotonic status cache；损坏时 fail closed
    def read_execution_status(
        self,
        sid: str,
        request_id: str,
    ) -> ExecutionStatusProjection | None:
        path = self.approved_execution_binding_path(sid, request_id)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            digest = envelope["digest"]
            if not isinstance(payload, dict) or not isinstance(digest, str):
                raise ValueError("approved execution status corrupt")
            if _planning_digest(payload) != digest:
                raise ValueError("approved execution status envelope digest mismatch")
            binding = ApprovedExecutionBinding.model_validate(payload["binding"])
            binding.verify_digest()
            status_payload = payload.get("status")
            if status_payload is None:
                return None
            return ExecutionStatusProjection.model_validate(status_payload)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("approved execution status corrupt") from exc

    # 以 create-once 语义落盘 admission binding，禁止冲突 request 覆盖
    def write_approved_execution_binding(self, binding: ApprovedExecutionBinding) -> None:
        binding.verify_digest()
        path = self.approved_execution_binding_path(binding.session_id, binding.request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"binding": binding.model_dump(mode="json"), "status": None}
        envelope = {
            "payload": payload,
            "digest": _planning_digest(payload),
        }
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            current = self.read_approved_execution_binding(
                binding.session_id,
                binding.request_id,
            )
            if current == binding:
                return
            raise ValueError("approved execution binding conflict")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.link(temporary, path)
        except FileExistsError:
            current = self.read_approved_execution_binding(
                binding.session_id,
                binding.request_id,
            )
            if current != binding:
                raise ValueError("approved execution binding conflict") from None
        finally:
            temporary.unlink(missing_ok=True)

    # 更新 monotonic status projection，不改变 immutable binding authority
    def write_execution_status(
        self,
        sid: str,
        request_id: str,
        *,
        status: ExecutionStatus,
        status_revision: int,
        reason: str | None,
        authoritative: bool = False,
    ) -> ExecutionStatusProjection:
        binding = self.read_approved_execution_binding(sid, request_id)
        if binding is None:
            raise ValueError("approved execution binding is missing")
        if status_revision < 0:
            raise ValueError("invalid execution status revision")
        current = self.read_execution_status(sid, request_id)
        candidate = ExecutionStatusProjection(
            execution_id=binding.execution_id,
            run_id=binding.run_id,
            request_id=binding.request_id,
            projection_key=binding.projection_key,
            status=status,
            status_revision=status_revision,
            reason=reason,
            updated_at=_now(),
        )
        if current is not None:
            if not authoritative and status_revision < current.status_revision:
                raise ValueError("execution status regression")
            if not authoritative and status_revision == current.status_revision:
                if (
                    candidate.execution_id == current.execution_id
                    and candidate.run_id == current.run_id
                    and candidate.request_id == current.request_id
                    and candidate.projection_key == current.projection_key
                    and candidate.status == current.status
                    and candidate.reason == current.reason
                ):
                    return current
                raise ValueError("execution status conflict")
            if (
                not authoritative
                and
                current.status in TERMINAL_EXECUTION_STATUSES
                and status not in TERMINAL_EXECUTION_STATUSES
            ):
                raise ValueError("execution status regression")
            if (
                not authoritative
                and current.status == "running"
                and status == "admitted"
            ):
                raise ValueError("execution status regression")
            if (
                not authoritative
                and
                current.status in TERMINAL_EXECUTION_STATUSES
                and status in TERMINAL_EXECUTION_STATUSES
                and current.status != status
            ):
                raise ValueError("execution terminal status conflict")
        path = self.approved_execution_binding_path(sid, request_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        payload["status"] = candidate.model_dump(mode="json")
        envelope = {"payload": payload, "digest": _planning_digest(payload)}
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return candidate
