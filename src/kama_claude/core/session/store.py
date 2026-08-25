from __future__ import annotations

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

logger = logging.getLogger(__name__)

MessageContent = str | list[dict[str, Any]]


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
    ) -> None:
        row: dict[str, Any] = {"ts": _now(), "role": role, "content": content}
        if run_id is not None:
            row["run_id"] = run_id
        if projection_metadata:
            row["projection_metadata"] = dict(projection_metadata)
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "thread.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
            self.append_message(
                sid,
                role=str(msg["role"]),
                content=msg["content"],
                run_id=run_id,
            )

    # 读取完整 thread 并返回可直接传给 Anthropic 的 messages
    def read_messages(self, sid: str) -> list[dict[str, Any]]:
        messages = [
            {"role": row["role"], "content": row.get("content", "")}
            for row in self._read_thread_rows(sid)
        ]

        messages = self._trim_orphan_tool_use(messages)
        from kama_claude.core.compact.budget import truncate_tool_results
        return truncate_tool_results(messages)

    # 读取供 TUI 使用的 projection metadata；不作为 provider message 输入
    def read_history_projection(self, sid: str) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for row in self._read_thread_rows(sid):
            message = {"role": row["role"], "content": row.get("content", "")}
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
            projected.append(message)
        return projected

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
            if role not in ("user", "assistant"):
                logger.warning(
                    "skip unknown thread role sid=%s line=%s role=%s",
                    sid,
                    line_no,
                    role,
                )
                continue
            rows.append(row)
        return rows

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

    # 将压缩后的消息对覆盖写入 thread.jsonl，原文件备份为 thread_<ts>.jsonl.bak
    def write_compacted(self, sid: str, messages: list[dict[str, Any]]) -> None:
        path = self.session_dir(sid) / "thread.jsonl"
        ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        bak = self.session_dir(sid) / f"thread_{ts_str}.jsonl.bak"
        if path.exists():
            path.rename(bak)
        with path.open("w", encoding="utf-8") as f:
            for msg in messages:
                row: dict[str, Any] = {"ts": _now(), "role": msg["role"], "content": msg["content"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
