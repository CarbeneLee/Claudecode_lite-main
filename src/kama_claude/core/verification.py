from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from kama_claude.core.workspace.errors import SensitivePathError, WorkspaceEscapeError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy

SNAPSHOT_POLICY_VERSION = 1
MAX_SNAPSHOT_FILES = 100_000
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_VERIFICATION_TIMEOUT_S = 300
MAX_VERIFICATION_OUTPUT_BYTES = 64 * 1024
SNAPSHOT_HASH_CHUNK_BYTES = 64 * 1024
SNAPSHOT_WORKER_CANCEL_WAIT_S = 10.0
SNAPSHOT_WORKER_FORCE_WAIT_S = 1.0

_SNAPSHOT_PROCESS_START_METHOD = "spawn"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_REF_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXCLUDED_ROOTS = frozenset(
    {
        ".git",
        ".venv",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        "__pycache__",
    }
)
_EXCLUDED_PATHS = frozenset(
    {
        ".kama/daemon",
        ".kama/planning",
        ".kama/runs",
        ".kama/sessions",
        ".kama/verification",
    }
)


# 计算结构化 verification payload 的 canonical SHA-256 digest
def verification_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 返回当前 UTC 时间的稳定 ISO 文本
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 规范化并校验 verifier target，拒绝路径逃逸和 option injection
def normalize_verification_target(raw: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError("verification target must be a non-empty string")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("verification target must be workspace-relative")
    normalized = path.as_posix().removeprefix("./")
    if normalized in ("", "."):
        raise ValueError("verification target is empty")
    if any(part.startswith("-") for part in Path(normalized).parts):
        raise ValueError("verification target contains an option-looking component")
    return normalized


class VerificationError(RuntimeError):
    # 表示 verification domain 内的稳定错误基类
    pass


class SnapshotCaptureError(VerificationError):
    # 表示 output snapshot 无法形成一致 sealed artifact
    pass


class InvalidVerificationTarget(VerificationError):
    # 表示 target 不在 immutable snapshot policy 允许范围
    pass


class VerificationEnvironmentUnavailable(VerificationError):
    # 表示 trusted verification runtime 或 verifier tool 不可用
    pass


class VerificationSpecV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["pytest", "compileall"]
    targets: tuple[str, ...]
    timeout_s: StrictInt = Field(ge=1, le=MAX_VERIFICATION_TIMEOUT_S)
    spec_digest: str

    # 校验单次 spec 的 target 集合并消除重复输入
    @field_validator("targets", mode="before")
    @classmethod
    def _validate_targets(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("verification targets must be a sequence")
        normalized = [normalize_verification_target(item) for item in value]
        if not normalized:
            raise ValueError("verification requires at least one target")
        if len(set(normalized)) != len(normalized):
            raise ValueError("verification targets must be unique")
        return tuple(sorted(normalized))

    # 从 typed user request 创建 immutable single-step spec
    @classmethod
    def create(
        cls,
        *,
        kind: Literal["pytest", "compileall"],
        targets: Sequence[str],
        timeout_s: int,
    ) -> VerificationSpecV1:
        normalized = tuple(sorted(normalize_verification_target(item) for item in targets))
        if not normalized:
            raise ValueError("verification requires at least one target")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": kind,
            "targets": list(normalized),
            "timeout_s": timeout_s,
        }
        return cls.model_validate({**payload, "spec_digest": verification_digest(payload)})

    # 校验 spec digest，防止 canonical request 被静默修改
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"spec_digest"})
        if verification_digest(payload) != self.spec_digest:
            raise ValueError("verification spec digest mismatch")


class VerificationSnapshotPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: Literal[1] = 1
    excluded_roots: tuple[str, ...] = tuple(sorted(_EXCLUDED_ROOTS))
    max_files: StrictInt = Field(default=MAX_SNAPSHOT_FILES, ge=1)
    max_bytes: StrictInt = Field(default=MAX_SNAPSHOT_BYTES, ge=1)

    # 返回固定的 V1 verifier-visible tree policy
    @classmethod
    def default(cls) -> VerificationSnapshotPolicyV1:
        return cls()

    # 拒绝 production capture 通过模型替换固定的 runtime-private roots
    @field_validator("excluded_roots", mode="before")
    @classmethod
    def _validate_excluded_roots(cls, value: object) -> tuple[str, ...]:
        roots = tuple(sorted(str(item) for item in value)) if isinstance(value, Sequence) else ()
        if roots != tuple(sorted(_EXCLUDED_ROOTS)):
            raise ValueError("verification snapshot policy roots are fixed")
        return roots

    # 判断 logical path 是否属于 runtime-private root
    def is_excluded(self, logical_path: str) -> bool:
        normalized = Path(logical_path).as_posix()
        if any(
            normalized == prefix or normalized.startswith(f"{prefix}/")
            for prefix in _EXCLUDED_PATHS
        ):
            return True
        parts = Path(logical_path).parts
        excluded = set(self.excluded_roots)
        return any(part in excluded for part in parts)


class VerificationPathStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exists: bool
    kind: Literal["file", "directory", "absent", "symlink", "special", "error"]
    content_digest: str | None = None
    state_digest: str

    # 从 canonical state 字段创建并绑定 path-state digest
    @classmethod
    def create(
        cls,
        *,
        exists: bool,
        kind: Literal["file", "directory", "absent", "symlink", "special", "error"],
        content_digest: str | None = None,
    ) -> VerificationPathStateV1:
        if kind == "absent" and (exists or content_digest is not None):
            raise ValueError("absent path state must not report existence or content")
        if kind in {"file", "directory", "symlink", "special"} and not exists:
            raise ValueError("present path state must report existence")
        if kind != "file" and content_digest is not None:
            raise ValueError("only regular files may carry a content digest")
        if kind == "file" and (
            not exists
            or not content_digest
            or not _SHA256_RE.fullmatch(content_digest)
        ):
            raise ValueError("file path state requires a SHA-256 content digest")
        payload = {
            "exists": exists,
            "kind": kind,
            "content_digest": content_digest,
        }
        return cls.model_validate(
            {**payload, "state_digest": verification_digest(payload)}
        )

    # 从 execution PathState 或等价对象转换成同一 state domain
    @classmethod
    def from_path_state(cls, state: object) -> VerificationPathStateV1:
        exists = bool(getattr(state, "exists", False))
        is_symlink = bool(getattr(state, "is_symlink", False))
        is_file = bool(getattr(state, "is_file", False))
        error = getattr(state, "error", None)
        content_digest = getattr(state, "digest", None)
        if error:
            kind: Literal["file", "directory", "absent", "symlink", "special", "error"] = "error"
        elif is_symlink:
            kind = "symlink"
        elif not exists:
            kind = "absent"
        elif is_file:
            kind = "file"
        else:
            kind = "directory"
        return cls.create(
            exists=exists,
            kind=kind,
            content_digest=(
                content_digest
                if kind == "file" and isinstance(content_digest, str)
                else None
            ),
        )

    # 校验 state digest，避免 expected state 被静默替换
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"state_digest"})
        if verification_digest(payload) != self.state_digest:
            raise ValueError("verification path state digest mismatch")


class SnapshotFileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    kind: Literal["file", "directory"]
    digest: str
    size: StrictInt = Field(ge=0)


# 将 sealed file/directory entry 转换为 expected-state domain
def _entry_state(entry: SnapshotFileEntry | None) -> VerificationPathStateV1:
    if entry is None:
        return VerificationPathStateV1.create(exists=False, kind="absent")
    return VerificationPathStateV1.create(
        exists=True,
        kind="file" if entry.kind == "file" else "directory",
        content_digest=entry.digest if entry.kind == "file" else None,
    )


class ExecutionOutputSnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    snapshot_policy_version: Literal[1] = 1
    session_id: str
    request_id: str
    execution_id: str
    execution_run_id: str
    projection_key: str
    decision_id: str
    decision_version: StrictInt = Field(ge=1)
    decision_content_digest: str
    approval_record_digest: str
    commit_receipt_digest: str
    execution_scope_digest: str
    repository_snapshot_digest: str
    workspace_id: str
    expected_target_states: dict[str, VerificationPathStateV1]
    relevant_non_target_states: dict[str, VerificationPathStateV1]
    entries: tuple[SnapshotFileEntry, ...]
    file_count: StrictInt = Field(ge=0)
    total_bytes: StrictInt = Field(ge=0)
    sanitized_tree_digest: str
    snapshot_artifact_digest: str
    manifest_digest: str

    # 从已复制且二次校验的 entries 创建 canonical manifest
    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        request_id: str,
        execution_id: str,
        execution_run_id: str,
        projection_key: str,
        decision_id: str,
        decision_version: int,
        decision_content_digest: str,
        approval_record_digest: str,
        commit_receipt_digest: str,
        execution_scope_digest: str,
        repository_snapshot_digest: str,
        workspace_id: str,
        expected_target_states: Mapping[str, VerificationPathStateV1],
        relevant_non_target_states: Mapping[str, VerificationPathStateV1],
        entries: Sequence[SnapshotFileEntry],
    ) -> ExecutionOutputSnapshotManifest:
        ordered_entries = tuple(sorted(entries, key=lambda item: item.path))
        entry_payload = [item.model_dump(mode="json") for item in ordered_entries]
        tree_digest = verification_digest({"entries": entry_payload})
        total_bytes = sum(item.size for item in ordered_entries if item.kind == "file")
        artifact_digest = verification_digest(
            {
                "snapshot_policy_version": SNAPSHOT_POLICY_VERSION,
                "sanitized_tree_digest": tree_digest,
                "file_count": sum(item.kind == "file" for item in ordered_entries),
                "total_bytes": total_bytes,
            }
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "snapshot_policy_version": SNAPSHOT_POLICY_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "execution_id": execution_id,
            "execution_run_id": execution_run_id,
            "projection_key": projection_key,
            "decision_id": decision_id,
            "decision_version": decision_version,
            "decision_content_digest": decision_content_digest,
            "approval_record_digest": approval_record_digest,
            "commit_receipt_digest": commit_receipt_digest,
            "execution_scope_digest": execution_scope_digest,
            "repository_snapshot_digest": repository_snapshot_digest,
            "workspace_id": workspace_id,
            "expected_target_states": {
                path: state.model_dump(mode="json")
                for path, state in sorted(expected_target_states.items())
            },
            "relevant_non_target_states": {
                path: state.model_dump(mode="json")
                for path, state in sorted(relevant_non_target_states.items())
            },
            "entries": entry_payload,
            "file_count": sum(item.kind == "file" for item in ordered_entries),
            "total_bytes": total_bytes,
            "sanitized_tree_digest": tree_digest,
            "snapshot_artifact_digest": artifact_digest,
        }
        return cls.model_validate(
            {
                **payload,
                "entries": ordered_entries,
                "manifest_digest": verification_digest(payload),
            }
        )

    # 校验 manifest canonical digest，拒绝字段或 policy version 篡改
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"manifest_digest"})
        if verification_digest(payload) != self.manifest_digest:
            raise ValueError("snapshot manifest digest mismatch")

    # 校验 immutable artifact tree 与 manifest 完全一致
    def verify_artifact(self, tree_root: Path) -> None:
        self.verify_digest()
        entries = _enumerate_entries(tree_root, VerificationSnapshotPolicyV1.default())
        if tuple(entries) != self.entries:
            raise SnapshotCaptureError("snapshot artifact does not match manifest")


class ExecutionCompletionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    request_id: str
    execution_id: str
    execution_run_id: str
    projection_key: str
    snapshot_manifest_digest: str
    run_finished_event_id: str
    receipt_digest: str

    # 在 durable run.finished 之后 materialize derived completion receipt
    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        request_id: str,
        execution_id: str,
        execution_run_id: str,
        projection_key: str,
        snapshot_manifest_digest: str,
        run_finished_event_id: str,
    ) -> ExecutionCompletionReceipt:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "request_id": request_id,
            "execution_id": execution_id,
            "execution_run_id": execution_run_id,
            "projection_key": projection_key,
            "snapshot_manifest_digest": snapshot_manifest_digest,
            "run_finished_event_id": run_finished_event_id,
        }
        return cls.model_validate({**payload, "receipt_digest": verification_digest(payload)})

    # 校验 completion receipt 的 canonical digest
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"receipt_digest"})
        if verification_digest(payload) != self.receipt_digest:
            raise ValueError("execution completion receipt digest mismatch")


# 从 binding、sealed snapshot 和双 stream durable run.finished 重建 receipt
async def materialize_execution_completion_receipt(
    *,
    store: Any,
    journal: Any,
    session_id: str,
    request_id: str,
) -> ExecutionCompletionReceipt:
    binding = store.read_approved_execution_binding(session_id, request_id)
    if binding is None:
        raise ValueError("approved execution binding is missing")
    artifact = store.find_execution_output_snapshot(
        session_id,
        request_id=request_id,
        execution_id=binding.execution_id,
        execution_run_id=binding.run_id,
        projection_key=binding.projection_key,
    )
    manifest = artifact.manifest
    if (
        manifest.session_id != binding.session_id
        or manifest.request_id != binding.request_id
        or manifest.execution_id != binding.execution_id
        or manifest.execution_run_id != binding.run_id
        or manifest.projection_key != binding.projection_key
        or manifest.decision_id != binding.decision_id
        or manifest.decision_version != binding.decision_version
        or manifest.decision_content_digest != binding.decision_content_digest
        or manifest.approval_record_digest != binding.approval_record_digest
        or manifest.commit_receipt_digest != binding.commit_receipt_digest
        or manifest.repository_snapshot_digest != binding.snapshot_digest
        or manifest.workspace_id != binding.workspace_id
    ):
        raise ValueError("verification snapshot does not match execution binding")
    run_stream = f"run:{binding.run_id}"
    session_stream = f"session:{session_id}"
    run_replay = await journal.read_replay(
        run_stream,
        after_seq=0,
        high_watermark=journal.high_watermark(run_stream),
    )
    run_finished = [
        record
        for record in run_replay.records
        if record.event.get("type") == "run.finished"
        and record.event.get("run_id") == binding.run_id
    ]
    if len(run_finished) != 1:
        raise ValueError("durable run.finished evidence is missing or ambiguous")
    terminal = run_finished[0]
    if (
        terminal.event.get("status") != "success"
        or terminal.event.get("execution_status") != "completed_unverified"
        or terminal.event.get("execution_id") != binding.execution_id
    ):
        raise ValueError("run.finished is not completed_unverified evidence")
    session_replay = await journal.read_replay(
        session_stream,
        after_seq=0,
        high_watermark=journal.high_watermark(session_stream),
    )
    matching_session_records = [
        record
        for record in session_replay.records
        if record.event_id == terminal.event_id and record.event == terminal.event
    ]
    if len(matching_session_records) != 1:
        raise ValueError("session stream run.finished evidence is missing")
    receipt = ExecutionCompletionReceipt.create(
        session_id=binding.session_id,
        request_id=binding.request_id,
        execution_id=binding.execution_id,
        execution_run_id=binding.run_id,
        projection_key=binding.projection_key,
        snapshot_manifest_digest=manifest.manifest_digest,
        run_finished_event_id=terminal.event_id,
    )
    store.write_execution_completion_receipt(receipt, replace=True)
    return receipt


class VerificationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    verification_request_id: str
    verification_id: str
    execution_id: str
    input_digest: str
    spec_digest: str
    runtime_profile_digest: str
    expected_image_id: str
    admitted_at: str
    binding_digest: str

    # 校验 binding 中的 local image identity 格式
    @field_validator("expected_image_id")
    @classmethod
    def _validate_image_id(cls, value: str) -> str:
        if not _IMAGE_ID_RE.fullmatch(value):
            raise ValueError("expected_image_id must use sha256:<digest>")
        return value

    # 创建一次性 verification admission binding 并绑定全部输入身份
    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        verification_request_id: str,
        verification_id: str,
        execution_id: str,
        input_digest: str,
        spec_digest: str,
        runtime_profile_digest: str,
        expected_image_id: str,
        admitted_at: str | None = None,
    ) -> VerificationBinding:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "verification_request_id": verification_request_id,
            "verification_id": verification_id,
            "execution_id": execution_id,
            "input_digest": input_digest,
            "spec_digest": spec_digest,
            "runtime_profile_digest": runtime_profile_digest,
            "expected_image_id": expected_image_id,
            "admitted_at": admitted_at or _now(),
        }
        return cls.model_validate({**payload, "binding_digest": verification_digest(payload)})

    # 校验一次性 binding 的 canonical digest，损坏时拒绝 admission
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"binding_digest"})
        if verification_digest(payload) != self.binding_digest:
            raise ValueError("verification binding digest mismatch")


# 将 Docker resource quantity 转换成 bounded bytes
def _resource_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", value.upper())
    if match is None:
        raise ValueError("resource quantity must be a positive bounded quantity")
    amount = float(match.group(1))
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]
    return int(amount * multiplier)


class VerificationResourcePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pids_limit: StrictInt = Field(ge=1, le=100_000)
    memory: StrictStr
    cpus: StrictStr
    tmpfs: StrictStr

    # 校验 memory/tmpsfs quantity 在有限 Docker 资源范围内
    @field_validator("memory")
    @classmethod
    def _validate_memory(cls, value: str) -> str:
        size = _resource_bytes(value)
        if not 1024**2 <= size <= 64 * 1024**3:
            raise ValueError("memory resource is outside bounded range")
        return value

    # 校验 tmpfs quantity 在有限 Docker 资源范围内
    @field_validator("tmpfs")
    @classmethod
    def _validate_tmpfs(cls, value: str) -> str:
        size = _resource_bytes(value)
        if not 1024**2 <= size <= 4 * 1024**3:
            raise ValueError("tmpfs resource is outside bounded range")
        return value

    # 校验 CPU quantity 为正且不超过固定上限
    @field_validator("cpus")
    @classmethod
    def _validate_cpus(cls, value: str) -> str:
        try:
            cpus = float(value)
        except ValueError as exc:
            raise ValueError("cpus must be a positive decimal quantity") from exc
        if not 0 < cpus <= 64:
            raise ValueError("cpus resource is outside bounded range")
        return value


class VerificationRuntimeProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profile_id: str
    image_ref: str
    expected_image_id: str
    python_executable: str
    expected_python_identity: str
    resource_policy: VerificationResourcePolicyV1
    env_policy: dict[str, str]
    network_policy: Literal["none"] = "none"
    user_identity: str
    profile_digest: str

    # 校验 runtime profile 的 digest-pinned image 与 immutable identity
    @field_validator("image_ref")
    @classmethod
    def _validate_image_ref(cls, value: str) -> str:
        if not _IMAGE_REF_RE.fullmatch(value):
            raise ValueError("image_ref must use repository@sha256:<digest>")
        return value

    # 校验本地 Docker image ID 格式
    @field_validator("expected_image_id")
    @classmethod
    def _validate_image_id(cls, value: str) -> str:
        if not _IMAGE_ID_RE.fullmatch(value):
            raise ValueError("expected_image_id must use sha256:<digest>")
        return value

    # 拒绝 trusted verification profile 以 root 身份运行
    @field_validator("user_identity")
    @classmethod
    def _validate_user_identity(cls, value: str) -> str:
        if value in {"0", "0:0", "root"} or value.startswith("0:"):
            raise ValueError("verification runtime must be non-root")
        return value

    # 校验 trusted Python identity probe 不能为空且只使用第一行稳定身份
    @field_validator("expected_python_identity")
    @classmethod
    def _validate_python_identity(cls, value: str) -> str:
        first_line = value.strip().splitlines()[0] if value.strip() else ""
        if not first_line:
            raise ValueError("expected_python_identity must be non-empty")
        return first_line

    # 创建 server-controlled runtime profile 并计算 profile digest
    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        image_ref: str,
        expected_image_id: str,
        python_executable: str,
        expected_python_identity: str,
        resource_policy: Mapping[str, Any] | None = None,
        env_policy: Mapping[str, str] | None = None,
        user_identity: str = "65532:65532",
    ) -> VerificationRuntimeProfileV1:
        resources: dict[str, Any] = {
            "pids_limit": 256,
            "memory": "2g",
            "cpus": "2",
            "tmpfs": "64m",
        }
        if resource_policy is not None:
            if isinstance(resource_policy, VerificationResourcePolicyV1):
                resources = resource_policy.model_dump(mode="json")
            else:
                resources.update(resource_policy)
        typed_resources = VerificationResourcePolicyV1.model_validate(resources)
        if not isinstance(expected_python_identity, str):
            raise ValueError("expected_python_identity must be a string")
        identity_lines = expected_python_identity.strip().splitlines()
        if not identity_lines:
            raise ValueError("expected_python_identity must be non-empty")
        normalized_identity = identity_lines[0]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "profile_id": profile_id,
            "image_ref": image_ref,
            "expected_image_id": expected_image_id,
            "python_executable": python_executable,
            "expected_python_identity": normalized_identity,
            "resource_policy": typed_resources.model_dump(mode="json"),
            "env_policy": dict(env_policy or {}),
            "network_policy": "none",
            "user_identity": user_identity,
        }
        return cls.model_validate({**payload, "profile_digest": verification_digest(payload)})

    # 校验 profile digest，防止 runtime policy 被静默修改
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"profile_digest"})
        if verification_digest(payload) != self.profile_digest:
            raise ValueError("verification runtime profile digest mismatch")


# 在 create-once binding 前验证 spec、snapshot target 和 trusted profile identity
def _admit_verification_binding(
    *,
    session_id: str,
    verification_request_id: str,
    verification_id: str,
    execution_id: str,
    spec: VerificationSpecV1,
    snapshot_artifact: SnapshotArtifact,
    runtime_profile: VerificationRuntimeProfileV1,
) -> VerificationBinding:
    spec.verify_digest()
    runtime_profile.verify_digest()
    snapshot_artifact.manifest.verify_artifact(snapshot_artifact.tree_root)
    validate_targets_in_snapshot(snapshot_artifact.manifest, spec.targets)
    return VerificationBinding.create(
        session_id=session_id,
        verification_request_id=verification_request_id,
        verification_id=verification_id,
        execution_id=execution_id,
        input_digest=snapshot_artifact.manifest.manifest_digest,
        spec_digest=spec.spec_digest,
        runtime_profile_digest=runtime_profile.profile_digest,
        expected_image_id=runtime_profile.expected_image_id,
    )




class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    verification_id: str
    verification_request_id: str
    execution_id: str
    input_digest: str
    spec_digest: str
    runtime_profile_digest: str
    expected_image_id: str
    observed_container_image_id: str | None = None
    status: Literal[
        "verification_passed",
        "verification_failed",
        "verification_error",
        "cancelled",
        "interrupted",
    ]
    exit_code: StrictInt | None = None
    reason: str | None = None
    stdout_preview: str = ""
    stderr_preview: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    captured_stdout_digest: str | None = None
    captured_stderr_digest: str | None = None
    started_at: str
    finished_at: str
    result_digest: str

    # 创建 terminal observation 并计算 result digest
    @classmethod
    def create(cls, **values: Any) -> VerificationResult:
        payload = {"schema_version": 1, **values}
        candidate = cls.model_validate({**payload, "result_digest": ""})
        digest_payload = candidate.model_dump(mode="json", exclude={"result_digest"})
        return candidate.model_copy(
            update={"result_digest": verification_digest(digest_payload)}
        )

    # 校验 immutable result observation digest
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"result_digest"})
        if verification_digest(payload) != self.result_digest:
            raise ValueError("verification result digest mismatch")


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    # 保存已 sealed manifest、tree 和 manifest file 的外部 artifact 位置
    manifest: ExecutionOutputSnapshotManifest
    artifact_dir: Path
    tree_root: Path
    manifest_path: Path


# 计算一个 path state digest，供 approved execution expected state 绑定
def path_state_digest(state: object) -> str:
    return VerificationPathStateV1.from_path_state(state).state_digest


# 保存 snapshot enumeration 的全局 bounded budget 和取消信号
@dataclass
class _SnapshotBudget:
    policy: VerificationSnapshotPolicyV1
    cancel_event: Any | None = None
    file_count: int = 0
    total_bytes: int = 0

    # 在每个 entry/chunk 边界检查 worker cancellation
    def check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise SnapshotCaptureError("snapshot capture cancelled")

    # 在读取文件前预留 file/byte budget，避免先读后拒绝
    def reserve_file(self, size: int) -> None:
        self.check_cancelled()
        if self.file_count >= self.policy.max_files:
            raise SnapshotCaptureError("snapshot file limit exceeded")
        if size < 0 or self.total_bytes + size > self.policy.max_bytes:
            raise SnapshotCaptureError("snapshot byte limit exceeded")
        self.file_count += 1
        self.total_bytes += size


# 以固定 chunk hash 普通文件，避免把整文件读入内存
def _hash_file(path: Path, budget: _SnapshotBudget) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                budget.check_cancelled()
                chunk = source.read(SNAPSHOT_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SnapshotCaptureError("snapshot source read failed") from exc
    return digest.hexdigest()


# 捕获 source tree 的 deterministic entries，拒绝 symlink/special file 并流式限界
def _enumerate_entries(
    root: Path,
    policy: VerificationSnapshotPolicyV1,
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[SnapshotFileEntry, ...]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise SnapshotCaptureError("snapshot workspace root is not a directory")
    access_policy = WorkspaceAccessPolicy(root)
    budget = _SnapshotBudget(policy=policy, cancel_event=cancel_event)

    # 递归枚举普通文件并计算 direct-child directory digest
    def visit(directory: Path, prefix: str) -> list[SnapshotFileEntry]:
        children: list[SnapshotFileEntry] = []
        budget.check_cancelled()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise SnapshotCaptureError("snapshot source enumeration failed") from exc
        for entry in entries:
            budget.check_cancelled()
            logical = f"{prefix}/{entry.name}" if prefix else entry.name
            if policy.is_excluded(logical):
                continue
            if entry.is_symlink():
                raise SnapshotCaptureError(f"snapshot source contains symlink: {logical}")
            try:
                access_policy.ensure_allowed(logical, Path(entry.path))
            except SensitivePathError:
                continue
            except WorkspaceEscapeError as exc:
                raise SnapshotCaptureError("snapshot source path escaped workspace") from exc
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
                size = entry.stat(follow_symlinks=False).st_size
            except OSError as exc:
                raise SnapshotCaptureError("snapshot source stat failed") from exc
            if stat.S_ISDIR(mode):
                nested = visit(Path(entry.path), logical)
                children.extend(nested)
                direct = [
                    item
                    for item in nested
                    if Path(item.path).parent.as_posix() == logical
                ]
                directory_entry = SnapshotFileEntry(
                    path=logical,
                    kind="directory",
                    digest=verification_digest(
                        {
                            "children": [
                                {
                                    "path": item.path,
                                    "kind": item.kind,
                                    "digest": item.digest,
                                    "size": item.size,
                                }
                                for item in direct
                            ]
                        }
                    ),
                    size=sum(item.size for item in nested if item.kind == "file"),
                )
                children.append(directory_entry)
            elif stat.S_ISREG(mode):
                budget.reserve_file(size)
                digest = _hash_file(Path(entry.path), budget)
                file_entry = SnapshotFileEntry(
                    path=logical,
                    kind="file",
                    digest=digest,
                    size=size,
                )
                children.append(file_entry)
            else:
                raise SnapshotCaptureError(f"snapshot source contains special file: {logical}")
        return children

    entries = tuple(sorted(visit(root, ""), key=lambda item: item.path))
    return entries


# 将 source entries 复制到 staging tree，不允许 symlink 或特殊文件落入 artifact
def _copy_entries(
    source_root: Path,
    staging_tree: Path,
    entries: Sequence[SnapshotFileEntry],
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    staging_tree.mkdir(parents=True, exist_ok=True)
    for item in entries:
        if cancel_event is not None and cancel_event.is_set():
            raise SnapshotCaptureError("snapshot capture cancelled")
        target = staging_tree / item.path
        if item.kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = source_root / item.path
        if source.is_symlink() or not source.is_file():
            raise SnapshotCaptureError(f"snapshot source changed during copy: {item.path}")
        try:
            with source.open("rb") as source_file, target.open("wb") as target_file:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise SnapshotCaptureError("snapshot capture cancelled")
                    chunk = source_file.read(SNAPSHOT_HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    target_file.write(chunk)
        except OSError as exc:
            raise SnapshotCaptureError("snapshot copy failed") from exc


# 校验 target 真实存在于 immutable manifest 且未被 policy 排除
def validate_targets_in_snapshot(
    manifest: ExecutionOutputSnapshotManifest,
    targets: Sequence[str],
) -> None:
    entries = {item.path: item for item in manifest.entries}
    for raw in targets:
        target = normalize_verification_target(raw)
        item = entries.get(target)
        if item is None or item.kind not in {"file", "directory"}:
            raise InvalidVerificationTarget(
                f"verification target is absent from immutable snapshot: {target}"
            )


# 捕获、复制、复核并原子发布一份固定 V1 snapshot
def capture_execution_output_snapshot(
    *,
    workspace_root: Path,
    verification_root: Path,
    session_id: str,
    request_id: str,
    execution_id: str,
    execution_run_id: str,
    projection_key: str,
    decision_id: str,
    decision_version: int,
    decision_content_digest: str,
    approval_record_digest: str,
    commit_receipt_digest: str,
    execution_scope_digest: str,
    repository_snapshot_digest: str,
    workspace_id: str,
    expected_target_states: Mapping[str, VerificationPathStateV1],
    relevant_non_target_states: Mapping[str, VerificationPathStateV1],
) -> SnapshotArtifact:
    return _capture_execution_output_snapshot(
        workspace_root=workspace_root,
        verification_root=verification_root,
        session_id=session_id,
        request_id=request_id,
        execution_id=execution_id,
        execution_run_id=execution_run_id,
        projection_key=projection_key,
        decision_id=decision_id,
        decision_version=decision_version,
        decision_content_digest=decision_content_digest,
        approval_record_digest=approval_record_digest,
        commit_receipt_digest=commit_receipt_digest,
        execution_scope_digest=execution_scope_digest,
        repository_snapshot_digest=repository_snapshot_digest,
        workspace_id=workspace_id,
        expected_target_states=expected_target_states,
        relevant_non_target_states=relevant_non_target_states,
        policy=VerificationSnapshotPolicyV1.default(),
        cancel_event=None,
    )


# 将 snapshot 请求转换成 spawn-safe 的有限进程输入
def _serialize_snapshot_worker_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    serialized = dict(kwargs)
    for key in ("workspace_root", "verification_root"):
        serialized[key] = str(serialized[key])
    for key in ("expected_target_states", "relevant_non_target_states"):
        states = serialized[key]
        serialized[key] = {
            path: state.model_dump(mode="json")
            for path, state in states.items()
        }
    return serialized


# 在 owned snapshot process 中执行 capture，并只写入内部结果 envelope
def _snapshot_worker_entry(
    payload: Mapping[str, Any],
    result_path: str,
    cancel_event: Any,
) -> None:
    try:
        worker_kwargs = dict(payload)
        worker_kwargs["workspace_root"] = Path(worker_kwargs["workspace_root"])
        worker_kwargs["verification_root"] = Path(worker_kwargs["verification_root"])
        for key in ("expected_target_states", "relevant_non_target_states"):
            worker_kwargs[key] = {
                path: VerificationPathStateV1.model_validate(state)
                for path, state in worker_kwargs[key].items()
            }
        artifact = _capture_execution_output_snapshot(
            **worker_kwargs,
            policy=VerificationSnapshotPolicyV1.default(),
            cancel_event=cancel_event,
        )
        result = {
            "ok": True,
            "manifest": artifact.manifest.model_dump(mode="json"),
            "artifact_dir": str(artifact.artifact_dir),
        }
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__}
    try:
        result_file = Path(result_path)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # 父进程会把缺失结果文件视为 worker failure
        pass


# 在独立 cleanup process 中删除 snapshot staging，避免阻塞 event loop
def _remove_snapshot_path_worker(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


# 等待 owned process 结束，避免把 join 放进不可控后台线程
async def _wait_owned_process(process: Any, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while process.is_alive():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    process.join(0)
    return True


# 对不合作的 snapshot process 执行有界 terminate/kill/reap
async def _terminate_owned_snapshot_process(process: Any) -> None:
    if not process.is_alive():
        process.join(0)
        return
    process.terminate()
    if await _wait_owned_process(process, SNAPSHOT_WORKER_FORCE_WAIT_S):
        return
    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()
    if not await _wait_owned_process(process, SNAPSHOT_WORKER_FORCE_WAIT_S):
        raise SnapshotCaptureError("snapshot worker termination could not be confirmed")


# 有界清理 snapshot staging，失败时不继续等待不可控的文件系统线程
async def _remove_snapshot_path_bounded(path: Path) -> None:
    if not path.exists():
        return
    context = multiprocessing.get_context(_SNAPSHOT_PROCESS_START_METHOD)
    process = context.Process(  # type: ignore[attr-defined]
        target=_remove_snapshot_path_worker,
        args=(str(path),),
        daemon=True,
    )
    started = False
    try:
        process.start()
        started = True
        completed = await _wait_owned_process(process, SNAPSHOT_WORKER_CANCEL_WAIT_S)
        if not completed:
            await _terminate_owned_snapshot_process(process)
    finally:
        if started and process.is_alive():
            await _terminate_owned_snapshot_process(process)
        if started:
            process.close()
    if path.exists():
        raise SnapshotCaptureError("snapshot staging cleanup could not be confirmed")


# 启动 owned snapshot process，并在取消时有界终止和清理 staging
async def capture_execution_output_snapshot_async(**kwargs: Any) -> SnapshotArtifact:
    verification_root = Path(kwargs["verification_root"])
    staging_id = uuid.uuid4().hex
    staging_path = verification_root / ".staging" / staging_id
    result_path = verification_root / ".staging" / f"{staging_id}.result.json"
    payload = _serialize_snapshot_worker_kwargs({**kwargs, "staging_id": staging_id})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context(_SNAPSHOT_PROCESS_START_METHOD)
    cancel_event = context.Event()
    process = context.Process(  # type: ignore[attr-defined]
        target=_snapshot_worker_entry,
        args=(payload, str(result_path), cancel_event),
        daemon=True,
    )
    started = False
    try:
        process.start()
        started = True
        completed = await _wait_owned_process(process, float("inf"))
        if not completed:
            raise SnapshotCaptureError("snapshot worker did not finish")
    except asyncio.CancelledError:
        cancel_event.set()
        try:
            try:
                completed = await _wait_owned_process(
                    process,
                    SNAPSHOT_WORKER_CANCEL_WAIT_S,
                )
                if not completed:
                    await _terminate_owned_snapshot_process(process)
            except Exception:
                # cancellation remains primary; missing cleanup is diagnosed by caller
                pass
        finally:
            try:
                await _remove_snapshot_path_bounded(staging_path)
            except Exception:
                pass
            result_path.unlink(missing_ok=True)
        raise
    except Exception:
        if started and process.is_alive():
            try:
                await _terminate_owned_snapshot_process(process)
            except Exception:
                pass
        try:
            await _remove_snapshot_path_bounded(staging_path)
        except Exception:
            pass
        result_path.unlink(missing_ok=True)
        raise
    finally:
        if started and process.is_alive():
            cancel_event.set()
            await _terminate_owned_snapshot_process(process)
        if started:
            process.close()

    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            raise SnapshotCaptureError("snapshot worker failed")
        manifest = ExecutionOutputSnapshotManifest.model_validate(raw["manifest"])
        manifest.verify_digest()
        artifact_dir = Path(raw["artifact_dir"])
        artifact = SnapshotArtifact(
            manifest=manifest,
            artifact_dir=artifact_dir,
            tree_root=artifact_dir / "tree",
            manifest_path=artifact_dir / "manifest.json",
        )
        manifest.verify_artifact(artifact.tree_root)
        return artifact
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotCaptureError("snapshot worker result is unavailable") from exc
    finally:
        result_path.unlink(missing_ok=True)


# 仅供内部 budget/cancellation 测试注入固定 V1 的更小资源上限
def _capture_execution_output_snapshot_with_policy(
    *,
    policy: VerificationSnapshotPolicyV1,
    cancel_event: Any | None = None,
    **kwargs: Any,
) -> SnapshotArtifact:
    return _capture_execution_output_snapshot(
        **kwargs,
        policy=policy,
        cancel_event=cancel_event,
    )


# 执行带内部 policy/cancellation 参数的 snapshot worker 核心
def _capture_execution_output_snapshot(
    *,
    workspace_root: Path,
    verification_root: Path,
    session_id: str,
    request_id: str,
    execution_id: str,
    execution_run_id: str,
    projection_key: str,
    decision_id: str,
    decision_version: int,
    decision_content_digest: str,
    approval_record_digest: str,
    commit_receipt_digest: str,
    execution_scope_digest: str,
    repository_snapshot_digest: str,
    workspace_id: str,
    expected_target_states: Mapping[str, VerificationPathStateV1],
    relevant_non_target_states: Mapping[str, VerificationPathStateV1],
    policy: VerificationSnapshotPolicyV1,
    cancel_event: Any | None,
    staging_id: str | None = None,
) -> SnapshotArtifact:
    source_root = workspace_root.resolve(strict=True)
    external_root = verification_root.resolve()
    if external_root == source_root or external_root.is_relative_to(source_root):
        raise SnapshotCaptureError("snapshot artifact root must be outside workspace")
    pre_entries = _enumerate_entries(
        source_root,
        policy,
        cancel_event=cancel_event,
    )
    staging_parent = verification_root / ".staging"
    staging = staging_parent / (staging_id or uuid.uuid4().hex)
    staging_tree = staging / "tree"
    try:
        _copy_entries(
            source_root,
            staging_tree,
            pre_entries,
            cancel_event=cancel_event,
        )
        post_entries = _enumerate_entries(
            source_root,
            policy,
            cancel_event=cancel_event,
        )
        if pre_entries != post_entries:
            raise SnapshotCaptureError("snapshot source changed during capture")
        copied_entries = _enumerate_entries(
            staging_tree,
            policy,
            cancel_event=cancel_event,
        )
        if copied_entries != pre_entries:
            raise SnapshotCaptureError("snapshot copied tree does not match source")
        manifest = ExecutionOutputSnapshotManifest.create(
            session_id=session_id,
            request_id=request_id,
            execution_id=execution_id,
            execution_run_id=execution_run_id,
            projection_key=projection_key,
            decision_id=decision_id,
            decision_version=decision_version,
            decision_content_digest=decision_content_digest,
            approval_record_digest=approval_record_digest,
            commit_receipt_digest=commit_receipt_digest,
            execution_scope_digest=execution_scope_digest,
            repository_snapshot_digest=repository_snapshot_digest,
            workspace_id=workspace_id,
            expected_target_states=expected_target_states,
            relevant_non_target_states=relevant_non_target_states,
            entries=pre_entries,
        )
        _validate_expected_states(manifest, expected_target_states)
        _validate_expected_states(manifest, relevant_non_target_states)
        staging.mkdir(parents=True, exist_ok=True)
        manifest_tmp = staging / "manifest.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        loaded = ExecutionOutputSnapshotManifest.model_validate(
            json.loads(manifest_tmp.read_text(encoding="utf-8"))
        )
        loaded.verify_digest()
        manifest_path = staging / "manifest.json"
        manifest_tmp.replace(manifest_path)
        final_dir = verification_root / "snapshots" / manifest.manifest_digest
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            existing = _read_snapshot_artifact(final_dir)
            if existing.manifest != manifest:
                raise SnapshotCaptureError("snapshot manifest publication conflict")
            shutil.rmtree(staging, ignore_errors=True)
            return existing
        staging.replace(final_dir)
        return SnapshotArtifact(
            manifest=manifest,
            artifact_dir=final_dir,
            tree_root=final_dir / "tree",
            manifest_path=final_dir / "manifest.json",
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# 校验 expected state digest 是否与 sealed tree 中的 path state 一致
def _validate_expected_states(
    manifest: ExecutionOutputSnapshotManifest,
    expected_states: Mapping[str, VerificationPathStateV1],
) -> None:
    entries = {item.path: item for item in manifest.entries}
    for raw_path, expected in expected_states.items():
        try:
            path = normalize_verification_target(raw_path)
            expected.verify_digest()
        except (AttributeError, TypeError, ValueError) as exc:
            raise SnapshotCaptureError("snapshot expected state is malformed") from exc
        actual = _entry_state(entries.get(path))
        if actual != expected:
            raise SnapshotCaptureError(f"snapshot expected state mismatch: {path}")


# 从 sealed artifact 读取 manifest 并验证所有 tree/file digests
def _read_snapshot_artifact(artifact_dir: Path) -> SnapshotArtifact:
    try:
        manifest = ExecutionOutputSnapshotManifest.model_validate(
            json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        manifest.verify_artifact(artifact_dir / "tree")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SnapshotCaptureError("snapshot artifact is corrupt") from exc
    return SnapshotArtifact(
        manifest=manifest,
        artifact_dir=artifact_dir,
        tree_root=artifact_dir / "tree",
        manifest_path=artifact_dir / "manifest.json",
    )


# 从 immutable source artifact 物化新的 writable runtime copy，并在 chunk 边界响应取消
def materialize_snapshot_copy(
    artifact: SnapshotArtifact,
    destination: Path,
    *,
    cancel_event: Any | None = None,
) -> None:
    artifact.manifest.verify_artifact(artifact.tree_root)
    if destination.exists():
        raise SnapshotCaptureError("runtime copy destination already exists")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for item in artifact.manifest.entries:
            if cancel_event is not None and cancel_event.is_set():
                raise SnapshotCaptureError("runtime copy cancelled")
            target = destination / item.path
            if item.kind == "directory":
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = artifact.tree_root / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_file, target.open("wb") as target_file:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise SnapshotCaptureError("runtime copy cancelled")
                    chunk = source_file.read(SNAPSHOT_HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    target_file.write(chunk)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
