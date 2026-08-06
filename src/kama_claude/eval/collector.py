from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.runner import AttemptExecution

MAX_WORKSPACE_FILES = 10_000
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_DIFF_BYTES = 64 * 1024
_DIFF_TRUNCATED = "\n... diff truncated ...\n"


class ArtifactCollectionError(RuntimeError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FileSnapshot(_StrictModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkspaceManifest(_StrictModel):
    files: list[FileSnapshot]
    total_bytes: int = Field(ge=0)


@dataclass(frozen=True)
class CollectedArtifacts:
    outcome_path: Path
    journal_path: Path
    trace_path: Path
    initial_manifest_path: Path
    final_manifest_path: Path
    workspace_diff_path: Path
    initial_manifest: WorkspaceManifest
    final_manifest: WorkspaceManifest
    workspace_diff: str


@dataclass(frozen=True)
class TimeoutPartialArtifacts:
    journal_path: Path | None
    trace_path: Path | None


# 以固定块大小流式计算文件哈希，避免按文件大小分配内存
def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# 按相对路径稳定遍历 workspace，拒绝 symlink 和非普通文件
def _workspace_files(root: Path) -> list[Path]:
    files: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name, reverse=True)
        except OSError as exc:
            raise ArtifactCollectionError("workspace cannot be scanned") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ArtifactCollectionError("workspace symlink is not allowed")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
            else:
                raise ArtifactCollectionError("workspace contains a non-regular entry")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


# 创建有文件数和总字节上限的稳定 workspace manifest
def snapshot_workspace(
    workspace: Path | str,
    *,
    max_files: int = MAX_WORKSPACE_FILES,
    max_total_bytes: int = MAX_WORKSPACE_BYTES,
) -> WorkspaceManifest:
    candidate = Path(workspace)
    if candidate.is_symlink():
        raise ArtifactCollectionError("workspace symlink is not allowed")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise ArtifactCollectionError("workspace is missing") from exc
    if not root.is_dir():
        raise ArtifactCollectionError("workspace must be a directory")
    paths = _workspace_files(root)
    if len(paths) > max_files:
        raise ArtifactCollectionError("workspace file limit exceeded")
    rows: list[FileSnapshot] = []
    total_bytes = 0
    for path in paths:
        try:
            size = path.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise ArtifactCollectionError("workspace file cannot be read") from exc
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ArtifactCollectionError("workspace byte limit exceeded")
        rows.append(
            FileSnapshot(
                path=path.relative_to(root).as_posix(),
                size=size,
                sha256=_hash_file(path),
            )
        )
    return WorkspaceManifest(files=rows, total_bytes=total_bytes)


# 读取文本文件；二进制或非法 UTF-8 文件返回 None 供 diff 写稳定摘要
def _read_text(path: Path) -> list[str] | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactCollectionError("workspace file cannot be read") from exc
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return None


# 将 diff 行编码到 byte cap 内，并在第一次截断时写入唯一 marker
def _bounded_text(lines: list[str], max_bytes: int) -> str:
    if max_bytes <= len(_DIFF_TRUNCATED.encode("utf-8")):
        raise ValueError("max_diff_bytes is too small")
    encoded = "".join(lines).encode("utf-8")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    marker = _DIFF_TRUNCATED.encode("utf-8")
    prefix = encoded[: max_bytes - len(marker)].decode("utf-8", errors="ignore")
    return prefix + _DIFF_TRUNCATED


# 对公开初始 fixture 和最终 workspace 生成稳定、有限的 unified diff
def build_workspace_diff(
    initial_workspace: Path,
    final_workspace: Path,
    initial: WorkspaceManifest,
    final: WorkspaceManifest,
    *,
    max_bytes: int = MAX_DIFF_BYTES,
) -> str:
    before = {item.path: item for item in initial.files}
    after = {item.path: item for item in final.files}
    output: list[str] = []
    for relative in sorted(before.keys() | after.keys()):
        old = before.get(relative)
        new = after.get(relative)
        if old is not None and new is not None and old.sha256 == new.sha256:
            continue
        old_lines = _read_text(initial_workspace / relative) if old is not None else []
        new_lines = _read_text(final_workspace / relative) if new is not None else []
        if old_lines is None or new_lines is None:
            output.append(f"Binary file changed: {relative}\n")
            continue
        output.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return _bounded_text(output, max_bytes)


# 将模型以稳定 JSON 写入 artifact
def _write_model(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 收集完成 attempt 的公开 runtime 证据，不执行 grader 或计算 task success
def collect_artifacts(
    execution: AttemptExecution,
    initial_fixture: Path,
    *,
    max_diff_bytes: int = MAX_DIFF_BYTES,
) -> CollectedArtifacts:
    prepared = execution.prepared
    if execution.worker_result is None:
        raise ArtifactCollectionError("worker outcome is missing")
    source_journal = (
        prepared.runs_dir / prepared.request.run_id / "events.v2.jsonl"
    )
    if not source_journal.is_file():
        raise ArtifactCollectionError("run journal is missing")
    if not prepared.trace_path.is_file():
        raise ArtifactCollectionError("run trace is missing")
    runtime_dir = prepared.attempt_dir / "runtime"
    public_dir = prepared.attempt_dir / "public"
    journal_path = runtime_dir / "events.v2.jsonl"
    shutil.copyfile(source_journal, journal_path)
    outcome_path = public_dir / "outcome.json"
    outcome_path.write_text(
        json.dumps(
            {
                "run_id": execution.worker_result.run_id,
                "runtime_status": execution.worker_result.runtime_status,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    initial = snapshot_workspace(initial_fixture)
    final = snapshot_workspace(prepared.workspace)
    initial_manifest_path = runtime_dir / "initial-workspace.json"
    final_manifest_path = runtime_dir / "final-workspace.json"
    _write_model(initial_manifest_path, initial)
    _write_model(final_manifest_path, final)
    workspace_diff = build_workspace_diff(
        Path(initial_fixture).resolve(strict=True),
        prepared.workspace.resolve(strict=True),
        initial,
        final,
        max_bytes=max_diff_bytes,
    )
    workspace_diff_path = runtime_dir / "workspace.diff"
    workspace_diff_path.write_text(workspace_diff, encoding="utf-8")
    return CollectedArtifacts(
        outcome_path=outcome_path,
        journal_path=journal_path,
        trace_path=prepared.trace_path,
        initial_manifest_path=initial_manifest_path,
        final_manifest_path=final_manifest_path,
        workspace_diff_path=workspace_diff_path,
        initial_manifest=initial,
        final_manifest=final,
        workspace_diff=workspace_diff,
    )


# 为 timeout attempt 尽力保存 identity observer 所需的 journal prefix 与 trace
def preserve_timeout_partial_evidence(
    execution: AttemptExecution,
) -> TimeoutPartialArtifacts:
    if (
        execution.failure_category is not FailureCategory.TIMEOUT
        or execution.worker_result is not None
    ):
        raise ArtifactCollectionError("partial evidence requires a timeout attempt")
    prepared = execution.prepared
    source_journal = (
        prepared.runs_dir / prepared.request.run_id / "events.v2.jsonl"
    )
    journal_path: Path | None = None
    if source_journal.is_file() and not source_journal.is_symlink():
        candidate = prepared.attempt_dir / "runtime" / "events.v2.jsonl"
        try:
            shutil.copyfile(source_journal, candidate)
        except OSError:
            pass
        else:
            journal_path = candidate
    trace_path = (
        prepared.trace_path
        if prepared.trace_path.is_file() and not prepared.trace_path.is_symlink()
        else None
    )
    return TimeoutPartialArtifacts(
        journal_path=journal_path,
        trace_path=trace_path,
    )
