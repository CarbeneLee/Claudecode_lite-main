from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from kama_claude.eval.schema import PrivateGraderSpec, PublicTaskSpec


@dataclass(frozen=True)
class LoadedTask:
    task_dir: Path
    public: PublicTaskSpec
    private: PrivateGraderSpec
    workspace_fixture: Path
    hidden_tests: Path | None


# 将候选路径解析到指定根目录内，拒绝缺失路径和 canonical 逃逸
def _resolve_within(root: Path, candidate: Path, *, label: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes its allowed root")
    return resolved


# 从 JSON 文件读取严格模型并将解析错误净化为稳定任务错误
def _load_model[TaskModel: (PublicTaskSpec, PrivateGraderSpec)](
    path: Path, model: type[TaskModel]
) -> TaskModel:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(value)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid evaluation file: {path.name}") from exc


# 递归拒绝 fixture 内任意 symlink，防止复制前跨越 public/private 数据边界
def _reject_symlink_entries(root: Path, *, label: str) -> None:
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"{label} contains a symlink")
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected") from exc


# 加载单个 Phase 8A task 并保持公开输入与私有 grader 的路径边界
def load_task(task_dir: Path | str) -> LoadedTask:
    root = Path(task_dir).resolve(strict=True)
    public_root = _resolve_within(root, root / "public", label="public task directory")
    private_root = _resolve_within(root, root / "private", label="private grader directory")
    public = _load_model(public_root / "task.json", PublicTaskSpec)
    private = _load_model(private_root / "grader.json", PrivateGraderSpec)
    if public.id != root.name:
        raise ValueError("public task id must match task directory")
    workspace_fixture = _resolve_within(
        public_root,
        root / public.workspace_fixture,
        label="workspace fixture",
    )
    if not workspace_fixture.is_dir():
        raise ValueError("workspace fixture must be a directory")
    _reject_symlink_entries(workspace_fixture, label="workspace fixture")
    hidden_candidate = private_root / "hidden_tests"
    hidden_tests = (
        _resolve_within(private_root, hidden_candidate, label="hidden tests")
        if hidden_candidate.exists()
        else None
    )
    if hidden_tests is not None and not hidden_tests.is_dir():
        raise ValueError("hidden tests must be a directory")
    return LoadedTask(
        task_dir=root,
        public=public,
        private=private,
        workspace_fixture=workspace_fixture,
        hidden_tests=hidden_tests,
    )
