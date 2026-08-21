from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kama_claude.eval.task import LoadedTask, load_task


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


SafeIdentifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
BenchmarkCategory = Literal[
    "bug_fixing",
    "feature_implementation",
    "test_generation",
]
CriterionGroup = Literal[
    "target_behavior",
    "regression",
    "generated_tests",
    "coverage",
]
CriterionIds = Annotated[list[SafeIdentifier], Field(min_length=1)]

_REQUIRED_GROUPS: dict[str, frozenset[str]] = {
    "bug_fixing": frozenset({"target_behavior", "regression"}),
    "feature_implementation": frozenset({"target_behavior", "regression"}),
    "test_generation": frozenset({"generated_tests", "regression", "coverage"}),
}


class BenchmarkTaskSpec(_StrictModel):
    schema_version: Literal[1]
    task_id: SafeIdentifier
    task_version: Annotated[int, Field(ge=1)]
    category: BenchmarkCategory
    criterion_groups: dict[CriterionGroup, CriterionIds]

    @model_validator(mode="after")
    # 校验每类任务只声明冻结分组，且每个 criterion 只归属一个分组
    def _groups_match_category(self) -> BenchmarkTaskSpec:
        required = _REQUIRED_GROUPS[self.category]
        if set(self.criterion_groups) != required:
            raise ValueError("criterion groups do not match benchmark category")
        criterion_ids = [
            criterion_id
            for group in self.criterion_groups.values()
            for criterion_id in group
        ]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion id must belong to exactly one group")
        return self


class SuiteManifest(_StrictModel):
    schema_version: Literal[1]
    suite_id: SafeIdentifier
    suite_version: Annotated[int, Field(ge=1)]
    task_ids: Annotated[list[SafeIdentifier], Field(min_length=1, max_length=20)]

    @model_validator(mode="after")
    # 校验 suite 中每个 task ID 唯一，避免同一任务被静默重复计分
    def _task_ids_are_unique(self) -> SuiteManifest:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("suite task ids must be unique")
        return self


@dataclass(frozen=True)
class LoadedBenchmarkTask:
    task_dir: Path
    metadata: BenchmarkTaskSpec
    evaluation_task: LoadedTask


@dataclass(frozen=True)
class LoadedBenchmarkSuite:
    suite_path: Path
    tasks_root: Path
    manifest: SuiteManifest
    tasks: tuple[LoadedBenchmarkTask, ...]


# 从 JSON 文件加载严格模型，并把解析细节净化为稳定 benchmark 错误
def _load_model[Model: (BenchmarkTaskSpec, SuiteManifest)](
    path: Path,
    model: type[Model],
) -> Model:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid benchmark file: {path.name}") from exc


# 将 task ID 解析为 tasks root 内真实目录，并拒绝 symlink canonical 逃逸
def _resolve_task_dir(tasks_root: Path, task_id: str) -> Path:
    candidate = tasks_root / task_id
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"benchmark task is missing: {task_id}") from exc
    if not resolved.is_relative_to(tasks_root):
        raise ValueError("benchmark task escapes tasks root")
    if not resolved.is_dir():
        raise ValueError(f"benchmark task must be a directory: {task_id}")
    return resolved


# 加载 suite 及其 evaluation tasks，并校验 observer metadata 与 private grader 对齐
def load_suite(
    suite_path: Path | str,
    tasks_root: Path | str,
) -> LoadedBenchmarkSuite:
    try:
        resolved_suite = Path(suite_path).resolve(strict=True)
        resolved_tasks_root = Path(tasks_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("benchmark suite path is missing") from exc
    if not resolved_suite.is_file():
        raise ValueError("benchmark suite must be a file")
    if not resolved_tasks_root.is_dir():
        raise ValueError("benchmark tasks root must be a directory")

    manifest = _load_model(resolved_suite, SuiteManifest)
    tasks: list[LoadedBenchmarkTask] = []
    for task_id in manifest.task_ids:
        task_dir = _resolve_task_dir(resolved_tasks_root, task_id)
        metadata = _load_model(task_dir / "benchmark.json", BenchmarkTaskSpec)
        evaluation_task = load_task(task_dir)
        grader_ids = {criterion.id for criterion in evaluation_task.private.criteria}
        grouped_ids = {
            criterion_id
            for group in metadata.criterion_groups.values()
            for criterion_id in group
        }
        if metadata.task_id != task_id or evaluation_task.public.id != task_id:
            raise ValueError("benchmark task identity mismatch")
        if grouped_ids != grader_ids:
            raise ValueError("criterion groups do not match private grader")
        tasks.append(
            LoadedBenchmarkTask(
                task_dir=task_dir,
                metadata=metadata,
                evaluation_task=evaluation_task,
            )
        )
    return LoadedBenchmarkSuite(
        suite_path=resolved_suite,
        tasks_root=resolved_tasks_root,
        manifest=manifest,
        tasks=tuple(tasks),
    )
