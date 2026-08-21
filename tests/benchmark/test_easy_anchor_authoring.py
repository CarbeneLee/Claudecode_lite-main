from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from kama_claude.benchmark.schema import (
    LoadedBenchmarkSuite,
    LoadedBenchmarkTask,
    load_suite,
)
from kama_claude.eval.graders import grade_rules

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SUITE_PATH = _REPOSITORY_ROOT / "benchmarks" / "suites" / "kama-coding-mvp-v1.json"
_TASKS_ROOT = _REPOSITORY_ROOT / "benchmarks" / "tasks"
_EASY_TASK_IDS = {
    "bugfix-subtract",
    "feature-low-stock",
    "testgen-normalize-username",
}
_DIFFICULTY_AXES = {
    "localization",
    "change_breadth",
    "contract_complexity",
    "oracle_depth",
    "issue_abstraction",
}
_JUNK_NAMES = {
    ".DS_Store",
    ".coverage",
    ".pytest_cache",
    "__pycache__",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _DifficultyReview(_StrictModel):
    level: Literal["easy"]
    axes: dict[str, int]
    total: int


class _RequirementMapping(_StrictModel):
    requirement_id: str
    public_contract: str
    criterion_ids: list[str]
    hidden_evidence: list[str]
    out_of_scope: str


class _WrongPatchProbe(_StrictModel):
    id: str
    patch: str
    expected_failing_groups: list[str]


class _SecurityReview(_StrictModel):
    trusted_fixture: Literal[True]
    offline: Literal[True]
    data_flow_isolation_only: Literal[True]
    process_isolation_is_sandbox: Literal[False]


class _AuthoringManifest(_StrictModel):
    schema_version: Literal[1]
    task_id: str
    review_status: Literal["author_validated_pending_external_review"]
    difficulty: _DifficultyReview
    requirement_test_matrix: list[_RequirementMapping]
    alternative_patch: str
    wrong_patch_probes: list[_WrongPatchProbe]
    forbidden_public_markers: list[str]
    forbidden_issue_markers: list[str]
    environment_file: str
    determinism_runs: Literal[3]
    security: _SecurityReview


@dataclass(frozen=True)
class _GradeSnapshot:
    criteria: tuple[tuple[str, bool], ...]
    groups: tuple[tuple[str, bool], ...]
    numeric_metrics: tuple[str, ...]


# 加载包含三个 easy anchors 的冻结 MVP suite
def _suite() -> LoadedBenchmarkSuite:
    return load_suite(_SUITE_PATH, _TASKS_ROOT)


# 从冻结 suite 中选择 Batch 0 的三个 easy anchors
def _easy_tasks() -> tuple[LoadedBenchmarkTask, ...]:
    tasks = tuple(
        task
        for task in _suite().tasks
        if task.metadata.task_id in _EASY_TASK_IDS
    )
    assert {task.metadata.task_id for task in tasks} == _EASY_TASK_IDS
    return tasks


# 读取单个 task 的结构化 authoring review evidence
def _authoring_manifest(task: LoadedBenchmarkTask) -> _AuthoringManifest:
    path = task.task_dir / "private" / "authoring" / "manifest.json"
    return _AuthoringManifest.model_validate_json(path.read_text(encoding="utf-8"))


# 将 private 相对路径限制在当前 task 的 private root 内
def _private_path(task: LoadedBenchmarkTask, relative: str) -> Path:
    private_root = (task.task_dir / "private").resolve(strict=True)
    candidate = (private_root / relative).resolve(strict=True)
    assert candidate.is_relative_to(private_root)
    assert candidate.is_file()
    assert not candidate.is_symlink()
    return candidate


# 使用 argv 方式把 authoring probe patch 应用到 fresh workspace
def _apply_patch(workspace: Path, patch: Path) -> None:
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch)],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


# 在 fresh copy 上运行真实 evaluation rule grader 并保留确定性证据
async def _grade_snapshot(
    task: LoadedBenchmarkTask,
    root: Path,
    *,
    patch: Path | None,
) -> _GradeSnapshot:
    workspace = root / "workspace"
    shutil.copytree(task.evaluation_task.workspace_fixture, workspace)
    if patch is not None:
        _apply_patch(workspace, patch)
    grade = await grade_rules(
        task.evaluation_task,
        workspace,
        root / "grade",
    )
    by_id = {criterion.id: criterion.passed for criterion in grade.criteria}
    groups = tuple(
        (
            group,
            all(by_id[criterion_id] for criterion_id in criterion_ids),
        )
        for group, criterion_ids in sorted(task.metadata.criterion_groups.items())
    )
    command_rows = json.loads(
        grade.command_results_path.read_text(encoding="utf-8")
    )
    numeric_metrics = tuple(
        line
        for row in command_rows
        for line in row["stdout"].splitlines()
        if line.startswith("KAMA_BENCH_METRICS_V1=")
    )
    return _GradeSnapshot(
        criteria=tuple(
            (criterion.id, criterion.passed) for criterion in grade.criteria
        ),
        groups=groups,
        numeric_metrics=numeric_metrics,
    )


# 返回 snapshot 中按名称索引的 criterion group 结果
def _groups(snapshot: _GradeSnapshot) -> dict[str, bool]:
    return dict(snapshot.groups)


# 功能：验证每个 easy anchor 都提供完整、可执行的 authoring evidence contract
# 设计：解析 private 结构化 manifest 并解析其真实文件引用，避免只靠人类 prose 声称已审计
def test_easy_anchor_authoring_evidence_is_complete() -> None:
    for task in _easy_tasks():
        manifest = _authoring_manifest(task)
        grader_ids = {
            criterion.id for criterion in task.evaluation_task.private.criteria
        }
        mapped_ids = {
            criterion_id
            for row in manifest.requirement_test_matrix
            for criterion_id in row.criterion_ids
        }

        assert manifest.task_id == task.metadata.task_id
        assert set(manifest.difficulty.axes) == _DIFFICULTY_AXES
        assert all(
            isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 2
            for score in manifest.difficulty.axes.values()
        )
        assert manifest.difficulty.total == sum(manifest.difficulty.axes.values())
        assert 0 <= manifest.difficulty.total <= 3
        assert mapped_ids == grader_ids
        assert len(manifest.wrong_patch_probes) >= 2
        assert (task.task_dir / "private" / "authoring-review.md").is_file()
        _private_path(task, manifest.alternative_patch)
        for row in manifest.requirement_test_matrix:
            assert row.requirement_id
            assert row.public_contract
            assert row.criterion_ids
            assert row.hidden_evidence
            assert row.out_of_scope
            for evidence in row.hidden_evidence:
                evidence_file = evidence.split("::", maxsplit=1)[0]
                _private_path(task, evidence_file)
        for probe in manifest.wrong_patch_probes:
            assert probe.id
            assert probe.expected_failing_groups
            assert set(probe.expected_failing_groups).issubset(
                task.metadata.criterion_groups
            )
            _private_path(task, probe.patch)


# 功能：验证 easy anchors 的 public bundle 没有 private leakage、root-cause hint 或 metadata junk
# 设计：扫描真实 task tree 与结构化 forbidden markers，同时解析公开环境合同而非比较 prose
def test_easy_anchor_public_bundle_is_hygienic_and_leak_free() -> None:
    for task in _easy_tasks():
        manifest = _authoring_manifest(task)
        for path in task.task_dir.rglob("*"):
            assert path.name not in _JUNK_NAMES
            assert path.suffix not in {".pyc", ".pyo"}
            assert not path.is_symlink()

        public_root = task.task_dir / "public"
        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(public_root.rglob("*"))
            if path.is_file()
        )
        for marker in manifest.forbidden_public_markers:
            assert marker not in public_text
        for marker in manifest.forbidden_issue_markers:
            assert marker not in task.evaluation_task.public.goal
        for private_marker in (
            "authoring-review",
            "grader.json",
            "hidden_tests",
            "reference.patch",
        ):
            assert private_marker not in public_text

        environment_path = (
            task.evaluation_task.workspace_fixture / manifest.environment_file
        )
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        assert environment == {
            "dependencies": ["pytest"],
            "network": "disabled",
            "python": "3.12",
            "schema_version": 1,
            "test_command": ["python", "-m", "pytest", "-q"],
        }


# 功能：验证 pristine/reference/alternative/wrong states 的 oracle 结果正确且三次 fresh-copy 一致
# 设计：所有 states 都运行真实 evaluation grader，按 group 断言并比较确定性 snapshot
@pytest.mark.asyncio
async def test_easy_anchor_validation_matrix_is_correct_and_deterministic(
    tmp_path: Path,
) -> None:
    for task in _easy_tasks():
        manifest = _authoring_manifest(task)
        states: list[
            tuple[str, Path | None, tuple[str, ...], bool]
        ] = [
            ("pristine", None, (), False),
            (
                "reference",
                task.task_dir / "private" / "reference.patch",
                (),
                True,
            ),
            (
                "alternative",
                _private_path(task, manifest.alternative_patch),
                (),
                True,
            ),
            *[
                (
                    probe.id,
                    _private_path(task, probe.patch),
                    tuple(probe.expected_failing_groups),
                    False,
                )
                for probe in manifest.wrong_patch_probes
            ],
        ]

        for state_id, patch, expected_failing_groups, should_pass in states:
            snapshots = [
                await _grade_snapshot(
                    task,
                    tmp_path
                    / task.metadata.task_id
                    / state_id
                    / f"run-{run}",
                    patch=patch,
                )
                for run in range(1, manifest.determinism_runs + 1)
            ]
            assert snapshots[1:] == snapshots[:-1], (
                task.metadata.task_id,
                state_id,
            )
            group_results = _groups(snapshots[0])
            assert group_results["regression"] is True
            if should_pass:
                assert all(group_results.values()), (
                    task.metadata.task_id,
                    state_id,
                    snapshots[0],
                )
            elif state_id == "pristine":
                assert any(
                    not passed
                    for group, passed in group_results.items()
                    if group != "regression"
                )
            else:
                assert expected_failing_groups
                assert all(
                    group_results[group] is False
                    for group in expected_failing_groups
                )
