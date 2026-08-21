from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
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
_TASKS_ROOT = _REPOSITORY_ROOT / "benchmarks" / "tasks"
_CHALLENGING_TASK_IDS = (
    "bugfix-retry-state-idempotency",
    "feature-inventory-reservation-lifecycle",
    "testgen-dependency-planner",
)
_ALL_TASK_IDS = (
    "bugfix-subtract",
    "feature-low-stock",
    "testgen-normalize-username",
    "bugfix-config-precedence",
    "feature-atomic-bulk-import",
    "testgen-quoted-query-parser",
    *_CHALLENGING_TASK_IDS,
)
_TASK_DOMAINS = {
    "bugfix-subtract": "arithmetic",
    "feature-low-stock": "stock-query",
    "testgen-normalize-username": "text-normalization",
    "bugfix-config-precedence": "configuration",
    "feature-atomic-bulk-import": "bulk-orders",
    "testgen-quoted-query-parser": "query-language",
    "bugfix-retry-state-idempotency": "job-lifecycle",
    "feature-inventory-reservation-lifecycle": "inventory-reservation",
    "testgen-dependency-planner": "dependency-graph",
}
_EXPECTED_GROUPS = {
    "bug_fixing": {"target_behavior", "regression"},
    "feature_implementation": {"target_behavior", "regression"},
    "test_generation": {"generated_tests", "regression", "coverage"},
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
    level: Literal["challenging"]
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


# 通过单 task 临时 suite 复用冻结 schema，不提前冻结正式 suite
def _challenging_task(tmp_path: Path, task_id: str) -> LoadedBenchmarkTask:
    suite_path = tmp_path / f"{task_id}-authoring-suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "evaluation-batch-two-authoring",
                "suite_version": 1,
                "task_ids": [task_id],
            }
        ),
        encoding="utf-8",
    )
    suite: LoadedBenchmarkSuite = load_suite(suite_path, _TASKS_ROOT)
    return suite.tasks[0]


# 读取单个 challenging task 的结构化 authoring review evidence
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


# 在 fresh copy 上运行真实 evaluation rule grader并保留确定性证据
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


# 功能：验证三个 challenging tasks 的结构、难度和独立 requirement evidence 完整
# 设计：通过临时 suite 复用冻结 schema，并要求四个 wrong probes 与 alternative patch
@pytest.mark.parametrize("task_id", _CHALLENGING_TASK_IDS)
def test_challenging_task_authoring_evidence_is_complete(
    tmp_path: Path,
    task_id: str,
) -> None:
    task = _challenging_task(tmp_path, task_id)
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
    assert 8 <= manifest.difficulty.total <= 10
    assert mapped_ids == grader_ids
    assert len(manifest.wrong_patch_probes) >= 4
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


# 功能：验证 challenging public bundles 不泄漏 private oracle、root cause 或 metadata junk
# 设计：扫描真实目录、结构化 forbidden markers 与固定环境合同，覆盖 data-flow boundary
@pytest.mark.parametrize("task_id", _CHALLENGING_TASK_IDS)
def test_challenging_task_public_bundles_are_hygienic_and_leak_free(
    tmp_path: Path,
    task_id: str,
) -> None:
    task = _challenging_task(tmp_path, task_id)
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


# 功能：验证 challenging tasks 的完整 candidate matrix 且三次 fresh-copy 结果一致
# 设计：对 pristine、双正确实现和至少四个 plausible wrong probes 运行真实规则 grader
@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", _CHALLENGING_TASK_IDS)
async def test_challenging_task_validation_matrix_is_correct_and_deterministic(
    tmp_path: Path,
    task_id: str,
) -> None:
    task = _challenging_task(tmp_path, task_id)
    manifest = _authoring_manifest(task)
    states: list[tuple[str, Path | None, tuple[str, ...], bool]] = [
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


# 功能：验证九任务在类别、难度、领域、oracle 和环境上满足冻结前平衡要求
# 设计：用临时 suite 只读加载全部任务，并对三乘三组合与跨任务合同做机械审计
def test_pre_freeze_suite_balance_is_consistent(tmp_path: Path) -> None:
    suite_path = tmp_path / "evaluation-pre-freeze-balance.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "evaluation-pre-freeze-balance",
                "suite_version": 1,
                "task_ids": list(_ALL_TASK_IDS),
            }
        ),
        encoding="utf-8",
    )
    suite = load_suite(suite_path, _TASKS_ROOT)
    category_difficulty: Counter[tuple[str, str]] = Counter()
    environments: set[str] = set()

    for task in suite.tasks:
        manifest_path = task.task_dir / "private" / "authoring" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        level = manifest["difficulty"]["level"]
        category_difficulty[(task.metadata.category, level)] += 1
        assert set(task.metadata.criterion_groups) == _EXPECTED_GROUPS[
            task.metadata.category
        ]
        environment_path = task.evaluation_task.workspace_fixture / "environment.json"
        environments.add(
            json.dumps(
                json.loads(environment_path.read_text(encoding="utf-8")),
                sort_keys=True,
            )
        )

    expected_pairs = {
        (category, difficulty)
        for category in _EXPECTED_GROUPS
        for difficulty in ("easy", "medium", "challenging")
    }
    assert set(category_difficulty) == expected_pairs
    assert set(category_difficulty.values()) == {1}
    assert len(set(_TASK_DOMAINS.values())) == len(_ALL_TASK_IDS)
    assert set(_TASK_DOMAINS) == set(_ALL_TASK_IDS)
    assert len(environments) == 1
