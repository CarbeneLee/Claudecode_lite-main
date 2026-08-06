from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError


# 加载尚待实现的 benchmark schema 模块，并把缺失模块转换为清晰的 RED 断言
def _schema_module() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.schema")
    except ModuleNotFoundError:
        pytest.fail("benchmark schema module is missing")


# 创建一个与 Phase 8A 合同兼容的最小 benchmark task
def _write_task(
    tasks_root: Path,
    *,
    task_id: str,
    category: str,
    criterion_groups: dict[str, list[str]],
) -> Path:
    task_dir = tasks_root / task_id
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "goal": "Complete the requested repository task.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    criteria = [
        {
            "id": criterion_id,
            "kind": "file_exists",
            "path": f"{criterion_id}.txt",
        }
        for criterion_id in dict.fromkeys(
            criterion_id
            for group in criterion_groups.values()
            for criterion_id in group
        )
    ]
    (private / "grader.json").write_text(
        json.dumps({"criteria": criteria}),
        encoding="utf-8",
    )
    (task_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task_id,
                "task_version": 1,
                "category": category,
                "criterion_groups": criterion_groups,
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 写入一个只包含 task ID 的最小 suite manifest
def _write_suite(path: Path, task_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "kama-internal-mvp",
                "suite_version": 1,
                "task_ids": task_ids,
            }
        ),
        encoding="utf-8",
    )


# 功能：验证 benchmark task schema 仅接受首批三类任务及各自冻结的 criterion groups
# 设计：使用三组手写合法输入和一个未批准类别，确保 schema 不提前扩展到五类任务
def test_task_schema_supports_only_three_initial_categories() -> None:
    schema = _schema_module()
    cases = [
        (
            "bug_fixing",
            {
                "target_behavior": ["target-tests"],
                "regression": ["regression-tests"],
            },
        ),
        (
            "feature_implementation",
            {
                "target_behavior": ["feature-tests"],
                "regression": ["regression-tests"],
            },
        ),
        (
            "test_generation",
            {
                "generated_tests": ["generated-tests"],
                "regression": ["regression-tests"],
                "coverage": ["coverage-threshold"],
            },
        ),
    ]

    for category, criterion_groups in cases:
        task = schema.BenchmarkTaskSpec.model_validate(
            {
                "schema_version": 1,
                "task_id": f"{category}-001",
                "task_version": 1,
                "category": category,
                "criterion_groups": criterion_groups,
            }
        )
        assert task.model_dump(mode="json")["criterion_groups"] == criterion_groups

    with pytest.raises(ValidationError):
        schema.BenchmarkTaskSpec.model_validate(
            {
                "schema_version": 1,
                "task_id": "refactor-001",
                "task_version": 1,
                "category": "refactoring",
                "criterion_groups": {
                    "target_behavior": ["target-tests"],
                    "regression": ["regression-tests"],
                },
            }
        )


# 功能：验证 benchmark metadata 拒绝未知字段、错误 group 组合和重复 criterion 映射
# 设计：分别破坏外层 strictness、类别必需分组和 criterion 唯一归属，锁定最小 observer 合同
def test_task_schema_rejects_scope_creep_and_ambiguous_groups() -> None:
    schema = _schema_module()
    base = {
        "schema_version": 1,
        "task_id": "bugfix-001",
        "task_version": 1,
        "category": "bug_fixing",
        "criterion_groups": {
            "target_behavior": ["target-tests"],
            "regression": ["regression-tests"],
        },
    }

    for invalid in (
        {**base, "model": "claude"},
        {
            **base,
            "criterion_groups": {"target_behavior": ["target-tests"]},
        },
        {
            **base,
            "criterion_groups": {
                "target_behavior": ["same-criterion"],
                "regression": ["same-criterion"],
            },
        },
    ):
        with pytest.raises(ValidationError):
            schema.BenchmarkTaskSpec.model_validate(invalid)


# 功能：验证 suite manifest 严格限制唯一 task ID、固定版本和最多二十个任务
# 设计：用重复 ID、未知字段与二十一个手写 ID 分别命中三个独立边界
def test_suite_manifest_is_strict_unique_and_bounded() -> None:
    schema = _schema_module()
    valid = {
        "schema_version": 1,
        "suite_id": "kama-internal-mvp",
        "suite_version": 1,
        "task_ids": ["bugfix-001", "feature-001", "testgen-001"],
    }

    manifest = schema.SuiteManifest.model_validate(valid)

    assert manifest.model_dump(mode="json") == valid
    for invalid in (
        {**valid, "provider": "anthropic"},
        {**valid, "task_ids": ["bugfix-001", "bugfix-001"]},
        {**valid, "task_ids": [f"task-{index}" for index in range(21)]},
        {**valid, "schema_version": 2},
    ):
        with pytest.raises(ValidationError):
            schema.SuiteManifest.model_validate(invalid)


# 功能：验证 suite loader 对齐 suite、benchmark metadata、Phase 8A task 与 private criteria
# 设计：创建三个真实临时 task 并加载，证明 benchmark 只在 Phase 8A task 外附加 observer metadata
def test_load_suite_reuses_phase8a_tasks_and_keeps_metadata_external(
    tmp_path: Path,
) -> None:
    schema = _schema_module()
    tasks_root = tmp_path / "tasks"
    cases = [
        (
            "bugfix-001",
            "bug_fixing",
            {
                "target_behavior": ["target-tests"],
                "regression": ["regression-tests"],
            },
        ),
        (
            "feature-001",
            "feature_implementation",
            {
                "target_behavior": ["feature-tests"],
                "regression": ["regression-tests"],
            },
        ),
        (
            "testgen-001",
            "test_generation",
            {
                "generated_tests": ["generated-tests"],
                "regression": ["regression-tests"],
                "coverage": ["coverage-threshold"],
            },
        ),
    ]
    for task_id, category, groups in cases:
        _write_task(
            tasks_root,
            task_id=task_id,
            category=category,
            criterion_groups=groups,
        )
    suite_path = tmp_path / "suites" / "mvp.json"
    _write_suite(suite_path, [task_id for task_id, _, _ in cases])

    loaded = schema.load_suite(suite_path, tasks_root)

    assert [task.metadata.task_id for task in loaded.tasks] == [
        "bugfix-001",
        "feature-001",
        "testgen-001",
    ]
    assert all(task.task_dir.parent == tasks_root.resolve() for task in loaded.tasks)
    assert all(
        "category" not in task.evaluation_task.public.model_dump(mode="json")
        for task in loaded.tasks
    )
    assert all(
        task.task_dir / "benchmark.json"
        != task.evaluation_task.workspace_fixture / "benchmark.json"
        for task in loaded.tasks
    )


# 功能：验证 suite loader 拒绝 metadata criterion 不存在和 task 目录 canonical 逃逸
# 设计：先制造 stale criterion mapping，再用 tasks root 内 symlink 指向外部合法 task 覆盖两类边界
def test_load_suite_rejects_stale_metadata_and_path_escape(tmp_path: Path) -> None:
    schema = _schema_module()
    tasks_root = tmp_path / "tasks"
    stale = _write_task(
        tasks_root,
        task_id="bugfix-001",
        category="bug_fixing",
        criterion_groups={
            "target_behavior": ["target-tests"],
            "regression": ["regression-tests"],
        },
    )
    grader_path = stale / "private" / "grader.json"
    grader = json.loads(grader_path.read_text(encoding="utf-8"))
    grader["criteria"] = grader["criteria"][:-1]
    grader_path.write_text(json.dumps(grader), encoding="utf-8")
    suite_path = tmp_path / "suites" / "mvp.json"
    _write_suite(suite_path, ["bugfix-001"])

    with pytest.raises(ValueError, match="criterion groups do not match private grader"):
        schema.load_suite(suite_path, tasks_root)

    outside_root = tmp_path / "outside"
    _write_task(
        outside_root,
        task_id="escaped-001",
        category="bug_fixing",
        criterion_groups={
            "target_behavior": ["target-tests"],
            "regression": ["regression-tests"],
        },
    )
    (tasks_root / "escaped-001").symlink_to(outside_root / "escaped-001")
    _write_suite(suite_path, ["escaped-001"])

    with pytest.raises(ValueError, match="benchmark task escapes tasks root"):
        schema.load_suite(suite_path, tasks_root)
