from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kama_claude.benchmark.cli import execute_benchmark
from kama_claude.benchmark.report import RepositoryState
from kama_claude.benchmark.schema import LoadedBenchmarkTask, load_suite
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.graders import grade_rules
from kama_claude.eval.models import WorkerResult
from kama_claude.eval.report import EvaluationReport
from kama_claude.eval.runner import AttemptExecution, prepare_attempt
from kama_claude.eval.task import load_task

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SUITE_PATH = _REPOSITORY_ROOT / "benchmarks" / "suites" / "kama-coding-mvp-v1.json"
_TASKS_ROOT = _REPOSITORY_ROOT / "benchmarks" / "tasks"


# 加载仓库内冻结的九任务 MVP suite
def _suite() -> object:
    return load_suite(_SUITE_PATH, _TASKS_ROOT)


# 将 private reference patch 只应用到临时 grading workspace
def _apply_reference(task: LoadedBenchmarkTask, workspace: Path) -> None:
    reference_patch = task.task_dir / "private" / "reference.patch"
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(reference_patch)],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


# 为 reference evaluator 写入最小合法 Phase 8A run journal
def _write_runtime_evidence(execution: AttemptExecution, goal: str) -> None:
    assert execution.worker_result is not None
    prepared = execution.prepared
    run_dir = prepared.runs_dir / prepared.request.run_id
    run_dir.mkdir()
    events = [
        {
            "type": "run.started",
            "run_id": prepared.request.run_id,
            "goal": goal,
            "ts": "t1",
        },
        {
            "type": "step.started",
            "run_id": prepared.request.run_id,
            "step": 1,
            "ts": "t2",
        },
        {
            "type": "step.finished",
            "run_id": prepared.request.run_id,
            "step": 1,
            "ts": "t3",
        },
        {
            "type": "run.finished",
            "run_id": prepared.request.run_id,
            "status": "success",
            "reason": None,
            "steps": 1,
            "ts": "t4",
        },
    ]
    rows = [
        {
            "event_id": f"event-{index}",
            "stream_id": f"run:{prepared.request.run_id}",
            "seq": index,
            "event": event,
        }
        for index, event in enumerate(events, 1)
    ]
    (run_dir / "events.v2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepared.trace_path.write_text("{}\n", encoding="utf-8")


# 使用 private reference patch 构造真实 Phase 8A evaluate_task 的 deterministic execution seam
async def _reference_evaluator(
    task_dir: Path | str,
    output_root: Path | str,
) -> EvaluationReport:
    loaded = load_task(task_dir)
    prepared = prepare_attempt(loaded, output_root)
    _apply_reference(
        LoadedBenchmarkTask(
            task_dir=Path(task_dir),
            metadata=next(
                task.metadata
                for task in _suite().tasks
                if task.metadata.task_id == loaded.public.id
            ),
            evaluation_task=loaded,
        ),
        prepared.workspace,
    )
    execution = AttemptExecution(
        prepared=prepared,
        worker_result=WorkerResult(
            run_id=prepared.request.run_id,
            runtime_status="success",
            result="reference validation",
        ),
        failure_category=FailureCategory.NONE,
        wall_latency_ms=1.0,
    )
    _write_runtime_evidence(execution, loaded.public.goal)

    # 将固定 reference execution 注入 Phase 8A 唯一 evaluator，不复制 collector/grader
    async def fake_attempt(
        _task: object,
        _output: object,
    ) -> AttemptExecution:
        return execution

    return await evaluate_task(
        task_dir,
        output_root,
        attempt_runner=fake_attempt,
    )


# 功能：验证 MVP suite 恰好包含批准的九个 task 与三个 category
# 设计：断言 Batch 0 至 2 的稳定顺序和完整集合，防止遗漏或混入未冻结 task
def test_mvp_suite_contains_exactly_nine_approved_tasks() -> None:
    suite = _suite()

    assert [task.metadata.task_id for task in suite.tasks] == [
        "bugfix-subtract",
        "feature-low-stock",
        "testgen-normalize-username",
        "bugfix-config-precedence",
        "feature-atomic-bulk-import",
        "testgen-quoted-query-parser",
        "bugfix-retry-state-idempotency",
        "feature-inventory-reservation-lifecycle",
        "testgen-dependency-planner",
    ]
    assert {task.metadata.category for task in suite.tasks} == {
        "bug_fixing",
        "feature_implementation",
        "test_generation",
    }


# 功能：验证 pristine fixture 的目标 criterion 失败但 regression criterion 通过
# 设计：运行真实 Phase 8A rule grader 并按 benchmark groups 解读，证明每个 task 非平凡且初态健康
@pytest.mark.asyncio
async def test_pristine_tasks_fail_targets_but_keep_regressions(
    tmp_path: Path,
) -> None:
    suite = _suite()

    for task in suite.tasks:
        grade = await grade_rules(
            task.evaluation_task,
            task.evaluation_task.workspace_fixture,
            tmp_path / task.metadata.task_id / "pristine",
        )
        by_id = {criterion.id: criterion.passed for criterion in grade.criteria}
        regression_ids = task.metadata.criterion_groups["regression"]
        assert all(by_id[criterion_id] for criterion_id in regression_ids)
        target_ids = [
            criterion_id
            for group, criterion_ids in task.metadata.criterion_groups.items()
            if group != "regression"
            for criterion_id in criterion_ids
        ]
        assert any(not by_id[criterion_id] for criterion_id in target_ids)


# 功能：验证 private reference patch 能使九个 task 的全部 Phase 8A criteria 通过
# 设计：只在 tmp copy 应用 patch 后运行真实 grader，证明 task 可解且不污染 public fixture
@pytest.mark.asyncio
async def test_reference_patches_satisfy_all_private_graders(tmp_path: Path) -> None:
    suite = _suite()

    for task in suite.tasks:
        workspace = tmp_path / task.metadata.task_id / "workspace"
        shutil.copytree(task.evaluation_task.workspace_fixture, workspace)
        _apply_reference(task, workspace)

        grade = await grade_rules(
            task.evaluation_task,
            workspace,
            tmp_path / task.metadata.task_id / "reference-grade",
        )

        assert grade.passed is True
        assert all(criterion.passed for criterion in grade.criteria)
        if task.metadata.category == "test_generation":
            commands = json.loads(
                grade.command_results_path.read_text(encoding="utf-8")
            )
            outputs = "\n".join(command["stdout"] for command in commands)
            assert "KAMA_BENCH_METRICS_V1=" in outputs
            assert '"coverage_delta":' in outputs


# 功能：验证九个 reference tasks 贯通 Phase 8A evaluator、analyzer 与 internal baseline report
# 设计：只替换 Agent execution 为 private reference patch，其余使用完整生产 observer pipeline
@pytest.mark.asyncio
async def test_nine_task_reference_pipeline_validates_framework(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reference-framework-validation"
    args = argparse.Namespace(
        command="run",
        suite=str(_SUITE_PATH),
        tasks_root=str(_TASKS_ROOT),
        output=str(output),
        repeats=1,
    )

    exit_code = await execute_benchmark(
        args,
        evaluator=_reference_evaluator,
        repository=RepositoryState(commit="e" * 40, dirty=False),
        model_id="reference-patch-framework-validation",
    )

    payload = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["scope"] == "fixed_task_internal_benchmark"
    assert payload["metrics"]["overall"]["scheduled_attempts"] == 9
    assert payload["metrics"]["overall"]["successful_attempts"] == 9
    assert set(payload["metrics"]["categories"]) == {
        "bug_fixing",
        "feature_implementation",
        "test_generation",
    }
    assert payload["external_benchmark_claim"] == (
        "not_swe_bench_or_general_coding_ability"
    )
