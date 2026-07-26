from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from kama_claude.eval.collector import (
    ArtifactCollectionError,
    collect_artifacts,
    preserve_timeout_partial_evidence,
)
from kama_claude.eval.failure import FailureCategory, select_failure_category
from kama_claude.eval.graders import (
    GraderExecutionError,
    RuleGrade,
    TraceGrade,
    grade_rules,
    grade_trace,
)
from kama_claude.eval.metrics import compute_basic_metrics
from kama_claude.eval.report import (
    EvaluationReport,
    PublicCriterionResult,
    write_report,
)
from kama_claude.eval.runner import AttemptExecution, run_attempt
from kama_claude.eval.task import LoadedTask, load_task

type AttemptRunner = Callable[
    [LoadedTask, Path | str],
    Awaitable[AttemptExecution],
]


# 写入不包含 final text、private criteria 或异常详情的公开 outcome
def _write_public_outcome(execution: AttemptExecution) -> None:
    public_dir = execution.prepared.attempt_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": execution.prepared.request.run_id,
        "runtime_status": (
            execution.worker_result.runtime_status
            if execution.worker_result is not None
            else execution.failure_category.value
        ),
    }
    (public_dir / "outcome.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 运行一个 Phase 8A task，收集客观证据并生成单 attempt 公开报告
async def evaluate_task(
    task_dir: Path | str,
    output_root: Path | str,
    *,
    attempt_runner: AttemptRunner = run_attempt,
) -> EvaluationReport:
    task = load_task(task_dir)
    execution = await attempt_runner(task, output_root)
    categories = {execution.failure_category}
    if execution.failure_category is FailureCategory.TIMEOUT:
        preserve_timeout_partial_evidence(execution)
    runtime_success = (
        execution.worker_result is not None
        and execution.worker_result.runtime_status == "success"
    )
    trace_grade = TraceGrade(passed=False, errors=["not evaluated"], events=[])
    rule_grade: RuleGrade | None = None
    if execution.worker_result is not None:
        try:
            artifacts = collect_artifacts(execution, task.workspace_fixture)
        except ArtifactCollectionError:
            categories.add(FailureCategory.INFRA_ERROR)
        else:
            trace_grade = grade_trace(
                artifacts.journal_path,
                expected_run_id=execution.prepared.request.run_id,
                expected_terminal_status=execution.worker_result.runtime_status,
            )
            if not trace_grade.passed:
                categories.add(FailureCategory.TRACE_INVALID)
            try:
                rule_grade = await grade_rules(
                    task,
                    execution.prepared.workspace,
                    execution.prepared.attempt_dir / "private",
                )
            except GraderExecutionError:
                categories.add(FailureCategory.GRADER_ERROR)
            else:
                if not rule_grade.passed:
                    categories.add(FailureCategory.TASK_FAILED)
    if not runtime_success and execution.failure_category is FailureCategory.NONE:
        categories.add(FailureCategory.RUNTIME_FAILED)
    failure_category = select_failure_category(categories)
    task_success = (
        runtime_success
        and trace_grade.passed
        and rule_grade is not None
        and rule_grade.passed
        and failure_category is FailureCategory.NONE
    )
    metrics = compute_basic_metrics(
        trace_grade.events if trace_grade.passed else [],
        task_success=task_success,
        runtime_success=runtime_success,
        wall_latency_ms=execution.wall_latency_ms,
        failure_category=failure_category,
    )
    criteria = (
        [
            PublicCriterionResult(
                id=criterion.id,
                kind=criterion.kind,
                passed=criterion.passed,
            )
            for criterion in rule_grade.criteria
        ]
        if rule_grade is not None
        else []
    )
    report = EvaluationReport(
        task_id=task.public.id,
        attempt_id=execution.prepared.attempt_id,
        task_success=task_success,
        runtime_success=runtime_success,
        trace_sanity_passed=trace_grade.passed,
        failure_category=failure_category,
        criteria=criteria,
        metrics=metrics,
    )
    _write_public_outcome(execution)
    metrics_path = execution.prepared.attempt_dir / "public" / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_root, report)
    return report
