from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from kama_claude.benchmark.orchestrator import BenchmarkAttempt
from kama_claude.benchmark.schema import BenchmarkTaskSpec
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import BasicMetrics, TokenUsage
from kama_claude.eval.report import EvaluationReport, PublicCriterionResult


# 加载尚待实现的 analyzer 模块，并把缺失模块转换为清晰的 RED 断言
def _analyzer_module() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.analyzers")
    except ModuleNotFoundError:
        pytest.fail("benchmark analyzer module is missing")


# 构造带指定 criteria 和基本指标的 Phase 8A report
def _report(
    task_id: str,
    *,
    attempt_id: str = "attempt-a",
    task_success: bool = True,
    failure_category: FailureCategory = FailureCategory.NONE,
    criteria: list[PublicCriterionResult],
    retry_count: int = 1,
) -> EvaluationReport:
    return EvaluationReport(
        task_id=task_id,
        attempt_id=attempt_id,
        task_success=task_success,
        runtime_success=True,
        trace_sanity_passed=True,
        failure_category=failure_category,
        criteria=criteria,
        metrics=BasicMetrics(
            task_success=task_success,
            runtime_success=True,
            step_count=3,
            tool_count=2,
            retry_count=retry_count,
            wall_latency_ms=25.0,
            token_usage=TokenUsage(
                input_tokens=20,
                output_tokens=5,
                cache_tokens=4,
            ),
            failure_category=failure_category,
        ),
    )


# 返回 test generation 的冻结 metadata
def _testgen_metadata() -> BenchmarkTaskSpec:
    return BenchmarkTaskSpec.model_validate(
        {
            "schema_version": 1,
            "task_id": "testgen-001",
            "task_version": 1,
            "category": "test_generation",
            "criterion_groups": {
                "generated_tests": ["generated-tests"],
                "regression": ["regression-tests"],
                "coverage": ["coverage-threshold"],
            },
        }
    )


# 在 Phase 8A attempt layout 中写入 analyzer 所需的 validated evidence
def _write_artifacts(
    evaluation_output: Path,
    report: EvaluationReport,
    *,
    metric_line: str | None,
) -> None:
    attempt_dir = (
        evaluation_output
        / "attempts"
        / report.task_id
        / report.attempt_id
    )
    runtime = attempt_dir / "runtime"
    private = attempt_dir / "private"
    runtime.mkdir(parents=True)
    private.mkdir()
    unchanged_hash = "1" * 64
    new_hash = "2" * 64
    (runtime / "initial-workspace.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "src/module.py", "size": 10, "sha256": unchanged_hash}
                ],
                "total_bytes": 10,
            }
        ),
        encoding="utf-8",
    )
    (runtime / "final-workspace.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "src/module.py", "size": 10, "sha256": unchanged_hash},
                    {
                        "path": "tests/test_generated.py",
                        "size": 20,
                        "sha256": new_hash,
                    },
                ],
                "total_bytes": 30,
            }
        ),
        encoding="utf-8",
    )
    (runtime / "workspace.diff").write_text(
        "--- a/tests/test_generated.py\n"
        "+++ b/tests/test_generated.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_generated():\n"
        "+    assert True\n",
        encoding="utf-8",
    )
    events = [
        {
            "type": "tool.call_failed",
            "tool_use_id": "tool-a",
            "tool_name": "bash",
            "error_class": "transient_error",
            "attempt": 1,
        },
        {
            "type": "tool.call_finished",
            "tool_use_id": "tool-a",
            "tool_name": "bash",
        },
        {
            "type": "tool.call_failed",
            "tool_use_id": "tool-b",
            "tool_name": "bash",
            "error_class": "permission_denied",
            "attempt": 1,
        },
    ]
    (runtime / "events.v2.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "event_id": f"event-{index}",
                    "stream_id": "run:test",
                    "seq": index,
                    "event": event,
                }
            )
            + "\n"
            for index, event in enumerate(events, 1)
        ),
        encoding="utf-8",
    )
    stdout = metric_line + "\n" if metric_line is not None else ""
    (private / "command-results.json").write_text(
        json.dumps(
            [
                {
                    "criterion_id": "coverage-threshold",
                    "exit_code": 0,
                    "timed_out": False,
                    "stdout": stdout,
                    "stderr": "",
                }
            ]
        ),
        encoding="utf-8",
    )


# 功能：验证 test generation analyzer 机械提取 criteria、workspace、retry 和数值 grader evidence
# 设计：同时放入 recovered transient failure 与未重试 permission failure，锁定 recovery 分母
def test_analyze_test_generation_attempt_uses_only_validated_evidence(
    tmp_path: Path,
) -> None:
    analyzers = _analyzer_module()
    metadata = _testgen_metadata()
    criteria = [
        PublicCriterionResult(id="generated-tests", kind="command_exit", passed=True),
        PublicCriterionResult(id="regression-tests", kind="command_exit", passed=True),
        PublicCriterionResult(id="coverage-threshold", kind="command_exit", passed=True),
    ]
    report = _report("testgen-001", criteria=criteria)
    evaluation_output = tmp_path / "evaluation"
    _write_artifacts(
        evaluation_output,
        report,
        metric_line=(
            'KAMA_BENCH_METRICS_V1={"tests_passed":7,'
            '"tests_failed":0,"coverage_delta":12.5}'
        ),
    )
    attempt = BenchmarkAttempt(
        task_id="testgen-001",
        category="test_generation",
        repeat=1,
        evaluation_output=evaluation_output,
        report=report,
    )

    result = analyzers.analyze_attempt(attempt, metadata)

    assert result.group_results == {
        "generated_tests": True,
        "regression": True,
        "coverage": True,
    }
    assert result.regression_introduced is False
    assert result.tests_passed == 7
    assert result.tests_failed == 0
    assert result.coverage_delta == 12.5
    assert result.changed_files == 1
    assert result.source_changed_files == 0
    assert result.test_changed_files == 1
    assert result.diff_additions == 2
    assert result.diff_deletions == 0
    assert result.retry_sequences == 1
    assert result.recovered_retries == 1
    assert result.retry_recovery_rate == 1.0
    assert result.step_count == report.metrics.step_count
    assert result.token_usage == report.metrics.token_usage


# 功能：验证 failed regression 保持 task failure，changed files 与 diff 不会覆盖 correctness
# 设计：构造 runtime success 但 regression criterion false 的 bug attempt，断言只做机械归因
def test_analyze_attempt_does_not_treat_small_diff_as_success(tmp_path: Path) -> None:
    analyzers = _analyzer_module()
    metadata = BenchmarkTaskSpec.model_validate(
        {
            "schema_version": 1,
            "task_id": "bugfix-001",
            "task_version": 1,
            "category": "bug_fixing",
            "criterion_groups": {
                "target_behavior": ["target-tests"],
                "regression": ["regression-tests"],
            },
        }
    )
    report = _report(
        "bugfix-001",
        task_success=False,
        failure_category=FailureCategory.TASK_FAILED,
        retry_count=0,
        criteria=[
            PublicCriterionResult(
                id="target-tests",
                kind="command_exit",
                passed=True,
            ),
            PublicCriterionResult(
                id="regression-tests",
                kind="command_exit",
                passed=False,
            ),
        ],
    )
    evaluation_output = tmp_path / "evaluation"
    _write_artifacts(evaluation_output, report, metric_line=None)
    attempt = BenchmarkAttempt(
        task_id="bugfix-001",
        category="bug_fixing",
        repeat=1,
        evaluation_output=evaluation_output,
        report=report,
    )

    result = analyzers.analyze_attempt(attempt, metadata)

    assert result.task_success is False
    assert result.group_results["target_behavior"] is True
    assert result.group_results["regression"] is False
    assert result.regression_introduced is True
    assert result.changed_files == 1


# 功能：验证 aggregate metrics 公开 scheduled 分母、category 成功率和 failure distribution
# 设计：从一成一败两个完整分析结果聚合，手算所有期望值避免复用实现逻辑
def test_aggregate_attempts_reports_fixed_denominators(tmp_path: Path) -> None:
    analyzers = _analyzer_module()
    metadata = _testgen_metadata()
    criteria = [
        PublicCriterionResult(id="generated-tests", kind="command_exit", passed=True),
        PublicCriterionResult(id="regression-tests", kind="command_exit", passed=True),
        PublicCriterionResult(id="coverage-threshold", kind="command_exit", passed=True),
    ]
    success_report = _report("testgen-001", criteria=criteria)
    first_output = tmp_path / "first"
    _write_artifacts(
        first_output,
        success_report,
        metric_line=(
            'KAMA_BENCH_METRICS_V1={"tests_passed":7,'
            '"tests_failed":0,"coverage_delta":12.5}'
        ),
    )
    success = analyzers.analyze_attempt(
        BenchmarkAttempt(
            task_id="testgen-001",
            category="test_generation",
            repeat=1,
            evaluation_output=first_output,
            report=success_report,
        ),
        metadata,
    )
    failed_report = _report(
        "testgen-001",
        attempt_id="attempt-b",
        task_success=False,
        failure_category=FailureCategory.TIMEOUT,
        criteria=[
            PublicCriterionResult(id="generated-tests", kind="command_exit", passed=False),
            PublicCriterionResult(id="regression-tests", kind="command_exit", passed=True),
            PublicCriterionResult(id="coverage-threshold", kind="command_exit", passed=False),
        ],
    )
    second_output = tmp_path / "second"
    _write_artifacts(
        second_output,
        failed_report,
        metric_line=(
            'KAMA_BENCH_METRICS_V1={"tests_passed":0,'
            '"tests_failed":1,"coverage_delta":0.0}'
        ),
    )
    failed = analyzers.analyze_attempt(
        BenchmarkAttempt(
            task_id="testgen-001",
            category="test_generation",
            repeat=2,
            evaluation_output=second_output,
            report=failed_report,
        ),
        metadata,
    )

    summary = analyzers.aggregate_attempts([success, failed])

    assert summary.overall.scheduled_attempts == 2
    assert summary.overall.successful_attempts == 1
    assert summary.overall.success_rate == 0.5
    assert summary.overall.total_input_tokens == 40
    assert summary.overall.median_wall_latency_ms == 25.0
    assert summary.overall.failure_categories == {"none": 1, "timeout": 1}
    assert summary.categories["test_generation"].scheduled_attempts == 2
    assert summary.categories["test_generation"].success_rate == 0.5


# 功能：验证缺失 journal 或 test-generation 数值 evidence 不会被静默解释为零
# 设计：分别删除已声明 sanity 的 journal 和 sentinel，断言 analyzer 以稳定 evidence error 失败
def test_analyzer_fails_closed_on_missing_required_evidence(tmp_path: Path) -> None:
    analyzers = _analyzer_module()
    metadata = _testgen_metadata()
    report = _report(
        "testgen-001",
        criteria=[
            PublicCriterionResult(id="generated-tests", kind="command_exit", passed=True),
            PublicCriterionResult(id="regression-tests", kind="command_exit", passed=True),
            PublicCriterionResult(id="coverage-threshold", kind="command_exit", passed=True),
        ],
    )
    missing_journal_output = tmp_path / "missing-journal"
    _write_artifacts(
        missing_journal_output,
        report,
        metric_line=(
            'KAMA_BENCH_METRICS_V1={"tests_passed":7,'
            '"tests_failed":0,"coverage_delta":12.5}'
        ),
    )
    attempt_dir = (
        missing_journal_output
        / "attempts"
        / report.task_id
        / report.attempt_id
    )
    (attempt_dir / "runtime" / "events.v2.jsonl").unlink()
    attempt = BenchmarkAttempt(
        task_id="testgen-001",
        category="test_generation",
        repeat=1,
        evaluation_output=missing_journal_output,
        report=report,
    )

    with pytest.raises(analyzers.BenchmarkAnalysisError, match="journal"):
        analyzers.analyze_attempt(attempt, metadata)

    missing_metrics_output = tmp_path / "missing-metrics"
    _write_artifacts(missing_metrics_output, report, metric_line=None)
    attempt = BenchmarkAttempt(
        task_id="testgen-001",
        category="test_generation",
        repeat=1,
        evaluation_output=missing_metrics_output,
        report=report,
    )

    with pytest.raises(analyzers.BenchmarkAnalysisError, match="numeric metrics"):
        analyzers.analyze_attempt(attempt, metadata)


# 功能：验证 timeout 且 trace 未形成时仍保留 attempt，不要求不存在的 workspace 与数值 evidence
# 设计：构造真实 Phase 8A timeout report 形状且不写任何 artifact，锁定 reliability 分母不会丢失
def test_analyzer_keeps_pre_artifact_timeout_as_reportable_attempt(
    tmp_path: Path,
) -> None:
    analyzers = _analyzer_module()
    metadata = _testgen_metadata()
    report = EvaluationReport(
        task_id="testgen-001",
        attempt_id="attempt-timeout",
        task_success=False,
        runtime_success=False,
        trace_sanity_passed=False,
        failure_category=FailureCategory.TIMEOUT,
        criteria=[],
        metrics=BasicMetrics(
            task_success=False,
            runtime_success=False,
            step_count=0,
            tool_count=0,
            retry_count=0,
            wall_latency_ms=30000.0,
            token_usage=TokenUsage(
                input_tokens=0,
                output_tokens=0,
                cache_tokens=0,
            ),
            failure_category=FailureCategory.TIMEOUT,
        ),
    )
    attempt = BenchmarkAttempt(
        task_id="testgen-001",
        category="test_generation",
        repeat=1,
        evaluation_output=tmp_path / "missing-evaluation-artifacts",
        report=report,
    )

    result = analyzers.analyze_attempt(attempt, metadata)

    assert result.task_success is False
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.group_results == {
        "generated_tests": None,
        "regression": None,
        "coverage": None,
    }
    assert result.changed_files is None
    assert result.tests_passed is None
    assert result.coverage_delta is None
