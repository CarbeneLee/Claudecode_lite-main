from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kama_claude.benchmark.orchestrator import BenchmarkAttempt
from kama_claude.benchmark.schema import BenchmarkCategory, BenchmarkTaskSpec
from kama_claude.eval.collector import WorkspaceManifest
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import TokenUsage

_METRIC_PREFIX = "KAMA_BENCH_METRICS_V1="
_DIFF_TRUNCATED = "... diff truncated ..."


class BenchmarkAnalysisError(RuntimeError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _PrivateNumericMetrics(_StrictModel):
    tests_passed: int | None = Field(default=None, ge=0)
    tests_failed: int | None = Field(default=None, ge=0)
    coverage_delta: float | None = None


class AttemptAnalysis(_StrictModel):
    task_id: str
    category: BenchmarkCategory
    repeat: int = Field(ge=1)
    task_success: bool
    runtime_success: bool
    trace_sanity_passed: bool
    failure_category: FailureCategory
    step_count: int = Field(ge=0)
    tool_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    wall_latency_ms: float = Field(ge=0)
    token_usage: TokenUsage
    group_results: dict[str, bool | None]
    regression_introduced: bool | None
    changed_files: int | None = Field(default=None, ge=0)
    source_changed_files: int | None = Field(default=None, ge=0)
    test_changed_files: int | None = Field(default=None, ge=0)
    diff_additions: int | None = Field(default=None, ge=0)
    diff_deletions: int | None = Field(default=None, ge=0)
    tests_passed: int | None = Field(default=None, ge=0)
    tests_failed: int | None = Field(default=None, ge=0)
    coverage_delta: float | None = None
    retry_sequences: int = Field(ge=0)
    recovered_retries: int = Field(ge=0)
    retry_recovery_rate: float | None = Field(default=None, ge=0, le=1)


class AggregateMetrics(_StrictModel):
    scheduled_attempts: int = Field(ge=1)
    successful_attempts: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    runtime_successful_attempts: int = Field(ge=0)
    median_wall_latency_ms: float = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cache_tokens: int = Field(ge=0)
    total_steps: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_retries: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    failure_categories: dict[str, int]


class BenchmarkMetrics(_StrictModel):
    overall: AggregateMetrics
    categories: dict[str, AggregateMetrics]


# 定位 Phase 8A 为指定 report 创建的唯一 attempt artifact 目录
def _attempt_dir(attempt: BenchmarkAttempt) -> Path:
    return (
        attempt.evaluation_output
        / "attempts"
        / attempt.task_id
        / attempt.report.attempt_id
    )


# 读取严格 workspace manifest，并将缺失或损坏统一分类为 evidence error
def _read_manifest(path: Path) -> WorkspaceManifest:
    try:
        return WorkspaceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise BenchmarkAnalysisError("workspace manifest evidence is invalid") from exc


# 比较初始和最终 manifest，返回所有内容或存在状态发生变化的路径
def _changed_paths(initial: WorkspaceManifest, final: WorkspaceManifest) -> set[str]:
    before = {item.path: item.sha256 for item in initial.files}
    after = {item.path: item.sha256 for item in final.files}
    return {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }


# 判断 changed path 是否属于常见测试目录或测试文件命名
def _is_test_path(path: str) -> bool:
    name = Path(path).name
    return (
        path.startswith(("tests/", "test/"))
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


# 从未截断 unified diff 机械统计新增和删除行
def _diff_counts(path: Path) -> tuple[int | None, int | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkAnalysisError("workspace diff evidence is invalid") from exc
    if _DIFF_TRUNCATED in text:
        return None, None
    additions = sum(
        line.startswith("+") and not line.startswith("+++")
        for line in text.splitlines()
    )
    deletions = sum(
        line.startswith("-") and not line.startswith("---")
        for line in text.splitlines()
    )
    return additions, deletions


# 从已通过 Phase 8A sanity 的 journal 计算发生真实 retry 的序列及恢复数
def _retry_evidence(path: Path) -> tuple[int, int]:
    failed_attempts: dict[str, int] = {}
    finished: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            event = row["event"]
            event_type = event.get("type")
            if event_type == "tool.call_failed":
                tool_id = str(event["tool_use_id"])
                failed_attempts[tool_id] = max(
                    failed_attempts.get(tool_id, 0),
                    int(event.get("attempt", 1)),
                )
            elif event_type == "tool.call_finished":
                finished.add(str(event["tool_use_id"]))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BenchmarkAnalysisError("journal evidence is invalid") from exc
    retry_sequences = {
        tool_id
        for tool_id, max_attempt in failed_attempts.items()
        if max_attempt > 1 or tool_id in finished
    }
    recovered = len(retry_sequences & finished)
    return len(retry_sequences), recovered


# 合并 private command stdout 中唯一、版本化且 allowlisted 的数值记录
def _private_numeric_metrics(path: Path) -> _PrivateNumericMetrics:
    values: dict[str, Any] = {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError
        for row in rows:
            stdout = row["stdout"]
            if not isinstance(stdout, str):
                raise TypeError
            for line in stdout.splitlines():
                if not line.startswith(_METRIC_PREFIX):
                    continue
                payload = json.loads(line.removeprefix(_METRIC_PREFIX))
                validated = _PrivateNumericMetrics.model_validate(payload)
                for key, value in validated.model_dump(exclude_none=True).items():
                    if key in values and values[key] != value:
                        raise ValueError("conflicting benchmark numeric metrics")
                    values[key] = value
        return _PrivateNumericMetrics.model_validate(values)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise BenchmarkAnalysisError("private numeric metrics evidence is invalid") from exc


# 将 Phase 8A report 与其 canonical artifacts 机械转换为单 attempt benchmark metrics
def analyze_attempt(
    attempt: BenchmarkAttempt,
    metadata: BenchmarkTaskSpec,
) -> AttemptAnalysis:
    if attempt.task_id != metadata.task_id or attempt.category != metadata.category:
        raise BenchmarkAnalysisError("attempt metadata identity mismatch")
    criterion_results = {criterion.id: criterion.passed for criterion in attempt.report.criteria}
    group_results: dict[str, bool | None] = {}
    for group, criterion_ids in metadata.criterion_groups.items():
        results = [criterion_results.get(criterion_id) for criterion_id in criterion_ids]
        group_results[group] = (
            all(results) if all(result is not None for result in results) else None
        )

    changed_files: int | None = None
    source_changed_files: int | None = None
    test_changed_files: int | None = None
    diff_additions: int | None = None
    diff_deletions: int | None = None
    retry_sequences = 0
    recovered_retries = 0
    numeric = _PrivateNumericMetrics()
    if attempt.report.trace_sanity_passed:
        root = _attempt_dir(attempt)
        runtime = root / "runtime"
        initial = _read_manifest(runtime / "initial-workspace.json")
        final = _read_manifest(runtime / "final-workspace.json")
        changed = _changed_paths(initial, final)
        changed_files = len(changed)
        test_changed_files = sum(_is_test_path(path) for path in changed)
        source_changed_files = sum(
            path.startswith("src/") and not _is_test_path(path) for path in changed
        )
        diff_additions, diff_deletions = _diff_counts(runtime / "workspace.diff")
        retry_sequences, recovered_retries = _retry_evidence(
            runtime / "events.v2.jsonl"
        )
        numeric = _private_numeric_metrics(root / "private" / "command-results.json")

    missing_testgen_metrics = (
        numeric.tests_passed is None
        or numeric.tests_failed is None
        or numeric.coverage_delta is None
    )
    if (
        attempt.report.trace_sanity_passed
        and metadata.category == "test_generation"
        and missing_testgen_metrics
    ):
        raise BenchmarkAnalysisError(
            "test generation numeric metrics evidence is missing"
        )
    recovery_rate = (
        recovered_retries / retry_sequences if retry_sequences > 0 else None
    )
    regression = group_results.get("regression")
    return AttemptAnalysis(
        task_id=attempt.task_id,
        category=attempt.category,
        repeat=attempt.repeat,
        task_success=attempt.report.task_success,
        runtime_success=attempt.report.runtime_success,
        trace_sanity_passed=attempt.report.trace_sanity_passed,
        failure_category=attempt.report.failure_category,
        step_count=attempt.report.metrics.step_count,
        tool_count=attempt.report.metrics.tool_count,
        retry_count=attempt.report.metrics.retry_count,
        wall_latency_ms=attempt.report.metrics.wall_latency_ms,
        token_usage=attempt.report.metrics.token_usage,
        group_results=group_results,
        regression_introduced=None if regression is None else not regression,
        changed_files=changed_files,
        source_changed_files=source_changed_files,
        test_changed_files=test_changed_files,
        diff_additions=diff_additions,
        diff_deletions=diff_deletions,
        tests_passed=numeric.tests_passed,
        tests_failed=numeric.tests_failed,
        coverage_delta=numeric.coverage_delta,
        retry_sequences=retry_sequences,
        recovered_retries=recovered_retries,
        retry_recovery_rate=recovery_rate,
    )


# 对一组固定 attempts 计算成功率、资源总量和失败分布
def _aggregate(rows: Sequence[AttemptAnalysis]) -> AggregateMetrics:
    if not rows:
        raise ValueError("cannot aggregate an empty benchmark attempt set")
    successful = sum(row.task_success for row in rows)
    failures = Counter(row.failure_category.value for row in rows)
    return AggregateMetrics(
        scheduled_attempts=len(rows),
        successful_attempts=successful,
        success_rate=successful / len(rows),
        runtime_successful_attempts=sum(row.runtime_success for row in rows),
        median_wall_latency_ms=float(median(row.wall_latency_ms for row in rows)),
        total_input_tokens=sum(row.token_usage.input_tokens for row in rows),
        total_output_tokens=sum(row.token_usage.output_tokens for row in rows),
        total_cache_tokens=sum(row.token_usage.cache_tokens for row in rows),
        total_steps=sum(row.step_count for row in rows),
        total_tool_calls=sum(row.tool_count for row in rows),
        total_retries=sum(row.retry_count for row in rows),
        timeout_count=sum(row.failure_category is FailureCategory.TIMEOUT for row in rows),
        failure_categories=dict(sorted(failures.items())),
    )


# 按实际出现的三类任务聚合 overall 与 category metrics
def aggregate_attempts(
    attempts: Sequence[AttemptAnalysis],
) -> BenchmarkMetrics:
    if not attempts:
        raise ValueError("cannot aggregate an empty benchmark attempt set")
    by_category: dict[str, list[AttemptAnalysis]] = {}
    for attempt in attempts:
        by_category.setdefault(attempt.category, []).append(attempt)
    return BenchmarkMetrics(
        overall=_aggregate(attempts),
        categories={
            category: _aggregate(rows)
            for category, rows in sorted(by_category.items())
        },
    )
