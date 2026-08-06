from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from kama_claude.benchmark.analyzers import aggregate_attempts, analyze_attempt
from kama_claude.benchmark.experiment import (
    EvidenceMode,
    ExperimentIdentityMismatch,
    ObservedExperimentIdentity,
    RepositoryIdentity,
    build_verified_experiment_identity,
    capture_declared_identity,
    collect_observed_identity,
    load_experiment_profile,
    require_identity_match,
    resolve_experiment_credential,
    scoped_experiment_environment,
    validate_experiment_output,
    write_declared_experiment,
    write_invalid_experiment,
)
from kama_claude.benchmark.orchestrator import Evaluator, run_suite
from kama_claude.benchmark.report import (
    RepositoryState,
    build_baseline_report,
    probe_repository,
    write_baseline_report,
)
from kama_claude.benchmark.schema import load_suite
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.report import EvaluationReport


# 解析唯一 profile-driven run 命令，不暴露 ad-hoc runtime 或 comparison 参数
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kama-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--experiment", required=True)
    run.add_argument("--output", required=True)
    return parser.parse_args(argv)


# 仅为语义自洽的 timeout report 选择 partial identity evidence 模式
def _identity_evidence_mode(report: EvaluationReport) -> EvidenceMode:
    metrics = report.metrics
    tokens = metrics.token_usage
    if report.failure_category is not FailureCategory.TIMEOUT:
        return EvidenceMode.COMPLETE
    if (
        report.runtime_success
        or report.task_success
        or report.trace_sanity_passed
        or report.criteria
        or metrics.runtime_success
        or metrics.task_success
        or metrics.failure_category is not FailureCategory.TIMEOUT
        or metrics.step_count != 0
        or metrics.tool_count != 0
        or metrics.retry_count != 0
        or tokens.input_tokens != 0
        or tokens.output_tokens != 0
        or tokens.cache_tokens != 0
    ):
        raise ValueError("timeout report is inconsistent with partial evidence")
    return EvidenceMode.TIMEOUT_PARTIAL


# 运行 versioned experiment、逐 attempt 验证身份并只为 valid matrix 写 baseline
async def execute_experiment(
    args: argparse.Namespace,
    *,
    evaluator: Evaluator | None = None,
    repository: RepositoryState | None = None,
    repository_root: Path | str | None = None,
    installed_sdk_version: str | None = None,
) -> int:
    loaded = load_experiment_profile(args.experiment)
    suite = load_suite(loaded.suite_path, loaded.tasks_root)
    root = Path.cwd() if repository_root is None else Path(repository_root)
    observed_repository = (
        probe_repository(root) if repository is None else repository
    )
    sdk_version = (
        version(loaded.profile.provider.sdk_distribution)
        if installed_sdk_version is None
        else installed_sdk_version
    )
    declared = capture_declared_identity(
        loaded,
        repository_root=root,
        repository=RepositoryIdentity(
            commit=observed_repository.commit,
            dirty=observed_repository.dirty,
        ),
        installed_sdk_version=sdk_version,
    )
    credential = resolve_experiment_credential(loaded.profile, root)
    output = validate_experiment_output(
        args.output,
        root,
        loaded.profile.artifacts,
    )
    write_declared_experiment(output, declared)
    evaluate = evaluate_task if evaluator is None else evaluator
    observations: list[ObservedExperimentIdentity] = []

    # 在真实 Phase 8A evaluator 后只读 artifact，identity mismatch 不进入 analyzer/metrics
    async def evaluate_and_verify(
        task_dir: Path | str,
        evaluation_output: Path | str,
    ) -> EvaluationReport:
        report = await evaluate(task_dir, evaluation_output)
        attempt_root = (
            Path(evaluation_output)
            / "attempts"
            / report.task_id
            / report.attempt_id
        )
        try:
            observed = collect_observed_identity(
                attempt_root,
                evidence_mode=_identity_evidence_mode(report),
            )
            require_identity_match(declared, observed)
        except ExperimentIdentityMismatch:
            raise
        except ValueError as exc:
            raise ExperimentIdentityMismatch(["identity_evidence"]) from exc
        observations.append(observed)
        return report

    try:
        with scoped_experiment_environment(
            loaded.profile,
            credential=credential,
        ):
            run = await run_suite(
                suite,
                output / "run",
                repeats=loaded.profile.schedule.repeats,
                evaluator=evaluate_and_verify,
            )
    except ExperimentIdentityMismatch as exc:
        write_invalid_experiment(output, list(exc.mismatches))
        return 2
    metadata_by_id = {task.metadata.task_id: task.metadata for task in suite.tasks}
    attempts = [
        analyze_attempt(attempt, metadata_by_id[attempt.task_id])
        for attempt in run.attempts
    ]
    identity = build_verified_experiment_identity(declared, observations)
    report = build_baseline_report(
        identity,
        attempts,
        aggregate_attempts(attempts),
    )
    write_baseline_report(output, report)
    return 0 if report.metrics.overall.successful_attempts == len(attempts) else 1


# 执行 kama-bench 并用退出码区分完整成功、task failure 与 invalid/infra error
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(execute_experiment(args))
    except (OSError, ValueError, RuntimeError):
        print("benchmark failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
