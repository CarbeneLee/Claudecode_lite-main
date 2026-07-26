from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from kama_claude.benchmark.analyzers import aggregate_attempts, analyze_attempt
from kama_claude.benchmark.orchestrator import Evaluator, run_suite
from kama_claude.benchmark.report import (
    RepositoryState,
    build_baseline_report,
    capture_experiment_identity,
    probe_repository,
    write_baseline_report,
)
from kama_claude.benchmark.schema import load_suite
from kama_claude.core.config import get_config


# 解析唯一固定 suite run 命令，不暴露 provider、model、prompt 或 comparison 参数
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kama-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("--tasks-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", required=True, type=int, choices=(1, 2, 3))
    return parser.parse_args(argv)


# 运行固定 suite、分析 Phase 8A artifacts 并写入 internal baseline report
async def execute_benchmark(
    args: argparse.Namespace,
    *,
    evaluator: Evaluator | None = None,
    repository: RepositoryState | None = None,
    model_id: str | None = None,
) -> int:
    suite = load_suite(args.suite, args.tasks_root)
    observed_repository = (
        probe_repository(Path.cwd()) if repository is None else repository
    )
    observed_model = get_config().llm.default_model if model_id is None else model_id
    identity = capture_experiment_identity(
        suite,
        repeats=args.repeats,
        model_id=observed_model,
        repository=observed_repository,
    )
    run = await run_suite(
        suite,
        args.output,
        repeats=args.repeats,
        evaluator=evaluator,
    )
    metadata_by_id = {task.metadata.task_id: task.metadata for task in suite.tasks}
    attempts = [
        analyze_attempt(attempt, metadata_by_id[attempt.task_id])
        for attempt in run.attempts
    ]
    report = build_baseline_report(
        identity,
        attempts,
        aggregate_attempts(attempts),
    )
    write_baseline_report(args.output, report)
    return 0 if report.metrics.overall.successful_attempts == len(attempts) else 1


# 执行 kama-bench 并用退出码区分完整成功、task failure 与 experiment error
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(execute_benchmark(args))
    except (OSError, ValueError, RuntimeError):
        print("benchmark failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
