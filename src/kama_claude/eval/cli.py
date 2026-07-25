from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory


# 解析唯一的 Phase 8A run 子命令，不暴露 provider、dataset 或 comparison 参数
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kama-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--output", required=True)
    return parser.parse_args(argv)


# 执行单 task/attempt 并用退出码区分成功、任务失败和 infrastructure failure
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = asyncio.run(evaluate_task(args.task, args.output))
    except (OSError, ValueError):
        print("evaluation failed", file=sys.stderr)
        return 2
    if report.task_success:
        return 0
    if report.failure_category in {
        FailureCategory.INFRA_ERROR,
        FailureCategory.GRADER_ERROR,
        FailureCategory.TRACE_INVALID,
    }:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
