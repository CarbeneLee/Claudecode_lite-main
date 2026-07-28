from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kama_claude.benchmark.schema import (
    BenchmarkCategory,
    LoadedBenchmarkSuite,
)
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.report import EvaluationReport

type Evaluator = Callable[
    [Path | str, Path | str],
    Awaitable[EvaluationReport],
]


@dataclass(frozen=True)
class BenchmarkAttempt:
    task_id: str
    category: BenchmarkCategory
    repeat: int
    evaluation_output: Path
    report: EvaluationReport


@dataclass(frozen=True)
class BenchmarkRun:
    suite_id: str
    suite_version: int
    repeats: int
    output_root: Path
    attempts: tuple[BenchmarkAttempt, ...]


# 验证当前内部 benchmark 只允许一到三次独立 repeat
def _validate_repeats(repeats: int) -> None:
    if isinstance(repeats, bool) or not 1 <= repeats <= 3:
        raise ValueError("repeats must be between 1 and 3")


# 为单 task/repeat 生成唯一的 Phase 8A evaluation artifact 目录
def _evaluation_output(root: Path, task_id: str, repeat: int) -> Path:
    return root / "tasks" / task_id / f"repeat-{repeat:02d}" / "evaluation"


# 按 suite 顺序串行调用 Phase 8A evaluate_task，并保留 caller cancellation
async def run_suite(
    suite: LoadedBenchmarkSuite,
    output_root: Path | str,
    *,
    repeats: int,
    evaluator: Evaluator | None = None,
) -> BenchmarkRun:
    _validate_repeats(repeats)
    root = Path(output_root).resolve(strict=False)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ValueError("benchmark output root must be new and writable") from exc

    evaluate = evaluate_task if evaluator is None else evaluator
    attempts: list[BenchmarkAttempt] = []
    for task in suite.tasks:
        for repeat in range(1, repeats + 1):
            evaluation_output = _evaluation_output(root, task.metadata.task_id, repeat)
            report = await evaluate(task.task_dir, evaluation_output)
            if report.task_id != task.metadata.task_id:
                raise ValueError("evaluation report task identity mismatch")
            attempts.append(
                BenchmarkAttempt(
                    task_id=task.metadata.task_id,
                    category=task.metadata.category,
                    repeat=repeat,
                    evaluation_output=evaluation_output,
                    report=report,
                )
            )
    return BenchmarkRun(
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.suite_version,
        repeats=repeats,
        output_root=root,
        attempts=tuple(attempts),
    )
