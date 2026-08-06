from __future__ import annotations

import hashlib
import json
import sys
import trace
from pathlib import Path

import pytest

_SOURCE_FILES = (
    "__init__.py",
    "errors.py",
    "models.py",
    "planner.py",
    "validation.py",
)
_TARGET_LINES = {
    "validation.py": frozenset({15, 16, 19, 20}),
    "planner.py": frozenset({18, 19, 20, 21, 25, 26}),
}
_BASELINE_COVERED_LINES = 6
_MINIMUM_COVERED_RATIO = 0.9


class _Collector:
    # 初始化测试收集计数器
    def __init__(self) -> None:
        self.count = 0

    # 记录 pytest 实际收集的测试数量
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.count = len(session.items)


# 计算文件内容的稳定 SHA-256
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 验证 production 文件集合与每个内容 hash 均未变化
def _source_is_unchanged(root: Path) -> bool:
    actual_root = root / "dependency_planner"
    expected_root = (
        Path(__file__).with_name("expected_source") / "dependency_planner"
    )
    actual_files = tuple(path.name for path in sorted(actual_root.glob("*.py")))
    if actual_files != _SOURCE_FILES:
        return False
    return all(
        _digest(actual_root / name) == _digest(expected_root / name)
        for name in _SOURCE_FILES
    )


# 收集 validation/planner 中关键分支与异常合同的执行行
def _covered_target_lines(
    counts: dict[tuple[str, int], int],
    source_root: Path,
) -> set[tuple[str, int]]:
    covered: set[tuple[str, int]] = set()
    for (filename, line), count in counts.items():
        path = Path(filename).resolve()
        if count <= 0 or path.parent != source_root:
            continue
        if line in _TARGET_LINES.get(path.name, frozenset()):
            covered.add((path.name, line))
    return covered


# 验证候选测试增加关键验证、循环和错误分支覆盖
def main() -> int:
    root = Path.cwd()
    if not _source_is_unchanged(root):
        return 1
    sys.path.insert(0, str(root))
    collector = _Collector()
    tracer = trace.Trace(count=True, trace=False)
    exit_code = tracer.runfunc(pytest.main, ["-q", "tests"], [collector])
    source_root = (root / "dependency_planner").resolve()
    covered = _covered_target_lines(tracer.results().counts, source_root)
    target_count = sum(len(lines) for lines in _TARGET_LINES.values())
    coverage_delta = (
        (len(covered) - _BASELINE_COVERED_LINES) / target_count * 100.0
    )
    metrics = {
        "tests_passed": collector.count if exit_code == 0 else 0,
        "tests_failed": 0 if exit_code == 0 else 1,
        "coverage_delta": round(coverage_delta, 3),
    }
    print("KAMA_BENCH_METRICS_V1=" + json.dumps(metrics, sort_keys=True))
    return (
        0
        if exit_code == 0
        and len(covered) / target_count >= _MINIMUM_COVERED_RATIO
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
