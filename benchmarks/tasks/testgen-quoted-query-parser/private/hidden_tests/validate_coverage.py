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
    "parser.py",
    "tokenizer.py",
)
_TARGET_LINES = {
    "parser.py": frozenset(
        {
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
        }
    ),
    "tokenizer.py": frozenset(
        {
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
        }
    ),
}
_BASELINE_COVERED_LINES = 27
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
    actual_root = root / "querylang"
    expected_root = Path(__file__).with_name("expected_source") / "querylang"
    actual_files = tuple(
        path.name for path in sorted(actual_root.glob("*.py"))
    )
    if actual_files != _SOURCE_FILES:
        return False
    return all(
        _digest(actual_root / name) == _digest(expected_root / name)
        for name in _SOURCE_FILES
    )


# 收集 parser/tokenizer 中与公开 boundary 和 error contract 相关的执行行
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


# 验证候选测试相对 pristine suite 增加关键 branch/error coverage
def main() -> int:
    root = Path.cwd()
    if not _source_is_unchanged(root):
        return 1
    sys.path.insert(0, str(root))
    collector = _Collector()
    tracer = trace.Trace(count=True, trace=False)
    exit_code = tracer.runfunc(
        pytest.main,
        ["-q", "tests"],
        [collector],
    )
    source_root = (root / "querylang").resolve()
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
