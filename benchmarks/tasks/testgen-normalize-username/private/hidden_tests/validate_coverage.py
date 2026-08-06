from __future__ import annotations

import json
import sys
import trace
from pathlib import Path

import pytest

_TARGET_LINES = frozenset({2, 3, 4, 5, 6, 7})
_BASELINE_COVERED_LINES = 4
_MINIMUM_DELTA = 30.0


class _Collector:
    def __init__(self) -> None:
        self.count = 0

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.count = len(session.items)


def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    target = (root / "text_utils.py").resolve()
    expected = Path(__file__).with_name("expected_text_utils.py")
    if target.read_text(encoding="utf-8") != expected.read_text(encoding="utf-8"):
        return 1
    collector = _Collector()
    tracer = trace.Trace(count=True, trace=False)
    exit_code = tracer.runfunc(
        pytest.main,
        ["-q", "tests/test_text_utils.py"],
        [collector],
    )
    covered_lines = {
        line
        for (filename, line), count in tracer.results().counts.items()
        if count > 0 and Path(filename).resolve() == target and line in _TARGET_LINES
    }
    coverage_delta = (
        (len(covered_lines) - _BASELINE_COVERED_LINES)
        / len(_TARGET_LINES)
        * 100.0
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
        and collector.count >= 4
        and coverage_delta >= _MINIMUM_DELTA
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
