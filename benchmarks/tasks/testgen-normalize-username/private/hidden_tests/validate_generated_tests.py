from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_MUTANTS = (
    "allows_empty.py",
    "allows_internal_spaces.py",
    "preserves_underscores.py",
)


class _Collector:
    # 初始化测试收集计数器
    def __init__(self) -> None:
        self.count = 0

    # 记录 pytest 实际收集的测试数量
    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.count = len(session.items)


# 验证候选测试能够杀死每个单缺陷实现
def _rejects_curated_mutants(root: Path) -> bool:
    mutants_root = Path(__file__).with_name("mutants")
    for mutant_name in _MUTANTS:
        with tempfile.TemporaryDirectory(prefix="kama-testgen-mutant-") as temp:
            mutant_root = Path(temp)
            tests_root = mutant_root / "tests"
            tests_root.mkdir()
            shutil.copy2(root / "tests" / "test_text_utils.py", tests_root)
            shutil.copy2(mutants_root / mutant_name, mutant_root / "text_utils.py")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_text_utils.py",
                ],
                cwd=mutant_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return False
    return True


# 验证源码未变、候选测试通过且覆盖所有公开要求
def main() -> int:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    target = root / "text_utils.py"
    expected = Path(__file__).with_name("expected_text_utils.py")
    if target.read_text(encoding="utf-8") != expected.read_text(encoding="utf-8"):
        return 1
    collector = _Collector()
    exit_code = pytest.main(["-q", "tests/test_text_utils.py"], plugins=[collector])
    return (
        0
        if exit_code == 0
        and collector.count >= 4
        and _rejects_curated_mutants(root)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
