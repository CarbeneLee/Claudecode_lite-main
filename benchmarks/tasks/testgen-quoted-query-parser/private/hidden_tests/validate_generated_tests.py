from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SOURCE_FILES = (
    "__init__.py",
    "errors.py",
    "models.py",
    "parser.py",
    "tokenizer.py",
)
_MUTANTS = (
    "splits_quoted_delimiter",
    "preserves_escapes",
    "drops_empty_quoted",
    "accepts_unclosed_quote",
)


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


# 在隔离进程中运行候选测试，避免 import cache 跨 mutant 污染
def _run_tests(workspace: Path) -> int:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
        env=environment,
    )
    return result.returncode


# 验证候选测试能杀死每个公开合同对应的单缺陷实现
def _rejects_curated_mutants(root: Path) -> bool:
    expected_root = Path(__file__).with_name("expected_source")
    mutants_root = Path(__file__).with_name("mutants")
    for mutant_name in _MUTANTS:
        with tempfile.TemporaryDirectory(prefix="kama-query-mutant-") as temp:
            mutant_workspace = Path(temp)
            shutil.copytree(root / "tests", mutant_workspace / "tests")
            shutil.copytree(
                expected_root / "querylang",
                mutant_workspace / "querylang",
            )
            for replacement in (mutants_root / mutant_name).glob("*.py"):
                shutil.copy2(
                    replacement,
                    mutant_workspace / "querylang" / replacement.name,
                )
            if _run_tests(mutant_workspace) == 0:
                return False
    return True


# 验证 source integrity、正确实现通过和四个行为 mutant 被拒绝
def main() -> int:
    root = Path.cwd()
    if not _source_is_unchanged(root):
        return 1
    if _run_tests(root) != 0:
        return 1
    return 0 if _rejects_curated_mutants(root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
