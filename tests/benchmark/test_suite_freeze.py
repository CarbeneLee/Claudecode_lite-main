from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.benchmark.report import (
    RepositoryState,
    capture_experiment_identity,
)
from kama_claude.benchmark.schema import load_suite

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TASKS_ROOT = _REPOSITORY_ROOT / "benchmarks" / "tasks"
_SUITE_PATH = (
    _REPOSITORY_ROOT / "benchmarks" / "suites" / "kama-coding-mvp-v1.json"
)
_FREEZE_PATH = (
    _REPOSITORY_ROOT
    / "benchmarks"
    / "suites"
    / "kama-coding-mvp-v1.freeze.json"
)
_TASK_IDS = (
    "bugfix-subtract",
    "feature-low-stock",
    "testgen-normalize-username",
    "bugfix-config-precedence",
    "feature-atomic-bulk-import",
    "testgen-quoted-query-parser",
    "bugfix-retry-state-idempotency",
    "feature-inventory-reservation-lifecycle",
    "testgen-dependency-planner",
)
_EXPECTED_GROUPS = {
    "bug_fixing": {"target_behavior", "regression"},
    "feature_implementation": {"target_behavior", "regression"},
    "test_generation": {"generated_tests", "regression", "coverage"},
}
_JUNK_NAMES = {
    ".DS_Store",
    ".coverage",
    ".pytest_cache",
    "__pycache__",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_HexDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _FrozenTask(_StrictModel):
    task_id: str
    task_version: int
    category: Literal[
        "bug_fixing",
        "feature_implementation",
        "test_generation",
    ]
    difficulty: Literal["easy", "medium", "challenging"]
    task_hash: _HexDigest
    grader_hash: _HexDigest
    reference_hash: _HexDigest


class _FreezeManifest(_StrictModel):
    schema_version: Literal[1]
    suite_id: Literal["kama-coding-mvp"]
    suite_version: Literal[1]
    runtime_suite: Literal["kama-coding-mvp-v1.json"]
    suite_hash: _HexDigest
    task_hash_algorithm: Literal["sha256-path-content-v1"]
    grader_hash_algorithm: Literal["sha256-path-content-v1"]
    reference_hash_algorithm: Literal["sha256-bytes-v1"]
    review_status: Literal["frozen_for_first_real_baseline"]
    tasks: tuple[_FrozenTask, ...]


# 对任意单文件内容计算稳定 SHA-256
def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 对指定 root 下的排序文件路径和内容计算稳定 SHA-256
def _path_content_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# 对 suite model 的 canonical JSON 计算与 baseline identity 相同的哈希
def _suite_hash(suite: object) -> str:
    manifest = suite.manifest  # type: ignore[attr-defined]
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 读取并严格校验独立 freeze evidence manifest
def _freeze_manifest() -> _FreezeManifest:
    return _FreezeManifest.model_validate_json(_FREEZE_PATH.read_text(encoding="utf-8"))


# 功能：验证 runtime suite 冻结为批准的九任务顺序与稳定 identity
# 设计：通过真实 load_suite 读取生产入口，防止 freeze evidence 与实际 baseline 输入分叉
def test_runtime_suite_freezes_approved_order_and_identity() -> None:
    suite = load_suite(_SUITE_PATH, _TASKS_ROOT)

    assert suite.manifest.suite_id == "kama-coding-mvp"
    assert suite.manifest.suite_version == 1
    assert tuple(task.metadata.task_id for task in suite.tasks) == _TASK_IDS


# 功能：验证 freeze manifest 的 suite、task、完整 grader bundle 与 reference hashes 匹配
# 设计：task hash 复用 baseline identity，grader 覆盖 criteria 和 hidden tests，reference 按 bytes
def test_freeze_manifest_matches_suite_and_content_hashes() -> None:
    suite = load_suite(_SUITE_PATH, _TASKS_ROOT)
    freeze = _freeze_manifest()
    identity = capture_experiment_identity(
        suite,
        repeats=1,
        model_id="freeze-audit-no-model-run",
        repository=RepositoryState(commit="0" * 40, dirty=False),
    )

    assert freeze.suite_id == suite.manifest.suite_id
    assert freeze.suite_version == suite.manifest.suite_version
    assert freeze.suite_hash == _suite_hash(suite)
    assert tuple(task.task_id for task in freeze.tasks) == _TASK_IDS
    for frozen, task in zip(freeze.tasks, suite.tasks, strict=True):
        assert frozen.task_version == task.metadata.task_version
        assert frozen.category == task.metadata.category
        assert frozen.task_hash == identity.task_hashes[frozen.task_id]
        private_root = task.task_dir / "private"
        hidden_root = private_root / "hidden_tests"
        assert frozen.grader_hash == _path_content_hash(
            task.task_dir,
            [
                private_root / "grader.json",
                *(
                    path
                    for path in hidden_root.rglob("*")
                    if path.is_file()
                ),
            ],
        )
        assert frozen.reference_hash == _file_hash(
            task.task_dir / "private" / "reference.patch"
        )


# 功能：验证九任务形成三类别乘三难度且 criterion 与 regression 语义一致
# 设计：按 canonical groups 和真实 grader argv 审计，不把 task-local criterion ID 当聚合指标
def test_frozen_suite_cross_task_contracts_are_consistent() -> None:
    suite = load_suite(_SUITE_PATH, _TASKS_ROOT)
    freeze = _freeze_manifest()
    pairs: Counter[tuple[str, str]] = Counter()

    for frozen, task in zip(freeze.tasks, suite.tasks, strict=True):
        pairs[(frozen.category, frozen.difficulty)] += 1
        assert set(task.metadata.criterion_groups) == _EXPECTED_GROUPS[
            task.metadata.category
        ]
        assert task.metadata.criterion_groups["regression"] == [
            "regression-tests"
        ]
        criteria = {
            criterion.id: criterion
            for criterion in task.evaluation_task.private.criteria
        }
        regression = criteria["regression-tests"]
        assert regression.kind == "command_exit"
        assert list(regression.argv[:5]) == [
            "python",
            "-m",
            "pytest",
            "-q",
            "tests",
        ]
        assert regression.expected_exit_code == 0
        for criterion in criteria.values():
            assert criterion.kind == "command_exit"
            assert criterion.expected_exit_code == 0

    expected_pairs = {
        (category, difficulty)
        for category in _EXPECTED_GROUPS
        for difficulty in ("easy", "medium", "challenging")
    }
    assert set(pairs) == expected_pairs
    assert set(pairs.values()) == {1}


# 功能：验证 public 无 private oracle 泄漏且 private authoring evidence 完整
# 设计：聚合每个 authoring manifest 的显式 markers、路径引用和 security boundary
def test_frozen_suite_public_private_boundary_is_complete() -> None:
    suite = load_suite(_SUITE_PATH, _TASKS_ROOT)

    for task in suite.tasks:
        public_root = task.task_dir / "public"
        private_root = task.task_dir / "private"
        authoring_path = private_root / "authoring" / "manifest.json"
        authoring = json.loads(authoring_path.read_text(encoding="utf-8"))

        for path in task.task_dir.rglob("*"):
            assert path.name not in _JUNK_NAMES
            assert path.suffix not in {".pyc", ".pyo"}
            assert not path.is_symlink()
        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(public_root.rglob("*"))
            if path.is_file()
        )
        for marker in authoring["forbidden_public_markers"]:
            assert marker not in public_text
        for marker in authoring["forbidden_issue_markers"]:
            assert marker not in task.evaluation_task.public.goal
        for marker in (
            "hidden_tests",
            "reference.patch",
            "authoring-review",
            "expected_source",
            "wrong-",
        ):
            assert marker not in public_text

        assert (private_root / "grader.json").is_file()
        assert (private_root / "hidden_tests").is_dir()
        assert (private_root / "reference.patch").is_file()
        assert (private_root / "authoring-review.md").is_file()
        assert authoring["review_status"] == (
            "author_validated_pending_external_review"
        )
        assert authoring["security"] == {
            "trusted_fixture": True,
            "offline": True,
            "data_flow_isolation_only": True,
            "process_isolation_is_sandbox": False,
        }
        evidence_paths = {
            authoring["alternative_patch"],
            *(
                probe["patch"]
                for probe in authoring["wrong_patch_probes"]
            ),
            *(
                evidence.split("::", maxsplit=1)[0]
                for row in authoring["requirement_test_matrix"]
                for evidence in row["hidden_evidence"]
            ),
        }
        for relative in evidence_paths:
            evidence = (private_root / relative).resolve(strict=True)
            assert evidence.is_relative_to(private_root.resolve(strict=True))
            assert evidence.is_file()


# 功能：验证所有 frozen task 声明三-copy candidate evidence 且 probe 数符合难度
# 设计：完整执行一致性由三个 authoring matrix tests 提供，本测试防止 freeze 遗漏其输入证据
def test_frozen_suite_determinism_evidence_is_complete() -> None:
    suite = load_suite(_SUITE_PATH, _TASKS_ROOT)
    minimum_probes = {
        "easy": 2,
        "medium": 3,
        "challenging": 4,
    }

    for frozen, task in zip(_freeze_manifest().tasks, suite.tasks, strict=True):
        authoring = json.loads(
            (
                task.task_dir / "private" / "authoring" / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert authoring["determinism_runs"] == 3
        assert len(authoring["wrong_patch_probes"]) >= minimum_probes[
            frozen.difficulty
        ]
        assert (
            task.task_dir / "private" / authoring["alternative_patch"]
        ).is_file()
