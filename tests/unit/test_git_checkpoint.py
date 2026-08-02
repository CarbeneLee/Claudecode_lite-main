from __future__ import annotations

import pytest

from kama_claude.core.git.checkpoint import (
    GitCheckpoint,
    checkpoint_ref,
    pre_run_ref,
)


# 功能：验证 GitCheckpoint 模型字段完整且不可变（快照记录契约）
# 设计：构造全字段实例断言值保留，修改尝试抛 FrozenInstanceError 证明 frozen
def test_checkpoint_fields_and_frozen() -> None:
    cp = GitCheckpoint(
        run_id="r1",
        step=0,
        label="baseline",
        kind="internal",
        commit_sha="a" * 40,
        ref="refs/kama/checkpoints/r1/0",
        short_sha="a" * 7,
        diff_stat="f.txt | 1 +",
        ts="2026-08-02T00:00:00+00:00",
        dirty=False,
    )
    assert cp.run_id == "r1"
    assert cp.step == 0
    assert cp.label == "baseline"
    assert cp.kind == "internal"
    assert cp.commit_sha == "a" * 40
    assert cp.ref == "refs/kama/checkpoints/r1/0"
    assert cp.short_sha == "a" * 7
    assert cp.diff_stat == "f.txt | 1 +"
    assert cp.ts == "2026-08-02T00:00:00+00:00"
    assert cp.dirty is False
    with pytest.raises(Exception):
        cp.dirty = True  # type: ignore[misc]


# 功能：验证 checkpoint ref 路径生成（命名空间/run/step 三层）
# 设计：参数化不同 run_id/step 断言 ref 形状，供 manager 与测试共用
@pytest.mark.parametrize(
    ("namespace", "run_id", "step", "expected"),
    [
        ("refs/kama", "r1", 0, "refs/kama/checkpoints/r1/0"),
        ("refs/kama", "run-42", 3, "refs/kama/checkpoints/run-42/3"),
        ("refs/ci", "r1", 10, "refs/ci/checkpoints/r1/10"),
    ],
)
def test_checkpoint_ref_path(namespace: str, run_id: str, step: int, expected: str) -> None:
    assert checkpoint_ref(namespace, run_id, step) == expected


# 功能：验证 pre-run 快照 ref 路径生成（独立于 step 链）
# 设计：参数化命名空间断言形状，与 checkpoint 链 ref 区分
@pytest.mark.parametrize(
    ("namespace", "run_id", "expected"),
    [
        ("refs/kama", "r1", "refs/kama/r1/pre-run"),
        ("refs/ci", "run-42", "refs/ci/run-42/pre-run"),
    ],
)
def test_pre_run_ref_path(namespace: str, run_id: str, expected: str) -> None:
    assert pre_run_ref(namespace, run_id) == expected
