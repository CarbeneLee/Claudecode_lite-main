from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kama_claude.eval.failure import FailureCategory, select_failure_category
from kama_claude.eval.schema import (
    CommandExitCriterion,
    FileContainsCriterion,
    PublicTaskSpec,
)
from kama_claude.eval.task import load_task


# 功能：验证公开任务只接受 id、goal、workspace_fixture 和 timeout_s 四个冻结字段
# 设计：从最小合法输入逐一注入 runtime 与 grader 字段，防止 Evaluation 演化成第二套运行配置
def test_public_task_schema_is_minimal_and_strict() -> None:
    raw = {
        "id": "bugfix-pytest",
        "goal": "Fix calculator.py.",
        "workspace_fixture": "public/workspace",
        "timeout_s": 300.0,
    }

    task = PublicTaskSpec.model_validate(raw)

    assert task.model_dump(mode="json") == raw
    for forbidden in ("grader", "provider", "model", "tools", "permission", "max_steps"):
        with pytest.raises(ValidationError):
            PublicTaskSpec.model_validate({**raw, forbidden: {}})


# 功能：验证 private grader criterion 仅支持文件状态与 argv 命令且拒绝 shell 字符串
# 设计：直接构造允许与禁止的判别类型，锁定 Phase 8A grader 能力而不依赖 runner
def test_private_criteria_are_rule_only_and_command_uses_argv() -> None:
    file_criterion = FileContainsCriterion(
        id="contains-result",
        kind="file_contains",
        path="result.txt",
        text="done",
    )
    command_criterion = CommandExitCriterion(
        id="tests-pass",
        kind="command_exit",
        argv=["python", "-m", "pytest", "-q"],
        expected_exit_code=0,
    )

    assert file_criterion.kind == "file_contains"
    assert command_criterion.argv[0] == "python"
    with pytest.raises(ValidationError):
        CommandExitCriterion.model_validate(
            {
                "id": "unsafe",
                "kind": "command_exit",
                "command": "pytest -q",
                "expected_exit_code": 0,
            }
        )


# 功能：验证 task loader 分离 public task、private grader 和 hidden tests 并拒绝 fixture 逃逸
# 设计：使用真实目录与 JSON 文件覆盖 canonical containment，确保 private 数据不会被折叠进公开模型
def test_load_task_separates_public_and_private_data(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-a"
    public_dir = task_dir / "public"
    private_dir = task_dir / "private"
    workspace = public_dir / "workspace"
    hidden_tests = private_dir / "hidden_tests"
    workspace.mkdir(parents=True)
    hidden_tests.mkdir(parents=True)
    (workspace / "input.txt").write_text("input", encoding="utf-8")
    (hidden_tests / "test_hidden.py").write_text("def test_hidden(): pass\n", encoding="utf-8")
    (public_dir / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Create result.txt.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    (private_dir / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "result-exists",
                        "kind": "file_exists",
                        "path": "result.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_task(task_dir)

    assert loaded.public.id == "task-a"
    assert loaded.workspace_fixture == workspace.resolve()
    assert loaded.hidden_tests == hidden_tests.resolve()
    assert loaded.private.criteria[0].id == "result-exists"
    assert "criteria" not in loaded.public.model_dump(mode="json")

    outside = tmp_path / "outside"
    outside.mkdir()
    (public_dir / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Escape.",
                "workspace_fixture": "../outside",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid evaluation file"):
        load_task(task_dir)


# 功能：验证 public workspace 内部 symlink 会在 worker 启动前被 task loader 拒绝
# 设计：让公开 fixture 的链接直接指向 private secret，锁定 data-flow 隔离发生在复制与执行之前
def test_load_task_rejects_symlink_inside_public_workspace(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-a"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    secret = private / "secret.txt"
    secret.write_text("hidden", encoding="utf-8")
    (workspace / "leak.txt").symlink_to(secret)
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Create result.txt.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "result-exists",
                        "kind": "file_exists",
                        "path": "result.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workspace fixture contains a symlink"):
        load_task(task_dir)


# 功能：验证 failure category 使用中央优先级且每个枚举值只出现一次
# 设计：将全部类别逆序输入同一选择器，断言最高优先级胜出并锁定 priority 的完备唯一性
def test_failure_category_priority_is_centralized_and_deterministic() -> None:
    assert select_failure_category(
        {
            FailureCategory.TASK_FAILED,
            FailureCategory.RUNTIME_FAILED,
            FailureCategory.TRACE_INVALID,
            FailureCategory.CANCELLED,
            FailureCategory.TIMEOUT,
            FailureCategory.GRADER_ERROR,
            FailureCategory.INFRA_ERROR,
        }
    ) is FailureCategory.INFRA_ERROR
    assert select_failure_category(set()) is FailureCategory.NONE
