from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


# 加载 repository grounding 模块，使缺失实现以明确 RED 失败呈现
def _grounding() -> ModuleType:
    return importlib.import_module("kama_claude.core.grounding")


# 功能：验证 root 与 target-local explicit sources 按目录作用域全部保留
# 设计：构造多 target 与同目录三种 compatibility 文件，断言窄规则不泄漏且没有 filename winner
def test_instruction_loader_preserves_sources_per_target_scope(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("root-agents\n", encoding="utf-8")
    (tmp_path / "AGENT.md").write_text("root-agent\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("root-claude\n", encoding="utf-8")
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "AGENTS.md").write_text("pkg-agents\n", encoding="utf-8")
    (package / "CLAUDE.md").write_text("pkg-claude\n", encoding="utf-8")
    sibling = tmp_path / "tests"
    sibling.mkdir()
    (sibling / "AGENT.md").write_text("tests-agent\n", encoding="utf-8")

    effective = _grounding().RepositoryInstructionLoader(tmp_path).load(
        ["src/pkg/module.py", "tests/test_module.py"]
    )

    assert [source.source_path for source in effective.root_sources] == [
        "AGENTS.md",
        "AGENT.md",
        "CLAUDE.md",
    ]
    assert [
        source.source_path
        for source in effective.sources_by_target["src/pkg/module.py"]
    ] == [
        "AGENTS.md",
        "AGENT.md",
        "CLAUDE.md",
        "src/pkg/AGENTS.md",
        "src/pkg/CLAUDE.md",
    ]
    assert [
        source.source_path
        for source in effective.sources_by_target["tests/test_module.py"]
    ] == ["AGENTS.md", "AGENT.md", "CLAUDE.md", "tests/AGENT.md"]
    assert effective.sources_by_target["src/pkg/module.py"][-1].content == (
        "pkg-claude\n"
    )
    assert all(source.kind == "explicit_instruction" for source in effective.root_sources)


# 功能：验证 instruction target 拒绝绝对路径和 workspace escape
# 设计：直接传入两种非法 logical path，锁定 loader 只沿 canonical workspace 内目录遍历
@pytest.mark.parametrize("target", ["/tmp/outside.py", "../outside.py"])
def test_instruction_loader_rejects_outside_target(
    tmp_path: Path,
    target: str,
) -> None:
    loader = _grounding().RepositoryInstructionLoader(tmp_path)

    with pytest.raises(ValueError, match="workspace-relative"):
        loader.load([target])


# 功能：验证 task-relevant snapshot 忽略无关文件但检测 evidence 和 planned-create 状态变化
# 设计：依次改动未纳入路径、已读源码和必须不存在的新 target，观察同一 predicate 的三种结果
def test_snapshot_current_only_depends_on_relevant_content(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("rules", encoding="utf-8")
    (tmp_path / "evidence.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "unrelated.log").write_text("first", encoding="utf-8")
    grounding = _grounding()
    instructions = grounding.RepositoryInstructionLoader(tmp_path).load([])
    builder = grounding.SnapshotBuilder(tmp_path)
    snapshot = builder.capture(
        instruction_sources=instructions.root_sources,
        grounding_paths=["evidence.py"],
        planned_existing_targets=["target.py"],
        planned_new_targets=["new_file.py"],
        git_head="a" * 40,
    )

    (tmp_path / "unrelated.log").write_text("second", encoding="utf-8")
    assert builder.is_current(snapshot) is True

    (tmp_path / "evidence.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert builder.is_current(snapshot) is False
    (tmp_path / "evidence.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert builder.is_current(snapshot) is True

    (tmp_path / "new_file.py").write_text("created elsewhere", encoding="utf-8")
    assert builder.is_current(snapshot) is False


# 功能：验证 snapshot digest 不绑定 head-only lifecycle identity
# 设计：用相同相关内容捕获两个不同 git_head，断言 provenance 变化但 stale digest 相同
def test_snapshot_digest_ignores_git_head_only_change(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("stable", encoding="utf-8")
    builder = _grounding().SnapshotBuilder(tmp_path)

    first = builder.capture(planned_existing_targets=["target.py"], git_head="a" * 40)
    second = builder.capture(planned_existing_targets=["target.py"], git_head="b" * 40)

    assert first.git_head != second.git_head
    assert first.snapshot_digest == second.snapshot_digest


# 功能：验证被篡改的 snapshot digest 即使相关文件未变也会 fail closed
# 设计：只替换 immutable model 的 digest 字段并复用真实文件，隔离内容 stale 与 artifact corruption
def test_snapshot_digest_tampering_is_not_current(tmp_path: Path) -> None:
    (tmp_path / "target.py").write_text("stable", encoding="utf-8")
    builder = _grounding().SnapshotBuilder(tmp_path)
    snapshot = builder.capture(planned_existing_targets=["target.py"])
    tampered = snapshot.model_copy(update={"snapshot_digest": "0" * 64})

    assert builder.is_current(tampered) is False


# 功能：验证 snapshot 只读取显式列出的 untracked target
# 设计：放置未列出的敏感命名文件与一个显式 untracked target，断言 snapshot map 只包含后者
def test_snapshot_does_not_scan_unrelated_untracked_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=not-read", encoding="utf-8")
    (tmp_path / "draft.py").write_text("draft", encoding="utf-8")

    snapshot = _grounding().SnapshotBuilder(tmp_path).capture(
        relevant_untracked_targets=["draft.py"]
    )

    assert snapshot.relevant_untracked_target_digests.keys() == {"draft.py"}
    assert ".env" not in snapshot.snapshot_digest
