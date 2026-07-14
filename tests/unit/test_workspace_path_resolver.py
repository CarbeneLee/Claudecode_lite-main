from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    WorkspaceEscapeError,
)
from kama_claude.core.workspace.resolver import WorkspacePathResolver


# 功能：验证 workspace 内已存在普通文件可解析为 canonical Path
# 设计：写入真实文件后比较 strict resolve 结果，覆盖 existing file 正常路径
def test_resolve_existing_allows_file(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("content", encoding="utf-8")

    resolved = WorkspacePathResolver(tmp_path).resolve_existing("file.txt")

    assert resolved == target.resolve(strict=True)


# 功能：验证 workspace 内已存在普通目录可以解析
# 设计：创建子目录并通过相对路径解析，避免把 resolver 限制为仅文件
def test_resolve_existing_allows_directory(tmp_path: Path) -> None:
    target = tmp_path / "src"
    target.mkdir()

    resolved = WorkspacePathResolver(tmp_path).resolve_existing("src")

    assert resolved == target.resolve(strict=True)


# 功能：验证绝对路径被硬拒绝
# 设计：传入 workspace 内文件的绝对路径，证明即使目标安全也必须使用相对参数
def test_resolve_existing_rejects_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(InvalidWorkspacePathError):
        WorkspacePathResolver(tmp_path).resolve_existing(str(target))


# 功能：验证父目录遍历不能逃逸 workspace
# 设计：创建真实 outside 文件使 strict resolve 成功，再断言 containment 拒绝
def test_resolve_existing_rejects_parent_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceEscapeError):
        WorkspacePathResolver(workspace).resolve_existing("../outside.txt")


# 功能：验证指向 workspace 内部的 symlink 可以读取
# 设计：通过文件 symlink 解析并断言返回真实内部目标而非 alias
def test_resolve_existing_allows_internal_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("content", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)

    resolved = WorkspacePathResolver(tmp_path).resolve_existing("alias.txt")

    assert resolved == target.resolve(strict=True)


# 功能：验证指向 workspace 外部的 symlink 被拒绝
# 设计：workspace 内 alias 指向相邻目录文件，断言 canonical containment 生效
def test_resolve_existing_rejects_external_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "alias.txt").symlink_to(outside)

    with pytest.raises(WorkspaceEscapeError):
        WorkspacePathResolver(workspace).resolve_existing("alias.txt")


# 功能：验证 write resolver 支持多层尚不存在的父目录
# 设计：不创建任何目标父目录，断言 strict=False 返回 workspace 内 canonical 目标且无副作用
def test_resolve_for_write_allows_missing_parent_chain(tmp_path: Path) -> None:
    resolved = WorkspacePathResolver(tmp_path).resolve_for_write("a/b/result.txt")

    assert resolved == tmp_path.resolve(strict=True) / "a" / "b" / "result.txt"
    assert not (tmp_path / "a").exists()


# 功能：验证 write path 经内部目录 symlink 时仍解析到 workspace 内
# 设计：alias 指向内部真实目录，目标文件尚不存在，覆盖 strict=False symlink component 行为
def test_resolve_for_write_allows_internal_directory_symlink(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (tmp_path / "alias").symlink_to(target_dir, target_is_directory=True)

    resolved = WorkspacePathResolver(tmp_path).resolve_for_write("alias/new.txt")

    assert resolved == target_dir.resolve(strict=True) / "new.txt"


# 功能：验证 write path 经外部目录 symlink 时被拒绝
# 设计：目标文件尚不存在但已有 symlink component 指向外部，断言 strict=False 仍能发现逃逸
def test_resolve_for_write_rejects_external_directory_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "alias").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceEscapeError):
        WorkspacePathResolver(workspace).resolve_for_write("alias/new.txt")


# 功能：验证普通文件不能作为 workspace root
# 设计：创建已存在文件排除 not-found 分支，断言构造阶段立即失败
def test_constructor_rejects_file_workspace_root(tmp_path: Path) -> None:
    workspace_file = tmp_path / "workspace.txt"
    workspace_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(InvalidWorkspacePathError):
        WorkspacePathResolver(workspace_file)


# 功能：验证不存在路径不能作为 workspace root
# 设计：传入缺失目录并断言构造失败，不允许隐式创建或 cwd fallback
def test_constructor_rejects_missing_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(InvalidWorkspacePathError):
        WorkspacePathResolver(tmp_path / "missing")
