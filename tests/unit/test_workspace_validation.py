from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.workspace.errors import InvalidWorkspaceError
from kama_claude.core.workspace.validation import validate_workspace_root


# 功能：验证已存在的绝对目录被转换为 canonical Path
# 设计：通过指向真实目录的 symlink 输入，断言返回值等于 strict resolve 结果
def test_validate_workspace_root_returns_canonical_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)

    assert validate_workspace_root(str(alias)) == workspace.resolve(strict=True)


# 功能：验证相对 workspace_root 被拒绝为 not_absolute
# 设计：使用不依赖当前 cwd 内容的相对路径，直接断言稳定 reason
def test_validate_workspace_root_rejects_relative_path() -> None:
    with pytest.raises(InvalidWorkspaceError) as exc:
        validate_workspace_root("project")

    assert exc.value.reason == "not_absolute"


# 功能：验证不存在的绝对 workspace_root 被拒绝为 not_found
# 设计：在 tmp_path 下构造不创建的子目录，避免依赖宿主机固定路径
def test_validate_workspace_root_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(InvalidWorkspaceError) as exc:
        validate_workspace_root(str(missing))

    assert exc.value.reason == "not_found"


# 功能：验证指向普通文件的 workspace_root 被拒绝为 not_directory
# 设计：创建真实文件以区分 exists 与 is_dir 检查，锁定校验顺序
def test_validate_workspace_root_rejects_file(tmp_path: Path) -> None:
    file_path = tmp_path / "project.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(InvalidWorkspaceError) as exc:
        validate_workspace_root(str(file_path))

    assert exc.value.reason == "not_directory"
