from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path, PurePath

import pytest

import kama_claude.core.tools.builtin.search_code as search_code_module
from kama_claude.core.tools.builtin.search_code import (
    SearchCodeTool,
)
from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver
from tests.unit._search_code_test_support import _footer, _records, _tool


@pytest.mark.parametrize("root_kind", ["file", "directory"])
# 功能：验证显式 file/directory root 不可读时直接传播 PermissionError
# 设计：在 fd open seam 注入确定性拒绝，区分调用级 root 失败与 recursive child best-effort skip
async def test_explicit_unreadable_root_propagates_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    target = tmp_path / "target"
    if root_kind == "directory":
        target.mkdir()
    else:
        target.write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    real_open = tool._open_workspace_fd

    # 只拒绝显式 root 的安全 fd open
    def deny_root(path: Path, *, directory: bool) -> int:
        if path == target.resolve(strict=True):
            raise PermissionError("private platform detail")
        return real_open(path, directory=directory)

    monkeypatch.setattr(tool, "_open_workspace_fd", deny_root)

    with pytest.raises(PermissionError, match="private platform detail"):
        await tool.invoke({"query": "needle", "path": "target"})


# 功能：验证递归 unreadable child directory 仍被跳过且搜索继续
# 设计：只拒绝排序靠前的 child directory fd，确认 visible sibling 仍产出结果与 skip 计数
async def test_recursive_unreadable_child_directory_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "a-blocked"
    blocked.mkdir()
    (blocked / "hidden.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "z-visible.txt").write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    real_open = tool._open_workspace_fd

    # 只拒绝 recursive child directory，不影响显式 workspace root
    def deny_child(path: Path, *, directory: bool) -> int:
        if directory and path == blocked.resolve(strict=True):
            raise PermissionError("denied child")
        return real_open(path, directory=directory)

    monkeypatch.setattr(tool, "_open_workspace_fd", deny_child)
    result = await tool.invoke({"query": "needle"})

    assert _records(result.content) == ["z-visible.txt:1: needle"]
    assert _footer(result.content)["skipped_unreadable"] == 1


@pytest.mark.parametrize("root_kind", ["fifo", "socket"])
# 功能：验证显式 FIFO/socket root 返回固定 invalid_input 且绝不尝试 open
# 设计：创建真实 POSIX special file 并将 fd helper 设为 fail-fast，避免依赖 timeout 证明不阻塞
async def test_explicit_special_root_is_invalid_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    short_workspace: tempfile.TemporaryDirectory[str] | None = None
    workspace = tmp_path
    if root_kind == "socket":
        short_workspace = tempfile.TemporaryDirectory(prefix="ks-", dir="/tmp")
        workspace = Path(short_workspace.name)
    target = workspace / root_kind
    unix_socket: socket.socket | None = None
    if root_kind == "fifo":
        os.mkfifo(target)
    else:
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_socket.bind(str(target))
    tool = _tool(workspace)
    opened = False

    # special root 若进入 fd open 即证明存在阻塞风险
    def unexpected_open(_path: Path, *, directory: bool) -> int:
        nonlocal opened
        opened = True
        raise AssertionError(f"special root reached fd open: {directory}")

    monkeypatch.setattr(tool, "_open_workspace_fd", unexpected_open)
    try:
        result = await tool.invoke({"query": "needle", "path": root_kind})
    finally:
        if unix_socket is not None:
            unix_socket.close()
        if short_workspace is not None:
            short_workspace.cleanup()

    assert opened is False
    assert result.is_error
    assert result.error_type == "invalid_input"
    assert result.content == "search path must be a regular file or directory"


@pytest.mark.parametrize(
    "missing_capability",
    ["O_NOFOLLOW", "O_DIRECTORY", "open_dir_fd", "scandir_fd"],
)
# 功能：验证 hardened search 缺少任一 POSIX capability 时 direct invoke fail closed
# 设计：逐项移除常量或 capability set，锁定平台不支持不能退化为成功空 footer
async def test_missing_posix_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    (tmp_path / "hit.txt").write_text("needle", encoding="utf-8")
    if missing_capability in {"O_NOFOLLOW", "O_DIRECTORY"}:
        monkeypatch.delattr(search_code_module.os, missing_capability)
    elif missing_capability == "open_dir_fd":
        monkeypatch.setattr(search_code_module.os, "supports_dir_fd", frozenset())
    else:
        monkeypatch.setattr(search_code_module.os, "supports_fd", frozenset())

    with pytest.raises(
        RuntimeError,
        match="secure search_code POSIX capabilities are unavailable",
    ):
        await _tool(tmp_path).invoke({"query": "needle"})


# 功能：验证递归遇到外部 symlink 与敏感候选时不搜索也不泄露
# 设计：在可搜索普通文件旁放置外部 file/dir alias 与 .env，确认 candidate policy 不可被删除
async def test_nested_external_symlinks_and_sensitive_candidates_do_not_leak(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "visible.txt").write_text("public-needle", encoding="utf-8")
    secret = outside / "secret.txt"
    secret.write_text("secret-needle", encoding="utf-8")
    (workspace / "external-file").symlink_to(secret)
    (workspace / "external-dir").symlink_to(outside, target_is_directory=True)
    (workspace / ".env").write_text("TOKEN=secret-needle", encoding="utf-8")

    result = await _tool(workspace).invoke({"query": "needle"})

    assert _records(result.content) == ["visible.txt:1: public-needle"]
    assert _footer(result.content)["directory_entries"] == 4
    assert "secret-needle" not in result.content
    assert str(outside.resolve(strict=True)) not in result.content


@pytest.mark.parametrize("swap_component", ["final", "ancestor"])
# 功能：验证 candidate 通过 policy 后替换 final/ancestor 组件都不能读取外部内容
# 设计：在真实 ensure_allowed 返回后换成外部 symlink，锁定逐组件 no-follow 不可被简化
async def test_candidate_symlink_swap_after_policy_cannot_escape_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_component: str,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    target_directory = workspace if swap_component == "final" else workspace / "nested"
    target_directory.mkdir(exist_ok=True)
    victim = target_directory / "victim.txt"
    victim.write_text("public", encoding="utf-8")
    secret = outside / "victim.txt"
    secret.write_text("secret-needle", encoding="utf-8")
    logical_victim = victim.relative_to(workspace).as_posix()
    resolver = WorkspacePathResolver(workspace)
    policy = WorkspaceAccessPolicy(workspace)
    ensure_allowed = policy.ensure_allowed
    swapped = False

    def swap_after_policy(logical_path: str, resolved_path: Path) -> None:
        nonlocal swapped
        ensure_allowed(logical_path, resolved_path)
        if logical_path == logical_victim and not swapped:
            if swap_component == "final":
                victim.unlink()
                victim.symlink_to(secret)
            else:
                target_directory.rename(workspace / "parked")
                target_directory.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(policy, "ensure_allowed", swap_after_policy)
    result = await SearchCodeTool(resolver, policy).invoke({"query": "needle"})

    assert swapped
    assert _records(result.content) == []
    assert _footer(result.content)["skipped_unreadable"] == 1
    assert "secret-needle" not in result.content
    assert str(outside.resolve(strict=True)) not in result.content


# 功能：验证已发现的目录在进入前消失时按 unreadable 跳过而非抛 execution_error
# 设计：在递归入口前删除空目录，精确命中 strict resolve/open race 且不依赖线程时序
async def test_directory_disappearing_before_entry_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disappearing = tmp_path / "disappearing"
    disappearing.mkdir()
    (tmp_path / "visible.txt").write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    walk_directory = tool._walk_directory

    def remove_before_walk(*args: object, **kwargs: object) -> bool:
        directory = args[0]
        logical_directory = args[1]
        if logical_directory == PurePath("disappearing"):
            assert isinstance(directory, Path)
            directory.rmdir()
        return walk_directory(*args, **kwargs)

    monkeypatch.setattr(tool, "_walk_directory", remove_before_walk)
    result = await tool.invoke({"query": "needle"})

    assert _records(result.content) == ["visible.txt:1: needle"]
    assert _footer(result.content)["skipped_unreadable"] == 1


# 功能：验证显式 sensitive root/descendant 被拒绝且两个 env 模板例外允许
# 设计：参数化 .git、根 .env 和嵌套 .env.local，再分别从例外文件作为 root 搜索
async def test_sensitive_roots_rejected_but_env_examples_allowed(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("needle", encoding="utf-8")
    (tmp_path / ".env").write_text("needle", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env.local").write_text("needle", encoding="utf-8")
    for name in (".env.example", ".env.template"):
        (tmp_path / name).write_text("needle", encoding="utf-8")

    sensitive_paths = (".git", str(Path(".git") / "HEAD"), ".env", "config/.env.local")
    for path in sensitive_paths:
        with pytest.raises(SensitivePathError):
            await _tool(tmp_path).invoke({"query": "needle", "path": path})

    for name in (".env.example", ".env.template"):
        result = await _tool(tmp_path).invoke({"query": "needle", "path": name})
        assert _records(result.content) == [f"{name}:1: needle"]


# 功能：验证 workspace 内部 file/dir symlink 可搜索且目录环不重复遍历
# 设计：通过显式 internal file alias 和指回 root 的 directory cycle，覆盖允许与去重两类边界
async def test_internal_symlinks_allowed_and_directory_cycle_deduplicated(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.txt"
    target.write_text("needle", encoding="utf-8")
    (real / "back").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "file-alias.txt").symlink_to(target)

    file_result = await _tool(tmp_path).invoke({"query": "needle", "path": "file-alias.txt"})
    (tmp_path / "file-alias.txt").unlink()
    tree_result = await _tool(tmp_path).invoke({"query": "needle"})

    assert _records(file_result.content) == ["file-alias.txt:1: needle"]
    assert _records(tree_result.content) == ["real/target.txt:1: needle"]
    assert _footer(tree_result.content)["visited_directories"] == 2


# 功能：验证同一 canonical directory 仅沿确定性排序后首个 logical alias 遍历
# 设计：让 alias 名按 case-sensitive 顺序早于真实目录，断言只输出 alias path 且不重复匹配
async def test_canonical_directory_dedup_uses_first_sorted_logical_path(
    tmp_path: Path,
) -> None:
    real = tmp_path / "z-real"
    real.mkdir()
    (real / "hit.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "a-alias").symlink_to(real, target_is_directory=True)

    result = await _tool(tmp_path).invoke({"query": "needle"})

    assert _records(result.content) == ["a-alias/hit.txt:1: needle"]
    footer = _footer(result.content)
    assert footer["matched_lines"] == 1
    assert footer["visited_directories"] == 2
    assert "z-real/hit.txt" not in result.content
    assert str(real.resolve(strict=True)) not in result.content


@pytest.mark.parametrize(
    ("root_kind", "expected"),
    [
        ("absolute", InvalidWorkspacePathError),
        ("parent", WorkspaceEscapeError),
        ("external_file", WorkspaceEscapeError),
        ("external_dir", WorkspaceEscapeError),
    ],
)
# 功能：验证显式 root 拒绝绝对路径、parent traversal 和外部 file/dir symlink
# 设计：参数化四种根路径逃逸形态，锁定 invocation 前的 resolver containment
async def test_explicit_root_rejects_workspace_escape_matrix(
    tmp_path: Path,
    root_kind: str,
    expected: type[Exception],
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret-needle", encoding="utf-8")
    (workspace / "file-link").symlink_to(outside_file)
    (workspace / "dir-link").symlink_to(outside, target_is_directory=True)
    roots = {
        "absolute": str(workspace),
        "parent": "../outside",
        "external_file": "file-link",
        "external_dir": "dir-link",
    }

    with pytest.raises(expected):
        await _tool(workspace).invoke({"query": "needle", "path": roots[root_kind]})
