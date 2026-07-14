from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver


# 构造绑定指定 workspace 的 read_file 工具
def _read_tool(workspace: Path) -> ReadFileTool:
    return ReadFileTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 功能：验证 process cwd=A 时 read_file 相对路径只读取 workspace=B
# 设计：daemon 与 workspace 写入同名冲突文件，显式绑定 B 后断言只返回 B 内容
async def test_read_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    (daemon_cwd / "hello.txt").write_text("daemon", encoding="utf-8")
    f = workspace / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    monkeypatch.chdir(daemon_cwd)

    result = await _read_tool(workspace).invoke({"path": "hello.txt"})

    assert not result.is_error
    assert result.content == "hello world"


# 功能：验证文件不存在时抛出 FileNotFoundError 而非返回错误 ToolResult
# 设计：传入不存在的路径，确认 ReadFileTool 不吞掉异常，让调用方（invoke_tool）负责错误分类和事件发布
async def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await _read_tool(tmp_path).invoke({"path": "missing.txt"})


# 功能：验证 read_file 拒绝绝对路径参数
# 设计：绝对路径即使位于 workspace 内也应失败，锁定仅 workspace-relative 的接口
async def test_absolute_path_rejected(tmp_path: Path) -> None:
    target = tmp_path / "inside.txt"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(InvalidWorkspacePathError):
        await _read_tool(tmp_path).invoke({"path": str(target)})


# 功能：验证包含 `..` 的路径被拒绝并抛出 PermissionError
# 设计：传入 `"../secret.txt"` 这种最典型的目录遍历形式，确认安全边界第一道防线有效
async def test_path_traversal_dotdot_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceEscapeError):
        await _read_tool(workspace).invoke({"path": "../secret.txt"})


# 功能：验证多级路径中嵌入的 `..` 经过路径规范化后也被正确检测
# 设计：使用 `"subdir/../../etc/passwd"` 测试路径 resolve 后的深度遍历，确认单层 `..` 过滤不足以覆盖此情况
async def test_path_traversal_nested_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "etc"
    workspace.mkdir()
    (workspace / "subdir").mkdir()
    outside.mkdir()
    (outside / "passwd").write_text("secret", encoding="utf-8")

    with pytest.raises(WorkspaceEscapeError):
        await _read_tool(workspace).invoke({"path": "subdir/../../etc/passwd"})


# 功能：验证 read_file 拒绝指向 workspace 外文件的 symlink
# 设计：workspace 内 alias 指向相邻 secret，确认工具只读取 resolver 返回的 contained path
async def test_external_symlink_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (workspace / "alias.txt").symlink_to(secret)

    with pytest.raises(WorkspaceEscapeError):
        await _read_tool(workspace).invoke({"path": "alias.txt"})


# 功能：验证 read_file 允许指向 workspace 内文件的 symlink
# 设计：内部 alias 解析到 canonical target 后读取，确认 symlink 策略不是全部禁用
async def test_internal_symlink_allowed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("inside", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)

    result = await _read_tool(tmp_path).invoke({"path": "alias.txt"})

    assert result.content == "inside"


# 功能：验证 read_file 拒绝敏感 .env 文件
# 设计：创建真实文件使 resolver 成功，再断言 access policy 执行硬拒绝
async def test_sensitive_env_rejected(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")

    with pytest.raises(SensitivePathError):
        await _read_tool(tmp_path).invoke({"path": ".env"})


# 功能：验证 read_file 允许 .env.example
# 设计：读取明确 policy 例外文件，防止 .env 前缀规则过宽
async def test_env_example_allowed(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("TOKEN=", encoding="utf-8")

    result = await _read_tool(tmp_path).invoke({"path": ".env.example"})

    assert result.content == "TOKEN="


# 功能：验证超过 512KB 的文件被截断并在末尾追加 [truncated] 标记
# 设计：写 600KB 文件，断言内容以 x×512KB 开头、以 [truncated] 结尾，确认截断不破坏前缀内容
async def test_truncation_over_512kb(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (600 * 1024))
    result = await _read_tool(tmp_path).invoke({"path": "big.txt"})
    assert not result.is_error
    assert result.content.endswith("[truncated]")
    # Actual text content is exactly 512KB worth of 'x' chars
    assert result.content.startswith("x" * (512 * 1024))


# 功能：验证恰好等于 512KB 的文件不被截断（边界值：超过而非大于等于）
# 设计：boundary check，确认截断阈值为"严格超过 512KB"，防止 off-by-one 错误
async def test_exact_512kb_is_not_truncated(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_bytes(b"y" * (512 * 1024))
    result = await _read_tool(tmp_path).invoke({"path": "exact.txt"})
    assert not result.is_error
    assert not result.content.endswith("[truncated]")
    assert len(result.content) == 512 * 1024


# 功能：验证空文件返回空字符串而非 None 或错误
# 设计：零字节文件确认 content="" 的正常返回，避免调用方（LLM prompt 组装）对空内容做额外 None 判断
async def test_empty_file_returns_empty_content(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = await _read_tool(tmp_path).invoke({"path": "empty.txt"})
    assert not result.is_error
    assert result.content == ""
