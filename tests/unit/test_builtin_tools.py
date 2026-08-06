from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from kama_claude.core.sandbox.errors import SandboxUnavailableError
from kama_claude.core.sandbox.executors import (
    CommandExecutor,
    ExecResult,
    HostExecutor,
)
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver


# 构造绑定指定 workspace 的 list_dir 工具
def _list_tool(workspace: Path) -> ListDirTool:
    return ListDirTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 构造绑定指定 workspace 的 write_file 工具
def _write_tool(workspace: Path) -> WriteFileTool:
    return WriteFileTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 构造从指定 workspace 启动的 bash 工具（显式注入宿主执行器）
def _bash_tool(workspace: Path) -> BashTool:
    return BashTool(HostExecutor(), workspace_root=workspace)


class _RecordingExecutor(CommandExecutor):
    # 记录 exec 调用并返回可配置结果，隔离 BashTool 的展示逻辑
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, float]] = []
        self.result = ExecResult(output=b"out", returncode=0, timed_out=False)
        self.failure: Exception | None = None

    async def exec(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        self.calls.append((command, cwd, timeout))
        if self.failure is not None:
            raise self.failure
        return self.result

# ── bash ──────────────────────────────────────────────────────────────────────

# 功能：验证成功命令的 stdout 出现在 ToolResult.content 中，is_error 为 False
# 设计：用 echo 命令避免外部依赖，直接比较输出内容，无需 mock
@pytest.mark.asyncio
async def test_bash_success_stdout(tmp_path: Path) -> None:
    result = await _bash_tool(tmp_path).invoke({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content


# 功能：验证非零退出码时 is_error=True 且 content 包含退出码标注
# 设计：`exit 2` 是最简单的非零退出；不依赖任何外部命令行为
@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_error(tmp_path: Path) -> None:
    result = await _bash_tool(tmp_path).invoke({"command": "exit 2"})
    assert result.is_error
    assert "[exit 2]" in result.content


# 功能：验证命令超时后 is_error=True，error_type 为 "timeout"
# 设计：timeout=1s 搭配 sleep 2 必然超时；验证 error_type 而非 content，避免超时消息格式耦合
@pytest.mark.asyncio
async def test_bash_timeout(tmp_path: Path) -> None:
    result = await _bash_tool(tmp_path).invoke({"command": "sleep 5", "timeout": 1})
    assert result.is_error
    assert result.error_type == "timeout"


# 功能：验证 stderr 被合并到 stdout 输出中
# 设计：只写 stderr 的命令（>&2 echo），输出应该出现在合并后的 content 里
@pytest.mark.asyncio
async def test_bash_stderr_merged(tmp_path: Path) -> None:
    result = await _bash_tool(tmp_path).invoke({"command": "echo err >&2"})
    assert not result.is_error
    assert "err" in result.content


# 功能：验证 Bash starts in workspace 而非 process cwd
# 设计：process cwd=A、workspace=B 时执行 pwd，断言输出 canonical B
@pytest.mark.asyncio
async def test_bash_starts_in_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(daemon_cwd)

    result = await _bash_tool(workspace).invoke({"command": "pwd"})

    assert result.content.strip() == str(workspace.resolve(strict=True))


# 功能：验证两个 BashTool 分别 starts in workspace A/B
# 设计：依次执行 pwd 并比较输出，证明实例不共享 cwd 状态
@pytest.mark.asyncio
async def test_bash_instances_start_in_distinct_workspaces(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    result_a = await _bash_tool(workspace_a).invoke({"command": "pwd"})
    result_b = await _bash_tool(workspace_b).invoke({"command": "pwd"})

    assert result_a.content.strip() == str(workspace_a.resolve(strict=True))
    assert result_b.content.strip() == str(workspace_b.resolve(strict=True))


# 功能：验证 Bash stdout/stderr 合并输出仍按 64KB 截断
# 设计：使用当前 Python 生成 70KB ASCII 输出，断言标记存在且前缀长度受限
@pytest.mark.asyncio
async def test_bash_output_truncation_unchanged(tmp_path: Path) -> None:
    command = f"{shlex.quote(sys.executable)} -c \"print('x' * 70000)\""

    result = await _bash_tool(tmp_path).invoke({"command": command})

    assert not result.is_error
    assert result.content.endswith("[truncated]")
    assert len(result.content) < 70000


# 功能：验证 BashTool 把命令、canonical workspace cwd 与 timeout 原样委托给 executor
# 设计：注入记录 executor，断言委托参数与结果展示——BashTool 不再直接碰 subprocess
@pytest.mark.asyncio
async def test_bash_delegates_command_and_cwd(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    executor.result = ExecResult(output=b"echoed", returncode=0, timed_out=False)

    result = await BashTool(executor, workspace_root=tmp_path).invoke(
        {"command": "echo hi", "timeout": 7}
    )

    assert not result.is_error
    assert result.content == "echoed"
    assert executor.calls == [("echo hi", tmp_path.resolve(), 7)]


# 功能：验证 executor 报告 timed_out 时映射为 timeout 错误结果
# 设计：注入 timed_out=True 的结果，断言展示层错误类型与超时消息
@pytest.mark.asyncio
async def test_bash_maps_executor_timeout_to_timeout_error(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    executor.result = ExecResult(output=b"", returncode=-1, timed_out=True)

    result = await BashTool(executor, workspace_root=tmp_path).invoke(
        {"command": "sleep 5", "timeout": 1}
    )

    assert result.is_error
    assert result.error_type == "timeout"
    assert "[timeout after 1s]" in result.content


# 功能：验证 executor 报告非零退出码时映射为 command_failed 错误结果
# 设计：注入 returncode=2 的结果，断言展示层保留退出码标注与输出
@pytest.mark.asyncio
async def test_bash_maps_executor_nonzero_to_command_failed(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    executor.result = ExecResult(output=b"boom", returncode=2, timed_out=False)

    result = await BashTool(executor, workspace_root=tmp_path).invoke(
        {"command": "exit 2"}
    )

    assert result.is_error
    assert result.error_type == "command_failed"
    assert "[exit 2]" in result.content
    assert "boom" in result.content


# 功能：验证 executor 抛出的沙箱异常原样传播（分类是 invocation 层的职责）
# 设计：注入 SandboxUnavailableError，断言 BashTool 不吞不转——展示层保持透明
@pytest.mark.asyncio
async def test_bash_propagates_executor_sandbox_error(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    executor.failure = SandboxUnavailableError("no daemon")

    with pytest.raises(SandboxUnavailableError):
        await BashTool(executor, workspace_root=tmp_path).invoke(
            {"command": "echo hi"}
        )


# ── write_file ────────────────────────────────────────────────────────────────

# 功能：验证 process cwd=A 时 write_file 只写入 workspace=B
# 设计：切换 daemon cwd 后使用相对路径，断言 B 创建文件且 A 不存在同名文件
@pytest.mark.asyncio
async def test_write_file_creates_and_returns_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(daemon_cwd)

    result = await _write_tool(workspace).invoke(
        {"path": "out.txt", "content": "hello world"}
    )

    assert not result.is_error
    assert "11" in result.content  # "hello world" = 11 bytes
    assert (workspace / "out.txt").read_text() == "hello world"
    assert not (daemon_cwd / "out.txt").exists()


# 功能：验证 write_file 自动创建不存在的父目录
# 设计：路径包含两层不存在的子目录，确认写入后目录结构被创建
@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    result = await _write_tool(tmp_path).invoke(
        {"path": "a/b/file.txt", "content": "x"}
    )
    assert not result.is_error
    assert target.exists()


# 功能：验证 write_file 拒绝包含 .. 的路径并抛出 PermissionError
# 设计：.. 路径遍历与 read_file 遵循相同规则，用相同的断言模式保持一致性
@pytest.mark.asyncio
async def test_write_file_rejects_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(WorkspaceEscapeError):
        await _write_tool(workspace).invoke(
            {"path": "../secret.txt", "content": "x"}
        )


# 功能：验证 write_file 拒绝绝对路径
# 设计：目标位于 workspace 内仍使用绝对参数，断言接口要求相对路径
@pytest.mark.asyncio
async def test_write_file_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(InvalidWorkspacePathError):
        await _write_tool(tmp_path).invoke(
            {"path": str(tmp_path / "out.txt"), "content": "x"}
        )


# 功能：验证 write_file 经外部目录 symlink 写入时拒绝
# 设计：alias 位于 workspace 内但 canonical parent 在外部，断言不会创建文件
@pytest.mark.asyncio
async def test_write_file_rejects_external_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "alias").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceEscapeError):
        await _write_tool(workspace).invoke(
            {"path": "alias/out.txt", "content": "secret"}
        )
    assert not (outside / "out.txt").exists()


# 功能：验证 write_file 经内部目录 symlink 写入允许
# 设计：alias 指向 workspace 内真实目录，断言 resolved target 收到内容
@pytest.mark.asyncio
async def test_write_file_allows_internal_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)

    result = await _write_tool(tmp_path).invoke(
        {"path": "alias/out.txt", "content": "inside"}
    )

    assert not result.is_error
    assert (target / "out.txt").read_text(encoding="utf-8") == "inside"


# 功能：验证 write_file 拒绝 .env、.git 和私钥路径
# 设计：参数化冻结的三类敏感 logical path，确认 policy 在创建父目录前执行
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [".env", str(Path(".git") / "config"), "keys/id_rsa"],
)
async def test_write_file_rejects_sensitive_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(SensitivePathError):
        await _write_tool(tmp_path).invoke({"path": path, "content": "secret"})


# 功能：验证 write_file 允许写入 .env.example
# 设计：使用明确 allowlist 例外，断言内容实际写入 workspace
@pytest.mark.asyncio
async def test_write_file_allows_env_example(tmp_path: Path) -> None:
    result = await _write_tool(tmp_path).invoke(
        {"path": ".env.example", "content": "TOKEN="}
    )

    assert not result.is_error
    assert (tmp_path / ".env.example").read_text(encoding="utf-8") == "TOKEN="


# 功能：验证 write_file 保持 1MB 内容上限
# 设计：写入超过限制一个字节的 ASCII 内容，断言 error ToolResult 且文件不存在
@pytest.mark.asyncio
async def test_write_file_size_limit_unchanged(tmp_path: Path) -> None:
    result = await _write_tool(tmp_path).invoke(
        {"path": "large.txt", "content": "x" * (1024 * 1024 + 1)}
    )

    assert result.is_error
    assert not (tmp_path / "large.txt").exists()


# 功能：验证 workspace A/B 的 WriteFileTool 实例互不串扰
# 设计：两个实例写同一 logical path，断言各自 workspace 保存独立内容
@pytest.mark.asyncio
async def test_write_file_instances_are_workspace_isolated(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()

    await _write_tool(workspace_a).invoke({"path": "result.txt", "content": "a"})
    await _write_tool(workspace_b).invoke({"path": "result.txt", "content": "b"})

    assert (workspace_a / "result.txt").read_text(encoding="utf-8") == "a"
    assert (workspace_b / "result.txt").read_text(encoding="utf-8") == "b"


# ── list_dir ──────────────────────────────────────────────────────────────────

# 功能：验证 process cwd=A 时 list_dir 只列出 workspace=B
# 设计：daemon 与 workspace 放置不同 marker，绑定 B 后交叉断言输出
@pytest.mark.asyncio
async def test_list_dir_shows_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    (daemon_cwd / "daemon.txt").write_text("x")
    (workspace / "foo.py").write_text("x")
    (workspace / "bar.md").write_text("y")
    monkeypatch.chdir(daemon_cwd)

    result = await _list_tool(workspace).invoke({"path": "."})

    assert not result.is_error
    assert "foo.py" in result.content
    assert "bar.md" in result.content
    assert "daemon.txt" not in result.content


# 功能：验证 list_dir 按 max_depth 限制递归深度（depth=1 时不展示孙级目录内容）
# 设计：创建 parent/child/grandchild 三层，depth=1 时 grandchild 不应出现在输出中
@pytest.mark.asyncio
async def test_list_dir_respects_max_depth(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("x")

    result = await _list_tool(tmp_path).invoke({"path": ".", "max_depth": 1})
    assert not result.is_error
    assert "child" in result.content
    assert "deep.txt" not in result.content


# 功能：验证对不存在的路径 list_dir 抛出 FileNotFoundError
# 设计：直接传入不存在的路径字符串，预期抛出标准异常（invocation.py 捕获后返回 error ToolResult）
@pytest.mark.asyncio
async def test_list_dir_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await _list_tool(tmp_path).invoke({"path": "missing"})


# 功能：验证 list_dir 拒绝包含 .. 的路径
# 设计：与 read_file 和 write_file 保持一致的安全规则
@pytest.mark.asyncio
async def test_list_dir_rejects_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(PermissionError):
        await _list_tool(workspace).invoke({"path": "../"})


# 功能：验证从 workspace 根列目录时不输出 .git 与 .env
# 设计：同时创建敏感目录和文件，断言名称及内部 marker 均不可见
@pytest.mark.asyncio
async def test_list_dir_filters_sensitive_entries(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")

    result = await _list_tool(tmp_path).invoke({"path": ".", "max_depth": 2})

    assert ".git" not in result.content
    assert ".env" not in result.content
    assert "HEAD" not in result.content
    assert "visible.txt" in result.content


# 功能：验证直接请求敏感 .git 目录被拒绝
# 设计：根路径本身先经过 policy，而不是依赖 child 过滤
@pytest.mark.asyncio
async def test_list_dir_rejects_sensitive_root(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    with pytest.raises(SensitivePathError):
        await _list_tool(tmp_path).invoke({"path": ".git"})


# 功能：验证指向 .git 的非敏感 alias 也不会出现在目录结果
# 设计：logical alias 不敏感但 canonical target 敏感，锁定 child canonical policy-check
@pytest.mark.asyncio
async def test_list_dir_filters_alias_to_sensitive_directory(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (tmp_path / "repo-data").symlink_to(git_dir, target_is_directory=True)

    result = await _list_tool(tmp_path).invoke({"path": "."})

    assert "repo-data" not in result.content


# 功能：验证指向 workspace 外部的 symlink 不输出且不递归
# 设计：外部目录包含 marker，断言 alias 名和 marker 都不可见
@pytest.mark.asyncio
async def test_list_dir_filters_external_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "external").symlink_to(outside, target_is_directory=True)

    result = await _list_tool(workspace).invoke({"path": ".", "max_depth": 2})

    assert "external" not in result.content
    assert "secret.txt" not in result.content


# 功能：验证指向 workspace 内普通目录的 symlink 可以输出并递归
# 设计：内部 alias 下放置 marker，断言 alias 与子文件均出现
@pytest.mark.asyncio
async def test_list_dir_allows_internal_directory_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)

    result = await _list_tool(tmp_path).invoke({"path": ".", "max_depth": 2})

    assert "alias/" in result.content
    assert "inside.txt" in result.content


# 功能：验证 list_dir 保持最多 200 entries 的输出限制
# 设计：创建 201 个普通文件，断言仅输出前 200 项并带 truncated 标记
@pytest.mark.asyncio
async def test_list_dir_respects_entry_limit(tmp_path: Path) -> None:
    for index in range(201):
        (tmp_path / f"file-{index:03}.txt").write_text("x", encoding="utf-8")

    result = await _list_tool(tmp_path).invoke({"path": "."})

    assert "... (truncated)" in result.content
    assert result.content.count(".txt") == 200
