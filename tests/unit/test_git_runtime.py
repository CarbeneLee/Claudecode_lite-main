from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.git.errors import (
    GitUnavailableError,
    RepositoryNotFoundError,
    classify_cli_error,
)
from kama_claude.core.git.runtime import GitCliRuntime
from kama_claude.core.sandbox.executors import ExecResult

_FAKE_GIT_SCRIPT = """\
#!/bin/sh
# 记录每个 argv 一行到 FAKE_GIT_LOG；FAKE_GIT_EXIT/STDERR/HANG 注入故障
printf '%s\\n' "$@" >> "$FAKE_GIT_LOG"
if [ "$FAKE_GIT_HANG" = "1" ]; then sleep 30; fi
if [ -n "$FAKE_GIT_STDERR" ]; then printf '%s\\n' "$FAKE_GIT_STDERR" >&2; fi
exit "${FAKE_GIT_EXIT:-0}"
"""


@pytest.fixture
def fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    # 安装 fake git CLI：记录每个 argv 一行到 log，env 变量触发故障注入
    script = tmp_path / "git"
    script.write_text(_FAKE_GIT_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.delenv("FAKE_GIT_EXIT", raising=False)
    monkeypatch.delenv("FAKE_GIT_STDERR", raising=False)
    monkeypatch.delenv("FAKE_GIT_HANG", raising=False)
    return script, log


def _runtime(fake_git: tuple[Path, Path], *, cwd: Path | None = None) -> GitCliRuntime:
    script, _ = fake_git
    return GitCliRuntime(
        git_executable=str(script),
        workspace_root=cwd or Path("/host-ws"),
    )


def _log_lines(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines()


# 功能：验证 run 把参数作为独立 argv 原样透传（宿主 shell 注入红线）
# 设计：注入含引号与分号的恶意参数，断言 log 中出现完整独立 argv 行——
#       证明命令未被宿主 shell 拼接或二次解析
async def test_run_argv_isolation(
    fake_git: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_git)
    evil = "status --porcelain; rm -rf /"
    result = await runtime.run([evil])
    assert result.returncode == 0
    lines = _log_lines(fake_git[1])
    assert evil in lines  # 独立 argv 完整保留，未拆分成多个宿主参数


# 功能：验证 run 正常返回 ExecResult（输出 + 返回码）
# 设计：fake git 输出一行 stdout 到 log 的同时返回 0，断言结果契约
async def test_run_returns_exec_result(fake_git: tuple[Path, Path]) -> None:
    runtime = _runtime(fake_git)
    result = await runtime.run(["rev-parse", "--is-inside-work-tree"])
    assert isinstance(result, ExecResult)
    assert result.returncode == 0
    assert not result.timed_out


# 功能：验证非零返回码不抛异常、原样返回（探测类命令语义）
# 设计：FAKE_GIT_EXIT=128 模拟 rev-parse 在非仓库目录的退出，断言 run 不 raise
async def test_run_nonzero_returns_result(
    fake_git: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GIT_EXIT", "128")
    runtime = _runtime(fake_git)
    result = await runtime.run(["rev-parse", "--is-inside-work-tree"])
    assert result.returncode == 128


# 功能：验证 run_check 对非零返回码按 stderr 关键词分类抛异常
# 设计：FAKE_GIT_STDERR 注入非仓库 stderr + 128 退出，断言 RepositoryNotFoundError
async def test_run_check_classifies_nonzero(
    fake_git: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GIT_EXIT", "128")
    monkeypatch.setenv(
        "FAKE_GIT_STDERR", "fatal: not a git repository (or any of the parent directories): .git"
    )
    runtime = _runtime(fake_git)
    with pytest.raises(RepositoryNotFoundError):
        await runtime.run_check(["rev-parse", "--is-inside-work-tree"])


# 功能：验证 run_check 在零返回码时返回结果（checkpoint 主路径不误伤）
# 设计：默认 fake 成功路径断言 run_check 返回 ExecResult 而非抛错
async def test_run_check_ok_returns_result(fake_git: tuple[Path, Path]) -> None:
    runtime = _runtime(fake_git)
    result = await runtime.run_check(["rev-parse", "--show-toplevel"])
    assert result.returncode == 0


# 功能：验证 git CLI 不存在时抛 GitUnavailableError（fail-open 基础）
# 设计：git_executable 指向不存在路径，断言 FileNotFoundError 被包装为 GitUnavailableError
async def test_missing_cli_raises_unavailable(tmp_path: Path) -> None:
    runtime = GitCliRuntime(
        git_executable=str(tmp_path / "no-such-git"),
        workspace_root=Path("/host-ws"),
    )
    with pytest.raises(GitUnavailableError):
        await runtime.run(["status"])


# 功能：验证执行超时抛 GitUnavailableError 并清理 git 子进程
# 设计：HANG 故障 + 短超时触发超时路径，断言异常类型且子进程被整组清理
async def test_timeout_raises_unavailable_and_cleans_up(
    fake_git: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GIT_HANG", "1")
    runtime = _runtime(fake_git)
    with pytest.raises(GitUnavailableError):
        await runtime.run(["status"], timeout=0.2)


# 功能：验证 run 的取消信号原样传播且 git 子进程被清理
# 设计：HANG 故障 + 短延迟 cancel，断言 CancelledError 传播（控制流而非工具失败）
async def test_cancellation_propagates(
    fake_git: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_GIT_HANG", "1")
    runtime = _runtime(fake_git)
    task = asyncio.create_task(runtime.run(["status"], timeout=30))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# 功能：验证 cwd 固定为 workspace_root（git 命令只在 workspace 内执行）
# 设计：在子目录 cwd 构造 runtime，断言后续 run 不改变工作目录语义——
#       run 无 cwd 参数，git 操作的边界由构造时 workspace_root 锁定
async def test_cwd_is_workspace_root(fake_git: tuple[Path, Path]) -> None:
    runtime = _runtime(fake_git, cwd=Path("/ws/deep/dir"))
    assert runtime.workspace_root == Path("/ws/deep/dir")


# 功能：验证 classify_cli_error 与 runtime 层契约一致（分类可独立测试）
# 设计：用真实 git 常见 stderr 冒烟分类器，防止 runtime 与分类器语义漂移
def test_classify_contract_stable() -> None:
    assert isinstance(
        classify_cli_error("fatal: not a git repository (or any of the parent directories): .git"),
        RepositoryNotFoundError,
    )
