from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.sandbox.errors import (
    ContainerNotReadyError,
    SandboxCreationFailedError,
    SandboxImageError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from kama_claude.core.sandbox.runtime import DockerCliRuntime

_FAKE_SCRIPT = """\
#!/bin/sh
printf '%s\\n' "$@" >> "$FAKE_DOCKER_LOG"
mode="${FAKE_DOCKER_MODE:-normal}"
state="${FAKE_DOCKER_STATE:-missing}"
# 全局故障：daemon 挂了所有命令都失败（stderr_* 模拟 daemon/镜像错误）
case "$mode" in
  exit_*) code="${mode#exit_}"; exit "$code" ;;
  stderr_*) echo "${mode#stderr_}" >&2; exit 1 ;;
esac
cmd="$1"
case "$cmd" in
  inspect)
    # 模拟 docker inspect 真实语义：missing 非零退出，running/stopped 输出 "ID state"
    case "$state" in
      running) echo "container123 true" ;;
      stopped) echo "container123 false" ;;
      missing) echo "No such container: kama-sandbox" >&2; exit 1 ;;
    esac ;;
  exec)
    case "$mode" in hang) sleep 30 ;; *) echo "fake-exec-output" ;; esac ;;
  run) echo "container123" ;;
  rm) : ;;
esac
exit 0
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    # 安装 fake docker CLI：记录每个 argv 一行到 log，FAKE_DOCKER_MODE 触发故障
    script = tmp_path / "docker"
    script.write_text(_FAKE_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("FAKE_DOCKER_MODE", "normal")
    return script, log


def _runtime(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    network: bool = True,
) -> DockerCliRuntime:
    script, _ = fake_docker
    return DockerCliRuntime(
        image="python:3.12-slim",
        workspace_root=Path("/host-ws"),
        network=network,
        docker_executable=str(script),
    )


def _log_lines(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines()


# 功能：验证 ensure_running 首次创建、二次探活后直接返回（幂等）
# 设计：容器缺失（missing）时创建；把 state 切为 running 后再次 ensure 只探活不重建——
#       生命周期幂等性的协议级证据
async def test_ensure_running_creates_once(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    monkeypatch.setenv("FAKE_DOCKER_STATE", "running")
    await runtime.ensure_running()
    lines = _log_lines(fake_docker[1])
    assert lines.count("run") == 1
    assert lines.count("inspect") == 2


# 功能：验证 inspect 探活时同步容器 ID，exec 无需重新创建即可使用
# 设计：容器残留（running）场景下跳过创建直接 exec，验证残留容器的复用路径
async def test_existing_container_reused_for_exec(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_STATE", "running")
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    result = await runtime.exec("echo hi", cwd="/workspace", timeout=5)
    assert result.output == b"fake-exec-output\n"
    lines = _log_lines(fake_docker[1])
    assert lines.count("run") == 0  # 容器已存在，不重建


# 功能：验证创建命令以 argv 数组构造（bind mount + workdir + 常驻命令）
# 设计：逐行断言完整 argv，确保无宿主 shell 拼接、挂载与工作目录符合设计
async def test_create_argv_shape(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    lines = _log_lines(fake_docker[1])
    run_line = next(i for i, line in enumerate(lines) if line == "run")
    assert lines[run_line + 1 : run_line + 9] == [
        "--detach",
        "--name",
        "kama-sandbox",
        "--volume",
        "/host-ws:/workspace",
        "--workdir",
        "/workspace",
        "python:3.12-slim",
    ]
    assert lines[run_line + 9:] == ["sleep", "infinity"]


# 功能：验证 network=False 时创建命令携带 --network none
# 设计：显式关网场景断言隔离参数存在，与默认联网路径区分
async def test_create_without_network(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch, network=False)
    await runtime.ensure_running()
    lines = _log_lines(fake_docker[1])
    assert "--network" in lines
    assert "none" in lines


# 功能：验证 inspect 输出非 true（容器 stopped）时走创建路径
# 设计：stopped 状态探活失败，断言随后发起 run——容器未运行不阻塞重建
async def test_inspect_unhealthy_triggers_create(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_STATE", "stopped")
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    lines = _log_lines(fake_docker[1])
    assert "run" in lines


# 功能：验证 daemon 不可用时 ensure_running 抛 SandboxUnavailableError
# 设计：stderr 关键词故障注入 inspect 路径，验证探活失败在 daemon 故障时 fail closed
async def test_daemon_unavailable_fails_closed(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FAKE_DOCKER_MODE",
        "stderr_Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
    )
    runtime = _runtime(fake_docker, monkeypatch)
    with pytest.raises(SandboxUnavailableError):
        await runtime.ensure_running()


# 功能：验证创建失败按 stderr 关键词分类为镜像错误或创建失败
# 设计：参数化 manifest unknown / pull access denied / 通用失败 三类 stderr，
#       验证 runtime 层错误分类契约
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("stderr_manifest for python:3.12-slim not found: manifest unknown", SandboxImageError),
        ("stderr_pull access denied for python:3.12-slim, repository does not exist", SandboxImageError),
        ("stderr_docker: Error response from daemon: OCI runtime create failed", SandboxCreationFailedError),
    ],
)
async def test_create_failure_classification(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: type[SandboxUnavailableError | SandboxImageError | SandboxCreationFailedError],
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_MODE", mode)
    runtime = _runtime(fake_docker, monkeypatch)
    with pytest.raises(expected):
        await runtime.ensure_running()


# 功能：验证 exec 把 command 作为独立 argv 原样透传（宿主 shell 注入红线）
# 设计：注入含引号与分号的恶意命令，断言 log 中出现完整独立 argv 行——
#       证明命令未被宿主 shell 拼接或二次解析
async def test_exec_argv_isolation(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    evil = "echo hello'; rm -rf /"
    result = await runtime.exec(evil, cwd="/workspace", timeout=5)
    assert result.returncode == 0
    assert result.output == b"fake-exec-output\n"
    lines = _log_lines(fake_docker[1])
    assert "exec" in lines
    assert evil in lines  # 独立 argv 完整保留，未拆分成多个宿主参数


# 功能：验证 exec 超时抛 SandboxTimeoutError 并清理 docker 子进程
# 设计：hang 故障 + 短超时触发超时路径，断言异常类型而非挂起
async def test_exec_timeout_raises_sandbox_timeout(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_MODE", "hang")
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    with pytest.raises(SandboxTimeoutError):
        await runtime.exec("sleep 30", cwd="/workspace", timeout=0.2)


# 功能：验证容器未就绪时 exec 抛 ContainerNotReadyError
# 设计：跳过 ensure_running 直接 exec，验证调用顺序前置条件
async def test_exec_before_ready(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch)
    with pytest.raises(ContainerNotReadyError):
        await runtime.exec("ls", cwd="/workspace", timeout=5)


# 功能：验证 close 幂等——重复调用只发起一次容器删除
# 设计：两次 close 后断言按容器 ID 的 rm（close 专属）仅一次；
#       创建前的 rm -f <name>（幂等清理）与 close 的 rm -f <ID> 是不同调用，需区分
async def test_close_idempotent(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    await runtime.close()
    await runtime.close()
    lines = _log_lines(fake_docker[1])
    assert lines.count("container123") == 1  # 只有 close 的 rm -f <ID> 会带容器 ID


# 功能：验证 exec 的取消信号原样传播且 docker 子进程被清理
# 设计：hang 故障 + 短延迟 cancel，断言 CancelledError 传播（控制流而非工具失败）
async def test_exec_cancellation_propagates(
    fake_docker: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_MODE", "hang")
    runtime = _runtime(fake_docker, monkeypatch)
    await runtime.ensure_running()
    task = asyncio.create_task(runtime.exec("sleep 30", cwd="/workspace", timeout=30))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
