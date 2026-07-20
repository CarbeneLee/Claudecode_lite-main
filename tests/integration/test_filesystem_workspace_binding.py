from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.runner import AgentRunner


# 从 LLM messages 中读取最近一次工具结果文本
def _latest_tool_result(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            result = block.get("content")
            if isinstance(result, str):
                return result
    return ""


class _ToolThenEndProvider:
    # 初始化单次工具调用定义与结果捕获状态
    def __init__(self, tool_name: str, tool_input: dict[str, object]) -> None:
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._called = False
        self.tool_result = ""

    # 首轮请求指定工具，次轮捕获真实 ToolResult 并结束
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if not self._called:
            self._called = True
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="tool-1",
                        name=self._tool_name,
                        input=self._tool_input,
                    )
                ],
            )
        self.tool_result = _latest_tool_result(messages)
        return LlmResponse(stop_reason="end_turn", text="done")


class _SubagentReadProvider:
    # 初始化 parent/child run 状态和 child read 结果
    def __init__(self) -> None:
        self._parent_run_id: str | None = None
        self._child_run_id: str | None = None
        self.child_read_result = ""

    # 驱动 parent spawn、child read、child 完成和 parent 完成四步流程
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if self._parent_run_id is None:
            self._parent_run_id = run_id
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="spawn-1",
                        name="spawn_agent",
                        input={
                            "description": "read marker",
                            "prompt": "Read marker.txt",
                        },
                    )
                ],
            )

        if run_id != self._parent_run_id:
            if self._child_run_id is None:
                self._child_run_id = run_id
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="read-1",
                            name="read_file",
                            input={"path": "marker.txt"},
                        )
                    ],
                )
            self.child_read_result = _latest_tool_result(messages)
            return LlmResponse(stop_reason="end_turn", text="child done")

        return LlmResponse(stop_reason="end_turn", text="parent done")


# 创建使用 fake provider 与显式 workspace 的真实 AgentRunner
def _runner(workspace: Path, provider: LLMProvider, runs_dir: Path) -> AgentRunner:
    config = KamaConfig()
    config.agent.max_steps = 6
    return AgentRunner(
        config,
        workspace_root=workspace,
        provider=provider,
        runs_dir=runs_dir,
    )


# 功能：验证 daemon cwd=A 时真实 loop 的 read_file 只读取 Runner workspace=B
# 设计：A/B 放置同名 marker，由 fake LLM 请求相对路径并捕获 ToolResult
async def test_runner_read_file_uses_workspace_not_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    (daemon_cwd / "marker.txt").write_text("daemon-marker", encoding="utf-8")
    (workspace / "marker.txt").write_text("workspace-marker", encoding="utf-8")
    monkeypatch.chdir(daemon_cwd)
    provider = _ToolThenEndProvider("read_file", {"path": "marker.txt"})

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture("read")

    assert outcome.status == "success"
    assert provider.tool_result == "workspace-marker"


# 功能：验证真实 Runner search_code 只搜索绑定 workspace 而非 daemon cwd
# 设计：两个根目录写入不同 marker，经 LLM→registry→tool 路径检查相对路径输出
async def test_runner_search_code_uses_workspace_not_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    (daemon_cwd / "daemon.txt").write_text("daemon-only-needle", encoding="utf-8")
    (workspace / "workspace.txt").write_text(
        "workspace-only-needle",
        encoding="utf-8",
    )
    monkeypatch.chdir(daemon_cwd)
    provider = _ToolThenEndProvider("search_code", {"query": "only-needle"})

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture(
        "search"
    )

    assert outcome.status == "success"
    assert "workspace.txt:1: workspace-only-needle" in provider.tool_result
    assert "daemon.txt" not in provider.tool_result
    assert str(workspace.resolve(strict=True)) not in provider.tool_result


# 功能：验证 workspace A/B 的真实 Runner 分别读取各自 marker
# 设计：两个 provider 和 Runner 使用相同 logical path，交叉比较 ToolResult
async def test_two_runners_read_distinct_workspace_markers(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    (workspace_a / "marker.txt").write_text("marker-a", encoding="utf-8")
    (workspace_b / "marker.txt").write_text("marker-b", encoding="utf-8")
    provider_a = _ToolThenEndProvider("read_file", {"path": "marker.txt"})
    provider_b = _ToolThenEndProvider("read_file", {"path": "marker.txt"})

    await _runner(workspace_a, provider_a, tmp_path / "runs-a").run_and_capture("read a")
    await _runner(workspace_b, provider_b, tmp_path / "runs-b").run_and_capture("read b")

    assert provider_a.tool_result == "marker-a"
    assert provider_b.tool_result == "marker-b"


# 功能：验证真实 loop 的 write_file 只在 Session workspace 创建文件
# 设计：process cwd 与 workspace 分离，fake LLM 写相对 result.txt 后检查两处文件系统
async def test_runner_write_file_creates_only_in_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    daemon_cwd.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(daemon_cwd)
    provider = _ToolThenEndProvider(
        "write_file",
        {"path": "result.txt", "content": "workspace-result"},
    )

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture("write")

    assert outcome.status == "success"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "workspace-result"
    assert not (daemon_cwd / "result.txt").exists()


# 功能：验证真实 loop 的 list_dir 不返回 .git 与 .env
# 设计：workspace 同时放置敏感和普通条目，由 fake LLM 捕获过滤后的 tree
async def test_runner_list_dir_filters_sensitive_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (workspace / "visible.txt").write_text("visible", encoding="utf-8")
    provider = _ToolThenEndProvider("list_dir", {"path": "."})

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture("list")

    assert outcome.status == "success"
    assert ".git" not in provider.tool_result
    assert ".env" not in provider.tool_result
    assert "visible.txt" in provider.tool_result


# 功能：验证真实 loop 的 Bash starts in workspace
# 设计：fake LLM 执行 pwd 并捕获 ToolResult，明确只测试启动目录而非 OS sandbox
async def test_runner_bash_starts_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ToolThenEndProvider("bash", {"command": "pwd"})

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture("pwd")

    assert outcome.status == "success"
    assert provider.tool_result.strip() == str(workspace.resolve(strict=True))


# 功能：验证 subagent read_file 继承 parent workspace
# 设计：同一 fake provider 驱动真实 spawn_agent 与 child read loop，捕获 child ToolResult
async def test_subagent_read_file_uses_parent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "marker.txt").write_text("parent-workspace-marker", encoding="utf-8")
    provider = _SubagentReadProvider()

    outcome = await _runner(workspace, provider, tmp_path / "runs").run_and_capture(
        "delegate read"
    )

    assert outcome.status == "success"
    assert provider.child_read_result == "parent-workspace-marker"
