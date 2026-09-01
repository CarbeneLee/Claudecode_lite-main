from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.agents.loader import AgentProfile
from kama_claude.core.bus.events import (
    LlmTokenEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.grounding import (
    ArchitectureSliceDraft,
    ArchitectureSliceService,
    ToolObservationCollector,
)
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runner import AgentRunner
from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.executors import ContainerExecutor, HostExecutor
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader
from kama_claude.core.subagent import tool as subagent_tool_module
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

_REQUIREMENT_CONTRACT = (
    "Before changing the workspace, create a concise requirement contract from every "
    "explicit acceptance criterion. For each item, record the required observable "
    "behavior, relevant failure or invalid-input behavior, any side-effect or state "
    "invariant, and the evidence you plan to use for verification. Keep this checklist "
    "visible in the conversation as you work, and update each item as implemented, "
    "verified, or unchecked. Before finishing, review every item. Do not assume unchecked "
    "items are complete: verify them when possible, otherwise clearly report the "
    "limitation. Keep the contract brief and auditable; do not expose private "
    "chain-of-thought or force any particular tool."
)
_STATE_TRANSITION_PROTOCOL = (
    "When a task changes persistent or shared state through multiple operations, briefly "
    "map the pre-state, each mutation point, every later operation that can fail, and the "
    "required post-state after success or failure. Before finishing, exercise at least "
    "one failure after an earlier mutation succeeds, and verify that rollback or "
    "compensation preserves the stated invariant. Do not apply this protocol to tasks "
    "without multi-step side effects."
)
_REPOSITORY_CHANGE_DISCIPLINE = """## Repository Change Discipline
Prefer editing existing files to creating new ones
Don't add features, refactor, or introduce abstractions beyond what the task requires
Don't design for hypothetical future requirements
A bug fix doesn't need surrounding cleanup"""


def _make_provider(result_text: str = "child done") -> Any:
    provider = AsyncMock()
    provider.chat = AsyncMock(
        return_value=LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text=result_text,
            usage=UsageStats(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                context_pct=0.01,
            ),
        )
    )
    return provider


def _make_tool(
    tmp_path: Path,
    provider: Any = None,
    depth: int = 0,
    journal: Any = None,
    sandbox_manager: SandboxManager | None = None,
    store: SessionStore | None = None,
    planning_only: bool = False,
) -> tuple[SpawnAgentTool, BackgroundTaskRegistry, EventBus]:
    bus = EventBus()
    registry = BackgroundTaskRegistry()
    tool = SpawnAgentTool(
        provider=provider or _make_provider(),
        workspace_root=tmp_path.resolve(),
        parent_bus=bus,
        parent_run_id="parent-run-01",
        permission_manager=None,
        max_steps=5,
        task_registry=registry,
        runs_dir=tmp_path,
        session_id="sess-test",
        depth=depth,
        journal=journal,
        sandbox_manager=sandbox_manager,
        store=store,
        planning_only=planning_only,
    )
    return tool, registry, bus


# 通过真实 grounding artifact 构造 trusted Planner 的 terminal submit 输入
async def _prepare_orchestrate_grounding(
    tmp_path: Path,
    *,
    selected_approach: str,
) -> tuple[Path, SessionStore, dict[str, object]]:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "target.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    collector = ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="read-target",
            tool_name="read_file",
            params={"path": "src/target.py"},
            ts="t1",
        )
    )
    await collector.handle(
        ToolCallFinishedEvent(
            run_id="explorer-run",
            tool_use_id="read-target",
            tool_name="read_file",
            elapsed_ms=1,
            output=source.read_text(encoding="utf-8"),
            ts="t2",
        )
    )
    store = SessionStore(tmp_path / "sessions")
    slice_service = ArchitectureSliceService(
        workspace_root=workspace,
        run_id="explorer-run",
        goal="Change target behavior",
        collector=collector,
        session_id="sess-orchestrate",
        store=store,
    )
    architecture_slice = slice_service.submit(
        ArchitectureSliceDraft(
            relevant_modules=("src/target.py",),
            related_tests=(),
            existing_patterns=("single module edit",),
            likely_change_targets=("src/target.py",),
            evidence_tool_call_ids=("read-target",),
            completeness="complete_for_task",
            confidence=0.9,
        )
    )
    draft: dict[str, object] = {
        "architecture_slice_id": architecture_slice.slice_id,
        "architecture_slice_version": architecture_slice.version,
        "architecture_mode": "preserve",
        "selected_approach": selected_approach,
        "existing_patterns_reused": ["single module edit"],
        "requirements": [
            {
                "requirement_id": "R1",
                "statement": "Change target behavior.",
                "required": True,
            }
        ],
        "intended_changes": [
            {
                "change_id": "C1",
                "description": "Update the existing target.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["read-target"],
            }
        ],
        "files_to_modify": ["src/target.py"],
        "files_to_create": [],
        "allowed_capabilities": ["read_file", "write_file"],
        "dependency_changes": [],
        "protocol_or_schema_changes": [],
        "verification_plan": [
            {"requirement_ids": ["R1"], "strategy": "Run focused tests."}
        ],
        "non_goals": ["No surrounding cleanup."],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
    }
    return workspace, store, draft


class _OrchestrateRuntimeProvider:
    MAX_ROOT_PROVIDER_CALLS = 4
    MAX_TOTAL_PROVIDER_CALLS = 8
    MAX_TOTAL_CHILD_SPAWNS = 3

    # 按 root/planner/executor tool schema 驱动真实三阶段 AgentLoop 流程
    def __init__(
        self,
        draft: dict[str, object],
        root_run_id: str,
        *,
        oversized: bool,
    ) -> None:
        self._draft = draft
        self._root_run_id = root_run_id
        self._oversized = oversized
        self._root_calls = 0
        self._planner_calls = 0
        self._executor_calls = 0
        self._total_provider_calls = 0
        self._child_spawns = 0
        self.planner_result_contents: list[str] = []
        self.executor_prompts: list[str] = []

    # 返回规划提交、协调派生或终态响应并记录 executor 是否被派生
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
        del bus, step, system
        self._total_provider_calls += 1
        if self._total_provider_calls > self.MAX_TOTAL_PROVIDER_CALLS:
            raise AssertionError(
                "orchestrate provider-call budget exceeded: "
                f"{self._total_provider_calls}>{self.MAX_TOTAL_PROVIDER_CALLS}"
            )
        names = {str(schema.get("name")) for schema in tool_schemas}
        if "planner_decision_submit" in names:
            if run_id == self._root_run_id:
                raise AssertionError("root unexpectedly received planner tool schema")
            self._planner_calls += 1
            if self._planner_calls == 1:
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id=f"planner-submit-{self._planner_calls}",
                            name="planner_decision_submit",
                            input=self._draft,
                        )
                    ],
                )
            if self._planner_calls == 2:
                return LlmResponse(
                    stop_reason="end_turn",
                    text="UNTRUSTED CHILD SUMMARY",
                )
            raise AssertionError(
                f"unexpected planner provider call #{self._planner_calls}"
            )
        if run_id == self._root_run_id:
            self._root_calls += 1
            if self._root_calls > self.MAX_ROOT_PROVIDER_CALLS:
                raise AssertionError(
                    "orchestrate root-call budget exceeded: "
                    f"{self._root_calls}>{self.MAX_ROOT_PROVIDER_CALLS}"
                )
            if self._root_calls == 1:
                self._child_spawns += 1
                if self._child_spawns > self.MAX_TOTAL_CHILD_SPAWNS:
                    raise AssertionError(
                        "orchestrate child-spawn budget exceeded: "
                        f"{self._child_spawns}>{self.MAX_TOTAL_CHILD_SPAWNS}"
                    )
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="spawn-planner",
                            name="spawn_agent",
                            input={
                                "description": "规划任务",
                                "prompt": "Change target behavior",
                                "subagent_type": "planner",
                            },
                        )
                    ],
                )
            if self._root_calls == 2:
                transcript = json.dumps(messages, ensure_ascii=False, default=str)
                self.planner_result_contents.extend(
                    str(block.get("content"))
                    for message in messages
                    for block in message.get("content", [])
                    if isinstance(block, dict)
                    and block.get("type") == "tool_result"
                )
                if self._oversized:
                    if "planner-result-too-large" not in transcript:
                        raise AssertionError(
                            "oversized scenario did not receive planner-result-too-large"
                        )
                    return LlmResponse(
                        stop_reason="end_turn",
                        text="planner failed; stop",
                    )
                prompt = transcript
                self.executor_prompts.append(prompt)
                self._child_spawns += 1
                if self._child_spawns > self.MAX_TOTAL_CHILD_SPAWNS:
                    raise AssertionError(
                        "orchestrate child-spawn budget exceeded: "
                        f"{self._child_spawns}>{self.MAX_TOTAL_CHILD_SPAWNS}"
                    )
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="spawn-executor",
                            name="spawn_agent",
                            input={
                                "description": "执行计划",
                                "prompt": prompt,
                                "subagent_type": "executor",
                            },
                        )
                    ],
                )
            if self._root_calls == 3 and not self._oversized:
                return LlmResponse(stop_reason="end_turn", text="orchestrate done")
            raise AssertionError(
                "unexpected root provider call "
                f"#{self._root_calls} for oversized={self._oversized}"
            )
        if {"bash", "write_file"}.issubset(names):
            self._executor_calls += 1
            if self._executor_calls == 1:
                return LlmResponse(stop_reason="end_turn", text="executor done")
            raise AssertionError(
                f"unexpected executor provider call #{self._executor_calls}"
            )
        raise AssertionError(f"unexpected child tool schema for run_id={run_id}")


class _MissingGroundingPlannerProvider:
    MAX_TOTAL_PROVIDER_CALLS = 2

    # 绑定两次固定 Planner submit，并在任何额外调用时立即失败
    def __init__(self, draft: dict[str, object]) -> None:
        self._draft = draft
        self.total_provider_calls = 0

    # 第一次制造缺失 grounding，第二次确认可操作反馈后重试并等待 runtime 终止
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
        del bus, run_id, step, system
        names = {str(schema.get("name")) for schema in tool_schemas}
        if "planner_decision_submit" not in names:
            raise AssertionError("missing-grounding provider reached a non-planner role")
        self.total_provider_calls += 1
        if self.total_provider_calls > self.MAX_TOTAL_PROVIDER_CALLS:
            raise AssertionError(
                "missing-grounding provider-call budget exceeded: "
                f"{self.total_provider_calls}>{self.MAX_TOTAL_PROVIDER_CALLS}"
            )
        if self.total_provider_calls == 2:
            transcript = json.dumps(messages, ensure_ascii=False, default=str)
            if "grounding artifact does not exist" not in transcript:
                raise AssertionError("planner did not receive the missing-grounding detail")
            if "subagent_type='explorer'" not in transcript:
                raise AssertionError("planner did not receive the Explorer recovery action")
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCallBlock(
                    id=f"missing-grounding-{self.total_provider_calls}",
                    name="planner_decision_submit",
                    input=self._draft,
                )
            ],
        )


class _PlannerExplorerGroundingProvider:
    MAX_TOTAL_PROVIDER_CALLS = 6
    MAX_PLANNER_CALLS = 3
    MAX_EXPLORER_CALLS = 3

    # 保存 exact decision draft 与每个角色的有限状态计数
    def __init__(self, draft: dict[str, object]) -> None:
        self._draft = draft
        self.total_provider_calls = 0
        self.planner_calls = 0
        self.explorer_calls = 0

    # 驱动 Planner→Explorer→grounding→decision 的唯一合法状态序列
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
        del bus, run_id, step
        self.total_provider_calls += 1
        if self.total_provider_calls > self.MAX_TOTAL_PROVIDER_CALLS:
            raise AssertionError(
                "planner-explorer provider-call budget exceeded: "
                f"{self.total_provider_calls}>{self.MAX_TOTAL_PROVIDER_CALLS}"
            )
        names = {str(schema.get("name")) for schema in tool_schemas}
        if "planner_decision_submit" in names:
            self.planner_calls += 1
            if self.planner_calls > self.MAX_PLANNER_CALLS:
                raise AssertionError("unexpected extra Planner provider call")
            if self.planner_calls == 1:
                contract = system or ""
                if 'subagent_type="explorer"' not in contract:
                    raise AssertionError("Planner contract omits the exact Explorer role")
                if "architecture_slice_submit" not in contract:
                    raise AssertionError("Planner contract omits the grounding commit")
                if "Do not guess" not in contract:
                    raise AssertionError("Planner contract does not forbid guessed identities")
                spawn_schema = next(
                    schema for schema in tool_schemas if schema.get("name") == "spawn_agent"
                )
                role_schema = spawn_schema["input_schema"]["properties"]["subagent_type"]
                if role_schema.get("enum") != ["explorer"]:
                    raise AssertionError("Planner spawn schema is not Explorer-only")
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="planner-spawn-explorer",
                            name="spawn_agent",
                            input={
                                "description": "Ground repository",
                                "prompt": "Change target behavior",
                                "subagent_type": "explorer",
                                "exploration_level": "standard",
                            },
                        )
                    ],
                )
            if self.planner_calls == 2:
                tool_results = [
                    str(block.get("content"))
                    for message in messages
                    for block in message.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                ]
                try:
                    slice_identity = next(
                        payload
                        for content in tool_results
                        for payload in [json.loads(content)]
                        if isinstance(payload, dict) and "slice_id" in payload
                    )
                except (json.JSONDecodeError, StopIteration) as exc:
                    raise AssertionError(
                        "Planner did not receive Explorer slice identity"
                    ) from exc
                self._draft["architecture_slice_id"] = slice_identity["slice_id"]
                self._draft["architecture_slice_version"] = slice_identity["version"]
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="planner-submit-decision",
                            name="planner_decision_submit",
                            input=self._draft,
                        )
                    ],
                )
            return LlmResponse(stop_reason="end_turn", text="plan committed")
        if "architecture_slice_submit" in names:
            self.explorer_calls += 1
            if self.explorer_calls > self.MAX_EXPLORER_CALLS:
                raise AssertionError("unexpected extra Explorer provider call")
            if self.explorer_calls == 1:
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="explorer-read",
                            name="read_file",
                            input={"path": "src/target.py"},
                        )
                    ],
                )
            if self.explorer_calls == 2:
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[
                        ToolCallBlock(
                            id="explorer-submit-slice",
                            name="architecture_slice_submit",
                            input={
                                "relevant_modules": ["src/target.py"],
                                "existing_patterns": ["single module edit"],
                                "likely_change_targets": ["src/target.py"],
                                "evidence_tool_call_ids": ["explorer-read"],
                                "completeness": "complete_for_task",
                                "confidence": 0.9,
                            },
                        )
                    ],
                )
            return LlmResponse(stop_reason="end_turn", text="grounding committed")
        raise AssertionError("provider reached an unexpected runtime role")


# 功能：验证 child run stream 注册严格早于 SubagentStartedEvent 发布
# 设计：fake journal 与真实 parent bus 共享顺序列表，执行最短前台 child lifecycle 比较边界顺序
async def test_child_stream_registers_before_subagent_started(tmp_path: Path) -> None:
    order: list[str] = []

    class RecordingJournal:
        # 记录 child run owner 与 parent session mapping 注册
        async def register_run(
            self,
            run_id: str,
            run_path: Path,
            *,
            session_id: str | None,
        ) -> object:
            order.append(f"register:{run_id}:{session_id}")
            return object()

    tool, _registry, bus = _make_tool(tmp_path, journal=RecordingJournal())

    async def collect(event: object) -> None:
        if isinstance(event, SubagentStartedEvent):
            order.append(f"event:{event.run_id}")

    bus.subscribe(collect)

    await tool.invoke({"description": "child", "prompt": "finish"})

    register_index = next(
        index for index, item in enumerate(order) if item.startswith("register:")
    )
    event_index = next(
        index for index, item in enumerate(order) if item.startswith("event:")
    )
    assert register_index < event_index


# 功能：验证 trusted Planner 在 child stream 注册前失败时不伪造 planner_run_id
# 设计：让 journal owner 在 pre-child registration 抛出稳定错误，直接检查 typed internal result 的身份边界
async def test_trusted_planner_pre_child_failure_has_no_planner_run_id(
    tmp_path: Path,
) -> None:
    from kama_claude.core.events.journal import JournalError

    class FailingJournal:
        # 在 child 创建前拒绝 journal registration
        async def register_run(
            self,
            run_id: str,
            run_path: Path,
            *,
            session_id: str | None,
        ) -> object:
            raise JournalError("registration failed")

    tool, _registry, _bus = _make_tool(
        tmp_path,
        journal=FailingJournal(),
        store=SessionStore(tmp_path / "sessions"),
    )

    result = await tool.run_trusted_planner_foreground(goal="inspect")

    assert result.status == "failed"
    assert result.planner_run_id is None
    assert result.failure_reason == "plan-event-append-failed"
    assert not hasattr(result, "summary")


# 功能：验证 trusted Planner typed result 不再携带自然语言 summary side channel
# 设计：直接构造 success/failure discriminated union，锁定唯一可返回的 identity/reason 字段
def test_trusted_planner_typed_result_has_no_summary_field() -> None:
    from kama_claude.core.planning import SubmittedDecisionIdentity
    from kama_claude.core.subagent.tool import (
        TrustedPlannerFailure,
        TrustedPlannerSuccess,
    )

    success = TrustedPlannerSuccess(
        status="success",
        planner_run_id="planner-1",
        decision_identity=SubmittedDecisionIdentity(
            decision_id="decision-1",
            version=1,
            snapshot_digest="snapshot",
            content_digest="content",
        ),
    )
    failure = TrustedPlannerFailure(
        status="failed",
        planner_run_id="planner-1",
        failure_reason="missing-terminal-decision",
    )

    assert not hasattr(success, "summary")
    assert not hasattr(failure, "summary")
    assert success.decision_identity.decision_id == "decision-1"
    assert failure.failure_reason == "missing-terminal-decision"


# 功能：验证 SpawnAgentTool 必须显式接收 workspace_root
# 设计：省略该 keyword 构造工具并断言 TypeError，防止引入 cwd fallback
def test_spawn_agent_requires_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        SpawnAgentTool(
            provider=_make_provider(),
            parent_bus=EventBus(),
            parent_run_id="parent",
            permission_manager=None,
            max_steps=5,
            task_registry=BackgroundTaskRegistry(),
            runs_dir=tmp_path,
            session_id="session",
        )


# 功能：验证 SpawnAgentTool 保存 canonical workspace root
# 设计：通过目录 symlink 构造工具并检查内部路径已 strict resolve
def test_spawn_agent_canonicalizes_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)

    tool, _, _ = _make_tool(alias)

    assert tool._workspace_root == workspace.resolve(strict=True)


# 功能：验证 nested SpawnAgentTool 继承 parent 的同一 canonical workspace
# 设计：直接构建 child registry 并检查其中 spawn_agent 的 workspace 状态
def test_nested_spawn_agent_inherits_workspace(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    nested = registry.get("spawn_agent")

    assert isinstance(nested, SpawnAgentTool)
    assert nested._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证注入 sandbox_manager 后 child registry 的 bash 使用容器执行器
# 设计：沿 child registry 组装路径检查 bash executor 类型，不执行命令
def test_child_registry_bash_uses_container_executor_when_sandbox_injected(
    tmp_path: Path,
) -> None:
    manager = SandboxManager(
        config=SandboxConfig(image="python:3.12-slim"),
        workspace_root=tmp_path.resolve(),
    )
    tool, _, _ = _make_tool(tmp_path, sandbox_manager=manager)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, ContainerExecutor)


# 功能：验证未注入 sandbox_manager 时 child registry 的 bash 保持宿主执行器
# 设计：默认路径断言 HostExecutor，防止子 agent 沙箱决策与顶层漂移
def test_child_registry_bash_uses_host_executor_without_sandbox(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, HostExecutor)


# 功能：验证 child registry 的 read/list 工具绑定 parent workspace
# 设计：直接检查 child registry 工具 resolver root，隔离 LLM 与文件系统执行
def test_child_registry_injects_workspace_into_read_and_list_tools(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    read_tool = registry.get("read_file")
    list_tool = registry.get("list_dir")

    assert isinstance(read_tool, ReadFileTool)
    assert isinstance(list_tool, ListDirTool)
    assert read_tool._resolver.root == tmp_path.resolve(strict=True)
    assert list_tool._resolver.root == tmp_path.resolve(strict=True)


# 功能：验证 Subagent child registry 注册绑定 parent workspace 的 search_code
# 设计：不运行 child loop，直接检查真实工具类型和 resolver canonical root
def test_child_registry_injects_workspace_into_search_code(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    search_tool = registry.get("search_code")

    assert isinstance(search_tool, SearchCodeTool)
    assert search_tool._resolver.root == tmp_path.resolve(strict=True)


# 功能：验证 Subagent profile allowlist 可单独允许或排除 search_code
# 设计：用两个最小 AgentProfile 构建 registry，交叉断言 search/read 工具存在性
def test_child_registry_filters_search_code_by_profile_allowlist(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)
    search_only = AgentProfile(
        name="searcher",
        description="",
        system_prompt="",
        allowed_tools=["search_code"],
    )
    read_only = AgentProfile(
        name="reader",
        description="",
        system_prompt="",
        allowed_tools=["read_file"],
    )

    search_registry = tool._build_child_registry(EventBus(), "search-run", search_only)
    read_registry = tool._build_child_registry(EventBus(), "read-run", read_only)

    assert isinstance(search_registry.get("search_code"), SearchCodeTool)
    assert search_registry.get("read_file") is None
    assert read_registry.get("search_code") is None
    assert isinstance(read_registry.get("read_file"), ReadFileTool)


# 功能：验证 child registry 的 write 工具与 Bash 继承 parent workspace
# 设计：检查 child 工具内部 root，不执行有副作用操作
def test_child_registry_injects_workspace_into_write_and_bash_tools(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    write_tool = registry.get("write_file")
    bash_tool = registry.get("bash")

    assert isinstance(write_tool, WriteFileTool)
    assert isinstance(bash_tool, BashTool)
    assert write_tool._resolver.root == tmp_path.resolve(strict=True)
    assert bash_tool._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证 nested subagent 的 filesystem 工具继续继承同一 workspace
# 设计：从 parent child registry 取 nested tool，再构建下一层 registry 检查四个工具 root
def test_nested_child_registry_keeps_workspace_bound_tools(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)
    child_registry = tool._build_child_registry(EventBus(), "child-run", None)
    nested = child_registry.get("spawn_agent")
    assert isinstance(nested, SpawnAgentTool)

    nested_registry = nested._build_child_registry(EventBus(), "nested-run", None)
    read_tool = nested_registry.get("read_file")
    write_tool = nested_registry.get("write_file")
    list_tool = nested_registry.get("list_dir")
    bash_tool = nested_registry.get("bash")

    assert isinstance(read_tool, ReadFileTool)
    assert isinstance(write_tool, WriteFileTool)
    assert isinstance(list_tool, ListDirTool)
    assert isinstance(bash_tool, BashTool)
    assert read_tool._resolver.root == tmp_path.resolve(strict=True)
    assert write_tool._resolver.root == tmp_path.resolve(strict=True)
    assert list_tool._resolver.root == tmp_path.resolve(strict=True)
    assert bash_tool._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证不同 workspace 的 subagent 分别加载各自 profile 与 project context
# 设计：A/B 使用同名 profile 和不同 context，捕获 provider system 参数并交叉排除污染
async def test_subagents_isolate_profile_and_context_by_workspace(tmp_path: Path) -> None:
    systems: list[str] = []
    for name in ("a", "b"):
        workspace = tmp_path / f"workspace-{name}"
        agents = workspace / ".kama" / "agents"
        agents.mkdir(parents=True)
        (workspace / ".kama" / "context.md").write_text(
            f"context-{name}",
            encoding="utf-8",
        )
        (agents / "planner.toml").write_text(
            '[agent]\ndescription = "local"\n'
            f'system_prompt = "profile-{name}"\nallowed_tools = ["read_file"]\n',
            encoding="utf-8",
        )
        provider = _make_provider()
        tool, _, _ = _make_tool(
            workspace,
            provider,
            store=SessionStore(tmp_path / f"sessions-{name}"),
        )

        await tool.invoke(
            {
                "description": "inspect",
                "prompt": "inspect workspace",
                "subagent_type": "planner",
            }
        )

        system = provider.chat.await_args.kwargs["system"]
        assert isinstance(system, str)
        systems.append(system)

    assert "profile-a" in systems[0]
    assert "context-a" in systems[0]
    assert "profile-b" not in systems[0]
    assert "context-b" not in systems[0]
    assert "profile-b" in systems[1]
    assert "context-b" in systems[1]
    assert "profile-a" not in systems[1]
    assert "context-a" not in systems[1]
    assert "PlannerDecision" in systems[0]
    assert "PlannerDecision" in systems[1]
    assert _STATE_TRANSITION_PROTOCOL not in systems[0]
    assert _STATE_TRANSITION_PROTOCOL not in systems[1]
    assert systems[0].count(_REPOSITORY_CHANGE_DISCIPLINE) == 1
    assert systems[1].count(_REPOSITORY_CHANGE_DISCIPLINE) == 1


# 功能：验证 runtime trusted planner contract 覆盖同名 custom profile 的 prompt
# 设计：local profile 只提供自定义文本，真实 child provider 捕获最终 system prompt 并检查提交要求
async def test_custom_planner_cannot_remove_trusted_contract(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner prompt"\n'
        'allowed_tools = ["read_file", "spawn_agent", "task_create", "bash"]\n',
        encoding="utf-8",
    )
    provider = _make_provider()
    tool, _, _ = _make_tool(
        tmp_path,
        provider,
        store=SessionStore(tmp_path / "sessions"),
    )

    result = await tool.invoke(
        {
            "description": "plan",
            "prompt": "inspect",
            "subagent_type": "planner",
        }
    )

    assert result.is_error is True
    system = provider.chat.await_args.kwargs["system"]
    assert "custom planner prompt" in system
    assert "planner_decision_submit" in system
    assert "terminal" in system.lower()
    assert 'subagent_type="explorer"' in system
    assert "architecture_slice_submit" in system
    assert "Do not guess" in system


# 功能：验证 trusted Planner 第二次无 grounding 提交后由既有 terminal_reason 边界立即停止
# 设计：两状态 provider 禁止第三次调用，并用真实 SpawnAgentTool/AgentLoop 断言单 child 有限失败
async def test_trusted_planner_repeated_missing_grounding_stops_after_two_calls(
    tmp_path: Path,
) -> None:
    workspace, store, draft = await _prepare_orchestrate_grounding(
        tmp_path,
        selected_approach="Keep the existing target module.",
    )
    provider = _MissingGroundingPlannerProvider(draft)
    tool, _, bus = _make_tool(
        workspace,
        provider,
        store=store,
        planning_only=True,
    )
    started_runs: list[str] = []

    # 记录真实 lifecycle 中创建的 child 数量，防止隐藏派生
    async def collect(event: object) -> None:
        if isinstance(event, SubagentStartedEvent):
            started_runs.append(event.run_id)

    bus.subscribe(collect)

    result = await tool.run_trusted_planner_foreground(goal="Change target behavior")

    assert result.status == "failed"
    assert result.failure_reason == "planning-grounding-missing"
    assert provider.total_provider_calls == provider.MAX_TOTAL_PROVIDER_CALLS
    assert len(started_runs) == 1


# 功能：验证 trusted Planner 依契约派生 Explorer 后能提交真实 grounding 与 terminal decision
# 设计：六状态 provider 驱动两个真实 child loop，并检查 store 产物与精确调用/派生预算
async def test_trusted_planner_grounds_with_explorer_before_terminal_submit(
    tmp_path: Path,
) -> None:
    workspace, store, draft = await _prepare_orchestrate_grounding(
        tmp_path,
        selected_approach="Keep the existing target module.",
    )
    runtime_draft = dict(draft)
    runtime_draft["intended_changes"] = [
        {
            "change_id": "C1",
            "description": "Update the existing target.",
            "requirement_ids": ["R1"],
            "target_paths": ["src/target.py"],
            "evidence_refs": ["explorer-read"],
        }
    ]
    provider = _PlannerExplorerGroundingProvider(runtime_draft)
    tool, _, bus = _make_tool(
        workspace,
        provider,
        store=store,
        planning_only=True,
    )
    started_runs: list[str] = []

    # 记录 Planner 与 Explorer 两层 child lifecycle，任何隐藏派生都会破坏精确计数
    async def collect(event: object) -> None:
        if isinstance(event, SubagentStartedEvent):
            started_runs.append(event.run_id)

    bus.subscribe(collect)

    result = await tool.run_trusted_planner_foreground(goal="Change target behavior")

    assert result.status == "success"
    assert provider.total_provider_calls == provider.MAX_TOTAL_PROVIDER_CALLS
    assert provider.planner_calls == provider.MAX_PLANNER_CALLS
    assert provider.explorer_calls == provider.MAX_EXPLORER_CALLS
    assert len(started_runs) == 2
    grounding = store.read_grounding("sess-test")
    assert grounding is not None
    assert len(grounding["architecture_slices"]) == 1
    assert len(store.list_decisions("sess-test")) == 1


@pytest.mark.parametrize(
    ("oversized", "executor_expected"),
    [(False, True), (True, False)],
)
# 功能：验证真实 /orchestrate AgentLoop 将完整 Planner ToolResult 交给 executor，超限则停止派生
# 设计：运行 root→planner→executor 的真实 SpawnAgent 链路，用同一 provider 记录 prompt 和 executor 派生边界
async def test_orchestrate_runtime_uses_full_planner_result_or_fails_closed(
    tmp_path: Path,
    oversized: bool,
    executor_expected: bool,
) -> None:
    approach = "x" * 20_000 if oversized else "Keep the existing target module."
    workspace, store, draft = await _prepare_orchestrate_grounding(
        tmp_path,
        selected_approach=approach,
    )
    session = Session(
        id="sess-orchestrate",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=workspace.resolve(),
    )
    store.write_meta(session)
    loader = SkillLoader(workspace)
    skill = loader.resolve("orchestrate")
    assert skill is not None
    root_run_id = "orchestrate-root"
    provider = _OrchestrateRuntimeProvider(
        draft,
        root_run_id,
        oversized=oversized,
    )
    journal = EventJournalCoordinator()
    await journal.register_session(session.id, store.session_dir(session.id))
    runner = AgentRunner(
        KamaConfig(),
        workspace_root=workspace,
        provider=provider,
        runs_dir=tmp_path / "runs",
        journal=journal,
    )

    try:
        outcome = await runner.run_and_capture(
            loader.render_prompt(skill, "Change target behavior"),
            run_id=root_run_id,
            session=session,
            store=store,
            system_prompt_override=skill.system_prompt_template,
            tool_whitelist=skill.allowed_tools,
        )
    finally:
        await journal.close()

    assert outcome.status == "success"
    assert bool(provider.executor_prompts) is executor_expected
    if executor_expected:
        planner_result_transcript = provider.executor_prompts[0]
        messages = json.loads(planner_result_transcript)
        tool_results = [
            block
            for message in messages
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        planner_result = next(
            json.loads(str(block["content"]))
            for block in tool_results
            if "planner_decision" in str(block.get("content"))
        )
        decision_payload = planner_result["planner_decision"]
        assert decision_payload["schema_version"] == 2
        assert "architecture_slice_content_digest" in decision_payload
        assert "verification_plan" in decision_payload
        assert "executor done" not in planner_result_transcript
    else:
        assert any(
            "planner-result-too-large" in content
            for content in provider.planner_result_contents
        )
    expected_root_calls = 3 if executor_expected else 2
    expected_provider_calls = 6 if executor_expected else 4
    assert provider._root_calls == expected_root_calls
    assert provider._total_provider_calls == expected_provider_calls
    assert provider._total_provider_calls <= provider.MAX_TOTAL_PROVIDER_CALLS
    assert provider._child_spawns == (2 if executor_expected else 1)
    assert provider._planner_calls == 2
    assert provider._executor_calls == int(executor_expected)
    with pytest.raises(AssertionError, match="unexpected root provider call"):
        await provider.chat(
            messages=[],
            tool_schemas=[],
            bus=EventBus(),
            run_id=root_run_id,
        )


# 功能：验证 trusted planner registry 移除 task 与 mutation 工具但保留 Explorer delegation
# 设计：使用同名 custom profile 请求超集 allowlist，检查 builtin 上界和 planner 专属工具集合
def test_trusted_planner_registry_excludes_task_and_mutation_tools(
    tmp_path: Path,
) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["read_file", "list_dir", "search_code", "spawn_agent", '
        '"planner_decision_submit", "task_create", "task_update", "bash", "write_file"]\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )
    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )

    assert registry.get("read_file") is not None
    assert registry.get("list_dir") is not None
    assert registry.get("search_code") is not None
    assert registry.get("spawn_agent") is not None
    assert registry.get("planner_decision_submit") is not None
    assert registry.get("task_create") is None
    assert registry.get("task_update") is None
    assert registry.get("bash") is None
    assert registry.get("write_file") is None


# 功能：验证 trusted Planner 看到的 spawn schema 只公开当前获准的 Explorer 角色
# 设计：构造真实 Planner child registry 并检查消费侧 schema，避免只测试 invoke 后的兜底拒绝
def test_trusted_planner_spawn_schema_exposes_only_explorer(tmp_path: Path) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )
    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )
    spawn = registry.get("spawn_agent")
    assert spawn is not None

    role_schema = spawn.input_schema["properties"]["subagent_type"]

    assert role_schema["enum"] == ["explorer"]
    assert "Only allowed in this context: explorer" in role_schema["description"]


# 功能：验证 custom Planner 空工具集合不会扩张，且无法提交时只返回安全失败摘要
# 设计：先检查最终 schema，再运行真实 child 断言 missing-terminal failure 不泄漏模型计划文本
async def test_custom_planner_empty_tools_remain_empty(tmp_path: Path) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = []\nallowed_subagent_types = []\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        _make_provider("直接修改 src/a.py，然后运行 bash ..."),
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )

    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )

    assert registry.tool_schemas() == []

    result = await tool.invoke(
        {
            "description": "plan",
            "prompt": "inspect",
            "subagent_type": "planner",
        }
    )

    assert result.is_error is True
    assert result.error_type == "command_failed"
    assert "missing-terminal-decision" in result.content
    assert "src/a.py" not in result.content
    assert "运行 bash" not in result.content


# 功能：验证 custom Planner 工具子集只保留显式声明且不补回其他 trusted capability
# 设计：只请求 read_file 与 submit，按 schema 顺序断言 list/search/spawn 和 mutation 均不可见
def test_custom_planner_tool_subset_is_preserved(tmp_path: Path) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["read_file", "planner_decision_submit"]\n'
        'allowed_subagent_types = []\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )

    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )

    assert [schema["name"] for schema in registry.tool_schemas()] == [
        "read_file",
        "planner_decision_submit",
    ]


# 功能：验证 custom Planner 工具超集最终只保留 trusted upper bound
# 设计：请求所有越权工具并保留合法工具，断言 schema 顺序稳定且越权名称全部消失
def test_custom_planner_tool_superset_is_narrowed(tmp_path: Path) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["read_file", "list_dir", "search_code", "spawn_agent", '
        '"planner_decision_submit", "bash", "write_file", "task_create", "task_update"]\n'
        'allowed_subagent_types = ["explorer", "executor", "reviewer"]\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )

    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )

    assert [schema["name"] for schema in registry.tool_schemas()] == [
        "read_file",
        "list_dir",
        "search_code",
        "planner_decision_submit",
        "spawn_agent",
    ]
    assert profile.allowed_subagent_types == ["explorer"]


# 功能：验证 spawn_agent 可见但空 child-type 集合不会自动恢复 Explorer
# 设计：保留 delegation schema，实际调用用稳定 invalid_input 检查 child authorization 与 tool visibility 分离
async def test_custom_planner_empty_child_types_reject_spawn(tmp_path: Path) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["spawn_agent"]\nallowed_subagent_types = []\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="inspect",
        run_id="planner-run",
    )
    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )
    spawn = registry.get("spawn_agent")
    assert spawn is not None

    result = await spawn.invoke(
        {
            "description": "explore",
            "prompt": "inspect",
            "subagent_type": "explorer",
        }
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "allowed subagent types: none" in result.content


# 功能：验证 custom Planner child-type 超集最多保留 Explorer
# 设计：通过真实 nested spawn 调用 executor/reviewer，断言 runtime authorization 仍受 trusted upper bound
async def test_custom_planner_child_type_superset_rejects_executor_and_reviewer(
    tmp_path: Path,
) -> None:
    from kama_claude.core.planning import PlannerDecisionService

    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["spawn_agent"]\n'
        'allowed_subagent_types = ["explorer", "executor", "reviewer"]\n',
        encoding="utf-8",
    )
    tool, _, _ = _make_tool(
        tmp_path,
        store=SessionStore(tmp_path / "sessions"),
    )
    profile = tool._profile_loader.load("planner")
    assert profile is not None
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=tool._store,
        goal="delegate",
        run_id="planner-run",
    )
    registry = tool._build_child_registry(
        EventBus(),
        "planner-run",
        profile,
        planner_service=service,
    )
    spawn = registry.get("spawn_agent")
    assert spawn is not None

    results = [
        await spawn.invoke(
            {
                "description": role,
                "prompt": "delegate",
                "subagent_type": role,
            }
        )
        for role in ("executor", "reviewer")
    ]

    assert all(result.is_error for result in results)
    assert all(result.error_type == "invalid_input" for result in results)
    assert all("allowed subagent types: explorer" in result.content for result in results)
    assert profile.allowed_subagent_types == ["explorer"]


# 功能：验证非 planner runtime role 不能通过 profile metadata 自称 trusted Planner
# 设计：custom 角色写入伪造 name 字段但请求 subagent_type=custom，断言不启用 terminal contract
async def test_runtime_role_identity_cannot_be_profile_forged(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "custom.toml").write_text(
        '[agent]\nname = "planner"\nsystem_prompt = "custom role"\n'
        'allowed_tools = ["read_file"]\n',
        encoding="utf-8",
    )
    provider = _make_provider("custom done")
    tool, _, _ = _make_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "custom",
            "prompt": "inspect",
            "subagent_type": "custom",
        }
    )

    assert result.is_error is False
    system = provider.chat.await_args.kwargs["system"]
    assert "Trusted Planner Contract" not in system


# 功能：验证 Planner terminal commit 后 read/search/spawn 等工具均被 invocation-time guard 拒绝
# 设计：直接构造真实 read tool 与已提交状态 service，隔离 registry allowlist 只测试终态语义
async def test_planner_terminal_guard_rejects_non_submit_tool(tmp_path: Path) -> None:
    from kama_claude.core.planning import (
        PlannerDecisionService,
        SubmittedDecisionIdentity,
    )
    from kama_claude.core.subagent.tool import _PlannerTerminalGuardTool

    tool, _, _ = _make_tool(tmp_path)
    service = PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-test",
        store=SessionStore(tmp_path / "sessions"),
        goal="inspect",
        run_id="planner-run",
    )
    service._terminal_decision = SubmittedDecisionIdentity(
        decision_id="decision_one",
        version=1,
        snapshot_digest="snapshot",
        content_digest="content",
    )
    guarded = _PlannerTerminalGuardTool(
        ReadFileTool(tool._path_resolver, tool._access_policy),
        service,
    )

    result = await guarded.invoke({"path": "missing.py"})

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "terminal decision already committed" in result.content


# 功能：验证 foreground planner 未提交 decision 时只返回安全 command_failed 摘要
# 设计：provider 输出含敏感执行计划的普通文本，断言 terminal gate 在 finished event 前覆盖该文本
async def test_planner_terminal_failure_sanitizes_foreground_result(
    tmp_path: Path,
) -> None:
    provider = _make_provider("直接修改 src/a.py，然后运行 bash ...")
    tool, _, bus = _make_tool(
        tmp_path,
        provider,
        store=SessionStore(tmp_path / "sessions"),
    )
    events: list[Any] = []

    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    result = await tool.invoke(
        {
            "description": "plan",
            "prompt": "inspect",
            "subagent_type": "planner",
        }
    )

    assert result.is_error is True
    assert result.error_type == "command_failed"
    assert "missing-terminal-decision" in result.content
    assert "src/a.py" not in result.content
    assert "运行 bash" not in result.content
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert len(finished) == 1
    assert finished[0].status == "failed"


# 功能：验证 background planner 使用同一 terminal gate 并通过 agent_result 返回安全失败
# 设计：后台 child 不提交 artifact，等待唯一 finished event 后查询 AgentResultTool 的稳定错误摘要
async def test_background_planner_terminal_failure_is_safe(tmp_path: Path) -> None:
    provider = _make_provider("直接修改 src/a.py，然后运行 bash ...")
    tool, registry, _ = _make_tool(
        tmp_path,
        provider,
        store=SessionStore(tmp_path / "sessions"),
    )
    spawn_result = await tool.invoke(
        {
            "description": "plan",
            "prompt": "inspect",
            "subagent_type": "planner",
            "run_in_background": True,
        }
    )
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]
    entry = registry.get(run_id)
    assert entry is not None
    task, _context = entry
    await task

    result = await AgentResultTool(registry).invoke({"run_id": run_id})
    assert result.is_error is True
    assert result.error_type == "command_failed"
    assert "missing-terminal-decision" in result.content
    assert "src/a.py" not in result.content
    assert "运行 bash" not in result.content


# 功能：验证 Planner terminal failure 不会先发布 transient success 或重复 finished event
# 设计：收集父 bus 的完整 lifecycle 顺序，只允许一个最终 failed SubagentFinishedEvent
async def test_planner_terminal_failure_emits_single_final_finished_event(
    tmp_path: Path,
) -> None:
    provider = _make_provider("untrusted plan")
    tool, _, bus = _make_tool(
        tmp_path,
        provider,
        store=SessionStore(tmp_path / "sessions"),
    )
    events: list[Any] = []

    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    await tool.invoke(
        {
            "description": "plan",
            "prompt": "inspect",
            "subagent_type": "planner",
        }
    )

    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]


# 功能：验证 planning-only invocation tree 不把 Planner child 的 LLM token 转发到 parent bus
# 设计：provider 主动向 child bus 发布秘密 token，再比较 planning-only 与普通 child 两条桥接边界
async def test_planning_only_token_bridge_isolated_without_changing_direct_children(
    tmp_path: Path,
) -> None:
    class _TokenProvider:
        # 在 child bus 发布一个仅用于隔离断言的 token 后结束本轮
        async def chat(self, messages: Any, tool_schemas: Any, bus: EventBus, run_id: str, **kwargs: Any) -> LlmResponse:
            await bus.publish(LlmTokenEvent(run_id=run_id, token="secret-token", ts="t"))
            return LlmResponse(stop_reason="end_turn", text="done")

    planning_events: list[Any] = []
    planning_bus = EventBus()

    async def collect_planning(event: Any) -> None:
        planning_events.append(event)

    planning_bus.subscribe(collect_planning)
    planning_tool, _, _ = _make_tool(
        tmp_path,
        _TokenProvider(),
        store=SessionStore(tmp_path / "planning-sessions"),
        planning_only=True,
    )
    planning_tool._parent_bus = planning_bus
    await planning_tool.invoke(
        {"description": "plan", "prompt": "inspect", "subagent_type": "planner"}
    )
    assert not any(isinstance(event, LlmTokenEvent) for event in planning_events)

    direct_events: list[Any] = []
    direct_bus = EventBus()

    async def collect_direct(event: Any) -> None:
        direct_events.append(event)

    direct_bus.subscribe(collect_direct)
    direct_tool, _, _ = _make_tool(tmp_path, _TokenProvider())
    direct_tool._parent_bus = direct_bus
    await direct_tool.invoke({"description": "child", "prompt": "inspect"})
    assert any(isinstance(event, LlmTokenEvent) for event in direct_events)


# 功能：验证 profiled child 也加载 workspace root explicit instructions
# 设计：让 profile 覆盖 base 并捕获 provider system，证明 root rules 来自独立 trusted slot
async def test_profiled_child_loads_root_repository_instructions(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("root-child-rule", encoding="utf-8")
    (agents / "custom.toml").write_text(
        '[agent]\nsystem_prompt = "custom-role"\nallowed_tools = ["read_file"]\n',
        encoding="utf-8",
    )
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, provider)

    await tool.invoke(
        {
            "description": "inspect",
            "prompt": "inspect",
            "subagent_type": "custom",
        }
    )

    system = provider.chat.await_args.kwargs["system"]
    assert system.startswith("custom-role\n\n" + _REPOSITORY_CHANGE_DISCIPLINE)
    assert "## Repository Instructions" in system
    assert "root-child-rule" in system


# 功能：验证未指定 profile 的 subagent 各继承一次 repaired default v1 与 v2
# 设计：执行真实前台 child loop 并捕获 provider system，区别于 profile override 的完全替换路径
async def test_unprofiled_subagent_inherits_v1_and_v2(tmp_path: Path) -> None:
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "inspect requirements",
            "prompt": "Implement behavior A and preserve invariant B.",
        }
    )

    assert result.is_error is False
    system = provider.chat.await_args.kwargs["system"]
    assert isinstance(system, str)
    assert system.count(_REQUIREMENT_CONTRACT) == 1
    assert system.count(_STATE_TRANSITION_PROTOCOL) == 1


# 功能：验证 subagent 模块不保留绑定项目目录的全局 profile loader
# 设计：直接检查模块命名空间，锁定跨 workspace 共享实例被移除
def test_subagent_has_no_module_profile_loader() -> None:
    assert not hasattr(subagent_tool_module, "_profile_loader")


# 功能：前台模式下 spawn_agent 应阻塞直到子 agent 完成并返回其结果
# 设计：使用返回 end_turn 的 mock provider，验证 tool_result.content 包含 provider 返回的文字
@pytest.mark.asyncio
async def test_foreground_returns_result(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path, _make_provider("analysis complete"))
    result = await tool.invoke({
        "description": "分析代码",
        "prompt": "分析 src/ 目录",
    })
    assert not result.is_error
    assert "analysis complete" in result.content


# 功能：后台模式应立即返回含 run_id 的消息，不阻塞等待子 agent
# 设计：run_in_background=true 后验证返回消息含 "run_id=" 并且任务注册表已有对应条目
@pytest.mark.asyncio
async def test_background_returns_run_id(tmp_path: Path) -> None:
    tool, registry, _ = _make_tool(tmp_path)
    result = await tool.invoke({
        "description": "后台任务",
        "prompt": "做点事",
        "run_in_background": True,
    })
    assert not result.is_error
    assert "run_id=" in result.content
    # extract run_id from message
    run_id = result.content.split("run_id=")[1].split(".")[0]
    assert registry.get(run_id) is not None


# 功能：验证后台 run_id 返回后立即取消仍进入生命周期边界并配对 finished
# 设计：不等待 child entered 信号而直接取消真实 registry task，稳定复现旧实现的首次调度竞态
async def test_background_immediate_cancellation_after_run_id_is_paired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never_release = asyncio.Event()

    # 保持 child loop 挂起，确保立即取消由 lifecycle owner 处理
    async def _wait_forever(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        await never_release.wait()

    monkeypatch.setattr(AgentLoop, "run", _wait_forever)
    tool, registry, bus = _make_tool(tmp_path)
    events: list[Any] = []

    # 收集真实父 bus 上的公开 lifecycle 事件
    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert context.status == "failed"
    assert context.reason == "cancelled"
    lifecycle_events = [
        event
        for event in events
        if isinstance(event, SubagentStartedEvent | SubagentFinishedEvent)
    ]
    assert [type(event) for event in lifecycle_events] == [
        SubagentStartedEvent,
        SubagentFinishedEvent,
    ]
    assert lifecycle_events[1].status == "failed"
    assert await AgentResultTool(registry).invoke({"run_id": run_id}) == ToolResult(
        content="Subagent was cancelled.",
        is_error=True,
        error_type="command_failed",
    )


# 功能：验证 spawn 在等待后台生命周期握手时取消会清理已注册 task 并保持取消身份
# 设计：受控 Event.wait 在真实 invoke_tool 任务内触发取消，检查公开 registry 与事件而非 task 私有字段
async def test_background_handshake_wait_cancellation_cleans_registered_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_blocker = asyncio.Event()
    handshake_blocker = asyncio.Event()
    observed_cancellations: list[asyncio.CancelledError] = []

    class _CancellingHandshake:
        # 保持受控握手未完成，使 invoke 停留在 wait 边界
        def set(self) -> None:
            return None

        # 在当前 invoke task 内注入取消并记录接收到的异常对象
        async def wait(self) -> None:
            current = asyncio.current_task()
            assert current is not None
            asyncio.get_running_loop().call_soon(current.cancel)
            try:
                await handshake_blocker.wait()
            except asyncio.CancelledError as exc:
                observed_cancellations.append(exc)
                raise

    # 保持 background task 活跃，供 invoke cancellation 路径负责清理
    async def _wait_forever(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        await child_blocker.wait()

    monkeypatch.setattr(AgentLoop, "run", _wait_forever)
    monkeypatch.setattr(subagent_tool_module.asyncio, "Event", _CancellingHandshake)
    tool, registry, bus = _make_tool(tmp_path)
    tool_registry = ToolRegistry()
    tool_registry.register(tool)
    events: list[Any] = []

    # 收集 tool 与 subagent 公开事件以排除伪造失败结果
    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await invoke_tool(
            tool_registry,
            ToolCallBlock(
                id="handshake-cancel",
                name="spawn_agent",
                input={
                    "description": "child",
                    "prompt": "work",
                    "run_in_background": True,
                },
            ),
            bus,
            run_id="parent-run-01",
        )

    assert observed_cancellations == [exc_info.value]
    entries = registry.all()
    assert len(entries) == 1
    task, context = entries[0]
    assert task.cancelled()
    assert context.status == "failed"
    assert context.reason == "cancelled"
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        SubagentStartedEvent,
        SubagentFinishedEvent,
    ]
    assert not any(isinstance(event, ToolCallFailedEvent) for event in events)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：后台任务未完成时 agent_result 应返回 "still running"
# 设计：用 Event 阻塞 provider.chat，在未等待任务完成时查询 agent_result
@pytest.mark.asyncio
async def test_agent_result_pending(tmp_path: Path) -> None:
    event = asyncio.Event()

    async def slow_chat(*args: Any, **kwargs: Any) -> LlmResponse:
        await event.wait()
        return LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text="done",
            usage=UsageStats(0, 0, 0, 0, 0.0),
        )

    provider = MagicMock()
    provider.chat = slow_chat

    tool, registry, _ = _make_tool(tmp_path, provider)
    spawn_result = await tool.invoke({
        "description": "slow task",
        "prompt": "do something slow",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert result.content == "still running"
    assert not result.is_error

    event.set()
    entry = registry.get(run_id)
    assert entry is not None
    task, _ = entry
    await task


# 功能：后台任务完成后 agent_result 应返回子 agent 的最终文本
# 设计：等待后台任务 task 完成后调用 agent_result，断言返回内容与 provider 结果一致
@pytest.mark.asyncio
async def test_agent_result_done(tmp_path: Path) -> None:
    tool, registry, _ = _make_tool(tmp_path, _make_provider("final answer"))
    spawn_result = await tool.invoke({
        "description": "bg task",
        "prompt": "do it",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    entry = registry.get(run_id)
    assert entry is not None
    task, _ = entry
    await asyncio.wait_for(task, timeout=5.0)

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert not result.is_error
    assert "final answer" in result.content


# 功能：depth=2 时调用 spawn_agent 应返回 is_error=True（嵌套限制）
# 设计：构造 depth=2 的工具，断言 invoke 直接返回错误而不调用 provider
@pytest.mark.asyncio
async def test_nesting_limit(tmp_path: Path) -> None:
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, provider, depth=2)
    result = await tool.invoke({
        "description": "nested",
        "prompt": "do nested work",
    })
    assert result.is_error
    assert result.error_type == "invalid_input"
    assert "nesting limit" in result.content
    provider.chat.assert_not_called()


# 功能：agent_result 查询不存在的 run_id 应返回 is_error=True
# 设计：空 registry 中查询随机 run_id，验证错误消息含 "Unknown"
@pytest.mark.asyncio
async def test_agent_result_unknown_run_id(tmp_path: Path) -> None:
    registry = BackgroundTaskRegistry()
    tool = AgentResultTool(registry)
    result = await tool.invoke({"run_id": "nonexistent-id"})
    assert result.is_error
    assert result.error_type == "not_found"
    assert "Unknown" in result.content


# 功能：SubagentStartedEvent 应在前台 spawn 时发布到父 bus
# 设计：订阅父 bus 收集所有事件，断言 subagent.started 出现，且 parent_run_id 和 description 正确
@pytest.mark.asyncio
async def test_foreground_publishes_started_event(tmp_path: Path) -> None:
    from kama_claude.core.bus.events import SubagentStartedEvent

    tool, _, bus = _make_tool(tmp_path)
    events: list[Any] = []

    async def _collect(e: Any) -> None:
        events.append(e)

    bus.subscribe(_collect)

    await tool.invoke({
        "description": "test task",
        "prompt": "test prompt",
    })
    started = [e for e in events if isinstance(e, SubagentStartedEvent)]
    assert len(started) == 1
    assert started[0].parent_run_id == "parent-run-01"
    assert started[0].description == "test task"
