from __future__ import annotations

import json
from pathlib import Path

import pytest

import kama_claude.core.grounding as grounding
from kama_claude.core.agents.loader import AgentProfileLoader
from kama_claude.core.bus.events import (
    SubagentStartedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.session.store import SessionStore
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import SpawnAgentTool
from kama_claude.core.tools.invocation import invoke_tool


# 构造只返回一次工具调用再结束的 explorer provider
class _ExplorerProvider:
    # 初始化 scripted responses 与 provider 收到的工具 schema
    def __init__(self, responses: list[LlmResponse]) -> None:
        self._responses = iter(responses)
        self.tool_names: list[str] = []

    # 捕获 registry 可见性并返回下一个无网络 scripted response
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
        self.tool_names = [str(schema["name"]) for schema in tool_schemas]
        return next(self._responses)


class _EvidenceRecoveryExplorerProvider:
    MAX_TOTAL_PROVIDER_CALLS = 4

    # 初始化固定四状态 Explorer 与显式调用计数
    def __init__(self) -> None:
        self.total_provider_calls = 0

    # 驱动 read→fabricated evidence submit→corrected submit→end_turn 的唯一有限路径
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
        if "architecture_slice_submit" not in names:
            raise AssertionError("evidence-recovery provider reached a non-Explorer role")
        self.total_provider_calls += 1
        if self.total_provider_calls > self.MAX_TOTAL_PROVIDER_CALLS:
            raise AssertionError(
                "evidence-recovery provider-call budget exceeded: "
                f"{self.total_provider_calls}>{self.MAX_TOTAL_PROVIDER_CALLS}"
            )
        if self.total_provider_calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-entry",
                        name="read_file",
                        input={"path": "entry.py"},
                    )
                ],
            )
        if self.total_provider_calls == 2:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="submit-without-evidence",
                        name="architecture_slice_submit",
                        input={
                            "relevant_modules": ["entry.py"],
                            "likely_change_targets": ["entry.py"],
                            "evidence_tool_call_ids": ["read_file:entry.py"],
                            "completeness": "complete_for_task",
                            "confidence": 0.9,
                        },
                    )
                ],
            )
        if self.total_provider_calls == 3:
            transcript = json.dumps(messages, ensure_ascii=False, default=str)
            if "unknown evidence tool call" not in transcript:
                raise AssertionError("Explorer did not receive the provenance rejection detail")
            if "read-entry" not in transcript:
                raise AssertionError("Explorer did not receive the exact available evidence ID")
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="submit-with-evidence",
                        name="architecture_slice_submit",
                        input={
                            "relevant_modules": ["entry.py"],
                            "likely_change_targets": ["entry.py"],
                            "evidence_tool_call_ids": ["read-entry"],
                            "completeness": "complete_for_task",
                            "confidence": 0.9,
                        },
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="grounding committed")


# 构造带 session store 的 root SpawnAgentTool
def _spawn_tool(
    workspace: Path,
    provider: object,
    *,
    max_steps: int = 5,
    permission_manager: PermissionManager | None = None,
    parent_bus: EventBus | None = None,
) -> SpawnAgentTool:
    bus = parent_bus or EventBus()
    return SpawnAgentTool(
        provider=provider,
        workspace_root=workspace,
        parent_bus=bus,
        parent_run_id="parent-run",
        permission_manager=permission_manager,
        max_steps=max_steps,
        task_registry=BackgroundTaskRegistry(),
        runs_dir=workspace / "runs",
        session_id="sess-1",
        store=SessionStore(workspace / "sessions"),
    )


# 功能：验证 builtin Explorer registry 仅包含 read/search 与 slice submit 工具
# 设计：执行真实 explorer child 并捕获 provider schemas，锁定 Bash/write/MCP/nested spawn 均不可见
async def test_explorer_registry_is_read_only_and_has_submit_tool(tmp_path: Path) -> None:
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="done")])
    tool = _spawn_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "inspect repository",
            "prompt": "Find entrypoints",
            "subagent_type": "explorer",
            "exploration_level": "light",
        }
    )

    assert set(provider.tool_names) == {
        "read_file",
        "list_dir",
        "search_code",
        "architecture_slice_submit",
    }
    assert {"bash", "write_file", "spawn_agent", "agent_result"}.isdisjoint(
        provider.tool_names
    )
    payload = json.loads(result.content)
    assert payload["completeness"] == "blocked"


# 功能：验证 builtin Planner 只能派生 Explorer 而不能派生 Executor
# 设计：从 planner child registry 取得真实 nested spawn tool，调用 executor 并断言 provider 未运行
async def test_planner_nested_spawn_rejects_non_explorer_type(tmp_path: Path) -> None:
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="unused")])
    root = _spawn_tool(tmp_path, provider)
    planner = AgentProfileLoader(tmp_path).load("planner")
    assert planner is not None
    registry = root._build_child_registry(EventBus(), "planner-run", planner)
    nested = registry.get("spawn_agent")
    assert isinstance(nested, SpawnAgentTool)

    result = await nested.invoke(
        {
            "description": "execute changes",
            "prompt": "write code",
            "subagent_type": "executor",
        }
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "allowed subagent types: explorer" in result.content
    assert provider.tool_names == []


# 功能：验证项目自定义同名 Planner 不能扩大 builtin 的 Explorer-only child contract
# 设计：local planner 显式请求 executor child 并保留 spawn 工具，断言 loader 将类型集合收窄为 explorer
def test_custom_planner_cannot_expand_allowed_subagent_types(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom planner"\n'
        'allowed_tools = ["read_file", "spawn_agent"]\n'
        'allowed_subagent_types = ["explorer", "executor"]\n',
        encoding="utf-8",
    )

    profile = AgentProfileLoader(tmp_path).load("planner")

    assert profile is not None
    assert profile.allowed_subagent_types == ["explorer"]


# 功能：验证同名 custom Explorer 不能通过 profile allowlist 重新获得 mutation 或 delegation 工具
# 设计：local explorer 请求 bash/write/spawn，再从真实 child registry 断言只保留 builtin read/search/submit 上界
def test_custom_explorer_cannot_expand_read_only_registry(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "explorer.toml").write_text(
        '[agent]\nsystem_prompt = "custom explorer"\n'
        'allowed_tools = ["read_file", "bash", "write_file", "spawn_agent"]\n',
        encoding="utf-8",
    )
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="unused")])
    root = _spawn_tool(tmp_path, provider)
    profile = AgentProfileLoader(tmp_path).load("explorer")
    assert profile is not None

    registry = root._build_child_registry(EventBus(), "explorer-run", profile)

    assert registry.get("read_file") is not None
    assert registry.get("bash") is None
    assert registry.get("write_file") is None
    assert registry.get("spawn_agent") is None


# 功能：验证 custom Explorer 的空 allowlist 表示零工具而非 unrestricted registry
# 设计：省略 allowed_tools 后构造真实 child registry，断言 Bash/write/read 全部不可见
def test_custom_explorer_empty_allowlist_exposes_no_tools(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "explorer.toml").write_text(
        '[agent]\nsystem_prompt = "custom explorer"\n',
        encoding="utf-8",
    )
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="unused")])
    root = _spawn_tool(tmp_path, provider)
    profile = AgentProfileLoader(tmp_path).load("explorer")
    assert profile is not None

    registry = root._build_child_registry(EventBus(), "explorer-run", profile)

    assert registry.tool_schemas() == []


# 功能：验证 v1 Explorer 拒绝 background 模式以避免绕过 typed terminal slice
# 设计：请求真实 explorer background run，断言 child 未启动且返回稳定 invalid_input
async def test_explorer_rejects_background_execution(tmp_path: Path) -> None:
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="unused")])
    tool = _spawn_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "inspect repository",
            "prompt": "Inspect",
            "subagent_type": "explorer",
            "run_in_background": True,
        }
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "foreground" in result.content
    assert provider.tool_names == []


# 功能：验证显式请求的 Explorer profile 损坏时不能降级为 unrestricted unprofiled child
# 设计：用 malformed local explorer 遮蔽 builtin，断言调用在 child 启动前 fail closed 且 provider 未运行
async def test_malformed_explorer_profile_fails_closed(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "explorer.toml").write_text("[agent\ninvalid", encoding="utf-8")
    provider = _ExplorerProvider([LlmResponse(stop_reason="end_turn", text="unused")])
    tool = _spawn_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "inspect repository",
            "prompt": "Inspect",
            "subagent_type": "explorer",
        }
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "profile is unavailable" in result.content
    assert provider.tool_names == []


# 功能：验证 semantic candidate 未经 read_file 回读不能成为 confirmed module
# 设计：向 collector 发布真实 semantic lifecycle，再用同一 draft 补 read lifecycle，观察拒绝转为有效 slice
async def test_semantic_candidate_requires_read_file_evidence(tmp_path: Path) -> None:
    source = tmp_path / "src" / "entry.py"
    source.parent.mkdir()
    source.write_text("def main():\n    return 1\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    bus = EventBus()
    bus.subscribe(collector.handle)
    await bus.publish(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="semantic-1",
            tool_name="search_semantic",
            params={"query": "entrypoint"},
            ts="t1",
        )
    )
    await bus.publish(
        ToolCallFinishedEvent(
            run_id="explorer-run",
            tool_use_id="semantic-1",
            tool_name="search_semantic",
            elapsed_ms=1,
            output="src/entry.py:1-2",
            ts="t2",
        )
    )
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Find entrypoint",
        collector=collector,
    )
    draft = grounding.ArchitectureSliceDraft(
        relevant_modules=["src/entry.py"],
        likely_change_targets=["src/entry.py"],
        evidence_tool_call_ids=["semantic-1"],
        completeness="complete_for_task",
        confidence=0.8,
    )

    with pytest.raises(ValueError, match="read_file evidence required"):
        service.submit(draft)

    await bus.publish(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="read-1",
            tool_name="read_file",
            params={"path": "src/entry.py"},
            ts="t3",
        )
    )
    await bus.publish(
        ToolCallFinishedEvent(
            run_id="explorer-run",
            tool_use_id="read-1",
            tool_name="read_file",
            elapsed_ms=1,
            output=source.read_text(encoding="utf-8"),
            ts="t4",
        )
    )
    accepted = service.submit(
        draft.model_copy(
            update={"evidence_tool_call_ids": ["semantic-1", "read-1"]}
        )
    )

    assert accepted.completeness == "complete_for_task"
    assert {ref.tool_call_id for ref in accepted.evidence_refs} == {
        "semantic-1",
        "read-1",
    }
    assert accepted.snapshot_digest


# 功能：验证源码在 read_file observation 后、slice submit 前变化会被判为 stale evidence
# 设计：先发布真实旧内容 lifecycle，再改写同一文件，断言 service 不把新 snapshot 与旧 observation 混绑
async def test_architecture_slice_rejects_stale_read_evidence(tmp_path: Path) -> None:
    source = tmp_path / "target.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="read-target",
            tool_name="read_file",
            params={"path": "target.py"},
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
    source.write_text("VALUE = 2\n", encoding="utf-8")
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect target",
        collector=collector,
    )

    with pytest.raises(ValueError, match="stale read_file evidence"):
        service.submit(
            grounding.ArchitectureSliceDraft(
                relevant_modules=("target.py",),
                likely_change_targets=("target.py",),
                evidence_tool_call_ids=("read-target",),
                completeness="complete_for_task",
                confidence=0.8,
            )
        )


# 功能：验证 task-local nested explicit instruction 必须由 Explorer 实际 read_file 回读
# 设计：目标目录放置 nested AGENTS，先只读源码观察拒绝，再补 instruction observation 后提交成功
async def test_architecture_slice_requires_nested_instruction_evidence(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    source = package / "target.py"
    instruction = package / "AGENTS.md"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    instruction.write_text("Keep package edits focused.\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    for tool_id, path in (("read-target", "src/pkg/target.py"),):
        await collector.handle(
            ToolCallStartedEvent(
                run_id="explorer-run",
                tool_use_id=tool_id,
                tool_name="read_file",
                params={"path": path},
                ts="t1",
            )
        )
        await collector.handle(
            ToolCallFinishedEvent(
                run_id="explorer-run",
                tool_use_id=tool_id,
                tool_name="read_file",
                elapsed_ms=1,
                output=(tmp_path / path).read_text(encoding="utf-8"),
                ts="t2",
            )
        )
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect package target",
        collector=collector,
    )
    draft = grounding.ArchitectureSliceDraft(
        relevant_modules=("src/pkg/target.py",),
        likely_change_targets=("src/pkg/target.py",),
        evidence_tool_call_ids=("read-target",),
        completeness="complete_for_task",
        confidence=0.8,
    )

    with pytest.raises(ValueError, match="task-local instructions"):
        service.submit(draft)

    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="read-instruction",
            tool_name="read_file",
            params={"path": "src/pkg/AGENTS.md"},
            ts="t3",
        )
    )
    await collector.handle(
        ToolCallFinishedEvent(
            run_id="explorer-run",
            tool_use_id="read-instruction",
            tool_name="read_file",
            elapsed_ms=1,
            output=instruction.read_text(encoding="utf-8"),
            ts="t4",
        )
    )
    accepted = service.submit(
        draft.model_copy(
            update={
                "evidence_tool_call_ids": ("read-target", "read-instruction")
            }
        )
    )
    assert "src/pkg/AGENTS.md" in accepted.effective_instruction_refs


# 功能：验证 fabricated tool_call_id 不能进入 ArchitectureSlice evidence
# 设计：使用空 collector 提交只含伪造 ID 的 draft，断言 provenance validation 在持久化前拒绝
def test_architecture_slice_rejects_fabricated_evidence(tmp_path: Path) -> None:
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect",
        collector=grounding.ToolObservationCollector(),
    )

    with pytest.raises(ValueError, match="unknown evidence tool call"):
        service.submit(
            grounding.ArchitectureSliceDraft(
                evidence_tool_call_ids=["invented"],
                completeness="partial",
                confidence=0.2,
            )
        )


# 功能：验证另一个 run 的真实 tool observation 不能作为当前 slice evidence
# 设计：collector 接收 successful foreign-run lifecycle，再由当前 service 引用同一 ID，隔离 fabricated 与 cross-run provenance
async def test_architecture_slice_rejects_cross_run_evidence(tmp_path: Path) -> None:
    collector = grounding.ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="other-run",
            tool_use_id="read-other",
            tool_name="read_file",
            params={"path": "other.py"},
            ts="t1",
        )
    )
    await collector.handle(
        ToolCallFinishedEvent(
            run_id="other-run",
            tool_use_id="read-other",
            tool_name="read_file",
            elapsed_ms=1,
            output="content",
            ts="t2",
        )
    )
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect",
        collector=collector,
    )

    with pytest.raises(ValueError, match="evidence belongs to another run"):
        service.submit(
            grounding.ArchitectureSliceDraft(
                evidence_tool_call_ids=["read-other"],
                completeness="partial",
                confidence=0.2,
            )
        )


# 功能：验证重复 evidence tool_call_id 在 slice 中只保留一个 canonical ref
# 设计：提交同一成功 observation 两次，锁定 digest/provenance 不受重复列表膨胀影响
async def test_architecture_slice_deduplicates_evidence_refs(tmp_path: Path) -> None:
    source = tmp_path / "target.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-run",
            tool_use_id="read-target",
            tool_name="read_file",
            params={"path": "target.py"},
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
    architecture_slice = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect target",
        collector=collector,
    ).submit(
        grounding.ArchitectureSliceDraft(
            relevant_modules=("target.py",),
            likely_change_targets=("target.py",),
            evidence_tool_call_ids=("read-target", "read-target"),
            completeness="complete_for_task",
            confidence=0.8,
        )
    )

    assert [ref.tool_call_id for ref in architecture_slice.evidence_refs] == [
        "read-target"
    ]


# 功能：验证 complete_for_task 至少引用一条本次 run 的成功 observation
# 设计：提交无 claims 的空 complete draft，排除用 schema-valid 空对象伪装已完成 grounding
def test_complete_architecture_slice_requires_recorded_evidence(tmp_path: Path) -> None:
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect",
        collector=grounding.ToolObservationCollector(),
    )

    with pytest.raises(ValueError, match="complete_for_task requires recorded evidence"):
        service.submit(
            grounding.ArchitectureSliceDraft(
                completeness="complete_for_task",
                confidence=0.5,
            )
        )


# 功能：验证 Explorer snapshot 保存 preflight 后 baseline HEAD 但不把它纳入 stale digest
# 设计：用相同 evidence 分别传入两个 head 构建 slice，断言 provenance 不同而相关内容摘要相同
async def test_architecture_slice_records_git_head_as_provenance_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "target.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    async def _submit(git_head: str) -> tuple[str | None, str]:
        collector = grounding.ToolObservationCollector()
        await collector.handle(
            ToolCallStartedEvent(
                run_id=f"run-{git_head[0]}",
                tool_use_id=f"read-{git_head[0]}",
                tool_name="read_file",
                params={"path": "target.py"},
                ts="t1",
            )
        )
        await collector.handle(
            ToolCallFinishedEvent(
                run_id=f"run-{git_head[0]}",
                tool_use_id=f"read-{git_head[0]}",
                tool_name="read_file",
                elapsed_ms=1,
                output=source.read_text(encoding="utf-8"),
                ts="t2",
            )
        )
        store = SessionStore(tmp_path / f"sessions-{git_head[0]}")
        result = grounding.ArchitectureSliceService(
            workspace_root=tmp_path,
            run_id=f"run-{git_head[0]}",
            goal="Inspect target",
            collector=collector,
            session_id="sess-1",
            store=store,
            git_head=git_head,
        ).submit(
            grounding.ArchitectureSliceDraft(
                relevant_modules=("target.py",),
                likely_change_targets=("target.py",),
                evidence_tool_call_ids=(f"read-{git_head[0]}",),
                completeness="complete_for_task",
                confidence=0.8,
            )
        )
        artifact = store.read_grounding("sess-1")
        assert artifact is not None
        snapshot = artifact["snapshots"][0]
        return snapshot["git_head"], result.snapshot_digest

    first_head, first_digest = await _submit("a" * 40)
    second_head, second_digest = await _submit("b" * 40)

    assert first_head == "a" * 40
    assert second_head == "b" * 40
    assert first_digest == second_digest


# 功能：验证无 SessionStore 时同一 service 的 slice lineage version 仍单调递增
# 设计：连续提交两个 partial version，并让第二个显式引用首个 slice_id，断言 runtime identity 不复用 v1
def test_in_memory_architecture_slice_revision_increments_version(tmp_path: Path) -> None:
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect",
        collector=grounding.ToolObservationCollector(),
    )
    first = service.submit(
        grounding.ArchitectureSliceDraft(
            completeness="partial",
            confidence=0.1,
        )
    )

    second = service.submit(
        grounding.ArchitectureSliceDraft(
            slice_id=first.slice_id,
            completeness="partial",
            confidence=0.2,
        )
    )

    assert first.version == 1
    assert second.version == 2
    assert first.content_digest != second.content_digest


# 功能：验证 model 不能通过任意 slice_id 创建不存在的 revision lineage
# 设计：首个 submit 显式提供伪造 ID，断言 runtime 不接受 model 自分配 identity
def test_architecture_slice_rejects_unknown_revision_lineage(tmp_path: Path) -> None:
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect",
        collector=grounding.ToolObservationCollector(),
    )

    with pytest.raises(ValueError, match="existing lineage"):
        service.submit(
            grounding.ArchitectureSliceDraft(
                slice_id="slice_fabricated",
                completeness="partial",
                confidence=0.1,
            )
        )


# 功能：验证预算耗尽且未 submit 时 Explorer 持久化 partial 而非伪 complete
# 设计：用 max_steps=1 的未知工具调用触发真实 exceeded_max_steps，读取 grounding artifact 的终态 slice
async def test_explorer_budget_exhaustion_records_partial_slice(tmp_path: Path) -> None:
    provider = _ExplorerProvider(
        [
            LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(id="unknown-1", name="unknown", input={})],
            )
        ]
    )
    tool = _spawn_tool(tmp_path, provider, max_steps=1)

    result = await tool.invoke(
        {
            "description": "inspect repository",
            "prompt": "Inspect until budget ends",
            "subagent_type": "explorer",
            "exploration_level": "deep",
        }
    )

    payload = json.loads(result.content)
    assert payload["completeness"] == "partial"
    grounding = SessionStore(tmp_path / "sessions").read_grounding("sess-1")
    assert grounding is not None
    assert grounding["architecture_slices"][-1]["completeness"] == "partial"


# 功能：验证 submit tool 经统一 invocation 返回 runtime 分配的 slice identity
# 设计：先执行真实 read_file 产生 evidence，再通过 registry submit schema 调用而非直接 service 方法
async def test_architecture_slice_submit_tool_returns_identity(tmp_path: Path) -> None:
    source = tmp_path / "entry.py"
    source.write_text("ENTRY = True\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    bus = EventBus()
    bus.subscribe(collector.handle)
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect entry",
        collector=collector,
    )
    from kama_claude.core.tools.builtin.read_file import ReadFileTool
    from kama_claude.core.tools.registry import ToolRegistry
    from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
    from kama_claude.core.workspace.resolver import WorkspacePathResolver

    resolver = WorkspacePathResolver(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(resolver, WorkspaceAccessPolicy(resolver.root)))
    registry.register(grounding.ArchitectureSliceSubmitTool(service))
    await invoke_tool(
        registry,
        ToolCallBlock(id="read-1", name="read_file", input={"path": "entry.py"}),
        bus,
        "explorer-run",
    )

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="submit-1",
            name="architecture_slice_submit",
            input={
                "relevant_entrypoints": ["entry.py"],
                "likely_change_targets": ["entry.py"],
                "evidence_tool_call_ids": ["read-1"],
                "completeness": "complete_for_task",
                "confidence": 0.9,
            },
        ),
        bus,
        "explorer-run",
    )

    identity = json.loads(result.content)
    assert result.is_error is False
    assert identity["slice_id"].startswith("slice_")
    assert identity["version"] == 1
    assert identity["content_digest"]


# 功能：验证 complete slice 缺少 evidence 时统一 invocation 返回可操作 invalid_input
# 设计：经真实 registry/invoke_tool 边界触发 service 拒绝，排除泛化 execution_error
async def test_architecture_slice_submit_reports_actionable_missing_evidence(
    tmp_path: Path,
) -> None:
    from kama_claude.core.tools.registry import ToolRegistry

    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect entry",
        collector=grounding.ToolObservationCollector(),
    )
    registry = ToolRegistry()
    registry.register(grounding.ArchitectureSliceSubmitTool(service))

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="submit-empty",
            name="architecture_slice_submit",
            input={
                "evidence_tool_call_ids": [],
                "completeness": "complete_for_task",
                "confidence": 0.9,
            },
        ),
        EventBus(),
        "explorer-run",
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "complete_for_task requires recorded evidence" in result.content
    assert "evidence_tool_call_ids" in result.content


# 功能：验证 fabricated evidence ID 在工具边界返回可恢复错误和真实候选 ID
# 设计：先经真实 read_file 记录 provenance，再提交人类可读伪 ID 锁定模型修正所需信息
async def test_architecture_slice_submit_reports_available_id_for_unknown_evidence(
    tmp_path: Path,
) -> None:
    from kama_claude.core.tools.builtin.read_file import ReadFileTool
    from kama_claude.core.tools.registry import ToolRegistry
    from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
    from kama_claude.core.workspace.resolver import WorkspacePathResolver

    source = tmp_path / "entry.py"
    source.write_text("ENTRY = True\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    bus = EventBus()
    bus.subscribe(collector.handle)
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect entry",
        collector=collector,
    )
    resolver = WorkspacePathResolver(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(resolver, WorkspaceAccessPolicy(resolver.root)))
    registry.register(grounding.ArchitectureSliceSubmitTool(service))
    await invoke_tool(
        registry,
        ToolCallBlock(id="read-entry", name="read_file", input={"path": "entry.py"}),
        bus,
        "explorer-run",
    )

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="submit-fabricated",
            name="architecture_slice_submit",
            input={
                "relevant_modules": ["entry.py"],
                "likely_change_targets": ["entry.py"],
                "evidence_tool_call_ids": ["read_file:entry.py"],
                "completeness": "complete_for_task",
                "confidence": 0.9,
            },
        ),
        bus,
        "explorer-run",
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "unknown evidence tool call: read_file:entry.py" in result.content
    assert "read-entry=entry.py" in result.content


# 功能：验证 complete slice 不能把 unresolved questions 带入父 Planner 后才失败
# 设计：先记录真实 read evidence，再经 submit tool 边界断言原 Explorer 内即可得到可恢复 invalid_input
async def test_architecture_slice_submit_rejects_complete_with_unresolved_questions(
    tmp_path: Path,
) -> None:
    from kama_claude.core.tools.builtin.read_file import ReadFileTool
    from kama_claude.core.tools.registry import ToolRegistry
    from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
    from kama_claude.core.workspace.resolver import WorkspacePathResolver

    source = tmp_path / "entry.py"
    source.write_text("ENTRY = True\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    bus = EventBus()
    bus.subscribe(collector.handle)
    resolver = WorkspacePathResolver(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(resolver, WorkspaceAccessPolicy(resolver.root)))
    registry.register(
        grounding.ArchitectureSliceSubmitTool(
            grounding.ArchitectureSliceService(
                workspace_root=tmp_path,
                run_id="explorer-run",
                goal="Inspect entry",
                collector=collector,
            )
        )
    )
    await invoke_tool(
        registry,
        ToolCallBlock(id="read-entry", name="read_file", input={"path": "entry.py"}),
        bus,
        "explorer-run",
    )

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="submit-unresolved",
            name="architecture_slice_submit",
            input={
                "relevant_modules": ["entry.py"],
                "likely_change_targets": ["entry.py"],
                "unresolved_questions": ["Which behavior is authoritative?"],
                "evidence_tool_call_ids": ["read-entry"],
                "completeness": "complete_for_task",
                "confidence": 0.9,
            },
        ),
        bus,
        "explorer-run",
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "requires unresolved_questions to be empty" in result.content
    assert "partial/blocked" in result.content


# 功能：验证 confirmed path 不能夹带 symbol 或描述并伪装为不存在的新文件
# 设计：真实读取 entry.py 后提交带冒号说明的 relevant_entrypoints，锁定 Explorer 内部 path 契约
async def test_architecture_slice_submit_rejects_nonexistent_confirmed_path(
    tmp_path: Path,
) -> None:
    from kama_claude.core.tools.builtin.read_file import ReadFileTool
    from kama_claude.core.tools.registry import ToolRegistry
    from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
    from kama_claude.core.workspace.resolver import WorkspacePathResolver

    source = tmp_path / "entry.py"
    source.write_text("def main():\n    return 0\n", encoding="utf-8")
    collector = grounding.ToolObservationCollector()
    bus = EventBus()
    bus.subscribe(collector.handle)
    resolver = WorkspacePathResolver(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(resolver, WorkspaceAccessPolicy(resolver.root)))
    registry.register(
        grounding.ArchitectureSliceSubmitTool(
            grounding.ArchitectureSliceService(
                workspace_root=tmp_path,
                run_id="explorer-run",
                goal="Inspect entry",
                collector=collector,
            )
        )
    )
    await invoke_tool(
        registry,
        ToolCallBlock(id="read-entry", name="read_file", input={"path": "entry.py"}),
        bus,
        "explorer-run",
    )

    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="submit-described-path",
            name="architecture_slice_submit",
            input={
                "relevant_entrypoints": ["entry.py:main"],
                "relevant_modules": ["entry.py"],
                "likely_change_targets": ["entry.py"],
                "evidence_tool_call_ids": ["read-entry"],
                "completeness": "complete_for_task",
                "confidence": 0.9,
            },
        ),
        bus,
        "explorer-run",
    )

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "confirmed repository path does not exist: entry.py:main" in result.content
    assert "without symbols or descriptions" in result.content


# 功能：验证 Explorer 收到缺 evidence 反馈后能有限修正并提交 complete slice
# 设计：四状态 provider 经真实 AgentLoop/PermissionManager 执行，精确断言调用与 child 预算
async def test_explorer_recovers_missing_evidence_without_permission_prompt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "entry.py"
    source.write_text("ENTRY = True\n", encoding="utf-8")
    provider = _EvidenceRecoveryExplorerProvider()
    bus = EventBus()
    started_runs: list[str] = []

    # 记录真实 child lifecycle，隐藏派生会破坏精确计数
    async def collect(event: object) -> None:
        if isinstance(event, SubagentStartedEvent):
            started_runs.append(event.run_id)

    bus.subscribe(collect)
    tool = _spawn_tool(
        tmp_path,
        provider,
        permission_manager=PermissionManager(timeout_s=0.01),
        parent_bus=bus,
    )

    result = await tool.invoke(
        {
            "description": "inspect entry",
            "prompt": "Inspect entry",
            "subagent_type": "explorer",
        }
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["completeness"] == "complete_for_task"
    assert payload["evidence_refs"] == [
        {"logical_path": "entry.py", "tool_call_id": "read-entry"}
    ]
    assert provider.total_provider_calls == provider.MAX_TOTAL_PROVIDER_CALLS
    assert len(started_runs) == 1


# 功能：验证 Explorer 实际消费的 tool schema 说明 evidence ID 必须来自成功 read_file
# 设计：检查注册工具的对外 schema 而非私有常量，锁定模型可见合同
def test_architecture_slice_schema_explains_evidence_tool_call_ids(
    tmp_path: Path,
) -> None:
    service = grounding.ArchitectureSliceService(
        workspace_root=tmp_path,
        run_id="explorer-run",
        goal="Inspect entry",
        collector=grounding.ToolObservationCollector(),
    )
    tool = grounding.ArchitectureSliceSubmitTool(service)

    field = tool.input_schema["properties"]["evidence_tool_call_ids"]

    assert "successful read_file tool call IDs" in field["description"]
    for name in ("relevant_entrypoints", "relevant_modules", "likely_change_targets"):
        path_field = tool.input_schema["properties"][name]
        description = path_field["description"].casefold()
        assert "exact workspace-relative file paths" in description
        assert "without symbols or descriptions" in description
    unresolved = tool.input_schema["properties"]["unresolved_questions"]
    assert "complete_for_task requires this field to be empty" in unresolved[
        "description"
    ]
