from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from kama_claude.core.bus.events import ToolCallFinishedEvent, ToolCallStartedEvent
from kama_claude.core.grounding import (
    ArchitectureSlice,
    ArchitectureSliceDraft,
    ArchitectureSliceService,
    ToolObservationCollector,
)
from kama_claude.core.session.store import SessionStore


# 延迟加载 planner decision 模块，使缺失实现以行为 RED 失败呈现
def _planning() -> ModuleType:
    return importlib.import_module("kama_claude.core.planning")


# 通过真实 tool lifecycle 与 explorer service 持久化 complete ArchitectureSlice
async def _ground_complete_slice(
    tmp_path: Path,
    *,
    likely_targets: tuple[str, ...] = ("src/target.py",),
) -> tuple[Path, SessionStore, ArchitectureSlice]:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "target.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    tests = workspace / "tests" / "test_target.py"
    tests.parent.mkdir()
    tests.write_text("def test_value():\n    assert True\n", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("Keep changes focused.\n", encoding="utf-8")
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
    service = ArchitectureSliceService(
        workspace_root=workspace,
        run_id="explorer-run",
        goal="Change target behavior",
        collector=collector,
        session_id="sess-1",
        store=store,
    )
    architecture_slice = service.submit(
        ArchitectureSliceDraft(
            relevant_modules=("src/target.py",),
            related_tests=("tests/test_target.py",),
            existing_patterns=("single module edit",),
            likely_change_targets=likely_targets,
            evidence_tool_call_ids=("read-target",),
            completeness="complete_for_task",
            confidence=0.9,
        )
    )
    return workspace, store, architecture_slice


# 构造满足最小 mapping/provenance contract 的 PlannerDecision draft
def _valid_draft(slice_: ArchitectureSlice, **updates: object) -> object:
    planning = _planning()
    values: dict[str, object] = {
        "architecture_slice_id": slice_.slice_id,
        "architecture_slice_version": slice_.version,
        "architecture_mode": "preserve",
        "selected_approach": "Edit the existing target module only.",
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
                "description": "Update existing target behavior.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["read-target"],
            }
        ],
        "files_to_modify": ["src/target.py"],
        "allowed_capabilities": ["read_file", "write_file"],
        "verification_plan": [
            {"requirement_ids": ["R1"], "strategy": "Run focused unit tests."}
        ],
        "non_goals": ["No surrounding cleanup."],
        "requires_user_approval": True,
    }
    values.update(updates)
    return planning.PlannerDecisionDraft.model_validate(values)


# 功能：验证 PlannerDecisionDraft 不接受模型伪造的 provenance digest
# 设计：直接向 strict Draft 注入两个可信字段，断言 schema 在 runtime lookup 前拒绝输入
def test_planner_draft_rejects_model_provenance_digests() -> None:
    planning = _planning()
    values = {
        "architecture_slice_id": "slice-1",
        "architecture_slice_version": 1,
        "snapshot_digest": "forged-snapshot",
        "architecture_slice_content_digest": "forged-slice",
    }

    with pytest.raises(ValueError):
        planning.PlannerDecisionDraft.model_validate(values)


# 功能：验证 PlannerDecisionDraft 只暴露 slice identity 而不暴露 snapshot digest
# 设计：检查生成的 submit tool schema，确保模型协议与 Draft 模型保持同一 authority 边界
def test_planner_submit_schema_excludes_provenance_digest_fields() -> None:
    planning = _planning()
    properties = planning.PlannerDecisionSubmitTool.input_schema["properties"]

    assert "architecture_slice_version" in properties
    assert "snapshot_digest" not in properties
    assert "architecture_slice_content_digest" not in properties


# 功能：验证旧 V1 payload 使用 legacy codec 解码且不会被 V2 字段污染
# 设计：从 exact decision 复制旧字段并重算 legacy digest，直接覆盖 persistence/history boundary
async def test_legacy_decision_codec_preserves_v1_payload_shape(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    exact = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    ).submit(_valid_draft(architecture_slice))
    legacy_payload = exact.model_dump(mode="json")
    legacy_payload.pop("schema_version")
    legacy_payload.pop("architecture_slice_version")
    legacy_payload.pop("architecture_slice_content_digest")
    legacy_payload.pop("content_digest")
    legacy_payload["content_digest"] = planning._legacy_content_digest(legacy_payload)

    record = planning.decode_planner_decision_record(legacy_payload)

    assert isinstance(record, planning.LegacyPlannerDecisionV1)
    assert record.content_digest == legacy_payload["content_digest"]
    assert "schema_version" not in record.model_dump(mode="json")


# 功能：验证固定旧版 V1 artifact 在升级后仍按原 canonical digest 解码
# 设计：使用不依赖当前 V2 model 的 literal payload 和 frozen digest，防止 codec 漂移被测试数据掩盖
def test_legacy_decision_codec_reads_frozen_v1_artifact() -> None:
    planning = _planning()
    legacy_payload = {
        "decision_id": "golden_legacy",
        "version": 1,
        "goal": "Inspect the repository.",
        "requirements": [],
        "architecture_slice_id": "slice-golden",
        "snapshot_digest": "snapshot-golden",
        "architecture_mode": "preserve",
        "selected_approach": "Read existing code.",
        "existing_patterns_reused": [],
        "intended_changes": [],
        "files_to_modify": [],
        "files_to_create": [],
        "allowed_capabilities": ["read_file"],
        "dependency_changes": [],
        "protocol_or_schema_changes": [],
        "verification_plan": [],
        "non_goals": [],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
        "content_digest": "f2eb94abc018a8a45cf9cd479b4b654f36f1fe1701b43ce116ccd339a9c3a8f3",
    }

    record = planning.decode_planner_decision_record(legacy_payload)

    assert isinstance(record, planning.LegacyPlannerDecisionV1)
    assert record.decision_id == "golden_legacy"
    assert record.content_digest == legacy_payload["content_digest"]


# 功能：验证 active Planner runtime 不会把 legacy record 当作 exact terminal artifact
# 设计：先在 persistence boundary 解码 V1，再调用 exact-only 收窄函数，锁定 union 不扩散到运行时
def test_active_runtime_rejects_legacy_decision_record() -> None:
    planning = _planning()
    legacy_payload = {
        "decision_id": "legacy_decision",
        "version": 1,
        "goal": "Inspect the repository.",
        "requirements": [],
        "architecture_slice_id": "slice-1",
        "snapshot_digest": "snapshot-1",
        "architecture_mode": "preserve",
        "selected_approach": "Inspect existing code.",
        "existing_patterns_reused": [],
        "intended_changes": [],
        "files_to_modify": [],
        "files_to_create": [],
        "allowed_capabilities": ["read_file"],
        "dependency_changes": [],
        "protocol_or_schema_changes": [],
        "verification_plan": [],
        "non_goals": [],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
    }
    legacy_payload["content_digest"] = planning._legacy_content_digest(legacy_payload)

    with pytest.raises(ValueError, match="terminal-runtime"):
        planning.decode_exact_planner_decision(legacy_payload)


# 功能：验证 legacy v1 lineage 可以继续 revision 为 exact v2 且旧 bytes 不变
# 设计：先直接写入 V1 immutable artifact，再由 active V2 service 使用同一 decision_id 生成下一版本
async def test_legacy_lineage_revisions_to_exact_v2(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    legacy_payload = {
        "decision_id": "legacy_decision",
        "version": 1,
        "goal": "Change target behavior",
        "requirements": [
            {"requirement_id": "R1", "statement": "Change target behavior.", "required": True}
        ],
        "architecture_slice_id": architecture_slice.slice_id,
        "snapshot_digest": architecture_slice.snapshot_digest,
        "architecture_mode": "preserve",
        "selected_approach": "Use the existing target module.",
        "existing_patterns_reused": ["single module edit"],
        "intended_changes": [
            {
                "change_id": "C1",
                "description": "Update target behavior.",
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
        "verification_plan": [{"requirement_ids": ["R1"], "strategy": "Run focused tests."}],
        "non_goals": ["No cleanup."],
        "assumptions": [],
        "unresolved_questions": [],
        "requires_user_approval": True,
    }
    legacy_payload["content_digest"] = planning._legacy_content_digest(legacy_payload)
    store.write_decision("sess-1", "legacy_decision", 1, legacy_payload)
    legacy_path = (
        store.session_dir("sess-1")
        / "planning"
        / "decisions"
        / "legacy_decision-v1.json"
    )
    legacy_bytes = legacy_path.read_bytes()

    exact = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    ).submit(_valid_draft(architecture_slice, decision_id="legacy_decision"))

    assert exact.schema_version == 2
    assert exact.decision_id == "legacy_decision"
    assert exact.version == 2
    assert legacy_path.read_bytes() == legacy_bytes
    assert [item["version"] for item in store.list_decisions("sess-1")] == [1, 2]


# 功能：验证合法 draft 生成 versioned/digest-bound decision 并通过五类验证状态
# 设计：从真实 persisted slice 提交 decision，再从 immutable file public reader 回读同一模型
async def test_valid_planner_decision_is_versioned_validated_and_persisted(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )

    decision = service.submit(_valid_draft(architecture_slice))

    assert decision.decision_id.startswith("decision_")
    assert decision.version == 1
    assert decision.content_digest
    assert service.last_validation == {
        "structure-valid": True,
        "reference-valid": True,
        "scope-valid": True,
        "provenance-valid": True,
        "snapshot-current": True,
    }
    stored = store.read_decision("sess-1", decision.decision_id, 1)
    assert stored == decision.model_dump(mode="json")
    decision_path = (
        store.session_dir("sess-1")
        / "planning"
        / "decisions"
        / f"{decision.decision_id}-v1.json"
    )
    assert decision_path.exists()


# 功能：验证 intended change 不能引用 ArchitectureSlice 外的 fabricated evidence
# 设计：只替换 evidence_refs 并保持其他 scope 合法，锁定 reference validation 分类
async def test_planner_decision_rejects_fabricated_evidence(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )
    draft = _valid_draft(
        architecture_slice,
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update target.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["invented"],
            }
        ],
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(draft)

    assert exc_info.value.category == "reference-invalid"
    assert store.list_decisions("sess-1") == []


# 功能：验证 files_to_modify 必须是已读、存在且位于 slice likely targets 的路径
# 设计：增加真实但未 grounding 的 other.py，排除 missing-file 分支并锁定 scope validation
async def test_planner_decision_rejects_unread_non_likely_modify_target(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    (workspace / "src" / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )
    draft = _valid_draft(
        architecture_slice,
        files_to_modify=["src/other.py"],
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update other module.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/other.py"],
                "evidence_refs": ["read-target"],
            }
        ],
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(draft)

    assert exc_info.value.category == "scope-invalid"


# 功能：验证 required requirement 必须同时映射 intended change 与 verification strategy
# 设计：分别移除 change mapping 和 verification mapping，参数化锁定两个独立 structure branches
@pytest.mark.parametrize("missing", ["change", "verification"])
async def test_planner_decision_requires_change_and_verification_mapping(
    tmp_path: Path,
    missing: str,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )
    updates: dict[str, object]
    if missing == "change":
        updates = {
            "intended_changes": [
                {
                    "change_id": "C1",
                    "description": "Unrelated change.",
                    "requirement_ids": [],
                    "target_paths": ["src/target.py"],
                    "evidence_refs": ["read-target"],
                }
            ]
        }
    else:
        updates = {"verification_plan": []}

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(_valid_draft(architecture_slice, **updates))

    assert exc_info.value.category == "structure-invalid"


# 功能：验证任何 unresolved question 在 v1 都阻止 decision validation
# 设计：保留完整 scope/evidence，只添加一个问题，采用保守 blocking 语义避免无分类字符串被忽略
async def test_planner_decision_blocks_on_unresolved_question(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(
            _valid_draft(
                architecture_slice,
                unresolved_questions=["Which protocol version applies?"],
            )
        )

    assert exc_info.value.category == "structure-invalid"


# 功能：验证 v1 decision 不能省略 approval requirement 或提交空 capability scope
# 设计：参数化两个 approval-chain 必需字段，确保尚未实施 backend 时 typed artifact 仍不可绕过边界
@pytest.mark.parametrize(
    "updates",
    [
        {"requires_user_approval": False},
        {"allowed_capabilities": []},
    ],
)
async def test_planner_decision_requires_approval_and_capability_scope(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(_valid_draft(architecture_slice, **updates))

    assert exc_info.value.category == "structure-invalid"


# 功能：验证相关 target 内容变化后 decision 在持久化前被判 stale
# 设计：grounding 完成后只改 files_to_modify 内容，再提交同一 snapshot-bound draft
async def test_planner_decision_rejects_stale_relevant_snapshot(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    (workspace / "src" / "target.py").write_text("VALUE = 2\n", encoding="utf-8")
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(_valid_draft(architecture_slice))

    assert exc_info.value.category == "snapshot-stale"
    assert store.list_decisions("sess-1") == []


# 功能：验证 revision 使用同一 decision ID 的下一 immutable version 且保留 v1 bytes
# 设计：记录 v1 原始文件，提交显式 revision，再比较版本、digest、v1 bytes 与两个文件列表
async def test_planner_decision_revision_preserves_immutable_versions(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-1",
    )
    first = service.submit(_valid_draft(architecture_slice))
    first_path = (
        store.session_dir("sess-1")
        / "planning"
        / "decisions"
        / f"{first.decision_id}-v1.json"
    )
    first_bytes = first_path.read_bytes()

    second_service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-2",
    )
    second = second_service.submit(
        _valid_draft(
            architecture_slice,
            decision_id=first.decision_id,
            selected_approach="Reuse the same module with a narrower edit.",
        )
    )

    assert second.decision_id == first.decision_id
    assert second.version == 2
    assert second.content_digest != first.content_digest
    assert first_path.read_bytes() == first_bytes
    assert [item["version"] for item in store.list_decisions("sess-1")] == [1, 2]


# 功能：验证 PlannerDecision 精确绑定 draft 指定的 slice version 而不是自动选择最新版本
# 设计：同一 slice lineage 生成 v1/v2 后分别提交两个 service，比较决策绑定版本与 evidence，排除 max(version) fallback
async def test_planner_decision_uses_exact_slice_lineage_version(
    tmp_path: Path,
) -> None:
    workspace, store, first_slice = await _ground_complete_slice(tmp_path)
    old_draft = _valid_draft(first_slice)
    collector = ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-revision",
            tool_use_id="read-revision",
            tool_name="read_file",
            params={"path": "src/target.py"},
            ts="t1",
        )
    )
    await collector.handle(
        ToolCallFinishedEvent(
            run_id="explorer-revision",
            tool_use_id="read-revision",
            tool_name="read_file",
            elapsed_ms=1,
            output=(workspace / "src" / "target.py").read_text(encoding="utf-8"),
            ts="t2",
        )
    )
    second_slice = ArchitectureSliceService(
        workspace_root=workspace,
        run_id="explorer-revision",
        goal="Change target behavior",
        collector=collector,
        session_id="sess-1",
        store=store,
    ).submit(
        ArchitectureSliceDraft(
            slice_id=first_slice.slice_id,
            relevant_modules=("src/target.py",),
            existing_patterns=("revised pattern",),
            likely_change_targets=("src/target.py",),
            evidence_tool_call_ids=("read-revision",),
            completeness="complete_for_task",
            confidence=0.95,
        )
    )
    planning = _planning()
    old_service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )

    assert second_slice.version == 2
    assert second_slice.snapshot_digest == first_slice.snapshot_digest
    old_decision = old_service.submit(old_draft)
    assert old_decision.architecture_slice_version == first_slice.version
    assert old_decision.architecture_slice_content_digest == first_slice.content_digest

    latest_draft = _valid_draft(
        second_slice,
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update existing target behavior.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["read-revision"],
            }
        ],
    )
    latest_service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )
    decision = latest_service.submit(latest_draft)
    assert decision.architecture_slice_id == second_slice.slice_id
    assert decision.architecture_slice_version == second_slice.version
    assert decision.architecture_slice_content_digest == second_slice.content_digest


# 功能：验证 terminal recheck 继续绑定已提交 decision 的旧 slice version
# 设计：提交 X:v1 后再写入 X:v2，检查同一 service 仍能通过 exact v1 artifact 的 terminal gate
async def test_terminal_recheck_keeps_submitted_slice_version(
    tmp_path: Path,
) -> None:
    workspace, store, first_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-terminal-lineage",
    )
    decision = service.submit(_valid_draft(first_slice))

    collector = ToolObservationCollector()
    await collector.handle(
        ToolCallStartedEvent(
            run_id="explorer-terminal-revision",
            tool_use_id="read-terminal-revision",
            tool_name="read_file",
            params={"path": "src/target.py"},
            ts="t1",
        )
    )
    await collector.handle(
        ToolCallFinishedEvent(
            run_id="explorer-terminal-revision",
            tool_use_id="read-terminal-revision",
            tool_name="read_file",
            elapsed_ms=1,
            output=(workspace / "src" / "target.py").read_text(encoding="utf-8"),
            ts="t2",
        )
    )
    second_slice = ArchitectureSliceService(
        workspace_root=workspace,
        run_id="explorer-terminal-revision",
        goal="Change target behavior",
        collector=collector,
        session_id="sess-1",
        store=store,
    ).submit(
        ArchitectureSliceDraft(
            slice_id=first_slice.slice_id,
            relevant_modules=("src/target.py",),
            existing_patterns=("newer pattern",),
            likely_change_targets=("src/target.py",),
            evidence_tool_call_ids=("read-terminal-revision",),
            completeness="complete_for_task",
            confidence=0.95,
        )
    )

    assert second_slice.version == first_slice.version + 1
    assert decision.architecture_slice_version == first_slice.version
    assert service.terminal_failure_reason() is None


# 功能：验证同名 immutable decision 文件若 bytes 不同则 corruption fail closed
# 设计：绕过 service 用 SessionStore 连续写同 identity 的不同 payload，锁定 create-once 边界
def test_decision_store_rejects_conflicting_existing_version(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.write_decision("sess-1", "decision_one", 1, {"version": 1, "value": "a"})

    with pytest.raises(ValueError, match="immutable decision conflict"):
        store.write_decision(
            "sess-1",
            "decision_one",
            1,
            {"version": 1, "value": "b"},
        )


# 功能：验证 grounded absent target 可进入 files_to_create，外部创建后同一 decision 变为 stale
# 设计：先提交合法 create scope，再只改变 absence state，区分 scope schema 与 relevant snapshot 时效性
async def test_planner_decision_tracks_planned_create_absence(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(
        tmp_path,
        likely_targets=("src/target.py", "src/new.py"),
    )
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    )
    draft = _valid_draft(
        architecture_slice,
        files_to_create=["src/new.py"],
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update target and add a grounded companion module.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py", "src/new.py"],
                "evidence_refs": ["read-target"],
            }
        ],
    )

    decision = service.submit(draft)
    assert decision.files_to_create == ("src/new.py",)

    (workspace / "src" / "new.py").write_text("NEW = True\n", encoding="utf-8")
    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(draft)
    assert exc_info.value.category == "snapshot-stale"


# 功能：验证 decision envelope 合法但内部 content_digest 被篡改时读取仍 fail closed
# 设计：直接重算外层 store digest 以绕过 envelope 检查，单独锁定 immutable model 自身摘要验证
async def test_decision_store_validates_internal_content_digest(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    decision = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    ).submit(_valid_draft(architecture_slice))
    path = (
        store.session_dir("sess-1")
        / "planning"
        / "decisions"
        / f"{decision.decision_id}-v1.json"
    )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["selected_approach"] = "tampered"
    encoded = json.dumps(
        envelope["payload"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope["digest"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="content digest mismatch"):
        store.read_decision("sess-1", decision.decision_id, 1)


# 功能：验证 typed submit tool 将 validation category 作为稳定 non-retryable ToolResult 返回
# 设计：用 fabricated evidence 触发真实 service validation，避免异常落入 generic execution_error
async def test_planner_submit_tool_returns_stable_validation_error(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    tool = planning.PlannerDecisionSubmitTool(
        planning.PlannerDecisionService(
            workspace_root=workspace,
            session_id="sess-1",
            store=store,
            goal="Change target behavior",
        )
    )
    draft = _valid_draft(
        architecture_slice,
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update target.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py"],
                "evidence_refs": ["fabricated"],
            }
        ],
    )

    result = await tool.invoke(draft.model_dump(mode="json"))

    assert result.is_error is True
    assert result.error_type == "invalid_input"
    assert "reference-invalid" in result.content


# 功能：验证第一次有效提交后相同 draft 重试只返回原 decision identity
# 设计：复用同一个 service 重复提交 canonical 相同内容，锁定 terminal commit 的安全幂等边界
async def test_planner_submit_exact_duplicate_is_idempotent(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-idempotent",
    )
    draft = _valid_draft(architecture_slice)

    first = service.submit(draft)
    second = service.submit(draft)

    assert second == first
    assert len(store.list_decisions("sess-1")) == 1
    assert service.is_terminal_committed is True


# 功能：验证结构化 PlanView 的 schema、section budget 与 JSON transport round-trip 保持一致
# 设计：从真实 immutable decision 生成 projection，再以 model_dump 重新解析，覆盖 JSON list 归一化边界
async def test_plan_view_transport_roundtrip_preserves_structured_sections(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    decision = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
    ).submit(_valid_draft(architecture_slice))

    plan = planning.build_plan_view(decision)
    restored = planning.PlanView.model_validate(plan.model_dump(mode="json"))

    assert restored == plan
    assert plan.schema_version == 1
    assert plan.section_budgets["requirements"] == len(plan.requirements)
    assert plan.section_budgets["verification_plan"] == len(plan.verification_plan)


# 功能：验证 terminal commit 后不同 draft 不能生成同一 run 的新 immutable version
# 设计：先完成一次合法提交，再只改变 selected_approach，断言稳定 terminal-contract violation
async def test_planner_submit_different_after_commit_is_rejected(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-terminal",
    )
    service.submit(_valid_draft(architecture_slice))

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(
            _valid_draft(
                architecture_slice,
                selected_approach="Use a different bounded edit.",
            )
        )

    assert exc_info.value.category == "terminal-contract-violation"
    assert len(store.list_decisions("sess-1")) == 1


# 功能：验证不同 Planner service 不能借用同一 session 中另一 run 的 terminal identity
# 设计：run A 提交后让 run B 直接做 terminal 校验，证明 identity 存在于 service 而非 session 全局
async def test_planner_terminal_identity_is_run_local(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service_a = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-a",
    )
    service_b = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-b",
    )

    service_a.submit(_valid_draft(architecture_slice))

    assert service_a.terminal_failure_reason() is None
    assert service_b.terminal_failure_reason() == "missing-terminal-decision"


# 功能：验证从未调用 planner_decision_submit 时 terminal reason 保持 missing-terminal-decision
# 设计：直接对全新 service 做 terminal 检查，排除历史 session artifact 对本次 run 的影响
async def test_planner_without_submit_is_missing_terminal_decision(tmp_path: Path) -> None:
    workspace, store, _architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-no-submit",
    )

    assert service.last_submit_outcome == "none"
    assert service.terminal_failure_reason() == "missing-terminal-decision"


# 功能：验证最后一次 incomplete validation 才能产生 planning-input-incomplete
# 设计：提交缺少 requirements 的 draft，断言 service-local 状态和 terminal reason 同步更新
async def test_planner_last_incomplete_validation_is_reported(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-incomplete",
    )

    with pytest.raises(planning.PlannerValidationError):
        service.submit(_valid_draft(architecture_slice, requirements=[]))

    assert service.last_submit_outcome == "incomplete"
    assert service.terminal_failure_reason() == "planning-input-incomplete"


# 功能：验证 incomplete 后的后续 invalid submit 会覆盖旧 incomplete 状态
# 设计：先触发显式 incomplete，再触发 fabricated evidence，确保 terminal gate 只看最后一次 submit
async def test_planner_later_invalid_submit_does_not_reuse_incomplete_reason(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-invalid-after-incomplete",
    )

    with pytest.raises(planning.PlannerValidationError):
        service.submit(_valid_draft(architecture_slice, requirements=[]))
    with pytest.raises(planning.PlannerValidationError):
        service.submit(
            _valid_draft(
                architecture_slice,
                intended_changes=[
                    {
                        "change_id": "C1",
                        "description": "Update target.",
                        "requirement_ids": ["R1"],
                        "target_paths": ["src/target.py"],
                        "evidence_refs": ["fabricated"],
                    }
                ],
            )
        )

    assert service.last_submit_outcome == "invalid"
    assert service.terminal_failure_reason() == "missing-terminal-decision"


# 功能：验证 schema-invalid submit 也会覆盖此前的 incomplete terminal 状态
# 设计：先走真实 incomplete draft，再用缺字段 payload 触发 tool schema 校验，覆盖 service 未收到 draft 的边界
async def test_planner_schema_invalid_submit_does_not_reuse_incomplete_reason(
    tmp_path: Path,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-schema-invalid-after-incomplete",
    )
    tool = planning.PlannerDecisionSubmitTool(service)

    incomplete = _valid_draft(architecture_slice, requirements=[])
    incomplete_result = await tool.invoke(incomplete.model_dump(mode="json"))
    assert incomplete_result.is_error is True
    assert service.last_submit_outcome == "incomplete"

    invalid_result = await tool.invoke({"requirements": []})

    assert invalid_result.is_error is True
    assert invalid_result.error_type == "invalid_input"
    assert service.last_submit_outcome == "invalid"
    assert service.terminal_failure_reason() == "missing-terminal-decision"


# 功能：验证 incomplete 后 valid submit 可以成功并覆盖之前的 validation 状态
# 设计：先触发可恢复的 incomplete，再提交原始合法 draft，锁定 retry 后的 terminal success
async def test_planner_valid_submit_after_incomplete_succeeds(tmp_path: Path) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target behavior",
        run_id="planner-run-valid-after-incomplete",
    )

    with pytest.raises(planning.PlannerValidationError):
        service.submit(_valid_draft(architecture_slice, requirements=[]))
    decision = service.submit(_valid_draft(architecture_slice))

    assert decision.version == 1
    assert service.last_submit_outcome == "accepted"
    assert service.terminal_failure_reason() is None


# 功能：验证 planner_decision_submit 的公开描述声明 terminal commit 语义
# 设计：直接读取 typed tool description，避免把安全语义只放在不可见实现细节中
def test_planner_submit_description_declares_terminal_commit() -> None:
    planning = _planning()
    service = object.__new__(planning.PlannerDecisionService)
    tool = planning.PlannerDecisionSubmitTool(service)

    assert "terminal" in tool.description.lower()
    assert "final" in tool.description.lower()


# 功能：验证 manifest 与 bus protocol target 必须在显式 declaration 列表中
# 设计：把机械可识别 target 纳入 grounded likely targets，再省略对应 declaration，参数化锁定两类规则
@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("pyproject.toml", "dependency_changes"),
        ("src/kama_claude/core/bus/events.py", "protocol_or_schema_changes"),
    ],
)
async def test_planner_decision_requires_explicit_special_change_declaration(
    tmp_path: Path,
    target: str,
    field: str,
) -> None:
    workspace, store, architecture_slice = await _ground_complete_slice(tmp_path)
    special = workspace / target
    special.parent.mkdir(parents=True, exist_ok=True)
    special.write_text("content\n", encoding="utf-8")
    # 先把 special target 纳入真实 read evidence 与新的 slice/snapshot。
    collector = ToolObservationCollector()
    for tool_id, path in (("read-target", "src/target.py"), ("read-special", target)):
        await collector.handle(
            ToolCallStartedEvent(
                run_id="explorer-special",
                tool_use_id=tool_id,
                tool_name="read_file",
                params={"path": path},
                ts="t1",
            )
        )
        await collector.handle(
            ToolCallFinishedEvent(
                run_id="explorer-special",
                tool_use_id=tool_id,
                tool_name="read_file",
                elapsed_ms=1,
                output=(workspace / path).read_text(encoding="utf-8"),
                ts="t2",
            )
        )
    grounded = ArchitectureSliceService(
        workspace_root=workspace,
        run_id="explorer-special",
        goal="Change target and special file",
        collector=collector,
        session_id="sess-1",
        store=store,
    ).submit(
        ArchitectureSliceDraft(
            relevant_modules=("src/target.py", target),
            likely_change_targets=("src/target.py", target),
            evidence_tool_call_ids=("read-target", "read-special"),
            completeness="complete_for_task",
            confidence=0.8,
        )
    )
    planning = _planning()
    service = planning.PlannerDecisionService(
        workspace_root=workspace,
        session_id="sess-1",
        store=store,
        goal="Change target and special file",
    )
    draft = _valid_draft(
        grounded,
        files_to_modify=["src/target.py", target],
        intended_changes=[
            {
                "change_id": "C1",
                "description": "Update target and declared boundary.",
                "requirement_ids": ["R1"],
                "target_paths": ["src/target.py", target],
                "evidence_refs": ["read-target", "read-special"],
            }
        ],
    )

    with pytest.raises(planning.PlannerValidationError) as exc_info:
        service.submit(draft)

    assert exc_info.value.category == "scope-invalid"
    assert field in str(exc_info.value)


# 功能：验证 planner_decision_submit tool 只出现在 session-backed Planner registry
# 设计：构造真实 root SpawnAgentTool 与 builtin Planner profile，直接检查 child registry 而不运行模型
def test_planner_profile_registry_has_typed_submit_tool(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    from kama_claude.core.agents.loader import AgentProfileLoader
    from kama_claude.core.events.bus import EventBus
    from kama_claude.core.subagent.registry import BackgroundTaskRegistry
    from kama_claude.core.subagent.tool import SpawnAgentTool

    planning = _planning()

    root = SpawnAgentTool(
        provider=AsyncMock(),
        workspace_root=tmp_path,
        parent_bus=EventBus(),
        parent_run_id="root-run",
        permission_manager=None,
        max_steps=5,
        task_registry=BackgroundTaskRegistry(),
        runs_dir=tmp_path / "runs",
        session_id="sess-1",
        store=SessionStore(tmp_path / "sessions"),
    )
    planner = AgentProfileLoader(tmp_path).load("planner")
    assert planner is not None

    service = planning.PlannerDecisionService(
        workspace_root=tmp_path,
        session_id="sess-1",
        store=root._store,
        goal="inspect",
        run_id="planner-run",
    )
    registry = root._build_child_registry(
        EventBus(),
        "planner-run",
        planner,
        planner_service=service,
    )

    assert registry.get("planner_decision_submit") is not None
    assert registry.get("write_file") is None
    assert registry.get("bash") is None
