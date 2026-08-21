from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.bus.events import ToolCallFinishedEvent, ToolCallStartedEvent
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

if TYPE_CHECKING:
    from kama_claude.core.session.store import SessionStore

_INSTRUCTION_FILENAMES = ("AGENTS.md", "AGENT.md", "CLAUDE.md")
_READ_FILE_MAX_BYTES = 512 * 1024
GroundingSourceKind = Literal[
    "explicit_instruction",
    "generated_context",
    "repository_evidence",
    "model_inference",
]


# 使用稳定 JSON 编码计算结构化内容摘要
def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# 计算文件原始字节的 SHA-256 摘要
def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 规范化 workspace root 并复用唯一的 workspace identity 算法
def workspace_identity(workspace_root: Path) -> tuple[Path, str]:
    root = WorkspacePathResolver(workspace_root).root
    workspace_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return root, workspace_id


class InstructionSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str
    scope_path: str
    kind: GroundingSourceKind
    content_digest: str
    content: str


class EffectiveInstructionSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    root_sources: tuple[InstructionSource, ...]
    sources_by_target: dict[str, tuple[InstructionSource, ...]] = Field(
        default_factory=dict
    )
    unresolved_conflicts: tuple[str, ...] = ()
    content_digest: str


class RepositorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    git_head: str | None = None
    instruction_file_digests: dict[str, str] = Field(default_factory=dict)
    grounding_file_digests: dict[str, str] = Field(default_factory=dict)
    planned_existing_target_digests: dict[str, str] = Field(default_factory=dict)
    planned_new_target_paths: tuple[str, ...] = ()
    relevant_manifest_digests: dict[str, str] = Field(default_factory=dict)
    relevant_untracked_target_digests: dict[str, str] = Field(default_factory=dict)
    snapshot_digest: str


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    tool_call_id: str
    tool_name: str
    logical_path: str | None = None
    content_digest: str


class ArchitectureSliceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice_id: str | None = None
    relevant_entrypoints: tuple[str, ...] = ()
    relevant_modules: tuple[str, ...] = ()
    relevant_symbols: tuple[str, ...] = ()
    confirmed_call_paths: tuple[tuple[str, ...], ...] = ()
    state_or_data_flow_summary: str = ""
    existing_patterns: tuple[str, ...] = ()
    extension_points: tuple[str, ...] = ()
    related_tests: tuple[str, ...] = ()
    relevant_invariants: tuple[str, ...] = ()
    likely_change_targets: tuple[str, ...] = ()
    affected_boundaries: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    evidence_tool_call_ids: tuple[str, ...] = ()
    completeness: Literal["complete_for_task", "partial", "blocked"]
    confidence: float = Field(ge=0.0, le=1.0)


class ArchitectureSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slice_id: str
    version: int
    goal: str
    snapshot_digest: str
    effective_instruction_refs: tuple[str, ...]
    relevant_entrypoints: tuple[str, ...]
    relevant_modules: tuple[str, ...]
    relevant_symbols: tuple[str, ...]
    confirmed_call_paths: tuple[tuple[str, ...], ...]
    state_or_data_flow_summary: str
    existing_patterns: tuple[str, ...]
    extension_points: tuple[str, ...]
    related_tests: tuple[str, ...]
    relevant_invariants: tuple[str, ...]
    likely_change_targets: tuple[str, ...]
    affected_boundaries: tuple[str, ...]
    non_goals: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    completeness: Literal["complete_for_task", "partial", "blocked"]
    confidence: float
    content_digest: str


# 返回 snapshot stale equality 使用的 canonical payload，明确排除 git_head
def _snapshot_payload(
    *,
    workspace_id: str,
    instruction_file_digests: dict[str, str],
    grounding_file_digests: dict[str, str],
    planned_existing_target_digests: dict[str, str],
    planned_new_target_paths: Sequence[str],
    relevant_manifest_digests: dict[str, str],
    relevant_untracked_target_digests: dict[str, str],
) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "instruction_file_digests": instruction_file_digests,
        "grounding_file_digests": grounding_file_digests,
        "planned_existing_target_digests": planned_existing_target_digests,
        "planned_new_target_paths": list(planned_new_target_paths),
        "relevant_manifest_digests": relevant_manifest_digests,
        "relevant_untracked_target_digests": relevant_untracked_target_digests,
    }


class RepositoryInstructionLoader:
    # 绑定 canonical workspace，作为 instruction discovery 的唯一根目录
    def __init__(self, workspace_root: Path) -> None:
        self._resolver = WorkspacePathResolver(workspace_root)
        self._root = self._resolver.root

    # 加载 root sources，并为每个 target 按 root 到 parent 深度收集适用来源
    def load(self, targets: Sequence[str]) -> EffectiveInstructionSet:
        root_sources = self._load_directory(Path("."))
        sources_by_target: dict[str, tuple[InstructionSource, ...]] = {}
        for target in targets:
            normalized = self._normalize_target(target)
            sources = list(root_sources)
            parent = Path(normalized).parent
            if parent != Path("."):
                for depth in range(1, len(parent.parts) + 1):
                    sources.extend(self._load_directory(Path(*parent.parts[:depth])))
            sources_by_target[normalized] = tuple(sources)
        digest_payload = {
            "root_sources": [source.model_dump(mode="json") for source in root_sources],
            "sources_by_target": {
                target: [source.model_dump(mode="json") for source in sources]
                for target, sources in sources_by_target.items()
            },
            "unresolved_conflicts": [],
        }
        return EffectiveInstructionSet(
            root_sources=tuple(root_sources),
            sources_by_target=sources_by_target,
            content_digest=canonical_digest(digest_payload),
        )

    # 将 target 归一化为 workspace 内 canonical relative path
    def _normalize_target(self, target: str) -> str:
        logical = Path(target)
        if not target or logical.is_absolute() or ".." in logical.parts:
            raise ValueError("target must be workspace-relative")
        resolved = self._resolver.resolve_for_write(target)
        relative = resolved.relative_to(self._root).as_posix()
        if relative == ".":
            raise ValueError("target must be workspace-relative")
        return relative

    # 读取目录内全部 compatibility instruction 文件且不设置文件名优先级
    def _load_directory(self, directory: Path) -> list[InstructionSource]:
        sources: list[InstructionSource] = []
        scope_path = directory.as_posix()
        for filename in _INSTRUCTION_FILENAMES:
            logical = (directory / filename).as_posix()
            candidate = self._root / logical
            if not candidate.exists():
                continue
            resolved = self._resolver.resolve_existing(logical)
            if not resolved.is_file():
                continue
            raw = resolved.read_bytes()
            sources.append(
                InstructionSource(
                    source_path=logical.removeprefix("./"),
                    scope_path=scope_path,
                    kind="explicit_instruction",
                    content_digest=hashlib.sha256(raw).hexdigest(),
                    content=raw.decode("utf-8"),
                )
            )
        return sources


# 将 instruction sources 渲染为带来源标识且不改写正文的 prompt 内容
def render_repository_instructions(sources: Sequence[InstructionSource]) -> str:
    sections = [
        f"### Source: {source.source_path} (scope: {source.scope_path})\n"
        + source.content
        for source in sources
    ]
    return "\n\n".join(sections)


class SnapshotBuilder:
    # 绑定 workspace resolver 与既有敏感路径 policy
    def __init__(self, workspace_root: Path) -> None:
        self._root, self._workspace_id = workspace_identity(workspace_root)
        self._resolver = WorkspacePathResolver(self._root)
        self._access_policy = WorkspaceAccessPolicy(self._root)

    # 只捕获显式列出的 instruction、evidence、target 与依赖相关内容
    def capture(
        self,
        *,
        instruction_sources: Sequence[InstructionSource] = (),
        grounding_paths: Sequence[str] = (),
        planned_existing_targets: Sequence[str] = (),
        planned_new_targets: Sequence[str] = (),
        relevant_manifests: Sequence[str] = (),
        relevant_untracked_targets: Sequence[str] = (),
        git_head: str | None = None,
    ) -> RepositorySnapshot:
        instruction_digests = self._digest_paths(
            [source.source_path for source in instruction_sources]
        )
        grounding_digests = self._digest_paths(grounding_paths)
        existing_target_digests = self._digest_paths(planned_existing_targets)
        manifest_digests = self._digest_paths(relevant_manifests)
        untracked_digests = self._digest_paths(relevant_untracked_targets)
        new_targets = tuple(sorted({self._normalize(path) for path in planned_new_targets}))
        for path in new_targets:
            candidate = self._resolver.resolve_for_write(path)
            self._access_policy.ensure_allowed(path, candidate)
            if candidate.exists() or candidate.is_symlink():
                raise ValueError(f"planned new target already exists: {path}")
        digest_payload = _snapshot_payload(
            workspace_id=self._workspace_id,
            instruction_file_digests=instruction_digests,
            grounding_file_digests=grounding_digests,
            planned_existing_target_digests=existing_target_digests,
            planned_new_target_paths=new_targets,
            relevant_manifest_digests=manifest_digests,
            relevant_untracked_target_digests=untracked_digests,
        )
        return RepositorySnapshot(
            workspace_id=self._workspace_id,
            git_head=git_head,
            instruction_file_digests=instruction_digests,
            grounding_file_digests=grounding_digests,
            planned_existing_target_digests=existing_target_digests,
            planned_new_target_paths=new_targets,
            relevant_manifest_digests=manifest_digests,
            relevant_untracked_target_digests=untracked_digests,
            snapshot_digest=canonical_digest(digest_payload),
        )

    # 重新读取 snapshot 的相关路径并判断内容与 absence 状态是否仍一致
    def is_current(self, snapshot: RepositorySnapshot) -> bool:
        if snapshot.workspace_id != self._workspace_id:
            return False
        digest_payload = _snapshot_payload(
            workspace_id=snapshot.workspace_id,
            instruction_file_digests=snapshot.instruction_file_digests,
            grounding_file_digests=snapshot.grounding_file_digests,
            planned_existing_target_digests=snapshot.planned_existing_target_digests,
            planned_new_target_paths=snapshot.planned_new_target_paths,
            relevant_manifest_digests=snapshot.relevant_manifest_digests,
            relevant_untracked_target_digests=(
                snapshot.relevant_untracked_target_digests
            ),
        )
        if canonical_digest(digest_payload) != snapshot.snapshot_digest:
            return False
        mappings = (
            snapshot.instruction_file_digests,
            snapshot.grounding_file_digests,
            snapshot.planned_existing_target_digests,
            snapshot.relevant_manifest_digests,
            snapshot.relevant_untracked_target_digests,
        )
        try:
            for expected in mappings:
                if self._digest_paths(expected) != expected:
                    return False
            for path in snapshot.planned_new_target_paths:
                candidate = self._resolver.resolve_for_write(path)
                self._access_policy.ensure_allowed(path, candidate)
                if candidate.exists() or candidate.is_symlink():
                    return False
        except (OSError, ValueError):
            return False
        return True

    # 归一化并校验一个 workspace-relative logical path
    def _normalize(self, logical_path: str) -> str:
        path = Path(logical_path)
        if not logical_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be workspace-relative")
        resolved = self._resolver.resolve_for_write(logical_path)
        normalized = resolved.relative_to(self._root).as_posix()
        if normalized == ".":
            raise ValueError("path must be workspace-relative")
        return normalized

    # 对显式路径集合做 canonical normalization、policy 检查和内容摘要
    def _digest_paths(self, paths: Sequence[str] | dict[str, str]) -> dict[str, str]:
        digests: dict[str, str] = {}
        for logical_path in paths:
            normalized = self._normalize(logical_path)
            resolved = self._resolver.resolve_existing(normalized)
            self._access_policy.ensure_allowed(normalized, resolved)
            if not resolved.is_file():
                raise ValueError(f"relevant path is not a regular file: {normalized}")
            digests[normalized] = _file_digest(resolved)
        return dict(sorted(digests.items()))


class ToolObservationCollector:
    # 初始化本次 explorer run 的 started 与 successful observation ledger
    def __init__(self) -> None:
        self._started: dict[str, ToolCallStartedEvent] = {}
        self._evidence: dict[str, EvidenceRef] = {}

    # 从既有 tool lifecycle 事件构建只含 provenance 与 output digest 的 evidence
    async def handle(self, event: BaseModel) -> None:
        if isinstance(event, ToolCallStartedEvent):
            self._started[event.tool_use_id] = event
            return
        if not isinstance(event, ToolCallFinishedEvent):
            return
        started = self._started.get(event.tool_use_id)
        if started is None or started.run_id != event.run_id:
            return
        logical_path: str | None = None
        raw_path = started.params.get("path")
        if isinstance(raw_path, str):
            logical_path = raw_path
        self._evidence[event.tool_use_id] = EvidenceRef(
            run_id=event.run_id,
            tool_call_id=event.tool_use_id,
            tool_name=event.tool_name,
            logical_path=logical_path,
            content_digest=hashlib.sha256(event.output.encode("utf-8")).hexdigest(),
        )

    # 按提交顺序解析 tool call IDs，未知或失败 observation 立即拒绝
    def resolve(self, tool_call_ids: Sequence[str]) -> tuple[EvidenceRef, ...]:
        resolved: list[EvidenceRef] = []
        seen: set[str] = set()
        for tool_call_id in tool_call_ids:
            if tool_call_id in seen:
                continue
            evidence = self._evidence.get(tool_call_id)
            if evidence is None:
                raise ValueError(f"unknown evidence tool call: {tool_call_id}")
            resolved.append(evidence)
            seen.add(tool_call_id)
        return tuple(resolved)


class ArchitectureSliceService:
    # 绑定单个 explorer run 的 workspace、goal、collector 与可选 session store
    def __init__(
        self,
        *,
        workspace_root: Path,
        run_id: str,
        goal: str,
        collector: ToolObservationCollector,
        session_id: str = "",
        store: SessionStore | None = None,
        git_head: str | None = None,
    ) -> None:
        self._root = WorkspacePathResolver(workspace_root).root
        self._run_id = run_id
        self._goal = goal
        self._collector = collector
        self._session_id = session_id
        self._store = store
        self._git_head = git_head
        self._submitted: ArchitectureSlice | None = None
        self._versions: dict[str, int] = {}

    @property
    # 返回本次 run 已提交或 runtime 记录的终态 slice
    def submitted(self) -> ArchitectureSlice | None:
        return self._submitted

    # 校验 evidence 与相关路径，分配 immutable slice identity 并持久化
    def submit(self, draft: ArchitectureSliceDraft) -> ArchitectureSlice:
        evidence_refs = self._collector.resolve(draft.evidence_tool_call_ids)
        for ref in evidence_refs:
            if ref.run_id != self._run_id:
                raise ValueError("evidence belongs to another run")
        if draft.completeness == "complete_for_task" and not evidence_refs:
            raise ValueError("complete_for_task requires recorded evidence")
        targets = tuple(
            dict.fromkeys(
                (
                    *draft.relevant_entrypoints,
                    *draft.relevant_modules,
                    *draft.likely_change_targets,
                )
            )
        )
        normalized_targets = tuple(self._normalize_target(path) for path in targets)
        read_paths = {
            self._normalize_target(ref.logical_path)
            for ref in evidence_refs
            if ref.tool_name == "read_file" and ref.logical_path is not None
        }
        for ref in evidence_refs:
            if ref.tool_name == "read_file":
                self._validate_read_evidence(ref)
        for path in normalized_targets:
            candidate = self._root / path
            if candidate.exists() and path not in read_paths:
                raise ValueError(f"read_file evidence required for confirmed path: {path}")
        instructions = RepositoryInstructionLoader(self._root).load(
            draft.likely_change_targets
        )
        grounding_paths = sorted(read_paths)
        existing_targets = [
            self._normalize_target(path)
            for path in draft.likely_change_targets
            if (self._root / self._normalize_target(path)).exists()
        ]
        new_targets = [
            self._normalize_target(path)
            for path in draft.likely_change_targets
            if not (self._root / self._normalize_target(path)).exists()
        ]
        instruction_sources: dict[str, InstructionSource] = {
            source.source_path: source for source in instructions.root_sources
        }
        for sources in instructions.sources_by_target.values():
            instruction_sources.update({source.source_path: source for source in sources})
        root_instruction_paths = {
            source.source_path for source in instructions.root_sources
        }
        nested_instruction_paths = set(instruction_sources) - root_instruction_paths
        unread_nested_instructions = nested_instruction_paths - read_paths
        if unread_nested_instructions:
            missing = ", ".join(sorted(unread_nested_instructions))
            raise ValueError(
                f"read_file evidence required for task-local instructions: {missing}"
            )
        snapshot = SnapshotBuilder(self._root).capture(
            instruction_sources=tuple(instruction_sources.values()),
            grounding_paths=grounding_paths,
            planned_existing_targets=existing_targets,
            planned_new_targets=new_targets,
            git_head=self._git_head,
        )
        slice_id = draft.slice_id or f"slice_{uuid.uuid4().hex}"
        version = self._next_version(slice_id, revision=draft.slice_id is not None)
        payload: dict[str, object] = {
            "slice_id": slice_id,
            "version": version,
            "goal": self._goal,
            "snapshot_digest": snapshot.snapshot_digest,
            "effective_instruction_refs": sorted(instruction_sources),
            "relevant_entrypoints": list(draft.relevant_entrypoints),
            "relevant_modules": list(draft.relevant_modules),
            "relevant_symbols": list(draft.relevant_symbols),
            "confirmed_call_paths": [list(path) for path in draft.confirmed_call_paths],
            "state_or_data_flow_summary": draft.state_or_data_flow_summary,
            "existing_patterns": list(draft.existing_patterns),
            "extension_points": list(draft.extension_points),
            "related_tests": list(draft.related_tests),
            "relevant_invariants": list(draft.relevant_invariants),
            "likely_change_targets": list(draft.likely_change_targets),
            "affected_boundaries": list(draft.affected_boundaries),
            "non_goals": list(draft.non_goals),
            "unresolved_questions": list(draft.unresolved_questions),
            "evidence_refs": [ref.model_dump(mode="json") for ref in evidence_refs],
            "completeness": draft.completeness,
            "confidence": draft.confidence,
        }
        architecture_slice = ArchitectureSlice(
            slice_id=slice_id,
            version=version,
            goal=self._goal,
            snapshot_digest=snapshot.snapshot_digest,
            effective_instruction_refs=tuple(sorted(instruction_sources)),
            relevant_entrypoints=draft.relevant_entrypoints,
            relevant_modules=draft.relevant_modules,
            relevant_symbols=draft.relevant_symbols,
            confirmed_call_paths=draft.confirmed_call_paths,
            state_or_data_flow_summary=draft.state_or_data_flow_summary,
            existing_patterns=draft.existing_patterns,
            extension_points=draft.extension_points,
            related_tests=draft.related_tests,
            relevant_invariants=draft.relevant_invariants,
            likely_change_targets=draft.likely_change_targets,
            affected_boundaries=draft.affected_boundaries,
            non_goals=draft.non_goals,
            unresolved_questions=draft.unresolved_questions,
            evidence_refs=evidence_refs,
            completeness=draft.completeness,
            confidence=draft.confidence,
            content_digest=canonical_digest(payload),
        )
        self._submitted = architecture_slice
        self._persist(architecture_slice, snapshot)
        return architecture_slice

    # 未显式 submit 时记录 partial 或 blocked 终态，禁止伪造 complete
    def record_incomplete(
        self,
        completeness: Literal["partial", "blocked"],
        reason: str,
    ) -> ArchitectureSlice:
        if self._submitted is not None:
            return self._submitted
        return self.submit(
            ArchitectureSliceDraft(
                completeness=completeness,
                confidence=0.0,
                unresolved_questions=(reason,),
            )
        )

    # 归一化 ArchitectureSlice 中的 workspace-relative path
    def _normalize_target(self, logical_path: str) -> str:
        path = Path(logical_path)
        if not logical_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("architecture path must be workspace-relative")
        resolved = (self._root / path).resolve(strict=False)
        if not resolved.is_relative_to(self._root):
            raise ValueError("architecture path must be workspace-relative")
        return resolved.relative_to(self._root).as_posix()

    # 重现 read_file 的 bounded text 输出并拒绝提交前已变化的 observation
    def _validate_read_evidence(self, evidence: EvidenceRef) -> None:
        if evidence.logical_path is None:
            raise ValueError("read_file evidence is missing logical path")
        logical_path = self._normalize_target(evidence.logical_path)
        path = WorkspacePathResolver(self._root).resolve_existing(logical_path)
        WorkspaceAccessPolicy(self._root).ensure_allowed(logical_path, path)
        raw = path.read_bytes()
        text = raw[:_READ_FILE_MAX_BYTES].decode("utf-8", errors="replace")
        if len(raw) > _READ_FILE_MAX_BYTES:
            text += "\n[truncated]"
        current_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if current_digest != evidence.content_digest:
            raise ValueError(f"stale read_file evidence: {logical_path}")

    # 从 session artifact 中计算同一 slice lineage 的下一版本
    def _next_version(self, slice_id: str, *, revision: bool) -> int:
        versions = [self._versions.get(slice_id, 0)]
        if self._store is not None and self._session_id:
            grounding = self._store.read_grounding(self._session_id) or {}
            slices = grounding.get("architecture_slices", [])
            if not isinstance(slices, list):
                raise ValueError("planning artifact is corrupt")
            versions.extend(
                int(item["version"])
                for item in slices
                if isinstance(item, dict) and item.get("slice_id") == slice_id
            )
        if revision and max(versions) == 0:
            raise ValueError("slice revision does not identify an existing lineage")
        version = max(versions) + 1
        self._versions[slice_id] = version
        return version

    # 将 slice 与 snapshot 追加到 grounding artifact，完整 ToolResult 仍留在 journal
    def _persist(
        self,
        architecture_slice: ArchitectureSlice,
        snapshot: RepositorySnapshot,
    ) -> None:
        if self._store is None or not self._session_id:
            return
        grounding = self._store.read_grounding(self._session_id) or {}
        slices = grounding.setdefault("architecture_slices", [])
        snapshots = grounding.setdefault("snapshots", [])
        if not isinstance(slices, list) or not isinstance(snapshots, list):
            raise ValueError("planning artifact is corrupt")
        slices.append(architecture_slice.model_dump(mode="json"))
        snapshots.append(snapshot.model_dump(mode="json"))
        self._store.write_grounding(self._session_id, grounding)


class ArchitectureSliceSubmitTool(BaseTool):
    name = "architecture_slice_submit"
    description = (
        "Submit the task-local ArchitectureSlice. Confirmed repository paths must be "
        "backed by successful read_file observations from this explorer run."
    )
    params_model = ArchitectureSliceDraft
    input_schema: dict[str, object] = ArchitectureSliceDraft.model_json_schema()

    # 绑定本次 explorer run 的 slice service
    def __init__(self, service: ArchitectureSliceService) -> None:
        self._service = service

    # 校验 draft 并返回 runtime 分配的 immutable slice identity
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        draft = ArchitectureSliceDraft.model_validate(params)
        architecture_slice = self._service.submit(draft)
        return ToolResult(
            content=json.dumps(
                {
                    "slice_id": architecture_slice.slice_id,
                    "version": architecture_slice.version,
                    "snapshot_digest": architecture_slice.snapshot_digest,
                    "content_digest": architecture_slice.content_digest,
                    "completeness": architecture_slice.completeness,
                },
                sort_keys=True,
            )
        )
