"""索引持久化（Repository pattern）：ChunkRecord JSONL + manifest，原子写与损坏检测"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.semantic.components.chunker import ChunkRecord
from kama_claude.core.semantic.components.embedding import SparseVector
from kama_claude.core.semantic.errors import IndexCorruptedError

_INDEX_VERSION = 1

# workspace 根 → 当前 HEAD 提交的提供器（由装配层注入，semantic 不依赖 git 模块）
GitHeadProvider = Callable[[Path], str | None]


@dataclass(frozen=True)
class FileStat:
    """文件指纹：修改时间（纳秒）+ 大小，用于增量检测"""

    mtime_ns: int
    size: int


@dataclass(frozen=True)
class IndexState:
    """解析后的 manifest：构建元数据 + 文件指纹快照"""

    version: int
    strategy: str
    built_at: str
    git_head: str | None
    files: dict[str, FileStat]


class SemanticIndex:
    """ChunkRecord 仓库：records.jsonl（每行一条）+ manifest.json（提交点）"""

    def __init__(
        self,
        index_dir: Path,
        *,
        git_head_provider: GitHeadProvider | None = None,
    ) -> None:
        self._dir = index_dir
        self._records_path = index_dir / "records.jsonl"
        self._manifest_path = index_dir / "manifest.json"
        self._git_head_provider = git_head_provider

    def write(
        self,
        records: list[ChunkRecord],
        *,
        files: dict[str, FileStat],
        git_head: str | None,
        strategy: str,
    ) -> None:
        """原子落盘：先 records 后 manifest；任一失败不留半成品"""
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": _INDEX_VERSION,
            "strategy": strategy,
            "built_at": datetime.now(UTC).isoformat(),
            "git_head": git_head,
            "files": {rel: {"mtime_ns": s.mtime_ns, "size": s.size} for rel, s in files.items()},
        }
        lines = "\n".join(json.dumps(_record_to_dict(r), ensure_ascii=False) for r in records)
        _atomic_write(self._records_path, lines + "\n")
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        _atomic_write(self._manifest_path, manifest_text)

    def read_state(self) -> IndexState | None:
        """读 manifest；缺失返回 None，损坏/版本不符抛 IndexCorruptedError"""
        if not self._manifest_path.exists():
            return None
        try:
            return _parse_state(json.loads(self._manifest_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise IndexCorruptedError(f"manifest unreadable: {exc}") from exc

    def read_records(self) -> list[ChunkRecord]:
        """读全部记录；行损坏/字段缺失抛 IndexCorruptedError"""
        if not self._records_path.exists():
            return []
        records: list[ChunkRecord] = []
        try:
            raw = self._records_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                if not line.strip():
                    continue
                records.append(_record_from_dict(json.loads(line)))
        except (OSError, ValueError) as exc:
            raise IndexCorruptedError(f"records unreadable: {exc}") from exc
        return records

    def git_head_changed(self, root: Path, state: IndexState) -> bool:
        """HEAD 与 manifest 记录不一致 → 需全量重建（分支切换是整体变化）"""
        if self._git_head_provider is None:
            return False
        current = self._git_head_provider(root)
        if current is None:
            return False
        return current != state.git_head


def changed_paths(
    current: dict[str, FileStat], known: dict[str, FileStat]
) -> tuple[set[str], set[str]]:
    """指纹 diff：返回 (变化或新增, 已删除)；未变文件不参与重索引"""
    changed = {p for p, s in current.items() if known.get(p) != s}
    deleted = set(known) - set(current)
    return changed, deleted


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _record_to_dict(r: ChunkRecord) -> dict[str, object]:
    vector = None
    if r.vector is not None:
        vector = {"indices": list(r.vector.indices), "values": list(r.vector.values)}
    return {
        "chunk_id": r.chunk_id,
        "logical_path": r.logical_path,
        "start_line": r.start_line,
        "end_line": r.end_line,
        "text": r.text,
        "symbol_type": r.symbol_type,
        "symbol_name": r.symbol_name,
        "parent_symbol": r.parent_symbol,
        "language": r.language,
        "vector": vector,
    }


def _record_from_dict(data: object) -> ChunkRecord:
    if not isinstance(data, dict):
        raise ValueError("record must be a JSON object")
    vector = None
    vector_raw = data.get("vector")
    if vector_raw is not None:
        if not isinstance(vector_raw, dict):
            raise ValueError("record vector must be an object")
        indices_raw = vector_raw.get("indices")
        values_raw = vector_raw.get("values")
        if not isinstance(indices_raw, list) or not isinstance(values_raw, list):
            raise ValueError("record vector indices/values must be arrays")
        indices = tuple(_require_int(v, "vector index") for v in indices_raw)
        values = tuple(float(v) for v in values_raw)
        vector = SparseVector(indices, values)
    parent = data.get("parent_symbol")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("record parent_symbol must be a string or null")
    language = data.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("record language must be a string or null")
    return ChunkRecord(
        chunk_id=_require_str(data.get("chunk_id"), "chunk_id"),
        logical_path=_require_str(data.get("logical_path"), "logical_path"),
        start_line=_require_int(data.get("start_line"), "start_line"),
        end_line=_require_int(data.get("end_line"), "end_line"),
        text=_require_str(data.get("text"), "text"),
        symbol_type=_require_str(data.get("symbol_type"), "symbol_type"),
        symbol_name=_require_str(data.get("symbol_name"), "symbol_name"),
        parent_symbol=parent,
        language=language,
        vector=vector,
    )


def _parse_state(data: object) -> IndexState:
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    version = data.get("version")
    if version != _INDEX_VERSION:
        raise IndexCorruptedError(f"index version {version!r} != {_INDEX_VERSION}")
    files_raw = data.get("files")
    if not isinstance(files_raw, dict):
        raise ValueError("manifest 'files' must be an object")
    files: dict[str, FileStat] = {}
    for rel, stat in files_raw.items():
        if not isinstance(stat, dict):
            raise ValueError(f"manifest stat for {rel!r} must be an object")
        files[rel] = FileStat(
            mtime_ns=_require_int(stat.get("mtime_ns"), f"{rel} mtime_ns"),
            size=_require_int(stat.get("size"), f"{rel} size"),
        )
    git_head = data.get("git_head")
    if git_head is not None and not isinstance(git_head, str):
        raise ValueError("manifest git_head must be a string or null")
    return IndexState(
        version=_INDEX_VERSION,
        strategy=_require_str(data.get("strategy"), "strategy"),
        built_at=_require_str(data.get("built_at"), "built_at"),
        git_head=git_head,
        files=files,
    )


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value
