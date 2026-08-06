from __future__ import annotations

import dataclasses
import json

import pytest

from kama_claude.core.semantic.components import index as index_mod
from kama_claude.core.semantic.components.chunker import chunk_text
from kama_claude.core.semantic.components.embedding import LexicalEmbeddingStrategy
from kama_claude.core.semantic.components.index import (
    FileStat,
    SemanticIndex,
    changed_paths,
)
from kama_claude.core.semantic.errors import IndexCorruptedError


def _sample_records() -> list:
    src = (
        "import secrets\n"
        "\n"
        "def login(user):\n"
        "    return user.token\n"
        "\n"
        "class Session:\n"
        "    def refresh(self):\n"
        "        return None\n"
    )
    chunks = chunk_text(src, logical_path="auth.py")
    strategy = LexicalEmbeddingStrategy()
    strategy.fit([src])
    records = [dataclasses.replace(c, vector=strategy.embed(c.text)) for c in chunks]
    records.append(dataclasses.replace(records[0], vector=None))  # 无向量记录也须回环
    return records


# 功能：验证 ChunkRecord（含 hierarchy 元数据与向量）经 JSONL 完全等价回环
# 设计：构建函数/类/模块三种符号的带向量记录，write 后 read_records 与原记录相等
def test_round_trip_records_with_hierarchy_and_vectors(tmp_path) -> None:
    idx_dir = tmp_path / "idx"
    index = SemanticIndex(idx_dir)
    records = _sample_records()

    assert any(r.vector is not None for r in records)
    assert {r.symbol_type for r in records} >= {"function", "class", "module"}

    index.write(records, files={}, git_head="abc", strategy="lexical")

    assert index.read_records() == records


# 功能：验证 manifest 状态回环——版本/策略/git_head/文件指纹完整保留
# 设计：写入带 git_head 与文件指纹的索引，read_state 逐字段断言
def test_round_trip_manifest_state(tmp_path) -> None:
    idx_dir = tmp_path / "idx"
    index = SemanticIndex(idx_dir)
    files = {"a.py": FileStat(mtime_ns=123, size=456)}

    index.write([], files=files, git_head="abc123", strategy="lexical")

    state = index.read_state()
    assert state is not None
    assert state.version == 1
    assert state.strategy == "lexical"
    assert state.git_head == "abc123"
    assert state.files == files


# 功能：验证空索引——无 manifest 返回 None，无 records 返回空列表
# 设计：全新目录直接读取两种产物
def test_missing_index_returns_none_state_and_empty_records(tmp_path) -> None:
    index = SemanticIndex(tmp_path / "idx")

    assert index.read_state() is None
    assert index.read_records() == []


# 功能：验证空记录写入与空记录读取
# 设计：空记录写盘后回读为空列表（records.jsonl 仅换行）
def test_empty_records_round_trip(tmp_path) -> None:
    index = SemanticIndex(tmp_path / "idx")

    index.write([], files={}, git_head=None, strategy="lexical")

    assert index.read_records() == []
    assert (tmp_path / "idx" / "records.jsonl").exists()
    assert (tmp_path / "idx" / "manifest.json").exists()


# 功能：验证原子写——os.replace 失败时目标文件不残留、tmp 被清理
# 设计：fault injection——monkeypatch os.replace 抛 PermissionError，断言无产物
def test_atomic_write_no_residue_on_replace_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx_dir = tmp_path / "idx"
    index = SemanticIndex(idx_dir)

    def fail_replace(src: str, dst: str) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(index_mod.os, "replace", fail_replace)

    with pytest.raises(PermissionError):
        index.write([], files={}, git_head=None, strategy="lexical")

    assert not (idx_dir / "records.jsonl").exists()
    assert not (idx_dir / "manifest.json").exists()
    assert list(idx_dir.glob("*.tmp")) == []


# 功能：验证损坏检测——manifest 非 JSON 抛 IndexCorruptedError
# 设计：写非法 JSON 到 manifest.json，read_state 应上抛语义化异常
def test_corrupted_manifest_json_raises(tmp_path) -> None:
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    (idx_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(IndexCorruptedError):
        SemanticIndex(idx_dir).read_state()


# 功能：验证损坏检测——版本不符抛 IndexCorruptedError（触发全量重建的信号）
# 设计：把已写入 manifest 的 version 改为 999，read_state 应拒绝
def test_version_mismatch_raises(tmp_path) -> None:
    idx_dir = tmp_path / "idx"
    index = SemanticIndex(idx_dir)
    index.write([], files={}, git_head=None, strategy="lexical")
    manifest_path = idx_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexCorruptedError):
        index.read_state()


# 功能：验证损坏检测——records 行损坏抛 IndexCorruptedError
# 设计：参数化两类坏行（非法 JSON / 缺字段的合法 JSON），read_records 均应拒绝
@pytest.mark.parametrize("bad_line", ["{bad json", '{"chunk_id": 1}'])
def test_corrupted_records_line_raises(tmp_path, bad_line: str) -> None:
    idx_dir = tmp_path / "idx"
    index = SemanticIndex(idx_dir)
    index.write([], files={}, git_head=None, strategy="lexical")
    (idx_dir / "records.jsonl").write_text(bad_line + "\n", encoding="utf-8")

    with pytest.raises(IndexCorruptedError):
        index.read_records()


# 功能：验证 git_head 变化判定——提供器返回不同 HEAD 时需全量重建
# 设计：同 HEAD 不触发；提供器返回新 HEAD 触发；无提供器与提供器返回 None 均跳过
def test_git_head_change_triggers_rebuild(tmp_path) -> None:
    index = SemanticIndex(tmp_path / "idx", git_head_provider=lambda _p: "abc123")
    index.write([], files={}, git_head="abc123", strategy="lexical")
    state = index.read_state()
    assert state is not None

    assert index.git_head_changed(tmp_path, state) is False

    moved = SemanticIndex(tmp_path / "idx", git_head_provider=lambda _p: "def456")
    moved_state = moved.read_state()
    assert moved_state is not None
    assert moved.git_head_changed(tmp_path, moved_state) is True

    no_provider = SemanticIndex(tmp_path / "idx")
    no_state = no_provider.read_state()
    assert no_state is not None
    assert no_provider.git_head_changed(tmp_path, no_state) is False

    null_provider = SemanticIndex(tmp_path / "idx", git_head_provider=lambda _p: None)
    null_state = null_provider.read_state()
    assert null_state is not None
    assert null_provider.git_head_changed(tmp_path, null_state) is False


# 功能：验证指纹 diff——修改/新增/删除三类变化精确识别
# 设计：mtime_ns 变化 → changed；新文件 → changed；known 独有 → deleted；未变 → 忽略
def test_changed_paths_detects_modify_add_delete() -> None:
    current = {
        "a.py": FileStat(mtime_ns=10, size=100),
        "b.py": FileStat(mtime_ns=20, size=200),
        "c.py": FileStat(mtime_ns=30, size=300),
    }
    known = {
        "a.py": FileStat(mtime_ns=10, size=100),
        "b.py": FileStat(mtime_ns=15, size=200),
        "d.py": FileStat(mtime_ns=40, size=400),
    }

    changed, deleted = changed_paths(current, known)

    assert changed == {"b.py", "c.py"}
    assert deleted == {"d.py"}


# 功能：验证无变化的指纹 diff 返回空集
# 设计：完全相同（含 size 相同 mtime 相同的极端）不产生任何变化
def test_changed_paths_identical_is_empty() -> None:
    files = {"a.py": FileStat(mtime_ns=10, size=100)}

    changed, deleted = changed_paths(files, dict(files))

    assert changed == set()
    assert deleted == set()
