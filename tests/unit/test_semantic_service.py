from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time

import pytest

from kama_claude.core.semantic.components import embedding as embedding_mod
from kama_claude.core.semantic.components import index as index_mod
from kama_claude.core.semantic.config import SemanticConfig
from kama_claude.core.semantic.errors import IndexUnavailableError
from kama_claude.core.semantic.service import SemanticRetrievalService

AUTH_PY = (
    '"""auth module"""\n'
    "import os\n"
    "\n"
    "def reset_password(user):\n"
    "    token = user.token\n"
    "    return token\n"
    "\n"
    "class Session:\n"
    "    def refresh(self):\n"
    "        return None\n"
)

NOTES_MD = "# Project notes\n\nKeep tokens fresh.\n"


def _write_workspace(tmp_path, files: dict[str, str]):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return ws


def _make_service(
    tmp_path,
    *,
    files: dict[str, str] | None = None,
    provider=None,
    **config_overrides,
) -> SemanticRetrievalService:
    ws = _write_workspace(tmp_path, files or {})
    config = SemanticConfig(index_dir=str(tmp_path / "idx"), **config_overrides)
    return SemanticRetrievalService(
        config=config, workspace_root=ws, git_head_provider=provider
    )


# 功能：验证首建——全量扫描、索引落盘、search 命中符号 chunk
# 设计：两文件 workspace 首建后断言统计与 top-1 命中（score=1.0 的完全匹配）
async def test_first_build_indexes_workspace(tmp_path) -> None:
    service = _make_service(
        tmp_path, files={"auth.py": AUTH_PY, "notes.md": NOTES_MD}
    )

    await service.ensure_ready()

    assert service.stats().indexed_files == 2
    results = await service.search("reset_password")
    assert results
    assert results[0].record.symbol_name == "reset_password"
    assert results[0].score > 0.5
    assert all(results[0].score > r.score for r in results[1:])
    assert results[0].record.language == "python"


# 功能：验证增量——改动文件重索引、未变文件零重读（monkeypatch 证明）
# 设计：追加新函数后 ensure_ready，断言新符号可搜且未变文件未被读取
async def test_incremental_update_only_reads_changed_files(tmp_path) -> None:
    service = _make_service(
        tmp_path,
        files={"auth.py": AUTH_PY, "notes.md": NOTES_MD},
    )
    await service.ensure_ready()

    auth_path = tmp_path / "ws" / "auth.py"
    auth_path.write_text(AUTH_PY + "\ndef logout(user):\n    return None\n", encoding="utf-8")
    read_rels: list[str] = []
    original_read = service._read_workspace_file

    def recording_read(rel: str) -> bytes | None:
        read_rels.append(rel)
        return original_read(rel)

    service._read_workspace_file = recording_read  # type: ignore[method-assign]
    await service.ensure_ready()

    assert "notes.md" not in read_rels
    assert read_rels == ["auth.py"]
    results = await service.search("logout")
    assert results and results[0].record.symbol_name == "logout"


# 功能：验证无变化刷新零文件读取（仅 stat 扫描）
# 设计：首建后再 ensure_ready，断言 _read_workspace_file 完全未被调用
async def test_no_change_refresh_reads_no_files(tmp_path) -> None:
    service = _make_service(tmp_path, files={"auth.py": AUTH_PY})
    await service.ensure_ready()

    read_rels: list[str] = []
    service._read_workspace_file = lambda rel: (  # type: ignore[method-assign]
        read_rels.append(rel) or b""
    )
    await service.ensure_ready()

    assert read_rels == []


# 功能：验证删文件——chunks 随指纹 diff 清除
# 设计：删除文件后 search 不再命中其内容
async def test_deleted_file_chunks_removed(tmp_path) -> None:
    service = _make_service(
        tmp_path, files={"auth.py": AUTH_PY, "notes.md": NOTES_MD}
    )
    await service.ensure_ready()

    (tmp_path / "ws" / "auth.py").unlink()
    await service.ensure_ready()

    assert await service.search("reset_password") == []
    assert await service.search("notes") != []


# 功能：验证二进制文件跳过（不崩、计数）
# 设计：含 \x00 的文件不产生 chunk
async def test_binary_files_skipped(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    (tmp_path / "ws" / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    (tmp_path / "ws" / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00")
    service = _make_service(tmp_path)

    await service.ensure_ready()

    assert service.stats().skipped_non_text == 1
    assert service.stats().indexed_files == 1


# 功能：验证不可读文件跳过（fault injection：chmod 000）
# 设计：权限拒绝不中断构建，计入 skipped_unreadable
async def test_unreadable_file_skipped(tmp_path) -> None:
    ws = _write_workspace(tmp_path, {"auth.py": AUTH_PY, "secret.py": "def s():\n    pass\n"})
    secret = ws / "secret.py"
    secret.chmod(0)
    service = _make_service(tmp_path)

    await service.ensure_ready()

    assert service.stats().skipped_unreadable == 1
    assert service.stats().indexed_files == 1
    secret.chmod(0o644)


# 功能：验证超大文件跳过（配置字节上限生效）
# 设计：max_file_bytes 调小后超限文件计入 skipped_large
async def test_oversized_file_skipped(tmp_path) -> None:
    _write_workspace(tmp_path, {"auth.py": AUTH_PY, "huge.py": "x = 1\n" * 100})
    service = _make_service(tmp_path, max_file_bytes=200)

    await service.ensure_ready()

    assert service.stats().skipped_large == 1
    assert service.stats().indexed_files == 1
    assert await service.search("reset_password") != []


# 功能：验证空 workspace——可空构建；文件出现后增量索引
# 设计：空目录 ensure_ready 不崩、search 为空；写入文件后 search 命中
async def test_empty_workspace_then_add_file(tmp_path) -> None:
    service = _make_service(tmp_path)

    await service.ensure_ready()
    assert await service.search("anything") == []

    (tmp_path / "ws" / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    await service.ensure_ready()

    assert await service.search("reset_password") != []


# 功能：验证并发 ensure 串行化——仅一次全量构建、无重叠执行
# 设计：monkeypatch _build_full 加慢速包装（计数+活跃度），5 个并发 ensure
async def test_concurrent_ensure_serialized(tmp_path, monkeypatch) -> None:
    service = _make_service(tmp_path, files={"auth.py": AUTH_PY})
    real_build = service._build_full
    state = {"calls": 0, "active": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_build(files, stop_event):
        with lock:
            state["calls"] += 1
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return real_build(files, stop_event)

    monkeypatch.setattr(service, "_build_full", slow_build)
    await asyncio.gather(*(service.ensure_ready() for _ in range(5)))

    assert state["calls"] == 1
    assert state["max_active"] == 1


# 功能：验证 workspace 根扫描失败 → IndexUnavailableError（工具层据此降级）
# 设计：fault injection——根目录 chmod 000，ensure_ready 上抛语义化异常
async def test_scan_root_failure_raises_index_unavailable(tmp_path) -> None:
    ws = _write_workspace(tmp_path, {"auth.py": AUTH_PY})
    service = _make_service(tmp_path)

    ws.chmod(0)
    try:
        with pytest.raises(IndexUnavailableError):
            await service.ensure_ready()
    finally:
        ws.chmod(0o755)


# 功能：验证 mtime 回拨仍被指纹 diff 捕获
# 设计：同尺寸新内容（firs→seco 等长、无共享 gram）+ utime 回拨到更早时间，
#       刷新后命中新内容且旧内容无残留
async def test_mtime_rollback_detected(tmp_path) -> None:
    ws = _write_workspace(tmp_path, {"a.py": "def firs():\n    x = 1\n"})
    service = _make_service(tmp_path)
    await service.ensure_ready()
    assert await service.search("firs") != []

    a_path = ws / "a.py"
    rolled_back_ns = os.stat(a_path).st_mtime_ns - 1_000_000_000
    a_path.write_text("def seco():\n    x = 2\n", encoding="utf-8")
    os.utime(a_path, ns=(rolled_back_ns, rolled_back_ns))
    await service.ensure_ready()

    assert await service.search("seco") != []
    assert await service.search("firs") == []


# 功能：验证 git_head 变化触发全量重建（新实例 + 新提供器）
# 设计：monkeypatch 计数 _build_full；HEAD 变化后 ensure_ready 应走全量路径
async def test_git_head_change_triggers_full_rebuild(tmp_path, monkeypatch) -> None:
    _write_workspace(tmp_path, {"a.py": "def one():\n    pass\n"})
    service_a = _make_service(tmp_path, provider=lambda _p: "head-a")
    await service_a.ensure_ready()

    service_b = _make_service(
        tmp_path,
        files={"a.py": "def one():\n    pass\n", "b.py": "def two():\n    pass\n"},
        provider=lambda _p: "head-b",
    )
    build_count = {"calls": 0}
    real_build = service_b._build_full

    def counting_build(files, stop_event):
        build_count["calls"] += 1
        return real_build(files, stop_event)

    monkeypatch.setattr(service_b, "_build_full", counting_build)
    await service_b.ensure_ready()

    assert build_count["calls"] == 1
    assert await service_b.search("two") != []


# 功能：验证索引写失败 → IndexUnavailableError（save 抛 PermissionError 的 fault injection）
# 设计：monkeypatch 原子写抛 PermissionError，ensure_ready 上抛语义化异常
async def test_write_failure_raises_index_unavailable(tmp_path, monkeypatch) -> None:
    service = _make_service(tmp_path, files={"a.py": "def one():\n    pass\n"})

    def fail_write(path, content: str) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(index_mod, "_atomic_write", fail_write)

    with pytest.raises(IndexUnavailableError):
        await service.ensure_ready()


# 功能：验证 onnx 策略加载失败 fail-open 降级回 lexical
# 设计：monkeypatch onnx 导入抛 ImportError，构建成功且策略标记为 lexical
async def test_onnx_strategy_falls_back_to_lexical(tmp_path, monkeypatch) -> None:
    def fail_import() -> None:
        raise ImportError("no onnxruntime")

    monkeypatch.setattr(embedding_mod, "_import_onnxruntime", fail_import)
    service = _make_service(
        tmp_path, files={"a.py": "def one():\n    pass\n"}, strategy="onnx"
    )

    await service.ensure_ready()

    assert service._state is not None
    assert service._state[1].is_lexical is True
    assert await service.search("one") != []


# 功能：验证 search 的 top_k 与阈值过滤、按分数降序
# 设计：多函数文件查询共享词，top_k=1 取首个；高阈值过滤部分命中
async def test_search_respects_top_k_and_threshold(tmp_path) -> None:
    multi = "\n\n".join(
        f"def fn{i}():\n    token = {i}\n" for i in range(1, 5)
    )
    service = _make_service(tmp_path, files={"multi.py": multi})
    await service.ensure_ready()

    limited = await service.search("token", top_k=2)
    assert len(limited) == 2
    assert limited[0].score == pytest.approx(limited[1].score)
    assert limited[0].score > 0.4

    partial = await service.search("token refresh")
    assert all(r.score < 1.0 for r in partial)

    strict = dataclasses.replace(service._config, similarity_threshold=0.9)
    service._config = strict
    assert await service.search("token refresh") == []


# 功能：验证无 gram 查询返回空（退化查询不产生假命中）
# 设计：标点查询与空串查询均返回空列表
async def test_search_gramless_query_returns_empty(tmp_path) -> None:
    service = _make_service(tmp_path, files={"a.py": "def one():\n    pass\n"})

    assert await service.search("!!!") == []
    assert await service.search("") == []


# 功能：验证 search 隐式 ensure（无需显式调用 ensure_ready）
# 设计：直接 search 完成首建并命中
async def test_search_implicitly_ensures_ready(tmp_path) -> None:
    service = _make_service(tmp_path, files={"a.py": "def one():\n    pass\n"})

    assert await service.search("one") != []


# 功能：验证索引目录按 workspace hash 隔离
# 设计：不同 workspace 根 → 不同子目录；同一根 → 同一目录（复用索引）
def test_index_dir_is_hashed_per_workspace(tmp_path) -> None:
    cfg = SemanticConfig(index_dir=str(tmp_path / "idx"))
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    service_a = SemanticRetrievalService(config=cfg, workspace_root=ws_a)
    service_b = SemanticRetrievalService(config=cfg, workspace_root=ws_b)

    assert service_a._index._dir != service_b._index._dir
    assert len(service_a._index._dir.name) == 12
    assert (
        SemanticRetrievalService(config=cfg, workspace_root=ws_a)._index._dir
        == service_a._index._dir
    )
