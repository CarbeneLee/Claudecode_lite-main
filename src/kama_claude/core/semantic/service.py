"""检索服务门面（Facade）：全量/增量索引构建、search 查询、并发串行化与取消

状态机：refresh 每次全目录 stat 指纹 diff —— 变化文件重 chunk+embed、删除清 chunk；
manifest 损坏或 git HEAD 变化 → 全量重建。所有磁盘/CPU 工作在 to_thread 中执行，
取消时通过 stop_event 协作停止。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from kama_claude.core.semantic.components.chunker import ChunkRecord, chunk_text
from kama_claude.core.semantic.components.embedding import (
    EmbeddingStrategy,
    LexicalEmbeddingStrategy,
    cosine_similarity,
    create_embedding_strategy,
)
from kama_claude.core.semantic.components.index import (
    FileStat,
    GitHeadProvider,
    IndexState,
    SemanticIndex,
    changed_paths,
)
from kama_claude.core.semantic.config import SemanticConfig
from kama_claude.core.semantic.errors import (
    EmbeddingStrategyUnavailableError,
    IndexCorruptedError,
    IndexUnavailableError,
)
from kama_claude.core.tools.builtin.search_code import (
    _IGNORED_DIRECTORIES,
    BINARY_PROBE_BYTES,
)


@dataclass(frozen=True)
class IndexStats:
    """一次 refresh 的构建统计（每次 refresh 重置）"""

    indexed_files: int = 0
    total_bytes: int = 0
    skipped_non_text: int = 0
    skipped_unreadable: int = 0
    skipped_large: int = 0


@dataclass(frozen=True)
class SearchResult:
    """检索命中：chunk 记录 + 相似度分数"""

    record: ChunkRecord
    score: float


class _Stopped(Exception):
    """协作取消信号：stop_event 置位后扫描/构建循环抛此异常退出"""


class SemanticRetrievalService:
    """代码检索门面：ensure_ready/refresh/search

    构造注入 SemanticConfig 与 git_head_provider（非 git workspace 可传 None）；
    索引目录 = <config.index_dir>/<workspace 根 sha256[:12]>，多 workspace 隔离。
    """

    def __init__(
        self,
        config: SemanticConfig,
        workspace_root: Path,
        *,
        git_head_provider: GitHeadProvider | None = None,
    ) -> None:
        self._config = config
        self._root = workspace_root.resolve()
        self._git_head_provider = git_head_provider
        self._index = SemanticIndex(
            self._index_dir(), git_head_provider=git_head_provider
        )
        self._lock = asyncio.Lock()
        # (IndexState, EmbeddingStrategy, records) 三元组；None 表示从未构建
        self._state: tuple[IndexState, EmbeddingStrategy, list[ChunkRecord]] | None = None
        self._strategy_cache: EmbeddingStrategy | None = None
        self._strategy_fitted = False
        self._stats = IndexStats()

    def _index_dir(self) -> Path:
        digest = hashlib.sha256(str(self._root).encode("utf-8")).hexdigest()[:12]
        return Path(self._config.index_dir).expanduser() / digest

    def stats(self) -> IndexStats:
        return self._stats

    async def ensure_ready(self) -> None:
        await self.refresh()

    async def refresh(self) -> None:
        """构建/增量更新索引（串行化 + to_thread 取消）；失败抛 IndexUnavailableError"""
        async with self._lock:
            stop_event = threading.Event()
            try:
                await asyncio.shield(asyncio.to_thread(self._refresh_sync, stop_event))
            except asyncio.CancelledError:
                stop_event.set()
                raise

    async def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        """查询索引（自动先 refresh）；无匹配/退化查询返回空列表"""
        await self.refresh()
        if len(query) > self._config.max_query_chars:
            query = query[: self._config.max_query_chars]
        assert self._state is not None
        strategy = self._state[1]
        if strategy.degraded_query(query):
            return []
        query_vector = strategy.embed(query)
        if not query_vector.indices:
            return []
        threshold = self._config.similarity_threshold
        limit = top_k if top_k is not None else self._config.default_top_k
        scored = [
            SearchResult(record, cosine_similarity(query_vector, record.vector))
            for record in self._state[2]
            if record.vector is not None and record.vector.indices
        ]
        scored = [r for r in scored if r.score >= threshold]
        scored.sort(key=lambda r: (-r.score, r.record.chunk_id))
        return scored[:limit]

    # ---- 同步刷新状态机（运行在 to_thread 线程中） ----

    def _refresh_sync(self, stop_event: threading.Event) -> None:
        self._stats = IndexStats()
        try:
            self._do_refresh(stop_event)
        except OSError as exc:
            raise IndexUnavailableError(f"semantic index refresh failed: {exc}") from exc

    def _do_refresh(self, stop_event: threading.Event) -> None:
        try:
            state = self._index.read_state()
        except IndexCorruptedError:
            state = None  # manifest 损坏 → 全量重建
        files = self._scan_workspace(stop_event)
        if state is None or self._index.git_head_changed(self._root, state):
            self._rebuild_all(files, stop_event)
            return
        changed, deleted = changed_paths(files, state.files)
        if not changed and not deleted:
            self._load_ready_state(state)
            return
        try:
            old_records = (
                self._state[2]
                if self._state is not None
                else self._index.read_records()
            )
        except IndexCorruptedError:
            self._rebuild_all(files, stop_event)
            return
        self._rebuild_incremental(files, changed, deleted, old_records, stop_event)

    def _load_ready_state(self, state: IndexState) -> None:
        """无变化的刷新：热态直接复用；冷启动从磁盘载入记录并 fit 策略"""
        if self._state is not None and self._state[0] == state:
            return
        try:
            records = self._index.read_records()
        except IndexCorruptedError:
            # 仅 records 损坏 → 全量重建
            stop_event = threading.Event()
            self._rebuild_all(self._scan_workspace(stop_event), stop_event)
            return
        strategy = self._strategy
        if not self._strategy_fitted and records:
            strategy.fit(r.text for r in records)
            self._strategy_fitted = True
        self._state = (state, strategy, records)

    def _rebuild_all(
        self, files: dict[str, FileStat], stop_event: threading.Event
    ) -> None:
        records = self._build_full(files, stop_event)
        self._finish_build(records, files)

    def _rebuild_incremental(
        self,
        files: dict[str, FileStat],
        changed: set[str],
        deleted: set[str],
        old_records: list[ChunkRecord],
        stop_event: threading.Event,
    ) -> None:
        strategy = self._strategy
        kept = [
            r
            for r in old_records
            if r.logical_path not in changed and r.logical_path not in deleted
        ]
        new_records: list[ChunkRecord] = []
        texts: list[str] = []
        for rel in sorted(changed):
            self._check_stop(stop_event)
            loaded = self._load_file(rel, stop_event)
            if loaded is None:
                continue
            chunks, text = loaded
            new_records.extend(chunks)
            texts.append(text)
        records = kept + new_records
        if texts:
            # IDF 随语料漂移：对全量记录文本重 fit 后统一重嵌，保证一致性
            strategy.fit(r.text for r in records)
            self._strategy_fitted = True
            records = [replace(r, vector=strategy.embed(r.text)) for r in records]
        self._stats = replace(self._stats, indexed_files=len(new_records))
        self._finish_build(records, files)

    def _finish_build(
        self, records: list[ChunkRecord], files: dict[str, FileStat]
    ) -> None:
        git_head = (
            self._git_head_provider(self._root)
            if self._git_head_provider is not None
            else None
        )
        self._index.write(
            records,
            files=files,
            git_head=git_head,
            strategy=self._config.strategy,
        )
        state = self._index.read_state()
        assert state is not None
        self._state = (state, self._strategy, records)

    def _build_full(
        self, files: dict[str, FileStat], stop_event: threading.Event
    ) -> list[ChunkRecord]:
        strategy = self._strategy
        records: list[ChunkRecord] = []
        texts: list[str] = []
        indexed = 0
        total_bytes = 0
        for rel in sorted(files):
            self._check_stop(stop_event)
            loaded = self._load_file(rel, stop_event)
            if loaded is None:
                continue
            chunks, text = loaded
            records.extend(chunks)
            texts.append(text)
            indexed += 1
            total_bytes += len(text.encode("utf-8"))
        strategy.fit(texts)
        self._strategy_fitted = True
        self._stats = replace(
            self._stats, indexed_files=indexed, total_bytes=total_bytes
        )
        if not records:
            return []
        return [replace(r, vector=strategy.embed(r.text)) for r in records]

    # ---- 文件访问与扫描 ----

    def _scan_workspace(self, stop_event: threading.Event) -> dict[str, FileStat]:
        """全目录 stat 指纹（忽略隐藏文件/目录与忽略目录）；根不可读抛 OSError

        os.walk 对顶层 scandir 错误静默跳过，须显式探测根目录可读性
        """
        with os.scandir(self._root):
            pass
        files: dict[str, FileStat] = {}
        for dirpath, dirnames, filenames in os.walk(self._root):
            self._check_stop(stop_event)
            dirnames[:] = [
                d
                for d in dirnames
                if d not in _IGNORED_DIRECTORIES and not d.startswith(".")
            ]
            for name in filenames:
                if name.startswith("."):
                    continue
                path = Path(dirpath) / name
                try:
                    st = path.stat()
                except OSError:
                    continue  # 目录/文件竞赛：本次扫描跳过
                if not stat.S_ISREG(st.st_mode):
                    continue
                rel = path.relative_to(self._root).as_posix()
                files[rel] = FileStat(mtime_ns=st.st_mtime_ns, size=st.st_size)
        return files

    def _load_file(
        self, rel: str, stop_event: threading.Event
    ) -> tuple[list[ChunkRecord], str] | None:
        """读取并分块单个文件；跳过（计入 stats）返回 None"""
        path = self._root / rel
        try:
            size = path.stat().st_size
        except OSError:
            self._bump("skipped_unreadable")
            return None
        if size > self._config.max_file_bytes:
            self._bump("skipped_large")
            return None
        data = self._read_workspace_file(rel)
        if data is None:
            self._bump("skipped_unreadable")
            return None
        if b"\x00" in data[:BINARY_PROBE_BYTES]:
            self._bump("skipped_non_text")
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            self._bump("skipped_non_text")
            return None
        chunks = chunk_text(
            text,
            logical_path=rel,
            chunk_size=self._config.chunk_size,
            min_chunk_lines=self._config.min_chunk_lines,
        )
        return chunks, text

    def _read_workspace_file(self, rel: str) -> bytes | None:
        try:
            return (self._root / rel).read_bytes()
        except OSError:
            return None

    # ---- 策略与协作取消 ----

    @property
    def _strategy(self) -> EmbeddingStrategy:
        if self._strategy_cache is None:
            try:
                self._strategy_cache = create_embedding_strategy(
                    self._config.strategy, ngram_n=self._config.ngram_n
                )
            except EmbeddingStrategyUnavailableError:
                # fail-open：onnx 后端缺失时降级回词法策略
                self._strategy_cache = LexicalEmbeddingStrategy(
                    ngram_n=self._config.ngram_n
                )
        return self._strategy_cache

    def _bump(self, field: str) -> None:
        self._stats = replace(self._stats, **{field: getattr(self._stats, field) + 1})

    @staticmethod
    def _check_stop(stop_event: threading.Event) -> None:
        if stop_event.is_set():
            raise _Stopped()
