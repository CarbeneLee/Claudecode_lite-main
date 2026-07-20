from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import pytest

import kama_claude.core.tools.builtin.search_code as search_code_module
from kama_claude.core.tools.builtin.search_code import (
    MAX_OUTPUT_BYTES,
    MAX_SNIPPET_CHARS,
    PER_DIRECTORY_ENTRY_LIMIT,
    PER_FILE_BYTE_LIMIT,
    SNIPPET_TARGET_CHARS,
)
from tests.unit._search_code_test_support import _footer, _records, _tool


class _OverflowEntry:
    name_reads = 0

    # 仅暴露有界枚举阶段允许读取的 entry name
    def __init__(self, index: int) -> None:
        self._name = f"entry-{index:05d}.txt"

    @property
    # 记录溢出 batch 是否被错误读取 name 进行排序或过滤
    def name(self) -> str:
        type(self).name_reads += 1
        return self._name


class _CountedScandir:
    # 构造可记录消费与关闭状态的 scandir iterator
    def __init__(self, count: int) -> None:
        self._count = count
        self.yielded = 0
        self.closed = False

    # 返回 scandir context manager 本身
    def __enter__(self) -> _CountedScandir:
        return self

    # 离开 context manager 时记录关闭
    def __exit__(self, *_args: object) -> None:
        self.close()

    # 返回用于 next 消费的 iterator
    def __iter__(self) -> _CountedScandir:
        return self

    # 每次仅产生一个 entry 并精确记录消费数
    def __next__(self) -> _OverflowEntry:
        if self.yielded >= self._count:
            raise StopIteration
        self.yielded += 1
        return _OverflowEntry(self.yielded)

    # 记录搜索是否主动关闭了有界 iterator
    def close(self) -> None:
        self.closed = True


class _GrowingFile:
    # 构造 stat 后增长的可控二进制文件
    def __init__(self, data: bytes, handle: BinaryIO) -> None:
        self._data = data
        self._handle = handle
        self._offset = 0

    # 返回二进制文件 context manager 本身
    def __enter__(self) -> _GrowingFile:
        return self

    # 离开 context manager 时关闭底层真实文件描述符
    def __exit__(self, *_args: object) -> None:
        self._handle.close()

    # 暴露真实已打开描述符供 final fstat 验证
    def fileno(self) -> int:
        return self._handle.fileno()

    # 按请求大小返回增长后的实际字节
    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _ShortReadFile:
    # 构造会连续返回短块但尚未到 EOF 的受控 BinaryIO
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._chunks = iter((b"ab", b"cd", b""))
        self.read_calls = 0

    # 返回受控 BinaryIO context manager 本身
    def __enter__(self) -> _ShortReadFile:
        return self

    # 离开 context manager 时关闭底层真实文件描述符
    def __exit__(self, *_args: object) -> None:
        self._handle.close()

    # 暴露与初始/final fstat 相同的真实文件描述符
    def fileno(self) -> int:
        return self._handle.fileno()

    # 按固定序列返回两个 short read 后再返回 EOF
    def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        return next(self._chunks)


# 功能：验证第 5,001 个 directory entry 立即停止且丢弃整个 batch
# 设计：可计数 fake iterator 额外提供第 5,002 项，断言它未消费、iterator 已关闭且候选未过滤
async def test_scandir_stops_at_5001_and_discards_overflowing_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _CountedScandir(PER_DIRECTORY_ENTRY_LIMIT + 2)

    # 返回具备显式 fd-capability 标记的受控 scandir iterator
    def fake_scandir(_path: object) -> _CountedScandir:
        return iterator

    monkeypatch.setattr(search_code_module.os, "scandir", fake_scandir)
    monkeypatch.setattr(
        search_code_module.os,
        "supports_fd",
        frozenset((*os.supports_fd, fake_scandir)),
    )
    _OverflowEntry.name_reads = 0
    tool = _tool(tmp_path)
    real_resolve = tool._resolver.resolve_existing
    resolved_paths: list[str] = []

    # 只允许显式 root 进入 resolver，任何 overflow candidate 调用都立即失败
    def guarded_resolve(path: str) -> Path:
        resolved_paths.append(path)
        if path != ".":
            raise AssertionError("overflowing batch must not resolve candidates")
        return real_resolve(path)

    monkeypatch.setattr(tool._resolver, "resolve_existing", guarded_resolve)

    result = await tool.invoke({"query": "needle"})

    assert iterator.yielded == PER_DIRECTORY_ENTRY_LIMIT + 1
    assert iterator.closed
    assert _OverflowEntry.name_reads == 0
    assert resolved_paths == ["."]
    assert _records(result.content) == []
    assert _footer(result.content) == {
        "matched_lines": 0,
        "directory_entries": PER_DIRECTORY_ENTRY_LIMIT + 1,
        "visited_directories": 1,
        "examined_files": 0,
        "examined_bytes": 0,
        "skipped_non_text": 0,
        "skipped_large": 0,
        "skipped_unreadable": 0,
        "truncated": "entry_limit",
    }


# 功能：验证全局 entry limit 在后续目录中触发时丢弃当前 batch
# 设计：缩小全局上限，让首个目录完整搜索、第二个目录的首 entry 成为 sentinel
async def test_total_entry_limit_preserves_prior_batches_and_discards_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "hit.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "z.txt").write_text("needle", encoding="utf-8")
    monkeypatch.setattr(search_code_module, "TOTAL_DIRECTORY_ENTRY_LIMIT", 4)

    result = await _tool(tmp_path).invoke({"query": "needle"})

    assert _records(result.content) == ["a/hit.txt:1: needle"]
    footer = _footer(result.content)
    assert footer["directory_entries"] == 5
    assert footer["matched_lines"] == 1
    assert footer["truncated"] == "entry_limit"


@pytest.mark.parametrize("resource", ["directory", "file"])
# 功能：验证 directory 与 file limit 都使用 limit+1 sentinel 并立即停止
# 设计：分别将上限缩到一，观测 sentinel counter 为二且后续候选不被搜索
async def test_directory_and_file_limits_count_the_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
) -> None:
    if resource == "directory":
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "hit.txt").write_text("needle", encoding="utf-8")
        (tmp_path / "z.txt").write_text("needle", encoding="utf-8")
        monkeypatch.setattr(search_code_module, "VISITED_DIRECTORY_LIMIT", 1)
    else:
        (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
        (tmp_path / "z.txt").write_text("needle", encoding="utf-8")
        monkeypatch.setattr(search_code_module, "EXAMINED_FILE_LIMIT", 1)

    footer = _footer((await _tool(tmp_path).invoke({"query": "needle"})).content)

    counter = "visited_directories" if resource == "directory" else "examined_files"
    reason = "directory_limit" if resource == "directory" else "file_limit"
    assert footer[counter] == 2
    assert footer["truncated"] == reason


# 功能：验证 entry 早于 ignore 且 regular file 早于 glob 计数
# 设计：同时放入 ignored directory、glob miss 和 glob hit，计数将过滤顺序变成可观测契约
async def test_counters_increment_before_ignore_and_glob(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.py").write_text("needle", encoding="utf-8")
    (tmp_path / "miss.md").write_text("needle", encoding="utf-8")
    (tmp_path / "hit.py").write_text("needle", encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "needle", "include_glob": "*.py"})

    assert _records(result.content) == ["hit.py:1: needle"]
    footer = _footer(result.content)
    assert footer["directory_entries"] == 3
    assert footer["visited_directories"] == 1
    assert footer["examined_files"] == 2
    assert footer["examined_bytes"] == len(b"needle")


# 功能：验证 short read 不等于 EOF，只有空 bytes 才结束完整文件读取
# 设计：受控 BinaryIO 依次返回 ab/cd/EOF，断言不会搜索首个部分块且精确读到第三次
async def test_short_read_continues_until_empty_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "short.txt"
    target.write_bytes(b"abcd")
    tool = _tool(tmp_path)
    real_open = tool._open_regular_file
    controlled: _ShortReadFile | None = None

    # 只替换目标文件的 read 行为并保留同一真实 fd
    def short_open(path: Path) -> tuple[BinaryIO, int]:
        nonlocal controlled
        handle, initial_size = real_open(path)
        controlled = _ShortReadFile(handle)
        return controlled, initial_size

    monkeypatch.setattr(tool, "_open_regular_file", short_open)
    result = await tool.invoke({"query": "abcd", "path": "short.txt", "case_sensitive": True})

    assert controlled is not None
    assert controlled.read_calls == 3
    assert _records(result.content) == ["short.txt:1: abcd"]
    assert _footer(result.content)["examined_bytes"] == 4


# 功能：验证最后一个完整文件恰好填满 total budget 时保留结果且不截断
# 设计：缩小 total=4 并使用真实同一 fd 的 tell/fstat 边界，旧无条件 >= 分支必然丢弃结果
async def test_exact_total_budget_keeps_complete_last_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "only.txt").write_bytes(b"abcd")
    monkeypatch.setattr(search_code_module, "PER_FILE_BYTE_LIMIT", 8)
    monkeypatch.setattr(search_code_module, "TOTAL_BYTE_LIMIT", 4)

    result = await _tool(tmp_path).invoke(
        {"query": "abcd", "path": "only.txt", "case_sensitive": True}
    )

    assert _records(result.content) == ["only.txt:1: abcd"]
    footer = _footer(result.content)
    assert footer["examined_bytes"] == 4
    assert footer["truncated"] == "none"


# 功能：验证完整文件填满 total budget 后在下一候选处触发 byte_limit
# 设计：按名称排序让 a.txt 完整命中、b.txt 需要下一次 read，断言不回滚已完成记录
async def test_exact_total_budget_stops_at_next_candidate_without_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"hit!")
    (tmp_path / "b.txt").write_bytes(b"hit")
    monkeypatch.setattr(search_code_module, "PER_FILE_BYTE_LIMIT", 8)
    monkeypatch.setattr(search_code_module, "TOTAL_BYTE_LIMIT", 4)

    result = await _tool(tmp_path).invoke({"query": "hit", "case_sensitive": True})

    assert _records(result.content) == ["a.txt:1: hit!"]
    footer = _footer(result.content)
    assert footer["examined_bytes"] == 4
    assert footer["truncated"] == "byte_limit"


# 功能：验证 total budget 处当前 fd 仍有尾部时丢弃整个文件
# 设计：5-byte 真实文件只允许读取 4 bytes，以同一 fd tell/fstat 证明仍未完整且不超读 payload
async def test_total_budget_discards_incomplete_current_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "partial.txt").write_bytes(b"abcdx")
    monkeypatch.setattr(search_code_module, "PER_FILE_BYTE_LIMIT", 8)
    monkeypatch.setattr(search_code_module, "TOTAL_BYTE_LIMIT", 4)

    result = await _tool(tmp_path).invoke(
        {"query": "abcd", "path": "partial.txt", "case_sensitive": True}
    )

    assert _records(result.content) == []
    footer = _footer(result.content)
    assert footer["examined_bytes"] == 4
    assert footer["truncated"] == "byte_limit"


# 功能：验证 binary、invalid UTF-8 与 unreadable 文件整份跳过且精确计数
# 设计：同时放置三类候选，用 open stub 稳定制造 PermissionError 而不依赖宿主权限
async def test_non_text_and_unreadable_files_are_never_partially_searched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = b"needle\x00"
    invalid = b"\xffneedle"
    (tmp_path / "binary.bin").write_bytes(binary)
    (tmp_path / "invalid.txt").write_bytes(invalid)
    unreadable = tmp_path / "unreadable.txt"
    unreadable.write_bytes(b"needle")
    tool = _tool(tmp_path)
    real_open = tool._open_regular_file

    # 只让指定候选的安全 fd open 边界稳定失败
    def fake_open(path: Path) -> tuple[BinaryIO, int]:
        if path == unreadable.resolve(strict=True):
            raise PermissionError("denied")
        return real_open(path)

    monkeypatch.setattr(tool, "_open_regular_file", fake_open)
    result = await tool.invoke({"query": "needle"})

    assert _records(result.content) == []
    footer = _footer(result.content)
    assert footer["examined_files"] == 3
    assert footer["examined_bytes"] == len(binary) + len(invalid)
    assert footer["skipped_non_text"] == 2
    assert footer["skipped_unreadable"] == 1


# 功能：验证恰好 1 MiB 允许，而实际读到第 1 MiB+1 字节会丢弃整个文件
# 设计：用精确边界文件和 stat 过期后增长 fake，防止实现仅依赖预检 stat
async def test_per_file_limit_uses_actual_limit_plus_one_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = tmp_path / "exact.txt"
    exact.write_bytes(b"x" * (PER_FILE_BYTE_LIMIT - len(b"needle")) + b"needle")
    growing = tmp_path / "growing.txt"
    growing.write_bytes(b"small")
    tool = _tool(tmp_path)
    real_open = tool._open_regular_file
    grown = b"needle" + b"x" * (PER_FILE_BYTE_LIMIT + 1 - len(b"needle"))

    # 让 growing.txt 保持已打开 fd 的过期小 stat 却返回超限实际字节
    def fake_open(path: Path) -> tuple[BinaryIO, int]:
        handle, initial_size = real_open(path)
        if path == growing.resolve(strict=True):
            return _GrowingFile(grown, handle), initial_size
        return handle, initial_size

    monkeypatch.setattr(tool, "_open_regular_file", fake_open)
    result = await tool.invoke({"query": "needle", "case_sensitive": True})

    records = _records(result.content)
    assert len(records) == 1
    assert records[0].startswith("exact.txt:1: ")
    assert "needle" in records[0]
    footer = _footer(result.content)
    assert footer["examined_bytes"] == 2 * PER_FILE_BYTE_LIMIT + 1
    assert footer["skipped_large"] == 1


# 功能：验证 total byte 在 per-file sentinel 之前耗尽时丢弃部分文件并记录 byte_limit
# 设计：将两个上限缩小到 4 字节，5 字节文件无法观测 large sentinel，锁定优先级
async def test_total_byte_exhaustion_before_large_sentinel_is_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "data.txt"
    target.write_bytes(b"hit!!")
    monkeypatch.setattr(search_code_module, "PER_FILE_BYTE_LIMIT", 4)
    monkeypatch.setattr(search_code_module, "TOTAL_BYTE_LIMIT", 4)
    tool = _tool(tmp_path)
    real_open = tool._open_regular_file

    # 使目标文件的 fd 预检 stat 不能替代实际读取边界
    def stale_size(path: Path) -> tuple[BinaryIO, int]:
        handle, initial_size = real_open(path)
        if path == target:
            return handle, 4
        return handle, initial_size

    monkeypatch.setattr(tool, "_open_regular_file", stale_size)

    result = await tool.invoke({"query": "hit", "path": "data.txt"})

    footer = _footer(result.content)
    assert _records(result.content) == []
    assert footer["examined_bytes"] == 4
    assert footer["skipped_large"] == 0
    assert footer["truncated"] == "byte_limit"


# 功能：验证 per-file sentinel 恰占 total budget 最后字节时先分类 large
# 设计：以 per-file=4、total=5 完整观测第五字节，确认数值耗尽本身不是截断
async def test_large_sentinel_on_last_total_byte_wins_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "data.txt"
    target.write_bytes(b"hit!!")
    monkeypatch.setattr(search_code_module, "PER_FILE_BYTE_LIMIT", 4)
    monkeypatch.setattr(search_code_module, "TOTAL_BYTE_LIMIT", 5)
    tool = _tool(tmp_path)
    real_open = tool._open_regular_file

    # 模拟文件在 fd 预检 stat 后增长一字节并保留其他路径的真实大小
    def stale_size(path: Path) -> tuple[BinaryIO, int]:
        handle, initial_size = real_open(path)
        if path == target:
            return handle, 4
        return handle, initial_size

    monkeypatch.setattr(tool, "_open_regular_file", stale_size)
    result = await tool.invoke({"query": "hit", "path": "data.txt"})

    footer = _footer(result.content)
    assert _records(result.content) == []
    assert footer["examined_bytes"] == 5
    assert footer["skipped_large"] == 1
    assert footer["truncated"] == "none"


@pytest.mark.parametrize(
    ("count", "returned", "matched", "truncated"),
    [(49, 49, 49, "none"), (50, 50, 50, "none"), (51, 50, 51, "max_results")],
)
# 功能：验证 max_results 只在实际观测第 N+1 个匹配时截断
# 设计：覆盖默认 49/50/51 三态，同时检查 sentinel 增加 matched 但不进入输出
async def test_default_max_results_observes_limit_plus_one_sentinel(
    tmp_path: Path,
    count: int,
    returned: int,
    matched: int,
    truncated: str,
) -> None:
    (tmp_path / "matches.txt").write_text("\n".join(["hit"] * count), encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "hit"})

    footer = _footer(result.content)
    assert len(_records(result.content)) == returned
    assert footer["matched_lines"] == matched
    assert footer["truncated"] == truncated


@pytest.mark.parametrize(
    ("text", "matched", "truncated"), [("hit", 1, "none"), ("hit\nhit", 2, "max_results")]
)
# 功能：验证 max_results=1 的 1/2 匹配边界与默认边界一致
# 设计：参数化最小可观测 sentinel，排除实现遇到第 N 条就过早截断
async def test_custom_max_results_one_boundary(
    tmp_path: Path,
    text: str,
    matched: int,
    truncated: str,
) -> None:
    (tmp_path / "matches.txt").write_text(text, encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "hit", "max_results": 1})

    footer = _footer(result.content)
    assert len(_records(result.content)) == 1
    assert footer["matched_lines"] == matched
    assert footer["truncated"] == truncated


# 功能：验证 256 个反斜线 query 保持合法并在 514 hard cap 内完整渲染
# 设计：将最坏 escaped match 放在长行中央，同时锁定双侧 ellipsis 和不裁断 match
async def test_maximum_escaped_match_preserves_match_and_both_ellipses(
    tmp_path: Path,
) -> None:
    query = "\\" * 256
    (tmp_path / "slashes.txt").write_text("left" * 100 + query + "right" * 100, encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": query, "case_sensitive": True})

    record = _records(result.content)[0]
    snippet = record.split(":1: ", 1)[1]
    escaped_match = "\\\\" * 256
    assert escaped_match in snippet
    assert snippet.startswith("…") and snippet.endswith("…")
    assert len(snippet) <= MAX_SNIPPET_CHARS == 514


# 功能：验证总输出按 UTF-8 bytes 限制且 footer 始终完整
# 设计：中文行同时锁定 byte cap 与 max_results 先触发时不被 output_limit 覆盖
async def test_output_limit_counts_utf8_bytes_and_keeps_complete_footer(tmp_path: Path) -> None:
    line = "针" * 400
    (tmp_path / "unicode.txt").write_text("\n".join([line] * 40), encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "针", "max_results": 200})

    encoded = result.content.encode("utf-8")
    footer = _footer(result.content)
    assert len(encoded) <= MAX_OUTPUT_BYTES
    assert footer["truncated"] == "output_limit"
    assert footer["matched_lines"] == 40
    assert 0 < len(_records(result.content)) < 40
    assert all(record.endswith(line) for record in _records(result.content))
    assert all(
        len(record.split(": ", 1)[1]) <= SNIPPET_TARGET_CHARS for record in _records(result.content)
    )
    assert result.content.endswith(result.content.splitlines()[-1])

    (tmp_path / "precedence.txt").write_text("\n".join([line] * 51), encoding="utf-8")
    precedence = await _tool(tmp_path).invoke({"query": "针", "path": "precedence.txt"})
    precedence_footer = _footer(precedence.content)
    assert precedence_footer["matched_lines"] == 51
    assert precedence_footer["truncated"] == "max_results"
    assert 0 < len(_records(precedence.content)) < 50


# 功能：验证 runs 不在 ignore list，且显式选择非敏感 ignored root 可覆盖递归忽略
# 设计：默认根搜索只命中 runs，随后以 build 为显式 root 锁定 override 语义
async def test_runs_is_searchable_and_explicit_ignored_root_is_allowed(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    ignored = tmp_path / "build"
    runs.mkdir()
    ignored.mkdir()
    (runs / "hit.txt").write_text("needle", encoding="utf-8")
    (ignored / "hit.txt").write_text("needle", encoding="utf-8")

    root_result = await _tool(tmp_path).invoke({"query": "needle"})
    explicit_result = await _tool(tmp_path).invoke({"query": "needle", "path": "build"})

    assert _records(root_result.content) == ["runs/hit.txt:1: needle"]
    assert _records(explicit_result.content) == ["build/hit.txt:1: needle"]
