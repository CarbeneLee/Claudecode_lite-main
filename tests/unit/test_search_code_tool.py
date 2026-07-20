from __future__ import annotations

import os
import re
import socket
import tempfile
from pathlib import Path, PurePath
from typing import BinaryIO

import pytest
from pydantic import ValidationError

import kama_claude.core.tools.builtin.search_code as search_code_module
from kama_claude.core.tools.builtin.search_code import (
    FILE_READ_CHUNK_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_SNIPPET_CHARS,
    PER_DIRECTORY_ENTRY_LIMIT,
    PER_FILE_BYTE_LIMIT,
    SNIPPET_TARGET_CHARS,
    SearchCodeParams,
    SearchCodeTool,
)
from kama_claude.core.workspace.errors import SensitivePathError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_FOOTER_RE = re.compile(
    r"^\[search_code\] matched_lines=(?P<matched_lines>\d+) "
    r"directory_entries=(?P<directory_entries>\d+) "
    r"visited_directories=(?P<visited_directories>\d+) "
    r"examined_files=(?P<examined_files>\d+) "
    r"examined_bytes=(?P<examined_bytes>\d+) "
    r"skipped_non_text=(?P<skipped_non_text>\d+) "
    r"skipped_large=(?P<skipped_large>\d+) "
    r"skipped_unreadable=(?P<skipped_unreadable>\d+) "
    r"truncated=(?P<truncated>\w+)$"
)


# 构造绑定指定 workspace 的 search_code 工具
def _tool(workspace: Path) -> SearchCodeTool:
    return SearchCodeTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 从搜索输出末行解析精确资源计数
def _footer(content: str) -> dict[str, int | str]:
    match = _FOOTER_RE.fullmatch(content.splitlines()[-1])
    assert match is not None, content
    values: dict[str, int | str] = {}
    for key, value in match.groupdict().items():
        values[key] = value if key == "truncated" else int(value)
    return values


# 返回除完整 footer 外的结果记录
def _records(content: str) -> list[str]:
    return content.splitlines()[:-1]


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


@pytest.mark.parametrize("path", ["", "   "])
# 功能：验证空 path 与全 whitespace path 在 schema 边界被拒绝
# 设计：直接调用公开 Pydantic model，使 mutation 接受空串时无需进入 filesystem 即失败
def test_path_rejects_empty_and_all_whitespace(path: str) -> None:
    with pytest.raises(ValidationError):
        SearchCodeParams.model_validate({"query": "needle", "path": path})


# 功能：验证带前后空格的真实文件名原样保留且可搜索
# 设计：创建实际 padded basename，同时断言 model 不 trim 与 resolver 能命中同一路径
async def test_path_preserves_spaces_for_real_filename(tmp_path: Path) -> None:
    padded_name = "  spaced file.txt  "
    (tmp_path / padded_name).write_text("needle", encoding="utf-8")

    params = SearchCodeParams.model_validate({"query": "needle", "path": padded_name})
    result = await _tool(tmp_path).invoke(params.model_dump())

    assert params.path == padded_name
    assert _records(result.content) == [f"{padded_name}:1: needle"]


# 功能：验证手写 input_schema 与 SearchCodeParams 的冻结 defaults/lengths 一致
# 设计：只断言用户可见 JSON schema 约束，防止 Pydantic validator 与工具描述发生漂移
def test_input_schema_declares_frozen_defaults_and_lengths() -> None:
    properties = SearchCodeTool.input_schema["properties"]
    assert isinstance(properties, dict)
    assert properties["query"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
        "description": "Literal text to search for.",
    }
    assert properties["path"] == {
        "type": "string",
        "default": ".",
        "minLength": 1,
        "maxLength": 1_024,
        "description": "Workspace-relative file or directory (default '.').",
    }
    assert properties["include_glob"] == {
        "type": ["string", "null"],
        "maxLength": 256,
        "description": "Optional case-sensitive basename glob.",
    }
    assert properties["case_sensitive"] == {
        "type": "boolean",
        "default": False,
        "description": "Use case-sensitive literal matching.",
    }
    assert properties["max_results"] == {
        "type": "integer",
        "default": 50,
        "minimum": 1,
        "maximum": 200,
        "description": "Maximum returned matching lines.",
    }


# 功能：验证搜索按 literal 匹配且 casefold expansion 正确映射回原文
# 设计：同时使用正则元字符和 Straße/STRASSE，断言 snippet 保留真实原文
async def test_literal_and_casefold_expansion_keep_original_match(tmp_path: Path) -> None:
    (tmp_path / "literal.txt").write_text("a[b] only\nleft Straße right", encoding="utf-8")

    literal = await _tool(tmp_path).invoke({"query": "a[b]", "case_sensitive": True})
    folded = await _tool(tmp_path).invoke({"query": "STRASSE", "case_sensitive": False})

    assert _records(literal.content) == ["literal.txt:1: a[b] only"]
    assert _records(folded.content) == ["literal.txt:2: left Straße right"]


# 功能：验证只有 LF 创建物理行，CR 与 Unicode line separator 仅被可视转义
# 设计：一个文件混合 CRLF、独立 CR、U+2028/U+2029 和无尾 LF 末行，直接锁定行号
async def test_physical_lines_are_lf_only_and_final_segment_is_searched(tmp_path: Path) -> None:
    text = "hit-one\r\nbefore\rhit-two\u2028hit-three\u2029tail\nlast-hit"
    (tmp_path / "lines.txt").write_text(text, encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "hit", "case_sensitive": True})

    assert _records(result.content) == [
        "lines.txt:1: hit-one",
        r"lines.txt:2: before\rhit-two\u2028hit-three\u2029tail",
        "lines.txt:3: last-hit",
    ]
    assert _footer(result.content)["matched_lines"] == 3


# 功能：验证文件以 LF 结尾时不搜索人工产生的空末段
# 设计：只在首段放置匹配并断言计数为一，锁定尾 LF 的 segment 语义
async def test_trailing_lf_does_not_create_searchable_empty_line(tmp_path: Path) -> None:
    (tmp_path / "trailing.txt").write_text("needle\n", encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "needle", "case_sensitive": True})

    assert _records(result.content) == ["trailing.txt:1: needle"]
    assert _footer(result.content)["matched_lines"] == 1


# 功能：验证 include_glob 仅匹配 case-sensitive basename 且不匹配相对路径
# 设计：根目录与子目录放置大小写不同后缀，使用 Python `[!a]` 字符类
async def test_include_glob_matches_case_sensitive_basename_only(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "alpha.py").write_text("needle", encoding="utf-8")
    (nested / "beta.py").write_text("needle", encoding="utf-8")
    (nested / "gamma.PY").write_text("needle", encoding="utf-8")

    result = await _tool(tmp_path).invoke(
        {"query": "needle", "include_glob": "[!a]*.py"}
    )

    assert _records(result.content) == ["nested/beta.py:1: needle"]
    assert _footer(result.content)["examined_files"] == 3


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

    result = await _tool(tmp_path).invoke(
        {"query": "needle", "include_glob": "*.py"}
    )

    assert _records(result.content) == ["hit.py:1: needle"]
    footer = _footer(result.content)
    assert footer["directory_entries"] == 3
    assert footer["visited_directories"] == 1
    assert footer["examined_files"] == 2
    assert footer["examined_bytes"] == len(b"needle")


# 功能：验证跨 64 KiB chunk 的 UTF-8 多字节字符只在整文件读完后解码
# 设计：将中文字符首字节精确放在第一个 chunk 末尾，独立 chunk decode 必然失败
async def test_utf8_character_crossing_read_chunk_is_searchable(tmp_path: Path) -> None:
    data = b"x" * (FILE_READ_CHUNK_BYTES - 1) + "中".encode() + b"-needle"
    (tmp_path / "chunk.txt").write_bytes(data)

    result = await _tool(tmp_path).invoke({"query": "中-needle", "case_sensitive": True})

    records = _records(result.content)
    assert len(records) == 1
    assert "中-needle" in records[0]
    assert _footer(result.content)["examined_bytes"] == len(data)


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
    result = await tool.invoke(
        {"query": "abcd", "path": "short.txt", "case_sensitive": True}
    )

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


@pytest.mark.parametrize("root_kind", ["file", "directory"])
# 功能：验证显式 file/directory root 不可读时直接传播 PermissionError
# 设计：在 fd open seam 注入确定性拒绝，区分调用级 root 失败与 recursive child best-effort skip
async def test_explicit_unreadable_root_propagates_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    target = tmp_path / "target"
    if root_kind == "directory":
        target.mkdir()
    else:
        target.write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    real_open = tool._open_workspace_fd

    # 只拒绝显式 root 的安全 fd open
    def deny_root(path: Path, *, directory: bool) -> int:
        if path == target.resolve(strict=True):
            raise PermissionError("private platform detail")
        return real_open(path, directory=directory)

    monkeypatch.setattr(tool, "_open_workspace_fd", deny_root)

    with pytest.raises(PermissionError, match="private platform detail"):
        await tool.invoke({"query": "needle", "path": "target"})


# 功能：验证递归 unreadable child directory 仍被跳过且搜索继续
# 设计：只拒绝排序靠前的 child directory fd，确认 visible sibling 仍产出结果与 skip 计数
async def test_recursive_unreadable_child_directory_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "a-blocked"
    blocked.mkdir()
    (blocked / "hidden.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "z-visible.txt").write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    real_open = tool._open_workspace_fd

    # 只拒绝 recursive child directory，不影响显式 workspace root
    def deny_child(path: Path, *, directory: bool) -> int:
        if directory and path == blocked.resolve(strict=True):
            raise PermissionError("denied child")
        return real_open(path, directory=directory)

    monkeypatch.setattr(tool, "_open_workspace_fd", deny_child)
    result = await tool.invoke({"query": "needle"})

    assert _records(result.content) == ["z-visible.txt:1: needle"]
    assert _footer(result.content)["skipped_unreadable"] == 1


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


@pytest.mark.parametrize(("text", "matched", "truncated"), [("hit", 1, "none"), ("hit\nhit", 2, "max_results")])
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
        len(record.split(": ", 1)[1]) <= SNIPPET_TARGET_CHARS
        for record in _records(result.content)
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


@pytest.mark.parametrize("root_kind", ["fifo", "socket"])
# 功能：验证显式 FIFO/socket root 返回固定 invalid_input 且绝不尝试 open
# 设计：创建真实 POSIX special file 并将 fd helper 设为 fail-fast，避免依赖 timeout 证明不阻塞
async def test_explicit_special_root_is_invalid_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    short_workspace: tempfile.TemporaryDirectory[str] | None = None
    workspace = tmp_path
    if root_kind == "socket":
        short_workspace = tempfile.TemporaryDirectory(prefix="ks-", dir="/tmp")
        workspace = Path(short_workspace.name)
    target = workspace / root_kind
    unix_socket: socket.socket | None = None
    if root_kind == "fifo":
        os.mkfifo(target)
    else:
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_socket.bind(str(target))
    tool = _tool(workspace)
    opened = False

    # special root 若进入 fd open 即证明存在阻塞风险
    def unexpected_open(_path: Path, *, directory: bool) -> int:
        nonlocal opened
        opened = True
        raise AssertionError(f"special root reached fd open: {directory}")

    monkeypatch.setattr(tool, "_open_workspace_fd", unexpected_open)
    try:
        result = await tool.invoke({"query": "needle", "path": root_kind})
    finally:
        if unix_socket is not None:
            unix_socket.close()
        if short_workspace is not None:
            short_workspace.cleanup()

    assert opened is False
    assert result.is_error
    assert result.error_type == "invalid_input"
    assert result.content == "search path must be a regular file or directory"


@pytest.mark.parametrize(
    "missing_capability",
    ["O_NOFOLLOW", "O_DIRECTORY", "open_dir_fd", "scandir_fd"],
)
# 功能：验证 hardened search 缺少任一 POSIX capability 时 direct invoke fail closed
# 设计：逐项移除常量或 capability set，锁定平台不支持不能退化为成功空 footer
async def test_missing_posix_capability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    (tmp_path / "hit.txt").write_text("needle", encoding="utf-8")
    if missing_capability in {"O_NOFOLLOW", "O_DIRECTORY"}:
        monkeypatch.delattr(search_code_module.os, missing_capability)
    elif missing_capability == "open_dir_fd":
        monkeypatch.setattr(search_code_module.os, "supports_dir_fd", frozenset())
    else:
        monkeypatch.setattr(search_code_module.os, "supports_fd", frozenset())

    with pytest.raises(
        RuntimeError,
        match="secure search_code POSIX capabilities are unavailable",
    ):
        await _tool(tmp_path).invoke({"query": "needle"})


# 功能：验证递归遇到外部 symlink 与敏感候选时不搜索也不泄露
# 设计：在可搜索普通文件旁放置外部 file/dir alias 与 .env，确认 candidate policy 不可被删除
async def test_nested_external_symlinks_and_sensitive_candidates_do_not_leak(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "visible.txt").write_text("public-needle", encoding="utf-8")
    secret = outside / "secret.txt"
    secret.write_text("secret-needle", encoding="utf-8")
    (workspace / "external-file").symlink_to(secret)
    (workspace / "external-dir").symlink_to(outside, target_is_directory=True)
    (workspace / ".env").write_text("TOKEN=secret-needle", encoding="utf-8")

    result = await _tool(workspace).invoke({"query": "needle"})

    assert _records(result.content) == ["visible.txt:1: public-needle"]
    assert _footer(result.content)["directory_entries"] == 4
    assert "secret-needle" not in result.content
    assert str(outside.resolve(strict=True)) not in result.content


@pytest.mark.parametrize("swap_component", ["final", "ancestor"])
# 功能：验证 candidate 通过 policy 后替换 final/ancestor 组件都不能读取外部内容
# 设计：在真实 ensure_allowed 返回后换成外部 symlink，锁定逐组件 no-follow 不可被简化
async def test_candidate_symlink_swap_after_policy_cannot_escape_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_component: str,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    target_directory = workspace if swap_component == "final" else workspace / "nested"
    target_directory.mkdir(exist_ok=True)
    victim = target_directory / "victim.txt"
    victim.write_text("public", encoding="utf-8")
    secret = outside / "victim.txt"
    secret.write_text("secret-needle", encoding="utf-8")
    logical_victim = victim.relative_to(workspace).as_posix()
    resolver = WorkspacePathResolver(workspace)
    policy = WorkspaceAccessPolicy(workspace)
    ensure_allowed = policy.ensure_allowed
    swapped = False

    def swap_after_policy(logical_path: str, resolved_path: Path) -> None:
        nonlocal swapped
        ensure_allowed(logical_path, resolved_path)
        if logical_path == logical_victim and not swapped:
            if swap_component == "final":
                victim.unlink()
                victim.symlink_to(secret)
            else:
                target_directory.rename(workspace / "parked")
                target_directory.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(policy, "ensure_allowed", swap_after_policy)
    result = await SearchCodeTool(resolver, policy).invoke({"query": "needle"})

    assert swapped
    assert _records(result.content) == []
    assert _footer(result.content)["skipped_unreadable"] == 1
    assert "secret-needle" not in result.content
    assert str(outside.resolve(strict=True)) not in result.content


# 功能：验证已发现的目录在进入前消失时按 unreadable 跳过而非抛 execution_error
# 设计：在递归入口前删除空目录，精确命中 strict resolve/open race 且不依赖线程时序
async def test_directory_disappearing_before_entry_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disappearing = tmp_path / "disappearing"
    disappearing.mkdir()
    (tmp_path / "visible.txt").write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    walk_directory = tool._walk_directory

    def remove_before_walk(*args: object, **kwargs: object) -> bool:
        directory = args[0]
        logical_directory = args[1]
        if logical_directory == PurePath("disappearing"):
            assert isinstance(directory, Path)
            directory.rmdir()
        return walk_directory(*args, **kwargs)

    monkeypatch.setattr(tool, "_walk_directory", remove_before_walk)
    result = await tool.invoke({"query": "needle"})

    assert _records(result.content) == ["visible.txt:1: needle"]
    assert _footer(result.content)["skipped_unreadable"] == 1


# 功能：验证显式 sensitive root/descendant 被拒绝且两个 env 模板例外允许
# 设计：参数化 .git、根 .env 和嵌套 .env.local，再分别从例外文件作为 root 搜索
async def test_sensitive_roots_rejected_but_env_examples_allowed(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("needle", encoding="utf-8")
    (tmp_path / ".env").write_text("needle", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env.local").write_text("needle", encoding="utf-8")
    for name in (".env.example", ".env.template"):
        (tmp_path / name).write_text("needle", encoding="utf-8")

    sensitive_paths = (".git", str(Path(".git") / "HEAD"), ".env", "config/.env.local")
    for path in sensitive_paths:
        with pytest.raises(SensitivePathError):
            await _tool(tmp_path).invoke({"query": "needle", "path": path})

    for name in (".env.example", ".env.template"):
        result = await _tool(tmp_path).invoke({"query": "needle", "path": name})
        assert _records(result.content) == [f"{name}:1: needle"]


# 功能：验证 workspace 内部 file/dir symlink 可搜索且目录环不重复遍历
# 设计：通过显式 internal file alias 和指回 root 的 directory cycle，覆盖允许与去重两类边界
async def test_internal_symlinks_allowed_and_directory_cycle_deduplicated(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.txt"
    target.write_text("needle", encoding="utf-8")
    (real / "back").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "file-alias.txt").symlink_to(target)

    file_result = await _tool(tmp_path).invoke(
        {"query": "needle", "path": "file-alias.txt"}
    )
    (tmp_path / "file-alias.txt").unlink()
    tree_result = await _tool(tmp_path).invoke({"query": "needle"})

    assert _records(file_result.content) == ["file-alias.txt:1: needle"]
    assert _records(tree_result.content) == ["real/target.txt:1: needle"]
    assert _footer(tree_result.content)["visited_directories"] == 2


# 功能：验证同一 canonical directory 仅沿确定性排序后首个 logical alias 遍历
# 设计：让 alias 名按 case-sensitive 顺序早于真实目录，断言只输出 alias path 且不重复匹配
async def test_canonical_directory_dedup_uses_first_sorted_logical_path(
    tmp_path: Path,
) -> None:
    real = tmp_path / "z-real"
    real.mkdir()
    (real / "hit.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "a-alias").symlink_to(real, target_is_directory=True)

    result = await _tool(tmp_path).invoke({"query": "needle"})

    assert _records(result.content) == ["a-alias/hit.txt:1: needle"]
    footer = _footer(result.content)
    assert footer["matched_lines"] == 1
    assert footer["visited_directories"] == 2
    assert "z-real/hit.txt" not in result.content
    assert str(real.resolve(strict=True)) not in result.content
