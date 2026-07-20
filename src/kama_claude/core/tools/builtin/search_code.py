from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.workspace.errors import InvalidWorkspacePathError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

DEFAULT_MAX_RESULTS = 50
MAX_RESULTS = 200
PER_DIRECTORY_ENTRY_LIMIT = 5_000
TOTAL_DIRECTORY_ENTRY_LIMIT = 10_000
VISITED_DIRECTORY_LIMIT = 2_000
EXAMINED_FILE_LIMIT = 5_000
PER_FILE_BYTE_LIMIT = 1 * 1024 * 1024
TOTAL_BYTE_LIMIT = 32 * 1024 * 1024
FILE_READ_CHUNK_BYTES = 64 * 1024
BINARY_PROBE_BYTES = 8 * 1024
# 普通短匹配优先将完整 snippet 控制在约 400 个转义后字符
SNIPPET_TARGET_CHARS = 400
# 最坏 512 字符 escaped match 加左右 ellipsis 的 correctness-preserving 硬上限
MAX_SNIPPET_CHARS = 514
MAX_OUTPUT_BYTES = 32 * 1024

_MAX_QUERY_CHARS = 256
_MAX_PATH_CHARS = 1_024
_MAX_GLOB_CHARS = 256
_INVALID_ROOT_MESSAGE = "search path must be a regular file or directory"
_POSIX_CAPABILITY_ERROR = "secure search_code POSIX capabilities are unavailable"
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".codegraph",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "build",
        "dist",
        "target",
        ".next",
        "mutants",
    }
)
_LOGGER = logging.getLogger(__name__)


# 判断参数文本是否包含不能跨越 schema 边界的控制字符
def _contains_frozen_control(value: str) -> bool:
    return any(ord(char) <= 0x1F or ord(char) == 0x7F or char in "\u2028\u2029" for char in value)


class SearchCodeParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    path: str = "."
    include_glob: str | None = None
    case_sensitive: bool = False
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=MAX_RESULTS)

    @field_validator("query")
    @classmethod
    # 校验 literal query 的长度、空白与控制字符约束
    def _validate_query(cls, value: str) -> str:
        if not value or value.isspace():
            raise ValueError("query must contain a non-whitespace character")
        if len(value) > _MAX_QUERY_CHARS:
            raise ValueError("query is too long")
        if _contains_frozen_control(value):
            raise ValueError("query contains unsupported control characters")
        return value

    @field_validator("path")
    @classmethod
    # 校验 workspace-relative path 的长度与控制字符约束
    def _validate_path(cls, value: str) -> str:
        if not value or value.isspace():
            raise ValueError("path must contain a non-whitespace character")
        if len(value) > _MAX_PATH_CHARS:
            raise ValueError("path is too long")
        if _contains_frozen_control(value):
            raise ValueError("path contains unsupported control characters")
        return value

    @field_validator("include_glob")
    @classmethod
    # 校验第一版仅支持单个 basename glob
    def _validate_include_glob(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.isspace():
            raise ValueError("include_glob must contain a non-whitespace character")
        if len(value) > _MAX_GLOB_CHARS:
            raise ValueError("include_glob is too long")
        if _contains_frozen_control(value):
            raise ValueError("include_glob contains unsupported control characters")
        if "/" in value or "\\" in value or "**" in value or value.startswith("!"):
            raise ValueError("include_glob must be a single basename pattern")
        return value


@dataclass
class _SearchState:
    directory_entries: int = 0
    visited_directories: int = 0
    examined_files: int = 0
    examined_bytes: int = 0
    matched_lines: int = 0
    skipped_non_text: int = 0
    skipped_large: int = 0
    skipped_unreadable: int = 0
    truncated: str = "none"
    records: list[str] = field(default_factory=list)
    visited_canonical_directories: set[tuple[int, int]] = field(default_factory=set)

    # 只记录搜索过程中第一个实际触发的资源上限
    def set_truncated(self, reason: str) -> None:
        if self.truncated == "none":
            self.truncated = reason


class _SearchStopped(Exception):
    pass


class SearchCodeTool(BaseTool):
    params_model = SearchCodeParams
    name = "search_code"
    description = (
        "Search UTF-8 source files for a literal string within the session workspace. "
        "The search is recursively bounded and returns workspace-relative paths only."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_QUERY_CHARS,
                "description": "Literal text to search for.",
            },
            "path": {
                "type": "string",
                "default": ".",
                "minLength": 1,
                "maxLength": _MAX_PATH_CHARS,
                "description": "Workspace-relative file or directory (default '.').",
            },
            "include_glob": {
                "type": ["string", "null"],
                "maxLength": _MAX_GLOB_CHARS,
                "description": "Optional case-sensitive basename glob.",
            },
            "case_sensitive": {
                "type": "boolean",
                "default": False,
                "description": "Use case-sensitive literal matching.",
            },
            "max_results": {
                "type": "integer",
                "default": DEFAULT_MAX_RESULTS,
                "minimum": 1,
                "maximum": MAX_RESULTS,
                "description": "Maximum returned matching lines.",
            },
        },
        "required": ["query"],
    }

    # 注入 workspace resolver 与敏感路径策略
    def __init__(
        self,
        resolver: WorkspacePathResolver,
        access_policy: WorkspaceAccessPolicy,
    ) -> None:
        self._resolver = resolver
        self._access_policy = access_policy

    # 在线程中执行搜索，并在取消后等待合作式 worker 到达终态
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        validated = SearchCodeParams.model_validate(params)
        stop_event = threading.Event()
        worker = asyncio.create_task(asyncio.to_thread(self._search_sync, validated, stop_event))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            stop_event.set()
            terminal_observed = False
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except _SearchStopped:
                    terminal_observed = True
                    break
                except Exception as exc:
                    _LOGGER.error(
                        "search worker failed during cancellation cleanup",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    terminal_observed = True
                    break
            if worker.done() and not worker.cancelled() and not terminal_observed:
                try:
                    worker.result()
                except _SearchStopped:
                    pass
                except Exception as exc:
                    _LOGGER.error(
                        "search worker failed during cancellation cleanup",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
            raise

    # 同步执行有界 workspace 搜索并构造 byte-bounded 结果
    def _search_sync(
        self,
        params: SearchCodeParams,
        stop_event: threading.Event,
    ) -> ToolResult:
        state = _SearchState()
        self._ensure_posix_capabilities()
        root = self._resolver.resolve_existing(params.path)
        self._access_policy.ensure_allowed(params.path, root)
        logical_root = PurePath(params.path)
        self._check_stop(stop_event)
        root_mode = root.stat().st_mode
        if stat.S_ISDIR(root_mode):
            self._walk_directory(
                root,
                logical_root,
                params,
                state,
                stop_event,
                explicit_root=True,
            )
        elif stat.S_ISREG(root_mode):
            self._search_file(
                root,
                logical_root,
                params,
                state,
                stop_event,
                explicit_root=True,
            )
        else:
            return ToolResult(
                content=_INVALID_ROOT_MESSAGE,
                is_error=True,
                error_type="invalid_input",
            )
        self._check_stop(stop_event)
        return ToolResult(content=self._assemble_output(state, stop_event))

    # 确认 hardened search 依赖的 POSIX fd 能力全部可用
    def _ensure_posix_capabilities(self) -> None:
        if (
            getattr(os, "O_NOFOLLOW", None) is None
            or getattr(os, "O_DIRECTORY", None) is None
            or os.open not in getattr(os, "supports_dir_fd", ())
            or os.scandir not in getattr(os, "supports_fd", ())
        ):
            raise RuntimeError(_POSIX_CAPABILITY_ERROR)

    # 在合作式边界观察异步调用方设置的停止信号
    def _check_stop(self, stop_event: threading.Event) -> None:
        if stop_event.is_set():
            raise _SearchStopped

    # 以有界 scandir batch 深度优先遍历确定性 logical path
    def _walk_directory(
        self,
        directory: Path,
        logical_directory: PurePath,
        params: SearchCodeParams,
        state: _SearchState,
        stop_event: threading.Event,
        *,
        explicit_root: bool = False,
    ) -> bool:
        self._check_stop(stop_event)
        entries: list[os.DirEntry[str]] = []
        directory_fd = -1
        try:
            directory_fd = self._open_workspace_fd(directory, directory=True)
            metadata = os.fstat(directory_fd)
            canonical_identity = (metadata.st_dev, metadata.st_ino)
            if canonical_identity in state.visited_canonical_directories:
                return False
            state.visited_canonical_directories.add(canonical_identity)
            state.visited_directories += 1
            if state.visited_directories > VISITED_DIRECTORY_LIMIT:
                state.set_truncated("directory_limit")
                return True

            with os.scandir(directory_fd) as iterator:
                while True:
                    self._check_stop(stop_event)
                    try:
                        entry = next(iterator)
                    except StopIteration:
                        break
                    self._check_stop(stop_event)
                    state.directory_entries += 1
                    entries.append(entry)
                    self._check_stop(stop_event)
                    if (
                        len(entries) > PER_DIRECTORY_ENTRY_LIMIT
                        or state.directory_entries > TOTAL_DIRECTORY_ENTRY_LIMIT
                    ):
                        state.set_truncated("entry_limit")
                        return True
        except OSError:
            if explicit_root:
                raise
            state.skipped_unreadable += 1
            return False
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)

        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            self._check_stop(stop_event)
            logical_child = logical_directory / entry.name
            logical_child_str = logical_child.as_posix()
            try:
                resolved_child = self._resolver.resolve_existing(logical_child_str)
                self._access_policy.ensure_allowed(logical_child_str, resolved_child)
                is_directory = resolved_child.is_dir()
                is_file = resolved_child.is_file()
            except (OSError, RuntimeError, InvalidWorkspacePathError):
                continue

            if is_directory:
                if entry.name in _IGNORED_DIRECTORIES:
                    continue
                if self._walk_directory(
                    resolved_child,
                    logical_child,
                    params,
                    state,
                    stop_event,
                ):
                    return True
                continue
            if not is_file:
                continue
            if self._search_file(
                resolved_child,
                logical_child,
                params,
                state,
                stop_event,
            ):
                return True
        return False

    # 搜索一个已通过 resolver 与 policy 的 regular-file candidate
    def _search_file(
        self,
        path: Path,
        logical_path: PurePath,
        params: SearchCodeParams,
        state: _SearchState,
        stop_event: threading.Event,
        *,
        explicit_root: bool = False,
    ) -> bool:
        self._check_stop(stop_event)
        state.examined_files += 1
        if state.examined_files > EXAMINED_FILE_LIMIT:
            state.set_truncated("file_limit")
            return True
        if params.include_glob is not None and not fnmatch.fnmatchcase(
            logical_path.name,
            params.include_glob,
        ):
            return False

        data, stop_search = self._read_file(
            path,
            state,
            stop_event,
            explicit_root=explicit_root,
        )
        if stop_search:
            return True
        if data is None:
            return False
        if b"\x00" in data[:BINARY_PROBE_BYTES]:
            state.skipped_non_text += 1
            return False
        try:
            text = bytes(data).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            state.skipped_non_text += 1
            return False
        self._check_stop(stop_event)
        return self._search_text(text, logical_path, params, state, stop_event)

    # 以实际读取字节同时执行 per-file 与 total-byte 边界
    def _read_file(
        self,
        path: Path,
        state: _SearchState,
        stop_event: threading.Event,
        *,
        explicit_root: bool = False,
    ) -> tuple[bytearray | None, bool]:
        data = bytearray()
        try:
            self._check_stop(stop_event)
            handle, initial_size = self._open_regular_file(path)
            with handle:
                if initial_size > PER_FILE_BYTE_LIMIT:
                    state.skipped_large += 1
                    return None, False
                while True:
                    self._check_stop(stop_event)
                    total_remaining = TOTAL_BYTE_LIMIT - state.examined_bytes
                    if total_remaining <= 0:
                        state.set_truncated("byte_limit")
                        return None, True
                    per_file_remaining = PER_FILE_BYTE_LIMIT + 1 - len(data)
                    read_size = min(
                        FILE_READ_CHUNK_BYTES,
                        total_remaining,
                        per_file_remaining,
                    )
                    chunk = handle.read(read_size)
                    state.examined_bytes += len(chunk)
                    self._check_stop(stop_event)
                    if not chunk:
                        if os.fstat(handle.fileno()).st_size > PER_FILE_BYTE_LIMIT:
                            state.skipped_large += 1
                            return None, False
                        return data, False
                    data.extend(chunk)
                    if len(data) > PER_FILE_BYTE_LIMIT:
                        state.skipped_large += 1
                        return None, False
                    if state.examined_bytes == TOTAL_BYTE_LIMIT:
                        if handle.tell() >= os.fstat(handle.fileno()).st_size:
                            return data, False
                        state.set_truncated("byte_limit")
                        return None, True
        except OSError:
            if explicit_root:
                raise
            state.skipped_unreadable += 1
            return None, False

    # 从 canonical workspace root 逐组件 no-follow 打开目标并绑定到文件描述符
    def _open_workspace_fd(self, path: Path, *, directory: bool) -> int:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if no_follow is None or directory_flag is None:
            raise OSError("safe no-follow workspace open is unsupported")
        try:
            relative_parts = path.relative_to(self._resolver.root).parts
        except ValueError as exc:
            raise OSError("workspace path escapes root") from exc

        base_flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
        current_fd = os.open(
            self._resolver.root,
            base_flags | directory_flag,
        )
        result_fd = -1
        try:
            for part in relative_parts[:-1]:
                next_fd = os.open(
                    part,
                    base_flags | directory_flag,
                    dir_fd=current_fd,
                )
                previous_fd = current_fd
                current_fd = next_fd
                os.close(previous_fd)
            if not relative_parts:
                result_fd = current_fd
                current_fd = -1
            else:
                final_flags = base_flags | (directory_flag if directory else 0)
                result_fd = os.open(
                    relative_parts[-1],
                    final_flags,
                    dir_fd=current_fd,
                )
            metadata = os.fstat(result_fd)
            expected_type = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(
                metadata.st_mode
            )
            if not expected_type:
                raise OSError("workspace candidate has unexpected file type")
            return result_fd
        except Exception:
            if result_fd >= 0:
                os.close(result_fd)
            raise
        finally:
            if current_fd >= 0:
                os.close(current_fd)

    # 安全打开 regular file 并返回与同一描述符绑定的预检大小
    def _open_regular_file(self, path: Path) -> tuple[BinaryIO, int]:
        file_fd = self._open_workspace_fd(path, directory=False)
        try:
            initial_size = os.fstat(file_fd).st_size
            return os.fdopen(file_fd, "rb"), initial_size
        except Exception:
            os.close(file_fd)
            raise

    # 按 LF 物理行搜索 literal query，并观测 max-results sentinel
    def _search_text(
        self,
        text: str,
        logical_path: PurePath,
        params: SearchCodeParams,
        state: _SearchState,
        stop_event: threading.Event,
    ) -> bool:
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        escaped_path = self._escape_path(logical_path.as_posix())
        for line_number, raw_line in enumerate(lines, start=1):
            self._check_stop(stop_event)
            line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
            match_span = self._literal_match_span(
                line,
                params.query,
                params.case_sensitive,
            )
            if match_span is None:
                continue
            state.matched_lines += 1
            if state.matched_lines > params.max_results:
                state.set_truncated("max_results")
                return True
            snippet = self._bounded_snippet(line, *match_span)
            state.records.append(f"{escaped_path}:{line_number}: {snippet}")
        return False

    # 返回 literal match 在原字符串中的准确 start/end offset
    def _literal_match_span(
        self,
        text: str,
        query: str,
        case_sensitive: bool,
    ) -> tuple[int, int] | None:
        if case_sensitive:
            start = text.find(query)
            return None if start < 0 else (start, start + len(query))

        folded_parts: list[str] = []
        original_offsets: list[int] = []
        for offset, char in enumerate(text):
            folded = char.casefold()
            folded_parts.append(folded)
            original_offsets.extend([offset] * len(folded))
        folded_text = "".join(folded_parts)
        folded_query = query.casefold()
        folded_start = folded_text.find(folded_query)
        if folded_start < 0:
            return None
        folded_end = folded_start + len(folded_query)
        original_start = original_offsets[folded_start]
        original_end = original_offsets[folded_end - 1] + 1
        return original_start, original_end

    # 以 400 字符为常规目标，在 514 硬上限内完整保留 escaped match 与必要 ellipsis
    def _bounded_snippet(self, line: str, match_start: int, match_end: int) -> str:
        pieces = [self._escape_character(char) for char in line]
        if sum(len(piece) for piece in pieces) <= SNIPPET_TARGET_CHARS:
            return "".join(pieces)

        match_pieces = pieces[match_start:match_end]
        match_chars = sum(len(piece) for piece in match_pieces)
        ellipsis_slots = int(match_start > 0) + int(match_end < len(pieces))
        assert match_chars + ellipsis_slots <= MAX_SNIPPET_CHARS
        cap = max(SNIPPET_TARGET_CHARS, match_chars + ellipsis_slots)
        context_budget = cap - match_chars - ellipsis_slots
        if match_start == 0:
            left_budget = 0
        elif match_end == len(pieces):
            left_budget = context_budget
        else:
            left_budget = context_budget // 2
        right_budget = context_budget - left_budget

        left_start = match_start
        left_used = 0
        while left_start > 0 and len(pieces[left_start - 1]) <= left_budget - left_used:
            left_start -= 1
            left_used += len(pieces[left_start])
        right_end = match_end
        right_used = 0
        while right_end < len(pieces) and len(pieces[right_end]) <= right_budget - right_used:
            right_used += len(pieces[right_end])
            right_end += 1

        unused = context_budget - left_used - right_used
        while left_start > 0 and len(pieces[left_start - 1]) <= unused:
            left_start -= 1
            piece_chars = len(pieces[left_start])
            left_used += piece_chars
            unused -= piece_chars
        while right_end < len(pieces) and len(pieces[right_end]) <= unused:
            piece_chars = len(pieces[right_end])
            right_used += piece_chars
            unused -= piece_chars
            right_end += 1

        left_ellipsis = "…" if left_start > 0 else ""
        right_ellipsis = "…" if right_end < len(pieces) else ""
        snippet = "".join(
            (
                left_ellipsis,
                *pieces[left_start:match_start],
                *match_pieces,
                *pieces[match_end:right_end],
                right_ellipsis,
            )
        )
        assert len(snippet) <= MAX_SNIPPET_CHARS
        return snippet

    # 将单个字符转换为稳定的单行可见表示
    def _escape_character(self, char: str) -> str:
        codepoint = ord(char)
        if char == "\\":
            return "\\\\"
        if char == "\n":
            return "\\n"
        if char == "\r":
            return "\\r"
        if char == "\t":
            return "\\t"
        if codepoint <= 0x1F or codepoint == 0x7F:
            return f"\\x{codepoint:02X}"
        if char == "\u2028":
            return "\\u2028"
        if char == "\u2029":
            return "\\u2029"
        return char

    # 转义 snippet 中的反斜线、控制字符与 Unicode 行分隔符
    def _escape_text(self, value: str) -> str:
        return "".join(self._escape_character(char) for char in value)

    # 转义 workspace-relative logical path 并额外保护 colon 分隔符
    def _escape_path(self, value: str) -> str:
        return self._escape_text(value).replace(":", "\\:")

    # 根据最终计数生成完整且不可截断的固定 footer
    def _footer(self, state: _SearchState) -> str:
        return (
            f"[search_code] matched_lines={state.matched_lines} "
            f"directory_entries={state.directory_entries} "
            f"visited_directories={state.visited_directories} "
            f"examined_files={state.examined_files} "
            f"examined_bytes={state.examined_bytes} "
            f"skipped_non_text={state.skipped_non_text} "
            f"skipped_large={state.skipped_large} "
            f"skipped_unreadable={state.skipped_unreadable} "
            f"truncated={state.truncated}"
        )

    # 在 UTF-8 byte cap 内只加入完整结果并始终保留完整 footer
    def _assemble_output(
        self,
        state: _SearchState,
        stop_event: threading.Event,
    ) -> str:
        self._check_stop(stop_event)
        footer = self._footer(state)
        complete = "\n".join((*state.records, footer))
        if len(complete.encode("utf-8")) <= MAX_OUTPUT_BYTES:
            return complete

        state.set_truncated("output_limit")
        accepted: list[str] = []
        for record in state.records:
            self._check_stop(stop_event)
            candidate_footer = self._footer(state)
            candidate = "\n".join((*accepted, record, candidate_footer))
            if len(candidate.encode("utf-8")) > MAX_OUTPUT_BYTES:
                break
            accepted.append(record)
        footer = self._footer(state)
        content = "\n".join((*accepted, footer))
        assert len(content.encode("utf-8")) <= MAX_OUTPUT_BYTES
        return content
