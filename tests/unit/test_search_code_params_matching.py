from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kama_claude.core.tools.builtin.search_code import (
    FILE_READ_CHUNK_BYTES,
    SearchCodeParams,
    SearchCodeTool,
)
from tests.unit._search_code_test_support import _footer, _records, _tool


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

    result = await _tool(tmp_path).invoke({"query": "needle", "include_glob": "[!a]*.py"})

    assert _records(result.content) == ["nested/beta.py:1: needle"]
    assert _footer(result.content)["examined_files"] == 3


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


# 功能：验证输出 logical path/snippet 转义 colon、backslash、control 与 Unicode separator
# 设计：使用 Unix 合法特殊文件名与混合控制内容，并硬断言 invocation 结果不泄露绝对路径
async def test_output_escapes_logical_path_and_never_leaks_absolute_path(tmp_path: Path) -> None:
    filename = "a:b\\c\n\x1b.py"
    (tmp_path / filename).write_text("needle:\t\r\x1b\u2028\u2029", encoding="utf-8")

    result = await _tool(tmp_path).invoke({"query": "needle", "case_sensitive": True})

    assert _records(result.content) == [r"a\:b\\c\n\x1B.py:1: needle:\t\r\x1B\u2028\u2029"]
    assert str(tmp_path.resolve(strict=True)) not in result.content
