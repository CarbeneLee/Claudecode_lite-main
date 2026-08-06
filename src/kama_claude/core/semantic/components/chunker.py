from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kama_claude.core.semantic.components.embedding import SparseVector

_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".txt": "text",
}

_PY_DECL = re.compile(r"^\s*(async\s+def|def|class)\s+([A-Za-z_]\w*)")
_JS_PREFIX = r"^\s*(?:export\s+)?(?:default\s+)?"
_JS_CLASS = re.compile(_JS_PREFIX + r"class\s+([A-Za-z_$]\w*)")
_JS_FUNC = re.compile(_JS_PREFIX + r"(?:async\s+)?function\s+([A-Za-z_$]\w*)")
_GO_FUNC = re.compile(r"^\s*func\s+(?:\(([^)]*)\)\s+)?([A-Za-z_]\w*)")
_GO_TYPE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)")
_RUST_FN = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)")
_RUST_ITEM = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+([A-Za-z_]\w*)")

# 语言是否支持符号级分块（无模式的语言整体为模块 chunk）
_DECL_LANGUAGES = frozenset({"python", "javascript", "go", "rust"})

# 语言注释起始标记（行首剔除后判断）
_COMMENT_PREFIXES = {
    "python": ("#",),
    "rust": ("#", "//", "/*", "*"),
    "javascript": ("//", "/*", "*"),
    "go": ("//", "/*", "*"),
}


@dataclass(frozen=True)
class ChunkRecord:
    """语义单元记录：符号级分块 + 层级元数据；vector 由索引构建时填充"""

    chunk_id: str
    logical_path: str
    start_line: int  # 1 基、闭区间
    end_line: int  # 1 基、闭区间
    text: str
    symbol_type: str  # "function" | "class" | "module"
    symbol_name: str
    parent_symbol: str | None
    language: str | None
    vector: SparseVector | None = None


def infer_language(logical_path: str) -> str | None:
    """按文件扩展名推断语言；未知扩展名返回 None"""
    return _LANGUAGE_BY_EXT.get(Path(logical_path).suffix.lower())


def _is_comment(line: str, language: str | None) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    prefixes = _COMMENT_PREFIXES.get(language or "", ())
    return any(stripped.startswith(p) for p in prefixes)


def _parse_decl(line: str, language: str) -> tuple[int, str, str, str | None] | None:
    """返回 (缩进, 符号名, 类型, go 接收者类型)；非声明行返回 None"""
    indent = len(line) - len(line.lstrip(" \t"))
    if language == "python":
        m = _PY_DECL.match(line)
        if not m:
            return None
        kind = "class" if m.group(1) == "class" else "function"
        return indent, m.group(2), kind, None
    if language == "javascript":
        m = _JS_CLASS.match(line)
        if m:
            return indent, m.group(1), "class", None
        m = _JS_FUNC.match(line)
        if m:
            return indent, m.group(1), "function", None
        return None
    if language == "go":
        m = _GO_TYPE.match(line)
        if m:
            return indent, m.group(1), "class", None
        m = _GO_FUNC.match(line)
        if m:
            receiver = m.group(1)
            parent = None
            if receiver:
                # "(u *User)" → "User"；方法父级即接收者类型
                parent = receiver.strip().split()[-1].lstrip("*")
            return indent, m.group(2), "function", parent
        return None
    if language == "rust":
        m = _RUST_FN.match(line)
        if m:
            return indent, m.group(1), "function", None
        m = _RUST_ITEM.match(line)
        if m:
            return indent, m.group(1), "class", None
        return None
    return None


def _brace_only(line: str) -> bool:
    """花括号收尾行（} 或 } 加注释）不触发缩进回退，归属当前 chunk"""
    stripped = line.strip()
    return stripped == "}" or (stripped.startswith("}") and "//" in stripped)


def chunk_text(
    text: str,
    *,
    logical_path: str,
    chunk_size: int = 200,
    min_chunk_lines: int = 5,
) -> list[ChunkRecord]:
    """将源码按符号切分为语义 chunk（分区：行号单调且拼接等于归一化原文）"""
    language = infer_language(logical_path)
    lines = text.rstrip("\n").split("\n")
    if not lines or all(not ln.strip() for ln in lines):
        return []

    module_name = Path(logical_path).stem

    # 第一遍：扫描声明（含多行签名延续与装饰器归属），计算父级
    decls: dict[int, tuple[int, int, str, str, str | None]] = {}
    stack: list[tuple[int, str]] = []  # (缩进, 符号名)
    sig_open = False
    sig_continuation: set[int] = set()  # 多行签名延续行（含收尾的 "):" 行）
    decorator_lines: set[int] = set()
    for idx, line in enumerate(lines):
        if _is_comment(line, language) or not line.strip():
            continue
        if sig_open:
            sig_continuation.add(idx)
            # 仅当闭合括号多于开启（如收尾的 "):"）才结束签名；无括号或平衡行保持延续
            if line.count(")") > line.count("("):
                sig_open = False
            continue
        if language not in _DECL_LANGUAGES:
            break
        parsed = _parse_decl(line, language)
        if parsed is None:
            continue
        indent, name, kind, receiver = parsed
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = receiver if receiver else (stack[-1][1] if stack else None)
        stack.append((indent, name))
        ext_start = idx
        while ext_start > 0 and lines[ext_start - 1].lstrip().startswith("@"):
            ext_start -= 1
            decorator_lines.add(ext_start)
        decls[idx] = (ext_start, indent, name, kind, parent)
        if line.count("(") > line.count(")"):
            sig_open = True

    # 第二遍：逐行归属（缩进回退结束符号块），得到连续分区
    pieces: list[tuple[int, int, str, str, str | None]] = []
    owner_stack: list[tuple[int, str, str, str | None]] = []  # (缩进, 名, 类型, 父级)
    # 文件头散落代码先归模块，首个声明（或装饰器行）到达时关闭
    current: tuple[int, str, str, str | None] | None = (0, module_name, "module", None)

    def close_piece(end_idx: int) -> None:
        nonlocal current
        if current is not None and end_idx >= current[0]:
            start, name, kind, parent = current
            pieces.append((start, end_idx, kind, name, parent))
        current = None

    for idx, line in enumerate(lines):
        if idx in decorator_lines or idx in sig_continuation:
            continue
        if idx in decls:
            ext_start, indent, name, kind, parent = decls[idx]
            while owner_stack and owner_stack[-1][0] >= indent:
                owner_stack.pop()
            owner_stack.append((indent, name, kind, parent))
            close_piece(ext_start - 1)
            current = (ext_start, name, kind, parent)
            continue
        stripped = line.strip()
        if not stripped or _is_comment(line, language) or _brace_only(line):
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        popped = False
        while owner_stack and owner_stack[-1][0] >= indent:
            owner_stack.pop()
            popped = True
        if popped and current is not None:
            close_piece(idx - 1)
            if owner_stack:
                current = (idx, owner_stack[-1][1], owner_stack[-1][2], owner_stack[-1][3])
            else:
                current = (idx, module_name, "module", None)
    close_piece(len(lines) - 1)

    records: list[ChunkRecord] = []
    for start, end, kind, name, parent in pieces:
        length = end - start + 1
        if length <= chunk_size:
            records.append(
                _make_record(logical_path, start, end, kind, name, parent, language, lines)
            )
            continue
        sub_starts = list(range(start, end + 1, chunk_size))
        sub_pieces = [
            (s, sub_starts[i + 1] - 1 if i + 1 < len(sub_starts) else end)
            for i, s in enumerate(sub_starts)
        ]
        if len(sub_pieces) > 1:
            last_start, last_end = sub_pieces[-1]
            if last_end - last_start + 1 < min_chunk_lines:
                sub_pieces = sub_pieces[:-2] + [(sub_pieces[-2][0], last_end)]
        for s, e in sub_pieces:
            records.append(_make_record(logical_path, s, e, kind, name, parent, language, lines))
    return records


def _make_record(
    logical_path: str,
    start: int,
    end: int,
    kind: str,
    name: str,
    parent: str | None,
    language: str | None,
    lines: list[str],
) -> ChunkRecord:
    text = "\n".join(lines[start : end + 1])
    return ChunkRecord(
        chunk_id=f"{logical_path}:{start + 1}-{end + 1}",
        logical_path=logical_path,
        start_line=start + 1,
        end_line=end + 1,
        text=text,
        symbol_type=kind,
        symbol_name=name,
        parent_symbol=parent,
        language=language,
    )
