from __future__ import annotations

import pytest

from kama_claude.core.semantic.components.chunker import (
    ChunkRecord,
    chunk_text,
    infer_language,
)

# 行号表（1 基）：
#  1  """module docstring"""
#  2  import os
#  3  (空)
#  4  CONST = 3
#  5  (空)
#  6  @decorator
#  7  def top_level(x):
#  8      return x + CONST
#  9  (空)
# 10  class UserManager:
# 11      """manages users"""
# 12  (空)
# 13      def __init__(self, name):
# 14          self.name = name
# 15  (空)
# 16      @property
# 17      def display(self):
# 18          return self.name.upper()
# 19  (空)
# 20  class Outer:
# 21      class Inner:
# 22          def helper(self):
# 23              return 1
# 24  (空)
# 25  trailing = 99
PY_SAMPLE = """\
\"\"\"module docstring\"\"\"
import os

CONST = 3

@decorator
def top_level(x):
    return x + CONST

class UserManager:
    \"\"\"manages users\"\"\"

    def __init__(self, name):
        self.name = name

    @property
    def display(self):
        return self.name.upper()

class Outer:
    class Inner:
        def helper(self):
            return 1

trailing = 99
"""

# 行号表：
#  1  // header comment
#  2  const VERSION = 1;
#  3  (空)
#  4  function greet(name) {
#  5    return "hi " + name;
#  6  }
#  7  (空)
#  8  class Store {
#  9    constructor(items) {
# 10      this.items = items;
# 11    }
# 12    find(id) {
# 13      return this.items[id];
# 14    }
# 15  }
JS_SAMPLE = """\
// header comment
const VERSION = 1;

function greet(name) {
  return "hi " + name;
}

class Store {
  constructor(items) {
    this.items = items;
  }
  find(id) {
    return this.items[id];
  }
}
"""

# 行号表：
#  1  package main
#  2  (空)
#  3  type User struct {
#  4      Name string
#  5  }
#  6  (空)
#  7  func NewUser(name string) *User {
#  8      return &User{Name: name}
#  9  }
# 10  (空)
# 11  func (u *User) Display() string {
# 12      return u.Name
# 13  }
GO_SAMPLE = """\
package main

type User struct {
    Name string
}

func NewUser(name string) *User {
    return &User{Name: name}
}

func (u *User) Display() string {
    return u.Name
}
"""

MD_SAMPLE = """\
# Title

Some **bold** text
"""

ALL_SAMPLES = [
    ("auth.py", PY_SAMPLE),
    ("app.js", JS_SAMPLE),
    ("user.go", GO_SAMPLE),
    ("README.md", MD_SAMPLE),
    ("data.xyz", "hello world\n"),
    ("misc.txt", "line one\nline two\n"),
    ("comment.py", "# def fake():\n# class Fake:\ndef real():\n    pass\n"),
    ("one.py", "x = 1\n"),
    ("func.py", "def f():\n    pass\n"),
    ("big.py", "def big():\n" + "\n".join(f"    pass  # {i}" for i in range(14)) + "\n"),
]


def _summary(
    records: list[ChunkRecord],
) -> list[tuple[str, str, int, int, str | None]]:
    return [
        (r.symbol_type, r.symbol_name, r.start_line, r.end_line, r.parent_symbol)
        for r in records
    ]


# 功能：验证按扩展名推断语言；未知扩展名返回 None
# 设计：参数化覆盖 .py/.js/.ts/.go/.md/.txt 与 Makefile/.bin 两类未知
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/auth.py", "python"),
        ("app.js", "javascript"),
        ("lib/util.ts", "typescript"),
        ("main.go", "go"),
        ("README.md", "markdown"),
        ("notes.txt", "text"),
        ("Makefile", None),
        ("data.bin", None),
    ],
)
def test_infer_language(path: str, expected: str | None) -> None:
    assert infer_language(path) == expected


# 功能：验证 Python 符号级分块——函数/类为独立 chunk，含嵌套层级与模块散落代码
# 设计：用带装饰器、嵌套类、类属性、尾块的样例，断言精确的 (类型, 名, 行号, 父级) 序列
def test_python_symbol_chunking() -> None:
    chunks = chunk_text(PY_SAMPLE, logical_path="auth.py")

    assert _summary(chunks) == [
        ("module", "auth", 1, 5, None),
        ("function", "top_level", 6, 9, None),
        ("class", "UserManager", 10, 12, None),
        ("function", "__init__", 13, 15, "UserManager"),
        ("function", "display", 16, 19, "UserManager"),
        ("class", "Outer", 20, 20, None),
        ("class", "Inner", 21, 21, "Outer"),
        ("function", "helper", 22, 24, "Inner"),
        ("module", "auth", 25, 25, None),
    ]


# 功能：验证装饰器行归属声明符号（含 @property 场景）且随符号整体移动
# 设计：断言 chunk 起点为装饰器首行、文本以 @ 开头，确认装饰器未被切到前一个 chunk
def test_decorator_lines_owned_by_following_symbol() -> None:
    chunks = chunk_text(PY_SAMPLE, logical_path="auth.py")
    top_level = next(c for c in chunks if c.symbol_name == "top_level")
    display = next(c for c in chunks if c.symbol_name == "display")

    assert top_level.start_line == 6
    assert top_level.text.startswith("@decorator\n")
    assert display.start_line == 16
    # 装饰器行保留原文缩进（concat 不变量），按 lstrip 后判断归属
    assert display.text.lstrip().startswith("@property\n")


# 功能：验证多行签名（括号跨行）不产生假符号，整段签名归属同一函数
# 设计：三行参数列表的 def，断言仅一个函数 chunk 且包含全部签名行
def test_multiline_signature_single_chunk() -> None:
    src = "def fetch_user(\n    user_id: int,\n    *,\n    refresh: bool = False,\n):\n    return user_id\n"

    chunks = chunk_text(src, logical_path="fetch.py")

    assert _summary(chunks) == [("function", "fetch_user", 1, 6, None)]
    assert chunks[0].text.startswith("def fetch_user(\n    user_id: int,")


# 功能：验证超大符号按行数硬切（无空行也不合并），子块继承符号元数据
# 设计：15 行函数以 chunk_size=6 硬切为 [1-6][7-15]（尾块 3 行 < min 5 并入前块）
def test_oversized_symbol_hard_split() -> None:
    src = "def big():\n" + "\n".join(f"    pass  # {i}" for i in range(14))

    chunks = chunk_text(src, logical_path="big.py", chunk_size=6, min_chunk_lines=5)

    assert _summary(chunks) == [
        ("function", "big", 1, 6, None),
        ("function", "big", 7, 15, None),
    ]
    assert all(c.symbol_type == "function" and c.parent_symbol is None for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 2


# 功能：验证尾块行数达到 min_chunk_lines 时保持独立子块
# 设计：同一样本 min_chunk_lines=3，尾块 3 行 >= 3 不并入 → 3 个子块
def test_hard_split_tail_kept_when_above_min() -> None:
    src = "def big():\n" + "\n".join(f"    pass  # {i}" for i in range(14))

    chunks = chunk_text(src, logical_path="big.py", chunk_size=6, min_chunk_lines=3)

    assert _summary(chunks) == [
        ("function", "big", 1, 6, None),
        ("function", "big", 7, 12, None),
        ("function", "big", 13, 15, None),
    ]


# 功能：验证 JavaScript 函数/类识别；类方法不单列（v1 限制，并入类 chunk）
# 设计：断言 module/function/class 三段边界；类 chunk 包含构造器与方法文本
def test_javascript_chunking() -> None:
    chunks = chunk_text(JS_SAMPLE, logical_path="app.js")

    assert _summary(chunks) == [
        ("module", "app", 1, 3, None),
        ("function", "greet", 4, 7, None),
        ("class", "Store", 8, 15, None),
    ]
    assert "constructor(items)" in chunks[-1].text
    assert "find(id)" in chunks[-1].text


# 功能：验证 Go 函数/类型识别；接收者方法父级为接收者类型
# 设计：断言 type User/NewUser/Display 三段边界，Display 的 parent 为接收者 User
def test_go_chunking_with_receiver_parent() -> None:
    chunks = chunk_text(GO_SAMPLE, logical_path="user.go")

    assert _summary(chunks) == [
        ("module", "user", 1, 2, None),
        ("class", "User", 3, 6, None),
        ("function", "NewUser", 7, 10, None),
        ("function", "Display", 11, 13, "User"),
    ]


# 功能：验证无符号语言（markdown）整体为单个模块 chunk
# 设计：md 全文应为一个 module chunk，语言标记为 markdown
def test_markdown_single_module_chunk() -> None:
    chunks = chunk_text(MD_SAMPLE, logical_path="README.md")

    assert _summary(chunks) == [("module", "README", 1, 3, None)]
    assert chunks[0].language == "markdown"


# 功能：验证未知扩展名按单模块 chunk 处理且语言为 None
# 设计：.xyz 文件不匹配任何语言，全文一个 module chunk
def test_unknown_language_single_module_chunk() -> None:
    chunks = chunk_text("hello world\n", logical_path="data.xyz")

    assert _summary(chunks) == [("module", "data", 1, 1, None)]
    assert chunks[0].language is None


# 功能：验证注释行（# 开头）不产生假符号
# 设计：注释中的 def/class 保持模块代码，仅真实 def 形成函数 chunk
def test_comment_lines_not_treated_as_decls() -> None:
    src = "# def fake():\n# class Fake:\ndef real():\n    pass\n"

    chunks = chunk_text(src, logical_path="comment.py")

    assert _summary(chunks) == [
        ("module", "comment", 1, 2, None),
        ("function", "real", 3, 4, None),
    ]


# 功能：验证空文件与纯空白文件不产生 chunk
# 设计：参数化空串/仅换行/空白行组合，全部应返回空列表
@pytest.mark.parametrize(
    "src",
    ["", "\n", "\n\n\n", "  \n", "  \n\t\n", "\t\t\n"],
)
def test_empty_and_whitespace_files_yield_no_chunks(src: str) -> None:
    assert chunk_text(src, logical_path="blank.py") == []


# 功能：验证单行文件与单行符号的正常分块
# 设计：单行赋值 → module 1-1；单行 def → function 1-1
@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("x = 1", [("module", "one", 1, 1, None)]),
        ("def f(): pass", [("function", "f", 1, 1, None)]),
    ],
)
def test_single_line_files(src: str, expected: list[tuple[str, str, int, int, str | None]]) -> None:
    logical_path = "one.py" if "module" in expected[0][0] else "f.py"

    assert _summary(chunk_text(src, logical_path=logical_path)) == expected


# 功能：验证强不变量——chunks 拼接（\n 连接）等于归一化原文
# 设计：参数化全部样例（含尾部换行输入），断言文本级拼接与原文逐字符一致
@pytest.mark.parametrize(("logical_path", "src"), ALL_SAMPLES)
def test_concat_invariant(logical_path: str, src: str) -> None:
    chunks = chunk_text(src, logical_path=logical_path)

    assert "\n".join(c.text for c in chunks) == src.rstrip("\n")


# 功能：验证强不变量——行号单调且完全覆盖（首个 1 基、相邻衔接、末行等于行数）
# 设计：参数化全部样例，断言分区性质：start<=end、后块 start == 前块 end+1
@pytest.mark.parametrize(("logical_path", "src"), ALL_SAMPLES)
def test_partition_invariant(logical_path: str, src: str) -> None:
    chunks = chunk_text(src, logical_path=logical_path)
    line_count = len(src.rstrip("\n").split("\n"))

    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == line_count
    for cur, nxt in zip(chunks, chunks[1:]):
        assert cur.start_line <= cur.end_line
        assert nxt.start_line == cur.end_line + 1


# 功能：验证 chunk_id 在同一文件内唯一
# 设计：参数化全部样例，断言 id 集合大小与 chunk 数一致
@pytest.mark.parametrize(("logical_path", "src"), ALL_SAMPLES)
def test_chunk_ids_unique(logical_path: str, src: str) -> None:
    chunks = chunk_text(src, logical_path=logical_path)

    assert len({c.chunk_id for c in chunks}) == len(chunks)


# 功能：验证尾部换行输入被归一化（行号与文本一致，不产生幽灵空行）
# 设计：带 \n 结尾的多行 def，断言 chunk 文本不含尾部空行且末行号正确
def test_trailing_newline_normalization() -> None:
    chunks = chunk_text("def f():\n    pass\n", logical_path="f.py")

    assert chunks[0].text == "def f():\n    pass"
    assert chunks[0].end_line == 2
