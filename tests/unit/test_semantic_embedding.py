from __future__ import annotations

import math

import pytest

from kama_claude.core.semantic.components import embedding as embedding_mod
from kama_claude.core.semantic.components.embedding import (
    LexicalEmbeddingStrategy,
    SparseVector,
    cosine_similarity,
    create_embedding_strategy,
)
from kama_claude.core.semantic.errors import EmbeddingStrategyUnavailableError

# 近义/无关 对照样本：near_b 与 near_a 共享多个字符 n-gram，far 与 near_a 几乎无共享
NEAR_FAR_ROWS = [
    ("create_user", "create_user_table", "send_email"),
    ("load_config", "read_config", "send_email"),
]


# 功能：验证 SparseVector 拒绝非法向量（长度不等/非递增/乱序/零值）
# 设计：参数化 4 类非法输入，全部应抛 ValueError
@pytest.mark.parametrize(
    ("indices", "values"),
    [
        ((1,), (1.0, 2.0)),
        ((1, 1), (1.0, 2.0)),
        ((2, 1), (1.0, 2.0)),
        ((1,), (0.0,)),
        ((1,), (-0.0,)),
    ],
)
def test_sparse_vector_rejects_invalid(
    indices: tuple[int, ...], values: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        SparseVector(indices, values)


# 功能：验证合法 SparseVector 原样保留
# 设计：单例冒烟，索引/取值与构造输入一致
def test_sparse_vector_accepts_valid() -> None:
    v = SparseVector((1, 3, 7), (0.5, 0.25, 0.125))

    assert v.indices == (1, 3, 7)
    assert v.values == (0.5, 0.25, 0.125)


# 功能：验证 cosine 的基本性质——相同向量为 1、空向量为 0、正交为 0
# 设计：直接构造 SparseVector 断言三种退化输入
def test_cosine_identical_and_orthogonal() -> None:
    a = SparseVector((1, 2, 3), (0.5, 0.5, 0.5))

    assert cosine_similarity(a, a) == pytest.approx(1.0)
    b = SparseVector((4, 5), (1.0, 1.0))
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_empty_vectors() -> None:
    empty = SparseVector((), ())
    nonempty = SparseVector((1,), (1.0,))

    assert cosine_similarity(empty, empty) == 0.0
    assert cosine_similarity(empty, nonempty) == 0.0
    assert cosine_similarity(nonempty, empty) == 0.0


# 功能：验证 cosine 对未归一化输入按模长归一（结果仍是 1.0）
# 设计：同索引不同模长的两个向量，点积除以模长积后应为 1
def test_cosine_handles_unnormalized_inputs() -> None:
    a = SparseVector((1,), (3.0,))
    b = SparseVector((1,), (4.0,))

    assert cosine_similarity(a, b) == pytest.approx(1.0)


# 功能：验证稀疏归并点积正确性（双指针只累加共享索引）
# 设计：交错索引手工计算点积/模长，断言与公式值一致
def test_cosine_sparse_merge() -> None:
    a = SparseVector((1, 3, 5), (1.0, 2.0, 3.0))
    b = SparseVector((2, 3, 5, 7), (4.0, 5.0, 6.0, 8.0))

    expected = (2 * 5 + 3 * 6) / math.sqrt(14 * 141)
    assert cosine_similarity(a, b) == pytest.approx(expected)


# 功能：验证 n-gram 近义/无关分离性质——近义对相似度高、无关对相似度低
# 设计：参数化 ngram_n(2/3/4) × 两组近义/无关样本，断言阈值与相对分离
@pytest.mark.parametrize("ngram_n", [2, 3, 4])
@pytest.mark.parametrize(("near_a", "near_b", "far"), NEAR_FAR_ROWS)
def test_near_far_separation(ngram_n: int, near_a: str, near_b: str, far: str) -> None:
    s = LexicalEmbeddingStrategy(ngram_n=ngram_n)

    cos_near = cosine_similarity(s.embed(near_a), s.embed(near_b))
    cos_far = cosine_similarity(s.embed(near_a), s.embed(far))

    assert cos_near > 0.4
    assert cos_far < 0.3
    assert cos_near > cos_far * 2


# 功能：验证中文每字成元——中文短语按单字 gram 匹配，与英文 token 零共享
# 设计：中文查询 vs 中文超集文档应显著高于 vs 纯英文文档
def test_chinese_chars_match_as_unigrams() -> None:
    s = LexicalEmbeddingStrategy()

    cos_chinese = cosine_similarity(s.embed("重置密码"), s.embed("重置密码失败重试"))
    cos_english = cosine_similarity(s.embed("重置密码"), s.embed("hello world"))

    assert cos_chinese > 0.3
    assert cos_chinese > cos_english


# 功能：验证确定性——同输入同输出，且跨实例一致（crc32 稳定，不受 hash 随机化影响）
# 设计：同实例两次 embed 相等；新建实例 embed 相等
def test_embed_deterministic() -> None:
    s = LexicalEmbeddingStrategy()

    a = s.embed("reset_password retry 密码")
    b = s.embed("reset_password retry 密码")

    assert a == b
    assert a == LexicalEmbeddingStrategy().embed("reset_password retry 密码")


# 功能：验证 IDF 按文件频次——罕见词权重严格高于常见词
# 设计：'quick'(df=1) vs 'the'(df=2) 两文档语料；查询含两词的文档时罕见词贡献更大
def test_idf_rare_words_weight_higher_than_common() -> None:
    s = LexicalEmbeddingStrategy()
    s.fit(["the quick brown fox jumps over the lazy dog", "the dog barks"])

    assert s.embed("quick") != s.embed("the")
    cos_rare = cosine_similarity(s.embed("quick"), s.embed("quick the"))
    cos_common = cosine_similarity(s.embed("the"), s.embed("quick the"))
    assert cos_rare > cos_common


# 功能：验证 L2 归一化——embed 输出模长为 1（含未见 gram 的 idf 回退）
# 设计：语料 fit 后多个查询（含未见词）的向量平方和均为 1
def test_embed_vectors_are_l2_normalized() -> None:
    s = LexicalEmbeddingStrategy()
    s.fit(["alpha beta gamma"])

    for text in ["alpha", "alpha beta", "gamma delta"]:
        v = s.embed(text)
        assert sum(x * x for x in v.values) == pytest.approx(1.0, rel=1e-6)


# 功能：验证 TF=log(1+count) 的精确权重——重复词对数饱和而非线性放大
# 设计：'x x y' vs 'x y y' 的余弦有闭式解 2·log3·log2/(log3²+log2²)（线性 TF 会得 0.8，
#       log 饱和得 0.9026）；另断言单 gram 重复向量方向不变
def test_tf_log_weighting_exact_value() -> None:
    s = LexicalEmbeddingStrategy()

    q = s.embed("x x y")
    d = s.embed("x y y")
    l3, l2 = math.log(3), math.log(2)
    expected = 2 * l3 * l2 / (l3 * l3 + l2 * l2)
    assert cosine_similarity(q, d) == pytest.approx(expected, rel=1e-9)

    assert s.embed("abc abc") == s.embed("abc")


# 功能：验证空语料 fit 与未 fit 直接 embed 均不崩且产出有效向量
# 设计：fit([]) 后 embed 非空；全新实例 embed 非空且值全部有限
def test_fit_empty_corpus_and_unfitted_strategy_work() -> None:
    s = LexicalEmbeddingStrategy()
    s.fit([])
    assert s.embed("hello").indices

    fresh = LexicalEmbeddingStrategy()
    v = fresh.embed("hello world")
    assert v.indices
    assert all(math.isfinite(x) for x in v.values)


# 功能：验证无 gram 文本产出空向量
# 设计：参数化空串/纯空白/纯标点，全部返回空向量
@pytest.mark.parametrize("text", ["", "   ", "!!!", "。。。", "!!?!"])
def test_embed_no_grams_yields_empty_vector(text: str) -> None:
    assert LexicalEmbeddingStrategy().embed(text) == SparseVector((), ())


# 功能：验证超短/无有效内容查询被标记为退化（建议字面量降级）
# 设计：参数化——短于 ngram_n 或提取不到 gram 的查询为 True，正常查询为 False
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", True),
        ("   ", True),
        ("ab", True),
        ("密", True),
        ("密码", True),
        ("!!!", True),
        ("reset", False),
        ("重置密码", False),
        ("create_user", False),
    ],
)
def test_degraded_query_flags_short_or_gramless_queries(
    query: str, expected: bool
) -> None:
    assert LexicalEmbeddingStrategy().degraded_query(query) is expected


# 功能：验证工厂默认创建 lexical 策略并透传 ngram_n
# 设计：断言类型标记与配置字段
def test_factory_lexical() -> None:
    s = create_embedding_strategy("lexical", ngram_n=4)

    assert s.is_lexical is True
    assert s.ngram_n == 4


# 功能：验证未知策略名抛 ValueError 且消息含可选值
# 设计：bogus 名触发校验错误
def test_factory_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="lexical.*onnx"):
        create_embedding_strategy("bogus")


# 功能：验证 onnx 后端加载失败抛 EmbeddingStrategyUnavailableError（fail-open 信号）
# 设计：monkeypatch 导入函数抛 ImportError，工厂应上抛语义化异常
def test_factory_onnx_import_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import() -> None:
        raise ImportError("no onnxruntime")

    monkeypatch.setattr(embedding_mod, "_import_onnxruntime", fail_import)

    with pytest.raises(EmbeddingStrategyUnavailableError):
        create_embedding_strategy("onnx")


# 功能：验证 onnx 成功导入时工厂明确未实现（预留位）
# 设计：monkeypatch 导入成功，工厂应抛 NotImplementedError
def test_factory_onnx_reserved_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_mod, "_import_onnxruntime", lambda: None)

    with pytest.raises(NotImplementedError):
        create_embedding_strategy("onnx")
