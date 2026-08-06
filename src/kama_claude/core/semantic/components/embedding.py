from __future__ import annotations

import itertools
import math
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from kama_claude.core.semantic.errors import EmbeddingStrategyUnavailableError


@dataclass(frozen=True)
class SparseVector:
    """稀疏向量：非零维的索引与取值；索引严格递增、值非零（L2 归一化后）"""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have equal length")
        if any(a >= b for a, b in itertools.pairwise(self.indices)):
            raise ValueError("indices must be strictly increasing")
        if any(v == 0.0 for v in self.values):
            raise ValueError("values must be non-zero")


# 英文/数字/下划线连续段为整 token（标识符敏感），CJK 单字成 token（每字成元）
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


class EmbeddingStrategy(Protocol):
    """可插拔检索策略接口（Strategy pattern）；实现须同时提供 is_lexical 标记"""

    @property
    def is_lexical(self) -> bool: ...

    def fit(self, documents: Iterable[str]) -> None: ...
    def embed(self, text: str) -> SparseVector: ...
    def degraded_query(self, query: str) -> bool: ...


@dataclass
class LexicalEmbeddingStrategy:
    """词法向量策略：字符 n-gram + TF=log(1+count) + IDF(按文件频次) + L2 归一化

    - fit 收集文档频次（每个 gram 每篇文档计一次）；空语料/未 fit 时权重退化为 1.0
    - 未见 gram 的 IDF 取 log(N+1)（最高权重，查询侧新词更显著）
    - gram→索引用 crc32 稳定映射，跨进程/跨实例可复现（确定性）
    """

    ngram_n: int = 3
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _n_docs: int = 0

    @property
    def is_lexical(self) -> bool:
        return True

    def fit(self, documents: Iterable[str]) -> None:
        docs = list(documents)
        df: dict[str, int] = {}
        for doc in docs:
            for gram in set(self._extract_grams(doc)):
                df[gram] = df.get(gram, 0) + 1
        n = len(docs)
        self._n_docs = n
        # log(1 + N/(1+df))：永不归零（单文档语料不塌缩），罕见词仍严格高于常见词
        self._idf = {g: math.log(1 + n / (1 + d)) for g, d in df.items()}

    def embed(self, text: str) -> SparseVector:
        counts: dict[str, int] = {}
        for gram in self._extract_grams(text):
            counts[gram] = counts.get(gram, 0) + 1
        if not counts:
            return SparseVector((), ())
        weights = {g: math.log(1 + c) * self._idf_weight(g) for g, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0.0:
            return SparseVector((), ())
        pairs = sorted((zlib.crc32(g.encode("utf-8")), w / norm) for g, w in weights.items())
        return SparseVector(tuple(i for i, _ in pairs), tuple(v for _, v in pairs))

    def degraded_query(self, query: str) -> bool:
        """查询过短（< ngram_n 字符）或提取不到 gram → 建议降级字面量检索"""
        stripped = query.strip()
        return len(stripped) < self.ngram_n or not self._extract_grams(stripped)

    def _idf_weight(self, gram: str) -> float:
        if self._n_docs == 0:
            return 1.0
        return self._idf.get(gram, math.log(1 + self._n_docs))

    def _extract_grams(self, text: str) -> list[str]:
        grams: list[str] = []
        for token in _TOKEN_RE.findall(text.lower()):
            if len(token) <= self.ngram_n:
                grams.append(token)
            else:
                grams.extend(
                    token[i : i + self.ngram_n] for i in range(len(token) - self.ngram_n + 1)
                )
        return grams


def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    """余弦相似度：双指针归并稀疏点积，除以模长积（容忍未归一化输入）"""
    if not a.indices or not b.indices:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values))
    norm_b = math.sqrt(sum(v * v for v in b.values))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = 0.0
    i = j = 0
    while i < len(a.indices) and j < len(b.indices):
        if a.indices[i] == b.indices[j]:
            dot += a.values[i] * b.values[j]
            i += 1
            j += 1
        elif a.indices[i] < b.indices[j]:
            i += 1
        else:
            j += 1
    return dot / (norm_a * norm_b)


def create_embedding_strategy(strategy: str, *, ngram_n: int = 3) -> EmbeddingStrategy:
    """策略工厂：lexical 直接可用；onnx 为预留位（导入失败抛可降级异常）"""
    if strategy == "lexical":
        return LexicalEmbeddingStrategy(ngram_n=ngram_n)
    if strategy == "onnx":
        try:
            _import_onnxruntime()
        except ImportError as exc:
            raise EmbeddingStrategyUnavailableError(
                "onnx strategy requires onnxruntime; install it or switch "
                "semantic.strategy back to 'lexical'"
            ) from exc
        raise NotImplementedError("onnx strategy is reserved; not implemented yet")
    raise ValueError(f"unknown embedding strategy: {strategy!r} (expected 'lexical' or 'onnx')")


def _import_onnxruntime() -> None:
    import onnxruntime  # type: ignore[import-not-found]  # noqa: F401
