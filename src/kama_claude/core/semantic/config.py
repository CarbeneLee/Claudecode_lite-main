from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_INDEX_DIR = "~/.kama/semantic"
_MIB = 1024 * 1024
_STRATEGIES = ("lexical", "onnx")
_DEGRADATIONS = ("literal_fallback", "fail_closed")


@dataclass(frozen=True)
class SemanticConfig:
    # 是否启用代码检索；false 时工具直接降级为字面量搜索
    enabled: bool = True
    # embedding 策略："lexical"（内置，零依赖）| "onnx"（预留，加载失败自动降级）
    strategy: str = "lexical"
    # 索引持久化根目录（按 workspace hash 分目录）
    index_dir: str = _DEFAULT_INDEX_DIR
    # 超大符号的行数硬切上限；常规函数/类按符号边界分块
    chunk_size: int = 200
    # 小于该行数的散落代码合并进模块级 chunk
    min_chunk_lines: int = 5
    # 字符 n-gram 阶数（lexical 策略）
    ngram_n: int = 3
    # search 默认返回 top-k 条结果
    default_top_k: int = 10
    # 相似度阈值下限，低于该值的结果不返回
    similarity_threshold: float = 0.10
    # 单文件索引入口上限
    max_index_files: int = 5000
    # 单文件字节上限（超出跳过）
    max_file_bytes: int = 1 * _MIB
    # 全部已索引文件字节累计上限（超出跳过剩余文件）
    total_index_bytes: int = 32 * _MIB
    # 索引不可用时的降级策略："literal_fallback"（回退 search_code + degraded 标记）| "fail_closed"
    degradation: str = "literal_fallback"
    # 查询长度上限（超出直接报错，避免超长向量计算）
    max_query_chars: int = 256

    def __post_init__(self) -> None:
        # 枚举与数值约束在构造时即拒绝非法取值，避免配置错误扩散到运行时
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {self.strategy!r}")
        if self.degradation not in _DEGRADATIONS:
            raise ValueError(
                f"degradation must be one of {_DEGRADATIONS}, got {self.degradation!r}"
            )
        if not 1 <= self.ngram_n <= 6:
            raise ValueError(f"ngram_n must be between 1 and 6, got {self.ngram_n!r}")
        for field, value in (
            ("chunk_size", self.chunk_size),
            ("min_chunk_lines", self.min_chunk_lines),
            ("default_top_k", self.default_top_k),
            ("max_index_files", self.max_index_files),
            ("max_file_bytes", self.max_file_bytes),
            ("total_index_bytes", self.total_index_bytes),
            ("max_query_chars", self.max_query_chars),
        ):
            if value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0.0, 1.0], got {self.similarity_threshold!r}"
            )
