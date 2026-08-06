from __future__ import annotations


class SemanticError(Exception):
    """代码检索错误基类：携带可选详情，供错误分类与降级决策使用"""

    def __init__(self, message: str, *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


class IndexUnavailableError(SemanticError):
    """索引目录不可用/索引构建失败（磁盘、权限、扫描异常）"""


class EmbeddingStrategyUnavailableError(SemanticError):
    """配置的 embedding 策略加载失败（如 onnx 后端未安装），可降级回 lexical"""


class IndexCorruptedError(SemanticError):
    """索引文件损坏（version 不符 / JSON 解析失败），需要全量重建"""
