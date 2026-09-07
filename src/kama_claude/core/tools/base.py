from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # 稳定 error_type taxonomy 定义在 tools.errors，None 会在 invocation 边界归一化
    error_type: str | None = None
    raw_truncated: bool = False
    original_size: int | None = None
    captured_size: int | None = None
    evidence_ref: str | None = None
    # 子代理终态的 bounded structured receipt；compare/repr 不影响 legacy text contract
    terminal_outcome: object | None = field(default=None, compare=False, repr=False)
    # 标记 parent ingress 是否应使用 terminal_outcome 的结构化 receipt 文本
    terminal_receipt: bool = field(default=False, compare=False, repr=False)


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, object]
    params_model: ClassVar[type[BaseModel] | None] = None

    # 执行工具调用，返回结果或错误
    @abstractmethod
    async def invoke(self, params: dict[str, object]) -> ToolResult: ...
