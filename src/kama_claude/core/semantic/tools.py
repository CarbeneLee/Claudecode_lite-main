"""search_semantic 工具：语义检索入口 + 索引不可用时的降级策略

默认 literal_fallback：索引构建失败时透明降级到 search_code 字面量检索，
结果末尾附加 [search_semantic] degraded=literal_fallback 标记；
fail_closed：返回 is_error + 稳定错误码 semantic_index_unavailable。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kama_claude.core.semantic.errors import IndexUnavailableError
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.search_code import (
    _MAX_QUERY_CHARS,
    MAX_OUTPUT_BYTES,
    _contains_frozen_control,
)

if TYPE_CHECKING:
    from kama_claude.core.semantic.service import SearchResult, SemanticRetrievalService
    from kama_claude.core.tools.builtin.search_code import SearchCodeTool

_MAX_TOP_K = 50

_DEGRADATIONS = frozenset({"literal_fallback", "fail_closed"})


class SearchSemanticParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    top_k: int | None = Field(default=None, ge=1, le=_MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        if not value or value.isspace():
            raise ValueError("query must contain a non-whitespace character")
        if len(value) > _MAX_QUERY_CHARS:
            raise ValueError("query is too long")
        if _contains_frozen_control(value):
            raise ValueError("query contains unsupported control characters")
        return value


class SearchSemanticTool(BaseTool):
    """自然语言/标识符代码检索：命中输出层级展示 + 稳定 footer"""

    params_model = SearchSemanticParams
    name = "search_semantic"
    description = (
        "Search the workspace codebase by meaning or identifier similarity using "
        "a local retrieval index. Returns ranked chunk locations with scores, "
        "e.g. 'auth.py:120-160 [score=0.87] UserManager -> reset_password()'. "
        "For exact literal matches prefer search_code."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_QUERY_CHARS,
                "description": "Natural-language or identifier query to retrieve.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_TOP_K,
                "description": "Maximum results to return (default from config).",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        service: SemanticRetrievalService,
        *,
        fallback: SearchCodeTool,
        degradation: str = "literal_fallback",
    ) -> None:
        if degradation not in _DEGRADATIONS:
            raise ValueError(
                f"unknown degradation mode: {degradation!r} "
                f"(expected one of {sorted(_DEGRADATIONS)})"
            )
        self._service = service
        self._fallback = fallback
        self._degradation = degradation

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = SearchSemanticParams.model_validate(params)
        try:
            results = await self._service.search(p.query, top_k=p.top_k)
        except IndexUnavailableError as exc:
            return await self._on_unavailable(p.query, exc)
        degraded = self._service.degraded_query(p.query)
        content = self._format_output(results, degraded="query" if degraded else "none")
        return ToolResult(content=content)

    async def _on_unavailable(self, query: str, exc: IndexUnavailableError) -> ToolResult:
        if self._degradation == "fail_closed":
            return ToolResult(
                content=f"semantic index unavailable: {exc}",
                is_error=True,
                error_type="semantic_index_unavailable",
            )
        result = await self._fallback.invoke({"query": query})
        if result.is_error:
            return result
        return ToolResult(content=result.content + "\n[search_semantic] degraded=literal_fallback")

    # ---- 输出组装 ----

    def _format_output(self, results: list[SearchResult], degraded: str) -> str:
        records = [self._format_result(r) for r in results]
        footer = self._footer(len(records), degraded, "none")
        complete = "\n".join((*records, footer))
        if len(complete.encode("utf-8")) <= MAX_OUTPUT_BYTES:
            return complete
        accepted: list[str] = []
        for record in records:
            candidate = "\n".join((*accepted, record, footer))
            if len(candidate.encode("utf-8")) > MAX_OUTPUT_BYTES:
                break
            accepted.append(record)
        footer = self._footer(len(accepted), degraded, "output_limit")
        content = "\n".join((*accepted, footer))
        assert len(content.encode("utf-8")) <= MAX_OUTPUT_BYTES
        return content

    def _format_result(self, result: SearchResult) -> str:
        record = result.record
        location = f"{record.logical_path}:{record.start_line}-{record.end_line}"
        score = f"[score={result.score:.2f}]"
        symbol = record.symbol_name
        if record.parent_symbol:
            symbol = f"{record.parent_symbol} → {symbol}"
        if record.symbol_type == "function":
            symbol = f"{symbol}()"
        elif record.symbol_type == "class":
            symbol = f"class {symbol}"
        elif record.symbol_type == "module":
            symbol = f"module {symbol}"
        return f"{location} {score} {symbol}"

    @staticmethod
    def _footer(results: int, degraded: str, truncated: str) -> str:
        return f"[search_semantic] results={results} degraded={degraded} truncated={truncated}"
