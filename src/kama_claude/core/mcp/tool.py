from __future__ import annotations

import asyncio
from typing import Any

from kama_claude.core.mcp.client import (
    McpClient,
    McpServerUnavailableError,
    McpToolDef,
    McpToolError,
)
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.evidence import ToolEvidenceBudget, bound_tool_output


# 将 MCP 工具包装为 BaseTool，使 ToolRegistry 可透明调用
class McpTool(BaseTool):
    params_model = None  # input_schema 来自 MCP tool_def，不使用 pydantic model

    # 初始化 MCP 工具包装器，工具名以 server_name__ 为前缀防止命名冲突
    def __init__(self, client: McpClient, server_name: str, tool_def: McpToolDef) -> None:
        self._client = client
        self._server_name = server_name
        self._tool_def = tool_def
        self.name = f"{server_name}__{tool_def.name}"
        self.description = tool_def.description or f"MCP tool from {server_name}"
        self.input_schema: dict[str, Any] = (
            tool_def.input_schema or {"type": "object", "properties": {}}
        )

    # 调用 MCP server 上的工具，连接不可用或工具执行失败时返回 is_error=True
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        try:
            content = await self._client.call_tool(self._tool_def.name, dict(params))
            raw_truncated_value = getattr(self._client, "last_call_truncated", False)
            raw_truncated = (
                raw_truncated_value if isinstance(raw_truncated_value, bool) else False
            )
            original_size = getattr(self._client, "last_call_original_size", None)
            if not isinstance(original_size, int):
                original_size = None
            receipt = bound_tool_output(
                content,
                budget=ToolEvidenceBudget(
                    producer_max_bytes=256 * 1024,
                    durable_max_bytes=256 * 1024,
                    model_max_chars=8_000,
                ),
                original_size=original_size,
                raw_truncated=raw_truncated,
            )
            return ToolResult(
                content=receipt.to_model_text(),
                raw_truncated=receipt.raw_truncated,
                original_size=receipt.original_size,
                captured_size=receipt.captured_size,
                evidence_ref=receipt.evidence_ref,
            )
        except asyncio.CancelledError:
            raise
        except McpServerUnavailableError:
            return ToolResult(
                content="MCP server is unavailable.",
                is_error=True,
                error_type="execution_error",
            )
        except McpToolError:
            return ToolResult(
                content="MCP tool reported an error.",
                is_error=True,
                error_type="command_failed",
            )
