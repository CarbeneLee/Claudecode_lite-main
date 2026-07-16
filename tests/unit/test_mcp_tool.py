from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kama_claude.core.mcp.client import McpClient, McpServerUnavailableError, McpToolDef
from kama_claude.core.mcp.tool import McpTool


def _make_tool(
    tool_name: str = "read_file",
    server_name: str = "filesystem",
) -> tuple[McpTool, AsyncMock]:
    client = AsyncMock(spec=McpClient)
    tool_def = McpToolDef(
        name=tool_name,
        description=f"Read a file via {server_name}",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    tool = McpTool(client, server_name, tool_def)
    return tool, client


# 功能：McpTool.invoke 应调用 client.call_tool 并将返回值封装为 ToolResult
# 设计：mock client.call_tool 返回固定字符串，验证 ToolResult.content 一致
@pytest.mark.asyncio
async def test_invoke_calls_mcp_client() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(return_value="file content here")
    result = await tool.invoke({"path": "/tmp/test.txt"})
    assert not result.is_error
    assert result.content == "file content here"
    client.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})


# 功能：工具名应以 {server_name}__ 为前缀防止命名冲突
# 设计：验证 McpTool.name 格式为 "filesystem__read_file"
def test_tool_name_prefixed() -> None:
    tool, _ = _make_tool("read_file", "filesystem")
    assert tool.name == "filesystem__read_file"


# 功能：client 抛 McpServerUnavailableError 时返回固定安全 execution_error
# 设计：异常携带 secret，断言 direct ToolResult 不回显 server 名、异常文本或内部状态
@pytest.mark.asyncio
async def test_unavailable_returns_error() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(
        side_effect=McpServerUnavailableError("/private/socket token=secret")
    )
    result = await tool.invoke({"path": "/tmp/x.txt"})
    assert result.is_error
    assert result.error_type == "execution_error"
    assert result.content == "MCP server is unavailable."
    assert "secret" not in result.content
    assert "filesystem" not in result.content


# 功能：client 抛其他异常时 McpTool direct invoke 原样传播
# 设计：保存异常对象并断言身份相同，证明 generic exception 留给中央 classifier
@pytest.mark.asyncio
async def test_runtime_error_propagates() -> None:
    tool, client = _make_tool()
    error = RuntimeError("unexpected failure")
    client.call_tool = AsyncMock(side_effect=error)

    with pytest.raises(RuntimeError) as exc_info:
        await tool.invoke({"path": "/tmp/y.txt"})

    assert exc_info.value is error


# 功能：input_schema 应直接使用 MCP tool_def 中的 schema，而非 pydantic model
# 设计：验证 params_model 为 None，input_schema 与 tool_def.input_schema 一致
def test_input_schema_from_tool_def() -> None:
    tool, _ = _make_tool()
    assert McpTool.params_model is None
    assert "path" in tool.input_schema.get("properties", {})
