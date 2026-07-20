from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_FOOTER_RE = re.compile(
    r"^\[search_code\] matched_lines=(?P<matched_lines>\d+) "
    r"directory_entries=(?P<directory_entries>\d+) "
    r"visited_directories=(?P<visited_directories>\d+) "
    r"examined_files=(?P<examined_files>\d+) "
    r"examined_bytes=(?P<examined_bytes>\d+) "
    r"skipped_non_text=(?P<skipped_non_text>\d+) "
    r"skipped_large=(?P<skipped_large>\d+) "
    r"skipped_unreadable=(?P<skipped_unreadable>\d+) "
    r"truncated=(?P<truncated>\w+)$"
)


# 构造绑定指定 workspace 的 search_code 工具
def _tool(workspace: Path) -> SearchCodeTool:
    return SearchCodeTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 从搜索输出末行解析精确资源计数
def _footer(content: str) -> dict[str, int | str]:
    match = _FOOTER_RE.fullmatch(content.splitlines()[-1])
    assert match is not None, content
    values: dict[str, int | str] = {}
    for key, value in match.groupdict().items():
        values[key] = value if key == "truncated" else int(value)
    return values


# 构造 search_code 的工具调用块
def _call(params: dict[str, object], uid: str = "search-1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name="search_code", input=params)


# 为 EventBus 注册按发布顺序保存事件的 collector
def _collect_events(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    # 保留事件对象以便检查 lifecycle 顺序和字段
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    return events


# 返回除完整 footer 外的搜索结果记录
def _records(content: str) -> list[str]:
    return content.splitlines()[:-1]
