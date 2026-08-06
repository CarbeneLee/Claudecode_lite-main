from __future__ import annotations

import re
from pathlib import Path

from kama_claude.core.semantic.config import SemanticConfig
from kama_claude.core.semantic.service import SemanticRetrievalService
from kama_claude.core.semantic.tools import SearchSemanticTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_FOOTER_RE = re.compile(
    r"^\[search_semantic\] results=(?P<results>\d+) "
    r"degraded=(?P<degraded>\w+) "
    r"truncated=(?P<truncated>\w+)$"
)


# 构造绑定指定 workspace 与索引目录的 search_semantic 工具（含字面量 fallback）
def _tool(
    workspace: Path,
    index_dir: Path,
    *,
    degradation: str = "literal_fallback",
) -> SearchSemanticTool:
    config = SemanticConfig(index_dir=str(index_dir))
    service = SemanticRetrievalService(config=config, workspace_root=workspace)
    fallback = SearchCodeTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )
    return SearchSemanticTool(service, fallback=fallback, degradation=degradation)


# 从工具输出末行解析精确计数
def _footer(content: str) -> dict[str, int | str]:
    match = _FOOTER_RE.fullmatch(content.splitlines()[-1])
    assert match is not None, content
    values: dict[str, int | str] = {}
    for key, value in match.groupdict().items():
        values[key] = value if key != "results" else int(value)
    return values


# 返回除完整 footer 外的结果记录
def _records(content: str) -> list[str]:
    return content.splitlines()[:-1]
