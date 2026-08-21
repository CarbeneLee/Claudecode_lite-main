from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentProfile:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    allowed_subagent_types: list[str] | None = None
    model: str = ""


# 按两级优先级（项目本地 > 用户全局 > 内建）查找并解析角色配置
class AgentProfileLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    # 绑定 canonical workspace，作为项目角色配置的唯一查找根目录
    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve(strict=True)

    # 查找指定角色配置；未找到返回 None
    def load(self, name: str) -> AgentProfile | None:
        for path in self._search_paths(name):
            if path.exists():
                try:
                    profile = self._parse(path, name)
                    return self._apply_builtin_child_constraint(profile, path)
                except Exception:
                    return None
        return None

    # 将 builtin child-type policy 作为同名 custom profile 不可扩大的上界
    def _apply_builtin_child_constraint(
        self,
        profile: AgentProfile,
        source_path: Path,
    ) -> AgentProfile:
        builtin_path = self._BUILTIN_DIR / f"{profile.name}.toml"
        if source_path == builtin_path or not builtin_path.exists():
            return profile
        builtin = self._parse(builtin_path, profile.name)
        maximum = builtin.allowed_subagent_types
        if maximum is None:
            return profile
        profile.allowed_tools = [
            name for name in builtin.allowed_tools if name in profile.allowed_tools
        ]
        requested = profile.allowed_subagent_types
        profile.allowed_subagent_types = (
            []
            if requested is None
            else [name for name in maximum if name in requested]
        )
        return profile

    # 返回 [项目本地, 用户全局, 内建] 路径；load() 返回第一个存在的，项目本地优先级最高
    def _search_paths(self, name: str) -> list[Path]:
        builtin = self._BUILTIN_DIR / f"{name}.toml"
        global_ = Path("~/.kama/agents").expanduser() / f"{name}.toml"
        local = self._workspace_root / ".kama" / "agents" / f"{name}.toml"
        return [local, global_, builtin]

    # 解析 TOML 角色配置文件
    def _parse(self, path: Path, name: str) -> AgentProfile:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        agent = data.get("agent", {})
        if not isinstance(agent, dict):
            raise ValueError("agent profile section must be a table")
        description = agent.get("description", "")
        system_prompt = agent.get("system_prompt", "")
        allowed_tools = agent.get("allowed_tools", [])
        allowed_subagent_types = agent.get("allowed_subagent_types")
        model = agent.get("model", "")
        if not isinstance(description, str) or not isinstance(system_prompt, str):
            raise ValueError("profile description and system_prompt must be strings")
        if not isinstance(allowed_tools, list) or not all(
            isinstance(item, str) for item in allowed_tools
        ):
            raise ValueError("profile allowed_tools must be a list of strings")
        if allowed_subagent_types is not None and (
            not isinstance(allowed_subagent_types, list)
            or not all(isinstance(item, str) for item in allowed_subagent_types)
        ):
            raise ValueError("profile allowed_subagent_types must be a list of strings")
        if not isinstance(model, str):
            raise ValueError("profile model must be a string")
        return AgentProfile(
            name=name,
            description=description,
            system_prompt=system_prompt.strip(),
            allowed_tools=allowed_tools,
            allowed_subagent_types=allowed_subagent_types,
            model=model,
        )
