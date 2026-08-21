from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.agents.loader import AgentProfileLoader


# 功能：内建 planner 角色配置应能被 AgentProfileLoader 加载
# 设计：直接调用 load("planner")，验证关键字段非空
def test_builtin_planner_found(tmp_path: Path) -> None:
    loader = AgentProfileLoader(tmp_path.resolve())
    profile = loader.load("planner")
    assert profile is not None
    assert profile.name == "planner"
    assert profile.system_prompt != ""
    assert "read_file" in profile.allowed_tools or len(profile.allowed_tools) > 0


# 功能：内建三种角色均可加载
# 设计：参数化测试所有内建角色名
@pytest.mark.parametrize("role", ["planner", "executor", "reviewer", "explorer"])
def test_all_builtin_roles_found(role: str, tmp_path: Path) -> None:
    loader = AgentProfileLoader(tmp_path.resolve())
    profile = loader.load(role)
    assert profile is not None, f"builtin role '{role}' not found"
    assert profile.allowed_tools  # 每个内建角色都有 allowed_tools
    assert "search_code" in profile.allowed_tools


# 功能：验证 builtin Planner 只允许 Explorer child 且 Explorer 自身不可继续派生
# 设计：读取真实 builtin profiles 并断言 profile-level child type contract，不检查 TOML 文本
def test_builtin_planner_and_explorer_child_type_contract(tmp_path: Path) -> None:
    loader = AgentProfileLoader(tmp_path.resolve())
    planner = loader.load("planner")
    explorer = loader.load("explorer")

    assert planner is not None
    assert explorer is not None
    assert planner.allowed_subagent_types == ["explorer"]
    assert "spawn_agent" in planner.allowed_tools
    assert "planner_decision_submit" in planner.allowed_tools
    assert "task_create" not in planner.allowed_tools
    assert "task_update" not in planner.allowed_tools
    assert explorer.allowed_subagent_types == []
    assert "spawn_agent" not in explorer.allowed_tools
    assert "architecture_slice_submit" in explorer.allowed_tools


# 功能：未知角色名应返回 None
# 设计：查找不存在的角色，断言返回 None 而非抛异常
def test_unknown_role_returns_none(tmp_path: Path) -> None:
    loader = AgentProfileLoader(tmp_path.resolve())
    result = loader.load("nonexistent_role_xyz")
    assert result is None


# 功能：TOML 角色配置文件应被正确解析
# 设计：写入临时 TOML 文件，通过 _parse 解析并验证所有字段
def test_toml_parsed(tmp_path: Path) -> None:
    content = """\
[agent]
description = "测试角色"
system_prompt = "你是测试助手。"
allowed_tools = ["read_file", "bash"]
model = "claude-sonnet-4-6"
"""
    p = tmp_path / "tester.toml"
    p.write_text(content, encoding="utf-8")
    loader = AgentProfileLoader(tmp_path.resolve())
    profile = loader._parse(p, "tester")
    assert profile.name == "tester"
    assert profile.description == "测试角色"
    assert profile.system_prompt == "你是测试助手。"
    assert "read_file" in profile.allowed_tools
    assert "bash" in profile.allowed_tools
    assert profile.model == "claude-sonnet-4-6"


# 功能：项目本地角色配置应覆盖内建同名配置
# 设计：将 daemon cwd 切到第三方目录，显式绑定 workspace 并断言加载项目版本
def test_project_overrides_builtin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    daemon_cwd = tmp_path / "daemon"
    local_agents = workspace / ".kama" / "agents"
    local_agents.mkdir(parents=True)
    daemon_cwd.mkdir()
    (local_agents / "planner.toml").write_text(
        '[agent]\ndescription = "local planner"\nsystem_prompt = "local prompt"\n'
        'allowed_tools = ["list_dir"]\nmodel = ""\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(daemon_cwd)
    loader = AgentProfileLoader(workspace.resolve())
    profile = loader.load("planner")
    assert profile is not None
    assert profile.description == "local planner"
    assert "list_dir" in profile.allowed_tools


# 功能：验证同名项目 profile 在 workspace A/B 中分别解析且不串扰
# 设计：为两个 workspace 写入不同 planner 配置，分别构造 loader 并比较 system_prompt
def test_profile_loaders_isolate_project_overrides(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    for workspace, prompt in ((workspace_a, "prompt-a"), (workspace_b, "prompt-b")):
        agents = workspace / ".kama" / "agents"
        agents.mkdir(parents=True)
        (agents / "planner.toml").write_text(
            '[agent]\ndescription = "local"\n'
            f'system_prompt = "{prompt}"\nallowed_tools = ["read_file"]\n',
            encoding="utf-8",
        )

    profile_a = AgentProfileLoader(workspace_a.resolve()).load("planner")
    profile_b = AgentProfileLoader(workspace_b.resolve()).load("planner")

    assert profile_a is not None
    assert profile_b is not None
    assert profile_a.system_prompt == "prompt-a"
    assert profile_b.system_prompt == "prompt-b"


# 功能：验证 custom Planner 缺失 child-type 字段时保持空权限而不补回 Explorer
# 设计：只提供工具子集并省略 child-type 配置，直接检查 loader 的 fail-closed 结果
def test_custom_planner_missing_child_types_stays_empty(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom"\nallowed_tools = ["read_file"]\n',
        encoding="utf-8",
    )

    profile = AgentProfileLoader(tmp_path.resolve()).load("planner")

    assert profile is not None
    assert profile.allowed_subagent_types == []


# 功能：验证 custom Planner 缺失工具字段时保持零工具而不补回 trusted allowlist
# 设计：只提供自定义 prompt，锁定配置缺失与显式空集合都不能触发 runtime 默认权限
def test_custom_planner_missing_tools_stays_empty(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom"\n',
        encoding="utf-8",
    )

    profile = AgentProfileLoader(tmp_path.resolve()).load("planner")

    assert profile is not None
    assert profile.allowed_tools == []


# 功能：验证 custom Planner 显式空工具和空 child-type 集合不被 loader 扩张
# 设计：分别写入两个空数组，锁定配置层保留 empty 与 missing 的收窄语义
def test_custom_planner_explicit_empty_capabilities_stay_empty(tmp_path: Path) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        '[agent]\nsystem_prompt = "custom"\n'
        'allowed_tools = []\nallowed_subagent_types = []\n',
        encoding="utf-8",
    )

    profile = AgentProfileLoader(tmp_path.resolve()).load("planner")

    assert profile is not None
    assert profile.allowed_tools == []
    assert profile.allowed_subagent_types == []


# 功能：验证 malformed allowlist 类型不会被当作可迭代配置放行
# 设计：参数化字符串和混合类型两类 TOML 输入，断言 loader fail closed 而非 substring 匹配
@pytest.mark.parametrize(
    "field_value",
    ['"read_file"', '["read_file", 1]'],
)
def test_custom_planner_malformed_tool_allowlist_fails_closed(
    tmp_path: Path,
    field_value: str,
) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        "[agent]\n"
        f"system_prompt = \"custom\"\nallowed_tools = {field_value}\n",
        encoding="utf-8",
    )

    assert AgentProfileLoader(tmp_path.resolve()).load("planner") is None


# 功能：验证 malformed child-type 类型不会被错误解释为 Explorer 授权
# 设计：覆盖字符串和混合列表输入，避免 `name in requested` 产生隐式 substring/类型语义
@pytest.mark.parametrize(
    "field_value",
    ['"explorer"', '["explorer", 1]'],
)
def test_custom_planner_malformed_child_types_fail_closed(
    tmp_path: Path,
    field_value: str,
) -> None:
    agents = tmp_path / ".kama" / "agents"
    agents.mkdir(parents=True)
    (agents / "planner.toml").write_text(
        "[agent]\n"
        'system_prompt = "custom"\nallowed_tools = ["spawn_agent"]\n'
        f"allowed_subagent_types = {field_value}\n",
        encoding="utf-8",
    )

    assert AgentProfileLoader(tmp_path.resolve()).load("planner") is None
