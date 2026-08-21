from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.skills.loader import Skill, SkillLoader


# 功能：内建 review skill 应能被 SkillLoader 查找到
# 设计：直接调用 resolve("review")，不依赖文件系统之外的任何状态
def test_builtin_skill_found(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path.resolve())
    skill = loader.resolve("review")
    assert skill is not None
    assert skill.name == "review"
    assert "审查" in skill.description or "review" in skill.description.lower()
    assert skill.system_prompt_template != ""


# 功能：内建 init / summarize / orchestrate skill 均可找到
# 设计：列举所有内建 skill 名，断言均能解析
@pytest.mark.parametrize("name", ["init", "review", "summarize", "orchestrate"])
def test_all_builtin_skills_found(name: str, tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path.resolve())
    skill = loader.resolve(name)
    assert skill is not None, f"builtin skill '{name}' not found"


# 功能：验证 orchestrate 在 Planner error 时停止，不把未经认证文本交给 executor
# 设计：读取真实 builtin skill 正文，锁定最小 prompt-level compatibility contract
def test_orchestrate_stops_after_planner_error(tmp_path: Path) -> None:
    skill = SkillLoader(tmp_path.resolve()).resolve("orchestrate")

    assert skill is not None
    assert "is_error=true" in skill.system_prompt_template
    assert "不得派生 executor 或 reviewer" in skill.system_prompt_template


# 功能：验证 orchestrate 将完整 ExactPlannerDecisionV2 传给 executor
# 设计：锁定 agent-facing renderer 与 bounded PlanView 的边界，防止 UI projection 被误当执行事实
def test_orchestrate_uses_full_agent_facing_decision(tmp_path: Path) -> None:
    skill = SkillLoader(tmp_path.resolve()).resolve("orchestrate")

    assert skill is not None
    assert "ExactPlannerDecisionV2" in skill.system_prompt_template
    assert "不是 bounded `PlanView`" in skill.system_prompt_template
    assert "不得使用 bounded PlanView" in skill.system_prompt_template


# 功能：验证需要代码探索的 init/review skill 声明 search_code
# 设计：只检查两个相关内建 skill 的解析结果，避免重复 getter 或文本快照测试
@pytest.mark.parametrize("name", ["init", "review"])
def test_code_exploration_skills_allow_search_code(name: str, tmp_path: Path) -> None:
    skill = SkillLoader(tmp_path.resolve()).resolve(name)

    assert skill is not None
    assert "search_code" in skill.allowed_tools


# 功能：不存在的 skill 名应返回 None
# 设计：查找一个不存在的名称，断言 resolve 返回 None 而非抛异常
def test_unknown_skill_returns_none(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path.resolve())
    result = loader.resolve("nonexistent_skill_xyz")
    assert result is None


# 功能：render_prompt 应将 $ARGUMENTS 替换为传入的参数字符串
# 设计：构造含 $ARGUMENTS 的 skill，验证 render_prompt 结果不含 "$ARGUMENTS" 且含参数值
def test_arguments_substituted(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path.resolve())
    skill = Skill(
        name="test",
        description="test skill",
        system_prompt_template="Review this: $ARGUMENTS\nPlease be thorough.",
        allowed_tools=[],
    )
    rendered = loader.render_prompt(skill, "src/foo.py")
    assert "$ARGUMENTS" not in rendered
    assert "src/foo.py" in rendered


# 功能：frontmatter 中的 allowed_tools 列表应被正确解析
# 设计：构造含 allowed_tools 的 Markdown 文件，通过 _parse_skill_file 解析并验证结果
def test_frontmatter_parsed(tmp_path: Path) -> None:
    from kama_claude.core.skills.loader import _parse_skill_file

    content = """\
---
name: custom
description: 自定义 skill 测试
allowed_tools:
  - read_file
  - bash
---
你是一个测试助手，目标：$ARGUMENTS
"""
    p = tmp_path / "custom.md"
    p.write_text(content, encoding="utf-8")
    skill = _parse_skill_file(p)
    assert skill.name == "custom"
    assert skill.description == "自定义 skill 测试"
    assert "read_file" in skill.allowed_tools
    assert "bash" in skill.allowed_tools
    assert "$ARGUMENTS" in skill.system_prompt_template


# 功能：无 frontmatter 的 Markdown 文件仍可加载，allowed_tools 为空列表
# 设计：写入纯正文 Markdown，断言解析成功且 allowed_tools=[]
def test_no_frontmatter(tmp_path: Path) -> None:
    from kama_claude.core.skills.loader import _parse_skill_file

    content = "你是助手，请帮助用户完成任务：$ARGUMENTS\n"
    p = tmp_path / "plain.md"
    p.write_text(content, encoding="utf-8")
    skill = _parse_skill_file(p)
    assert skill.name == "plain"
    assert skill.allowed_tools == []
    assert "你是助手" in skill.system_prompt_template


# 功能：项目本地 skill 应覆盖内建同名 skill
# 设计：daemon cwd 放置冲突版本，loader 显式绑定另一个 workspace，断言只读取绑定目录
def test_project_overrides_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    daemon_cwd = tmp_path / "daemon"
    local_skills = workspace / ".kama" / "skills"
    daemon_skills = daemon_cwd / ".kama" / "skills"
    local_skills.mkdir(parents=True)
    daemon_skills.mkdir(parents=True)
    (local_skills / "review.md").write_text(
        "---\nname: review\ndescription: local override\n---\nlocal system prompt $ARGUMENTS\n",
        encoding="utf-8",
    )
    (daemon_skills / "review.md").write_text(
        "---\nname: review\ndescription: daemon override\n---\ndaemon prompt\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(daemon_cwd)
    loader = SkillLoader(workspace.resolve())
    skill = loader.resolve("review")
    assert skill is not None
    assert skill.description == "local override"
    assert "local system prompt" in skill.system_prompt_template


# 功能：验证两个 SkillLoader 对同名项目 skill 保持 workspace 隔离
# 设计：A/B 各写不同描述，通过两个显式 root loader 解析并比较结果
def test_skill_loaders_isolate_project_overrides(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    for workspace, description in ((workspace_a, "skill-a"), (workspace_b, "skill-b")):
        skills = workspace / ".kama" / "skills"
        skills.mkdir(parents=True)
        (skills / "local.md").write_text(
            f"---\nname: local\ndescription: {description}\n---\n{description} $ARGUMENTS\n",
            encoding="utf-8",
        )

    skill_a = SkillLoader(workspace_a.resolve()).resolve("local")
    skill_b = SkillLoader(workspace_b.resolve()).resolve("local")

    assert skill_a is not None
    assert skill_b is not None
    assert skill_a.description == "skill-a"
    assert skill_b.description == "skill-b"
