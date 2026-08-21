from __future__ import annotations

from kama_claude.core.context import ExecutionContext

_REQUIREMENT_CONTRACT = (
    "Before changing the workspace, create a concise requirement contract from every "
    "explicit acceptance criterion. For each item, record the required observable "
    "behavior, relevant failure or invalid-input behavior, any side-effect or state "
    "invariant, and the evidence you plan to use for verification. Keep this checklist "
    "visible in the conversation as you work, and update each item as implemented, "
    "verified, or unchecked. Before finishing, review every item. Do not assume unchecked "
    "items are complete: verify them when possible, otherwise clearly report the "
    "limitation. Keep the contract brief and auditable; do not expose private "
    "chain-of-thought or force any particular tool."
)
_STATE_TRANSITION_PROTOCOL = (
    "When a task changes persistent or shared state through multiple operations, briefly "
    "map the pre-state, each mutation point, every later operation that can fail, and the "
    "required post-state after success or failure. Before finishing, exercise at least "
    "one failure after an earlier mutation succeeds, and verify that rollback or "
    "compensation preserves the stated invariant. Do not apply this protocol to tasks "
    "without multi-step side effects."
)
_REPOSITORY_CHANGE_DISCIPLINE = """## Repository Change Discipline
Prefer editing existing files to creating new ones
Don't add features, refactor, or introduce abstractions beyond what the task requires
Don't design for hypothetical future requirements
A bug fix doesn't need surrounding cleanup"""


# 构造最小 ExecutionContext 并允许测试覆盖任意字段
def _make_ctx(**kwargs) -> ExecutionContext:
    defaults = dict(run_id="r1", goal="test goal", max_steps=5)
    defaults.update(kwargs)
    return ExecutionContext(**defaults)


# 功能：验证三层记忆全部存在时都出现在 system prompt 中且顺序正确
# 设计：分别设置 global_context、project_context、session_notes，断言各 section 标题及内容依次出现
def test_all_layers_present() -> None:
    ctx = _make_ctx(
        global_context="global line",
        project_context="project line",
        session_notes="session note",
    )
    prompt = ctx.system_prompt("BASE")
    assert "BASE" in prompt
    assert "## Global Context\nglobal line" in prompt
    assert "## Project Context\nproject line" in prompt
    assert "## Session Notes\nsession note" in prompt
    # 顺序：global 在 project 之前，project 在 session 之前
    assert prompt.index("Global") < prompt.index("Project") < prompt.index("Session")


# 功能：验证无上下文时 system prompt 仍包含一次可信仓库变更纪律
# 设计：用最小 context 断言 base 后紧跟完整 policy，防止 profile/base 成为绕过点
def test_no_layers() -> None:
    ctx = _make_ctx()
    prompt = ctx.system_prompt("BASE_ONLY")
    assert prompt == "BASE_ONLY\n\n" + _REPOSITORY_CHANGE_DISCIPLINE


# 功能：验证只有 global_context 时只出现 Global section，其他 section 不出现
# 设计：只设置 global_context，断言 Project 和 Session 标题不在 prompt 中
def test_only_global() -> None:
    ctx = _make_ctx(global_context="global content")
    prompt = ctx.system_prompt("BASE")
    assert "## Global Context" in prompt
    assert "## Project Context" not in prompt
    assert "## Session Notes" not in prompt


# 功能：验证 session_notes 非空时包含 note_save 提示语
# 设计：只设置 session_notes，断言 prompt 含 note_save 相关提示
def test_session_notes_hint() -> None:
    ctx = _make_ctx(session_notes="some note")
    prompt = ctx.system_prompt("BASE")
    assert "note_save" in prompt


# 功能：验证 repaired v2 default base 之后仍按 Global、Project、Session 顺序追加 context
# 设计：把 base 与 exact v1/v2 传入真实 system_prompt，逐段定位并排除协议或 context 层重排
def test_v1_and_v2_default_base_precede_all_context_layers() -> None:
    base = "BASE\n\n" + _REQUIREMENT_CONTRACT + "\n\n" + _STATE_TRANSITION_PROTOCOL
    ctx = _make_ctx(
        global_context="global",
        project_context="project",
        session_notes="session",
    )

    prompt = ctx.system_prompt(base)

    assert prompt.startswith(base)
    assert prompt.count(_REPOSITORY_CHANGE_DISCIPLINE) == 1
    assert prompt.count(_REQUIREMENT_CONTRACT) == 1
    assert prompt.count(_STATE_TRANSITION_PROTOCOL) == 1
    assert prompt.index(_REQUIREMENT_CONTRACT) < prompt.index(
        _STATE_TRANSITION_PROTOCOL
    )
    assert prompt.index(_STATE_TRANSITION_PROTOCOL) < prompt.index(
        _REPOSITORY_CHANGE_DISCIPLINE
    )
    assert prompt.index(_REPOSITORY_CHANGE_DISCIPLINE) < prompt.index(
        "## Global Context"
    )
    assert prompt.index("## Global Context") < prompt.index("## Project Context")
    assert prompt.index("## Project Context") < prompt.index("## Session Notes")


# 功能：验证显式 override 替换 default base 但不能移除可信仓库变更纪律
# 设计：向真实 ExecutionContext 同时传 full base 和 override，锁定 trusted policy 与可替换 role slot 的边界
def test_override_excludes_v1_and_v2_but_keeps_context_order() -> None:
    base = "BASE\n\n" + _REQUIREMENT_CONTRACT + "\n\n" + _STATE_TRANSITION_PROTOCOL
    ctx = _make_ctx(
        system_prompt_override="OVERRIDE",
        global_context="global",
        project_context="project",
        session_notes="session",
    )

    prompt = ctx.system_prompt(base)

    assert prompt.startswith("OVERRIDE\n\n" + _REPOSITORY_CHANGE_DISCIPLINE)
    assert prompt.count(_REPOSITORY_CHANGE_DISCIPLINE) == 1
    assert _REQUIREMENT_CONTRACT not in prompt
    assert _STATE_TRANSITION_PROTOCOL not in prompt
    assert prompt.index("## Global Context") < prompt.index("## Project Context")
    assert prompt.index("## Project Context") < prompt.index("## Session Notes")


# 功能：验证上下文中引用相同四条规则时 composer 不删除或改写原始文本
# 设计：把 canonical block 作为 project context 原文输入，断言可信注入与引用各保留一份
def test_quoted_policy_in_project_context_is_preserved() -> None:
    quoted = "Discussion quote:\n" + _REPOSITORY_CHANGE_DISCIPLINE
    ctx = _make_ctx(project_context=quoted)

    prompt = ctx.system_prompt("BASE")

    assert prompt.count(_REPOSITORY_CHANGE_DISCIPLINE) == 2
    assert "## Project Context\n" + quoted in prompt


# 功能：验证 repository instructions 位于可信 policy 后且早于既有 context layers
# 设计：同时填充所有 slot 并比较索引，防止 repository instructions 被误放入可替换 base 或 generated context
def test_repository_instructions_have_dedicated_composition_slot() -> None:
    ctx = _make_ctx(
        repository_instructions="source: AGENTS.md\nroot rule",
        global_context="global",
        project_context="generated project context",
        session_notes="session",
    )

    prompt = ctx.system_prompt("BASE")

    assert "## Repository Instructions\nsource: AGENTS.md\nroot rule" in prompt
    assert prompt.index(_REPOSITORY_CHANGE_DISCIPLINE) < prompt.index(
        "## Repository Instructions"
    )
    assert prompt.index("## Repository Instructions") < prompt.index(
        "## Global Context"
    )
