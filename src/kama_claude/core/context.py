from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REPOSITORY_CHANGE_DISCIPLINE = """## Repository Change Discipline
Prefer editing existing files to creating new ones
Don't add features, refactor, or introduce abstractions beyond what the task requires
Don't design for hypothetical future requirements
A bug fix doesn't need surrounding cleanup"""


@dataclass # 唯一的、贯穿整个 run 生命周期的有状态对象（状态机、消息容器和系统提示词构建器）
class ExecutionContext:
    run_id: str # 本次运行的唯一标识
    goal: str # 用户的初始目标文本
    max_steps: int # 最大执行步数
    prefill_messages: list[dict[str, Any]] = field(default_factory=list) # session 回放的完整历史
    session_notes: str = "" # 持久化笔记
    global_context: str = "" # ~/.kama/context.md 内容
    project_context: str = "" # .kama/context.md 内容
    repository_instructions: str = "" # workspace 显式仓库规则及来源标识
    messages: list[dict[str, Any]] = field(default_factory=list) # ← 核心：完整的对话历史
    step: int = 0 # 当前步数计数器
    status: str = "running"  # "running" | "success" | "failed" 
    reason: str | None = None # 失败原因（成功时为 None）
    result: str = "" # 最终文本输出
    # skill 或 subagent 角色可覆盖默认 system prompt
    system_prompt_override: str | None = None #system_prompt_override 存在时，base prompt 被完全跳过
    # 每个上下文层都通过 .strip() 检查是否为空。
    # 空文件不会产生空的 ## Global Context\n 标题，保持最终 prompt 的整洁
    # 初始化消息历史，优先使用 session 完整回放内容
    def __post_init__(self) -> None:
        if self.prefill_messages: # prefill_message非空
            # 防御性拷贝，避免外部修改 prefill_messages 导致跨 run 数据污染
            self.messages = [dict(m) for m in self.prefill_messages]
        # 如果 prefill_messages 为空且 messages 不为空，则将 goal 作为用户消息追加
        elif not self.messages:
            self.messages.append({"role": "user", "content": self.goal})

    # 返回当前 run 的可信 policy 与上下文组合；override 只能替换 role/base 槽位
    def system_prompt(self, base: str) -> str:
        # base 硬编码在 AgentLoop.run() 中
        parts = [self.system_prompt_override if self.system_prompt_override else base]
        parts.append("\n\n" + REPOSITORY_CHANGE_DISCIPLINE)
        if self.repository_instructions.strip():
            parts.append(
                "\n\n## Repository Instructions\n" + self.repository_instructions
            )
        if self.global_context.strip(): #~/.kama/context.md,跨项目记录用户偏好和全局规则
            parts.append("\n\n## Global Context\n" + self.global_context.strip())
        # .kama/context.md 作用于当前项目，包含项目目标、约束和已知事实
        if self.project_context.strip():
            parts.append("\n\n## Project Context\n" + self.project_context.strip())
        if self.session_notes.strip(): #SessionStore 持久化，跨多轮对话持续记忆的事实
            # 运行时元指令提示 LLM 可以使用持久化记忆工具
            parts.append(
                "\n\n## Session Notes\n"
                + self.session_notes.strip()
                + "\n\nRemember important durable facts by calling note_save."
            )
        return "".join(parts)#因为每个部分都以 \n\n 开头，用空字符串 join 比 "\n\n".join() 更精确

    # 将 LLM 响应的 content blocks 追加为 assistant 消息
    def add_assistant_message(self, content: list[Any]) -> None:
        self.messages.append({"role": "assistant", "content": content})

    # 将工具调用结果追加为 user 消息；同一步的多个结果共享同一条消息
    def add_tool_result(# 同一步的多个工具调用结果共享同一条 user 消息，避免消息膨胀
        self, tool_use_id: str, content: str, is_error: bool = False
    ) -> None:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            # ToolResult.content 写回 ExecutionContext；错误会带 is_error=True
            block["is_error"] = True

        last = self.messages[-1] if self.messages else None
        # 五个条件全部满足时追加到上一条 user 消息，否则创建新消息
        if (
            last is not None #	messages 列表为空
            and last["role"] == "user" #上一条是 assistant（需要在新的 user 消息中追加）
            and isinstance(last["content"], list) #user content 是纯文本字符串（如初始 "请帮我..."）
            and last["content"] #content 列表为空（防御性检查）
            # content 中有 text 类型的 block（混合内容）
            and all(b.get("type") == "tool_result" for b in last["content"])
        ):
            last["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})

    # 返回 True 表示 loop 应停止（状态不再是 running）
    def is_done(self) -> bool:
        return self.status != "running"

    # 将 run 标记为成功
    def mark_success(self) -> None:
        self.status = "success"

    # 将 run 标记为失败并记录原因
    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason
'''
Anthropic API 要求同一步的多个 tool_result 必须放在同一条 role: user 消息的 content 数组中：


# ✅ 正确格式
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "tool_001", "content": "..."},
    {"type": "tool_result", "tool_use_id": "tool_002", "content": "..."},
]}

# ❌ 错误格式（会导致 API 拒绝）
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_001", ...}]}
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_002", ...}]}
'''

'''
当前 messages 末尾                     调用 add_tool_result 后
────────────────────────────────────    ──────────────────────────
[..., {role:assistant, content:[...tool_use...]}]
                                     → [..., {role:assistant, ...},
                                          {role:user, content:[tool_result_A]}]  ← 新建 user 消息

[..., {role:user, content:[tool_result_A]}]
                                     → [..., {role:user, content:[tool_result_A,
                                                                   tool_result_B]}]  ← 追加到同一条

[..., {role:user, content:"纯文本消息"}]
                                     → [..., {role:user, content:"纯文本"},
                                          {role:user, content:[tool_result_A]}]  ← 新建 user 消息
'''

'''
AgentRunner.run_and_capture()
  │  创建 ExecutionContext  ──────────────────────┐
  │  读取 context.status / context.result          │
  │                                                │
AgentLoop.run(context)                             │
  │  轮询 context.is_done()         ◄──────────────┤
  │  调用 context.system_prompt()                  │
  │  调用 context.add_assistant_message()          │
  │  调用 context.add_tool_result()                │
  │  设置 context.step += 1                        │
  │  设置 context.result / mark_success / mark_failed
  │                                                │
Compactor.compact(context, provider)               │
  │  读取 context.messages                         │
  │  就地替换 context.messages = [...]  ◄──────────┤
  │                                                │
EventWriter                                        │
  │  读取 context.run_id（事件关联）                │
  └────────────────────────────────────────────────┘
'''
