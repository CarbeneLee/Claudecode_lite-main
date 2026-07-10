from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kama_claude.core.bus.events import StepFinishedEvent, StepStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from kama_claude.core.compact.compactor import Compactor # TYPE_CHECKING 避免循环导入问题
    from kama_claude.core.permissions.manager import PermissionManager


log = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(UTC).isoformat()

# AgentLoop 是核心循环驱动器，负责执行 plan→act→observe 循环，直到上下文终止。
class AgentLoop:
    # 初始化循环依赖：LLM provider、工具注册表、事件总线和可选管理器
    def __init__(
        self,
        provider: LLMProvider, # LLMProvider 是一个抽象类，定义了与语言模型交互的接口
        registry: ToolRegistry, # ToolRegistry 是一个工具注册表，管理可用的工具和它们的模式
        bus: EventBus, # EventBus 是一个事件总线，用于在系统中发布和订阅事件
        *,
        permission_manager: PermissionManager | None = None, # PermissionManager 是一个权限管理器，用于控制工具调用的权限
        compactor: Compactor | None = None, # Compactor 是一个上下文压缩器，用于在对话中压缩上下文以节省token
        compact_threshold: float = 0.80, # compact_threshold 是一个浮点数，表示触发上下文压缩的阈值（百分比）  
        session_id: str = "", # session_id 是一个字符串，表示当前会话的唯一标识符
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._permission_manager = permission_manager
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._session_id = session_id

    # 驱动 plan→act→observe 循环直到上下文终止；CancelledError 向上传播
    # AgentLoop.run() 每步发布 step.started，调用 LLM：传入 context.messages、registry.tool_schemas()、system prompt
    async def run(self, context: ExecutionContext) -> None: 
        while not context.is_done(): #一旦 status 变为 "success" 或 "failed"，循环退出。这是一个由状态机驱动的循环，不是简单的计数循环。
            context.step += 1
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )

            # [plan] call LLM — API errors terminate the run
            try:
                response = await self._provider.chat(
                    messages=context.messages,#携带完整的对话消息列表，包括系统提示和用户输入。
                    tool_schemas=self._registry.tool_schemas(), #将所有注册的工具以 Anthropic API 格式传给 LLM
                    bus=self._bus,
                    run_id=context.run_id,
                    step=context.step,
                    system=context.system_prompt(
                        "You are a helpful AI assistant. "
                        "Use the available tools to complete the user's goal. "
                        "When the goal is fully achieved, respond with a final answer "
                        "and do not call any more tools."
                    ),
                )
            except asyncio.CancelledError: # 异常策略处理
                context.mark_failed("cancelled")
                raise
            except Exception:
                logging.getLogger(__name__).exception(
                    "LLM call failed run_id=%s step=%d", context.run_id, context.step
                )
                context.mark_failed("llm_error")
                break

            # [observe] append assistant content blocks to context
            # thinking blocks must come first and be preserved verbatim for extended thinking mode
            blocks: list[dict[str, object]] = list(response.thinking_blocks) # 从 LLM 的响应中提取思考块和工具调用，将思考块添加到结果列表中。
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
            context.add_assistant_message(blocks) #严格遵循 Anthropic API 的 content block 顺序要求：think blocks → text → tool_use。思考块必须在前，且原样保留，以便扩展思考模式使用。

            # [act] execute each requested tool; errors become tool results so loop continues
            if response.stop_reason == "tool_use":
                for tc in response.tool_calls:
                    result = await invoke_tool(
                        self._registry, tc, self._bus, context.run_id,
                        permission_manager=self._permission_manager,
                        session_id=self._session_id,
                    )
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)
            elif response.stop_reason == "max_tokens" and response.tool_calls: #LLM 在输出工具调用参数时被截断，tool_calls 列表里会有不完整的调用参数
                # Output token limit hit mid-tool-call; input is incomplete.
                # Add synthetic error results so the conversation stays balanced.
                for tc in response.tool_calls:
                    context.add_tool_result(
                        tc.id,
                        "Error: output token limit reached before this tool call could be "
                        "completed. "
                        "Please break the task into smaller steps and try again.",
                        is_error=True,
                    )

            # Termination check — end_turn wins over max_steps if both hit on same step
            if response.stop_reason == "end_turn": #end_turn 优先于 max_steps。
                context.result = response.text or "" #如果 LLM 在第 20 步给出了最终答案，不会被误判为超步数失败
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")

            # 工具结果追加完毕（messages 末尾为 user）后检查压缩，仅在 run 继续时触发
            # 此时压缩结果 [user_summary, assistant_ack] 对下一次 LLM 调用是合法输入
            if ( #压缩触发
                not context.is_done() #压缩只对继续运行的循环有意义
                and response.stop_reason == "tool_use" #只在工具调用后压缩，此时 messages 末尾是 user（含 tool_result），压缩结果为 [user_summary, assistant_ack]，对下一轮 LLM 调用是合法输入
                and self._compactor is not None # 压缩器已注入（单次执行不可用）
                and self._compact_threshold > 0 # 自动压缩已启用
                and response.usage is not None # 有使用量数据
                and response.usage.context_pct >= self._compact_threshold # 上下文使用率超过阈值
            ):
                await self._compactor.compact(context, self._provider)

            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
