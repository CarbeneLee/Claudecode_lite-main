from __future__ import annotations
# invocation.py 是 KamaClaude 的工具调用管线 (Tool Invocation Pipeline),它是 AgentLoop [Act] 阶段的核心实现——接收 LLM 产生的 ToolCallBlock
import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from kama_claude.core.bus.events import (
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.errors import RateLimitedError
from kama_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from kama_claude.core.permissions.manager import PermissionManager

_DEFAULT_TIMEOUT: float = 120.0 #单个工具调用最长 2 分钟。对于耗时操作（如 bash 执行大脚本），调用方可以传入更大的 timeout
_MAX_RETRIES: int = 2 #最多 3 次尝试（1 次初始 + 2 次重试）。不是 3 而是 2，符合 range(1, _MAX_RETRIES + 2) 的语义
_RETRY_BASE_S: float = 2.0  # backoff base; tests can monkeypatch to 0
_RETRYABLE: frozenset[str] = frozenset({"runtime_error", "rate_limited"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 发布 ToolCallFailedEvent 并返回对应 ToolResult
async def _fail(
    bus: EventBus,
    run_id: str,
    tool_call: ToolCallBlock,
    error_class: str,
    error_message: str,
    elapsed_ms: int,
    *,
    attempt: int = 1,
) -> ToolResult:
    await bus.publish(
        ToolCallFailedEvent( #发布 ToolCallFailedEvent——让外部观察者（TUI、日志、追踪系统）知道工具调用失败了
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            error_class=error_class,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
            ts=_now(),
        )
    )
    return ToolResult(content=error_message, is_error=True, error_type=error_class)


# 校验参数、检查权限、限时调用工具、发布进度事件，失败时指数退避重试，返回 ToolResult（不抛异常）
async def invoke_tool(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    bus: EventBus,
    run_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
    *,
    permission_manager: PermissionManager | None = None,
    session_id: str = "",
) -> ToolResult:
    t0 = time.monotonic()# monotonic 不受系统时间调整（NTP 校时、用户改时间）的影响，保证 elapsed_ms 始终准确——即使有人在工具执行过程中调整了系统时钟。

    await bus.publish(
        ToolCallStartedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            ts=_now(),
        )
    )

    def elapsed() -> int: #elapsed() 闭包,通过闭包捕获 t0，调用方不需要传递计时状态。每次调用返回的是从阶段 1 开始到当前的真实耗时（毫秒级），而非单次尝试的耗时
        return int((time.monotonic() - t0) * 1000)

    tool = registry.get(tool_call.name)
    if tool is None: #LLM 可能产生幻觉，调用不存在的工具。这里做防御性检查，返回 "unknown tool" 的明确错误，LLM 看到后会纠正自己的行为。
        return await _fail(
            bus, run_id, tool_call,
            "runtime_error", f"unknown tool: {tool_call.name}", elapsed(),
        )

    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input)) # Pydantic 的 model_validate
        except ValidationError as exc: #校验失败返回 schema_error，工具不会执行
            return await _fail(
                bus, run_id, tool_call,
                "schema_error", str(exc), elapsed(),
            )
# 权限检查
    if permission_manager is not None:
        async def _emit_permission(raw: dict[str, Any]) -> None:
            await bus.publish(PermissionRequestedEvent(**raw, run_id=run_id))

        allowed, decision = await permission_manager.check_and_wait( #若有 PermissionManager，调用 check_and_wait()
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            session_id=session_id,
            event_emitter=_emit_permission,
        )
        if allowed:
            if decision not in ("auto_allow",): #可能自动放行、自动拒绝，或发布 permission.requested 并等待客户端 permission.respond
                await bus.publish(
                    PermissionGrantedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        decision=decision,
                        ts=_now(),
                    )
                )
        else:
            if decision != "auto_deny":
                await bus.publish(
                    PermissionDeniedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        decision=decision,
                        ts=_now(),
                    )
                )
            return await _fail(# 请求被拒绝，明确告知 LLM 这是用户的决定（而非系统错误）
                bus, run_id, tool_call,
                "permission_denied",
                "Permission denied by user. You may not execute this command. "
                "Try an alternative approach or ask the user what to do.",
                elapsed(),
            )

    for attempt in range(1, _MAX_RETRIES + 2):# 唯一正常返回的路径——所有其他路径都通向 _fail()
        error_class: str | None = None
        error_message: str | None = None

        try:
            result = await asyncio.wait_for(
                tool.invoke(dict(tool_call.input)), timeout=timeout
            )
            ms = elapsed()

            if result.is_error:
                error_class = result.error_type or "runtime_error"
                error_message = result.content
            else:
                await bus.publish(
                    ToolCallFinishedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        tool_name=tool_call.name,
                        elapsed_ms=ms,
                        output=result.content,
                        ts=_now(),
                    )
                )
                return result

        except RateLimitedError as exc:
            error_class = "rate_limited"
            error_message = str(exc)
        except TimeoutError:
            return await _fail(
                bus, run_id, tool_call,
                "timeout", f"tool timed out after {timeout}s", elapsed(),
                attempt=attempt,
            )
        except Exception as exc:
            error_class = "runtime_error"
            error_message = str(exc)

        assert error_class is not None and error_message is not None
        ms = elapsed() # 

        if error_class in _RETRYABLE and attempt <= _MAX_RETRIES:
            await bus.publish(
                ToolCallFailedEvent( #每次重试前发布 ToolCallFailedEvent：这让监控系统可以追踪"这个工具在第几次尝试时失败了"
                    run_id=run_id,
                    tool_use_id=tool_call.id,
                    tool_name=tool_call.name,
                    error_class=error_class,
                    error_message=error_message,
                    elapsed_ms=ms,
                    attempt=attempt,
                    ts=_now(),
                )
            )
            await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
            continue

        return await _fail(
            bus, run_id, tool_call,
            error_class, error_message, ms,
            attempt=attempt,
        )

    # unreachable, but keeps mypy happy 理论上永远不会执行，for 循环的 _fail 或 return result 覆盖了所有路径），但类型检查器（mypy）需要看到函数在所有路径上都有返回值。
    return ToolResult(content="internal error", is_error=True, error_type="runtime_error")
'''
invoke_tool(tool_call)
│
├── [1] 计时启动 + ToolCallStartedEvent
├── [2] 工具查找 ──→ 不存在? → _fail("unknown tool")
├── [3] 参数校验(Pydantic) → 不合法? → _fail("schema_error")
├── [4] 权限检查(PermissionManager) → 被拒? → _fail("permission_denied")
│
└── [5] 重试循环 (attempt = 1, 2, 3)
    │
    ├── asyncio.wait_for(tool.invoke(...), timeout)
    │   │
    │   ├── 成功 → ToolCallFinishedEvent → return ToolResult
    │   ├── is_error=True → error_class → 判断是否可重试
    │   ├── RateLimitedError  → "rate_limited" → 可重试
    │   ├── TimeoutError      → _fail("timeout") → 退出
    │   └── 其他异常          → "runtime_error" → 可重试
    │
    ├── 可重试 && attempt ≤ 2?
    │   ├── YES → ToolCallFailedEvent + 指数退避 → continue
    │   └── NO  → _fail(最终失败) → return ToolResult
'''

'''
invoke_tool 是权限管线的触发者，PermissionManager 是执行者。

权限决策的 6 层递进
PermissionManager.check_and_wait() 按优先级从高到低评估：


Tier 1: deny_patterns（拒绝模式，bash 专有）      → auto_deny
Tier 2: OUTSIDE_CWD_HEURISTICS（工作目录外）      → 强制 ASK
Tier 3: session always 缓存（内存）                → auto_allow/auto_deny
Tier 4: persistent always（policy.toml 文件）      → auto_allow/auto_deny
Tier 5: allow_patterns（允许模式，bash 专有）      → auto_allow
Tier 6: tool default（策略文件中的默认值）          → auto_allow/auto_deny
────────────────────────────────────────────────────────────
None of the above（default=ASK 或无策略）          → 挂起等待用户
同步挂起机制

loop = asyncio.get_event_loop()
future: asyncio.Future[str] = loop.create_future()
self._pending[tool_use_id] = _PendingRequest(
    future=future, session_id=session_id, tool_name=tool_name,
)
await event_emitter({...})

try:
    if self._timeout_s > 0:
        raw = await asyncio.wait_for(future, timeout=self._timeout_s)
    else:
        raw = await future
except TimeoutError:
    self._pending.pop(tool_use_id, None)
    return False, "timeout"
这是一个异步生产者-消费者模式：

创建 Future 对象（消费者端的占位符）
通过 event_emitter 向客户端（TUI / API）发送审批请求（生产者触发）
await future——工具调用在此挂起，直到用户响应
客户端通过 PermissionManager.respond(tool_use_id, decision) 来 set_result，唤醒 Future
超时保护：asyncio.wait_for(future, timeout=self._timeout_s)，默认 60 秒。超时后返回 False, "timeout"——这算是一种安全的拒绝。

客户端断连保护

def cancel_session(self, session_id: str, reason: str = "client_disconnected") -> None:
    to_cancel = [
        uid for uid, req in self._pending.items()
        if req.session_id == session_id
    ]
    for uid in to_cancel:
        req = self._pending.pop(uid)
        if not req.future.done():
            req.future.set_result("deny_once")
当客户端断开连接时，该 session 的所有待审批请求被批量拒绝（set_result("deny_once")）。这防止了 Future 永久挂起导致的内存泄漏和工具调用死锁。
'''

'''
完整数据流图

AgentLoop.run()
  │
  │  response.tool_calls = [ToolCallBlock(id="tc1", name="bash", input={...})]
  │
  ▼
invoke_tool(registry, tool_call, bus, run_id)
  │
  ├── t0 = time.monotonic()
  ├── bus.publish(ToolCallStartedEvent)          ───► events.jsonl
  │
  ├── tool = registry.get("bash")
  │   └── not found? → _fail → ToolCallFailedEvent → ToolResult(is_error=True)
  │
  ├── tool.params_model.model_validate(input)
  │   └── invalid? → _fail("schema_error")
  │
  ├── permission_manager.check_and_wait(...)
  │   ├── Tier 1-6 决策 → auto_allow / auto_deny
  │   └── ASK → Future 挂起 → event_emitter → 客户端响应
  │       └── denied? → _fail("permission_denied")
  │       └── timeout? → _fail("timeout")
  │
  └── for attempt in 1..3:
      ├── asyncio.wait_for(tool.invoke(params), timeout=120s)
      │   ├── success → ToolCallFinishedEvent → return ToolResult
      │   ├── is_error → 继续重试判断
      │   ├── RateLimitedError → "rate_limited"
      │   ├── TimeoutError → _fail("timeout")
      │   └── Exception → "runtime_error"
      │
      ├── 可重试 && attempt ≤ 2?
      │   ├── YES → ToolCallFailedEvent + sleep(2^attempt) → continue
      │   └── NO  → _fail(error_class)
      │
      └── return ToolResult ──► AgentLoop.add_tool_result() ──► context.messages

      invoke_tool() 是 AgentLoop [Act] 的唯一入口：

AgentLoop.run()                         ← 调用方
  │
  └── invoke_tool(registry, tool_call, bus, run_id)
        │
        ├── registry.get()              ← 工具查找
        ├── tool.params_model.validate() ← Pydantic 校验
        ├── permission_manager          ← 权限审批（阻塞等待用户）
        │     ├── policy 评估
        │     ├── Future 挂起
        │     └── event_emitter → bus   ← 事件发布
        │
        ├── tool.invoke()               ← 实际工具执行
        │     └── (BashTool / ReadFileTool / WriteFileTool / ...)
        │
        ├── bus.publish(ToolCall*Event) ← 可观测性事件
        │
        └── return ToolResult           ← 结果返回 LLM
              │
              └── context.add_tool_result() ← 写入对话历史
'''

'''
模式/实践	                位置	                                价值
永不抛异常的设计	        整个函数都返回 ToolResult	            LLM 总能收到反馈，loop 不会因工具失败而崩溃
time.monotonic() 计时	t0 = time.monotonic()	                不受系统时间调整影响，耗时统计准确
闭包式 elapsed()	    内部函数捕获 t0	                            调用方无需传计时状态，接口简洁
Pydantic 校验先行	    权限检查之前做参数校验	                    无效参数直接拒绝，不浪费权限检查资源
auto_* 事件抑制	    decision not in ("auto_allow",)	            缓存命中不高亮，保持日志信噪比
依赖反转的权限回调	    _emit_permission 闭包	                    PermissionManager 不依赖 EventBus
按特定性排序的异常捕获	RateLimitedError → TimeoutError → Exception	    语义化异常 vs 兜底泛化异常
TimeoutError 不可重试	特殊分支直接 _fail	                        超时=任务太重，重试无意义
指数退避重试	        2.0 × 2^(attempt-1)	                        给上游 API 恢复时间
Mypy 兼容守卫	        # unreachable, but keeps mypy happy	        工程实际：有时需要为类型检查器妥协
客户端断连保护	        cancel_session	                            防止 Future 永久挂起导致内存泄漏
对 LLM 友好的错误信息	permission_denied 的指导语	                LLM 不只是看到错误，还能知道下一步怎么做
frozenset 不可变常量	_RETRYABLE	                                防止运行时意外修改配置
测试友好的可修改常量	# tests can monkeypatch to 0	                测试不走真实退避，保持快速
'''