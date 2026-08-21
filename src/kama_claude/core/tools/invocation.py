from __future__ import annotations

# invocation.py 是 KamaClaude 的工具调用管线 (Tool Invocation Pipeline)
# 它是 AgentLoop [Act] 阶段的核心实现，接收 LLM 产生的 ToolCallBlock
import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from kama_claude.core.bus.events import (
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.execution_scope import (
    AuthorizationAttempt,
    ExecutionScopeAuthorization,
    ExecutionScopeError,
    ScopedExecutionContext,
    ScopeRequiredError,
)
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.errors import (
    RETRYABLE_ERROR_TYPES,
    classify_tool_exception,
    normalize_tool_error,
)
from kama_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from kama_claude.core.permissions.manager import PermissionManager

# 单个工具调用最长 2 分钟，耗时操作可由调用方传入更大 timeout
_DEFAULT_TIMEOUT: float = 120.0
# 最多 3 次尝试，即 1 次初始调用加 2 次重试
_MAX_RETRIES: int = 2
_RETRY_BASE_S: float = 2.0  # backoff base; tests can monkeypatch to 0


def _now() -> str:
    return datetime.now(UTC).isoformat()


class InvocationAuthorization(Protocol):
    # 描述 shared invocation pipeline 的 scope authorization 三个生命周期钩子
    async def authorize_call(self, *, tool_call: ToolCallBlock, tool: Any = None) -> None: ...

    # 在每次 attempt 前检查当前 execution state
    async def before_attempt(
        self,
        *,
        tool_call: ToolCallBlock,
        tool: Any = None,
        attempt: int,
    ) -> None: ...

    # 在 retry decision 前审计 attempt 的实际结果
    async def after_attempt(
        self,
        *,
        tool_call: ToolCallBlock,
        tool: Any = None,
        attempt: int,
        outcome: object,
    ) -> AuthorizationAttempt: ...


class ToolInvoker(Protocol):
    # 描述 AgentLoop 必须注入的受控工具调用边界
    def tool_schemas(self) -> list[dict[str, object]]: ...

    # 执行一次已经过 lookup/schema/permission/scope 管线的工具调用
    async def invoke(self, tool_call: ToolCallBlock) -> ToolResult: ...

    # 返回不可继续执行的硬终止原因，普通工具错误返回 None
    def terminal_reason(self) -> str | None: ...


class UnscopedAuthorization:
    # 为 Direct invocation 提供不改变现有行为的 no-op authorization
    async def authorize_call(self, *, tool_call: ToolCallBlock, tool: Any = None) -> None:
        del tool_call, tool

    # Direct invocation 不维护 execution-local pre-state
    async def before_attempt(
        self,
        *,
        tool_call: ToolCallBlock,
        tool: Any = None,
        attempt: int,
    ) -> None:
        del tool_call, tool, attempt

    # Direct invocation 保留既有 retry policy
    async def after_attempt(
        self,
        *,
        tool_call: ToolCallBlock,
        tool: Any = None,
        attempt: int,
        outcome: object,
    ) -> AuthorizationAttempt:
        del tool_call, tool, attempt, outcome
        return AuthorizationAttempt()


class DirectToolInvoker:
    # 为 Direct Mode 封装既有 generic invoke_tool，避免 AgentLoop 直接依赖 registry
    def __init__(
        self,
        registry: ToolRegistry,
        bus: EventBus,
        run_id: str,
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        permission_manager: PermissionManager | None = None,
        session_id: str = "",
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._run_id = run_id
        self._timeout = timeout
        self._permission_manager = permission_manager
        self._session_id = session_id

    # Direct Mode 的 provider schema 与实际 generic registry 保持一致
    def tool_schemas(self) -> list[dict[str, object]]:
        return self._registry.tool_schemas()

    # 调用既有 unscoped pipeline，保持 Direct Mode 的 retry/permission 语义
    async def invoke(self, tool_call: ToolCallBlock) -> ToolResult:
        return await invoke_tool(
            self._registry,
            tool_call,
            self._bus,
            self._run_id,
            self._timeout,
            permission_manager=self._permission_manager,
            session_id=self._session_id,
        )

    # Direct Mode 没有 scoped hard terminal
    def terminal_reason(self) -> str | None:
        return None


# 将 scoped audit 失败标记为 execution secondary inconclusive 状态
def _mark_audit_inconclusive(authorization: InvocationAuthorization) -> None:
    marker = getattr(authorization, "_context", None)
    mutation_state = getattr(marker, "mutation_state", None)
    if mutation_state is not None:
        mutation_state.mark_audit_inconclusive()


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
        # 发布失败事件，让 TUI、日志和追踪系统知道工具调用失败
        ToolCallFailedEvent(
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


# 复用 lookup、schema、permission、attempt audit、retry 与 event 生命周期
async def _invoke_with_authorization(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    bus: EventBus,
    run_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
    *,
    permission_manager: PermissionManager | None = None,
    session_id: str = "",
    authorization: InvocationAuthorization,
) -> ToolResult:
    # monotonic 不受 NTP 或人工校时影响，保证 elapsed_ms 准确
    t0 = time.monotonic()

    await bus.publish(
        ToolCallStartedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            ts=_now(),
        )
    )

    # 通过闭包捕获 t0，返回从阶段 1 开始到当前的真实耗时
    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    tool = registry.get(tool_call.name)
    # 防御 LLM 调用不存在的工具，并返回明确错误供其纠正
    if tool is None:
        return await _fail(
            bus, run_id, tool_call,
            "unknown_tool", f"unknown tool: {tool_call.name}", elapsed(),
        )

    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input)) # Pydantic 的 model_validate
        except Exception as exc:
            validation_error_class, validation_error_message = classify_tool_exception(
                exc,
                validation_model=tool.params_model,
            )
            return await _fail(
                bus, run_id, tool_call,
                validation_error_class, validation_error_message, elapsed(),
            )

    try:
        await authorization.authorize_call(tool_call=tool_call, tool=tool)
    except ExecutionScopeError as exc:
        return await _fail(
            bus,
            run_id,
            tool_call,
            exc.error_type,
            str(exc),
            elapsed(),
        )

    # 权限检查
    if permission_manager is not None:
        async def _emit_permission(raw: dict[str, Any]) -> None:
            await bus.publish(PermissionRequestedEvent(**raw, run_id=run_id))

        # 若有 PermissionManager，则调用 check_and_wait()
        allowed, decision = await permission_manager.check_and_wait(
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            session_id=session_id,
            event_emitter=_emit_permission,
        )
        if allowed:
            # 可能自动决策，或发布 permission.requested 等待客户端响应
            if decision not in ("auto_allow",):
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

    for attempt in range(1, _MAX_RETRIES + 2):  # 唯一正常返回的路径——其余路径通向 _fail()
        error_class: str | None = None
        error_message: str | None = None
        result: ToolResult | None = None
        caught_exception: Exception | None = None

        try:
            await authorization.before_attempt(
                tool_call=tool_call,
                tool=tool,
                attempt=attempt,
            )
        except ExecutionScopeError as exc:
            return await _fail(
                bus,
                run_id,
                tool_call,
                exc.error_type,
                str(exc),
                elapsed(),
                attempt=attempt,
            )

        try:
            result = await asyncio.wait_for(
                tool.invoke(dict(tool_call.input)), timeout=timeout
            )
        except asyncio.CancelledError as cancellation:
            # 取消是 primary outcome；audit 失败只能记录 secondary inconclusive
            try:
                await authorization.after_attempt(
                    tool_call=tool_call,
                    tool=tool,
                    attempt=attempt,
                    outcome=cancellation,
                )
            except asyncio.CancelledError:
                _mark_audit_inconclusive(authorization)
                raise cancellation
            except Exception:
                _mark_audit_inconclusive(authorization)
            raise
        except Exception as exc:
            caught_exception = exc

        try:
            audit_result = await authorization.after_attempt(
                tool_call=tool_call,
                tool=tool,
                attempt=attempt,
                outcome=result if result is not None else caught_exception,
            )
        except ExecutionScopeError as exc:
            _mark_audit_inconclusive(authorization)
            return await _fail(
                bus,
                run_id,
                tool_call,
                exc.error_type,
                str(exc),
                elapsed(),
                attempt=attempt,
            )
        except Exception:
            _mark_audit_inconclusive(authorization)
            return await _fail(
                bus,
                run_id,
                tool_call,
                "scope_audit_failed",
                "scoped post-state audit failed",
                elapsed(),
                attempt=attempt,
            )

        ms = elapsed()
        if caught_exception is None and result is not None and not result.is_error:
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

        if caught_exception is not None:
            error_class, error_message = classify_tool_exception(caught_exception)
        else:
            assert result is not None
            error_class, error_message = normalize_tool_error(
                result.error_type,
                result.content,
            )

        assert error_class is not None and error_message is not None

        if (
            error_class in RETRYABLE_ERROR_TYPES
            and attempt <= _MAX_RETRIES
            and audit_result.retry_allowed
        ):
            await bus.publish(
                # 每次重试前发布失败事件，以追踪失败的 attempt
                ToolCallFailedEvent(
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

    # 理论上不可达，但 mypy 需要看到所有路径都有返回值
    return ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )


# 保持 Direct Mode 的原有无 scope invocation API 与 retry 语义
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
    return await _invoke_with_authorization(
        registry,
        tool_call,
        bus,
        run_id,
        timeout,
        permission_manager=permission_manager,
        session_id=session_id,
        authorization=UnscopedAuthorization(),
    )


class ScopedToolInvoker:
    # 构造必须绑定 non-null context，防止 scoped execution 退回 unscoped path
    def __init__(
        self,
        registry: ToolRegistry,
        bus: EventBus,
        run_id: str,
        context: ScopedExecutionContext,
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        permission_manager: PermissionManager | None = None,
        session_id: str = "",
    ) -> None:
        if context is None:
            raise ScopeRequiredError("scope-required")
        self._registry = registry
        self._bus = bus
        self._run_id = run_id
        self._timeout = timeout
        self._permission_manager = permission_manager
        self._session_id = session_id
        self._context = context
        self._authorization = ExecutionScopeAuthorization(context)
        # 未来如需并行工具执行，必须单独评审 authorization state 与 ledger attribution
        self._flight_lock = asyncio.Lock()

    # 使用与 generic invoke_tool 相同的 pipeline，并串行化单一 execution context 的调用
    async def invoke(self, tool_call: ToolCallBlock) -> ToolResult:
        async with self._flight_lock:
            return await _invoke_with_authorization(
                self._registry,
                tool_call,
                self._bus,
                self._run_id,
                self._timeout,
                permission_manager=self._permission_manager,
                session_id=self._session_id,
                authorization=self._authorization,
            )

    # 为 scoped provider 返回与实际 invocation 相同的受控 schema
    def tool_schemas(self) -> list[dict[str, object]]:
        return self._registry.tool_schemas()

    # 普通 scoped invocation 不声明 approved-run hard terminal
    def terminal_reason(self) -> str | None:
        return None

    @property
    # 暴露只读 execution context 供审计测试读取，不提供替换入口
    def context(self) -> ScopedExecutionContext:
        return self._context


class ApprovedScopedToolInvoker(ScopedToolInvoker):
    # 只接受不可变 trusted registry，防止 approved execution 退回普通 registry
    def __init__(self, registry: Any, *args: Any, **kwargs: Any) -> None:
        from kama_claude.core.execution import TrustedScopedToolRegistry

        if not isinstance(registry, TrustedScopedToolRegistry):
            raise ScopeRequiredError("trusted scoped registry is required")
        context = kwargs.get("context")
        if context is None and len(args) >= 3:
            context = args[2]
        if context is None or registry.workspace_root != context.workspace_root:
            raise ScopeRequiredError("trusted registry workspace does not match scope")
        self._trusted_registry = registry
        self._terminal_reason: str | None = None
        super().__init__(registry, *args, **kwargs)  # type: ignore[arg-type]

    # approved provider 只能看到构造时 sealed registry 的 exact schema
    def tool_schemas(self) -> list[dict[str, object]]:
        return self._trusted_registry.tool_schemas()

    # 返回 approved execution 已经锁存的 hard terminal reason
    def terminal_reason(self) -> str | None:
        return self._terminal_reason

    # 将 scope hard failure 锁存为不可继续执行的 terminal reason
    async def invoke(self, tool_call: ToolCallBlock) -> ToolResult:
        if self._terminal_reason is not None:
            return ToolResult(
                content="approved execution is terminal",
                is_error=True,
                error_type=self._terminal_reason,
            )
        result = await super().invoke(tool_call)
        if result.error_type in {
            "unknown_tool",
            "scope_denied",
            "external_workspace_drift",
        }:
            self._terminal_reason = "scope_denied"
        elif result.error_type in {
            "scope_mutation_inconclusive",
            "scope_audit_failed",
            "scope_required",
        }:
            self._terminal_reason = "inconclusive"
        elif self._context.mutation_state.status == "inconclusive":
            self._terminal_reason = "inconclusive"
        return result

    # 暴露 immutable scope 与 execution-local ledger 的只读入口
    @property
    def context(self) -> ScopedExecutionContext:
        return self._context
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
    │   └── 其他异常          → "execution_error" → 不重试
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
超时保护：asyncio.wait_for(future, timeout=self._timeout_s)，默认 60 秒。\
超时后返回 False, "timeout"——这算是一种安全的拒绝。

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
当客户端断开连接时，该 session 的所有待审批请求被批量拒绝（set_result("deny_once")）。\
这防止了 Future 永久挂起导致的内存泄漏和工具调用死锁。
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
      │   └── Exception → "execution_error"
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
永不抛异常的设计	        整个函数都返回 ToolResult	            LLM 总能收到反馈，\
loop 不会因工具失败而崩溃
time.monotonic() 计时	t0 = time.monotonic()	                不受系统时间调整影响，耗时统计准确
闭包式 elapsed()	    内部函数捕获 t0	                            调用方无需传计时状态，接口简洁
Pydantic 校验先行	    权限检查之前做参数校验	                    无效参数直接拒绝，\
不浪费权限检查资源
auto_* 事件抑制	    decision not in ("auto_allow",)	            缓存命中不高亮，保持日志信噪比
依赖反转的权限回调	    _emit_permission 闭包	                    PermissionManager 不依赖 EventBus
按特定性排序的异常捕获	RateLimitedError → TimeoutError → Exception	    语义化异常 vs 兜底泛化异常
TimeoutError 不可重试	特殊分支直接 _fail	                        超时=任务太重，重试无意义
指数退避重试	        2.0 × 2^(attempt-1)	                        给上游 API 恢复时间
Mypy 兼容守卫	        # unreachable, but keeps mypy happy	        工程实际：\
有时需要为类型检查器妥协
客户端断连保护	        cancel_session	                            防止 Future 永久挂起导致内存泄漏
对 LLM 友好的错误信息	permission_denied 的指导语	                LLM 不只是看到错误，\
还能知道下一步怎么做
frozenset 不可变常量	RETRYABLE_ERROR_TYPES	                     防止运行时意外修改配置
测试友好的可修改常量	# tests can monkeypatch to 0	                测试不走真实退避，保持快速
'''
