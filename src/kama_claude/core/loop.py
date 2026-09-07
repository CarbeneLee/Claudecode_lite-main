from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kama_claude.core.bus.events import StepFinishedEvent, StepStartedEvent
from kama_claude.core.compact.compactor import CompactionConflict
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.llm.gateway import (
    ContextAdmissionError,
    ProviderRequestGateway,
)
from kama_claude.core.tools.invocation import ToolInvoker

if TYPE_CHECKING:
    from kama_claude.core.compact.compactor import Compactor  # TYPE_CHECKING 避免循环导入问题


log = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(UTC).isoformat()


# 将step primary投影为不含异常内容的稳定日志分类
def _primary_failure_category(
    primary_failure: BaseException | None,
    context: ExecutionContext,
) -> str:
    if primary_failure is None:
        return "none"
    if isinstance(primary_failure, asyncio.CancelledError):
        return "cancellation"
    if context.reason == "llm_error":
        return "llm_error"
    return "propagated_exception"


# 单次发布step terminal并在调用方取消时等待同一publication task结束
async def _publish_step_finished_once(bus: EventBus, event: StepFinishedEvent) -> None:
    publication = asyncio.create_task(bus.publish(event))
    cancellation: asyncio.CancelledError | None = None
    while not publication.done():
        try:
            await asyncio.shield(publication)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                if cancellation is None:
                    cancellation = exc
                continue
            break
        except Exception:
            break

    delivery_failure: asyncio.CancelledError | Exception | None = None
    try:
        publication.result()
    except asyncio.CancelledError as exc:
        delivery_failure = exc
    except Exception as exc:
        delivery_failure = exc

    if cancellation is not None:
        if delivery_failure is not None:
            log.error(
                "step terminal delivery failure treated as secondary "
                "run_id=%s step=%d failure_role=secondary "
                "primary_category=cancellation",
                event.run_id,
                event.step,
            )
        raise cancellation
    if delivery_failure is not None:
        raise delivery_failure


# AgentLoop 是核心循环驱动器，负责执行 plan→act→observe 循环，直到上下文终止。
class AgentLoop:
    # 初始化循环依赖：LLM provider、强制工具调用器和事件总线
    def __init__(
        self,
        provider: LLMProvider, # LLMProvider 是一个抽象类，定义了与语言模型交互的接口
        tool_invoker: ToolInvoker, # ToolInvoker 同时提供 provider schema 与唯一执行入口
        bus: EventBus, # EventBus 是一个事件总线，用于在系统中发布和订阅事件
        *,
        # Compactor 在对话中压缩上下文以节省 token
        compactor: Compactor | None = None,
        # compact_threshold 表示触发上下文压缩的百分比阈值
        compact_threshold: float = 0.80,
        soft_trigger_ratio: float = 0.80,
        target_ratio: float = 0.60,
    ) -> None:
        if tool_invoker is None:
            raise TypeError("tool_invoker is required")
        # 所有 AgentLoop generation 统一经过 gateway 的 capacity admission
        self._provider = (
            provider
            if isinstance(provider, ProviderRequestGateway)
            else ProviderRequestGateway(
                provider,
                soft_trigger_ratio=soft_trigger_ratio,
                target_ratio=target_ratio,
            )
        )
        self._bus = bus
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._tool_invoker = tool_invoker

    # 调用 compactor 并兼容旧测试替身；真实实现始终携带 same-route layout
    async def _compact_context(
        self,
        context: ExecutionContext,
        *,
        system: str,
        immutable_system: str,
        semi_stable_context: str,
        tool_schemas: list[dict[str, object]],
    ) -> object:
        compactor = self._compactor
        if compactor is None:
            return None
        parameters: Mapping[str, inspect.Parameter] = {}
        try:
            parameters = inspect.signature(compactor.compact).parameters
        except (TypeError, ValueError):
            parameters = {}
        supports_layout = "same_route" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not supports_layout:
            return await compactor.compact(context, self._provider)
        return await compactor.compact(
            context,
            self._provider,
            same_route=True,
            system=system,
            immutable_system=immutable_system,
            semi_stable_context=semi_stable_context,
            tool_schemas=tool_schemas,
        )

    # 每步发布 step.started，并向 LLM 传入消息、工具 schema 和 system prompt
    # 驱动 plan→act→observe 循环直到上下文终止；CancelledError 向上传播
    async def run(self, context: ExecutionContext) -> None:
        # 状态变为 success 或 failed 后退出，由状态机而不是简单计数驱动
        while not context.is_done():
            context.step += 1
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )

            primary_failure: BaseException | None = None
            pending_terminal: str | None = None
            pending_result = ""
            try:
                # [plan] call LLM — API errors terminate the run
                admission_reductions = 0
                compaction_conflicts = 0
                proactive_reductions = 0
                provider_overflow_retries = 0
                base_prompt = (
                    "You are a helpful AI assistant. "
                    "Use the available tools to complete the user's goal. "
                    "When the goal is fully achieved, respond with a final answer "
                    "and do not call any more tools.\n\n"
                    "Before changing the workspace, create a concise requirement "
                    "contract from every explicit acceptance criterion. For each "
                    "item, record the required observable behavior, relevant failure "
                    "or invalid-input behavior, any side-effect or state invariant, "
                    "and the evidence you plan to use for verification. Keep this "
                    "checklist visible in the conversation as you work, and update "
                    "each item as implemented, verified, or unchecked. Before "
                    "finishing, review every item. Do not assume unchecked items are "
                    "complete: verify them when possible, otherwise clearly report "
                    "the limitation. Keep the contract brief and auditable; do not "
                    "expose private chain-of-thought or force any particular tool.\n\n"
                    "When a task changes persistent or shared state through multiple "
                    "operations, briefly map the pre-state, each mutation point, every "
                    "later operation that can fail, and the required post-state after "
                    "success or failure. Before finishing, exercise at least one failure "
                    "after an earlier mutation succeeds, and verify that rollback or "
                    "compensation preserves the stated invariant. Do not apply this "
                    "protocol to tasks without multi-step side effects."
                )
                while True:
                    # Admission reductions can install a new checkpoint projection;
                    # rerender Layer C before retrying instead of reusing stale system
                    # text from the rejected request.
                    context.surface_state.synchronize_epochs(
                        route_epoch=self._provider.route_epoch,
                        prefix_epoch=self._provider.prefix_epoch,
                    )
                    request_system = context.system_prompt(base_prompt)
                    request_immutable_system = context.immutable_system_prompt(base_prompt)
                    request_semi_stable_context = context.semi_stable_context()
                    request_tool_schemas = self._tool_invoker.tool_schemas()
                    try:
                        # Decide proactive pressure from the exact next request
                        # envelope, before generating a response.  Previous
                        # response usage is telemetry only and cannot describe
                        # the newly appended tool result or user steering.
                        next_envelope = self._provider.prepare_request(
                            messages=context.provider_messages(),
                            tool_schemas=request_tool_schemas,
                            system=request_system,
                            immutable_system=request_immutable_system,
                            semi_stable_context=request_semi_stable_context,
                            surface_revision=context.surface_state.surface_revision,
                        )
                        next_admission = self._provider.admit(next_envelope)
                        if (
                            next_admission.state == "SOFT_PRESSURED"
                            and self._compactor is not None
                            and self._compact_threshold > 0
                            and proactive_reductions < 1
                        ):
                            proactive_reductions += 1
                            try:
                                reduced = await self._compact_context(
                                    context,
                                    system=request_system,
                                    immutable_system=request_immutable_system,
                                    semi_stable_context=request_semi_stable_context,
                                    tool_schemas=request_tool_schemas,
                                )
                            except CompactionConflict:
                                reduced = None
                            if reduced is not None:
                                continue
                        response = await self._provider.chat(
                            # 携带重放 continuation 的 provider 消息列表
                            messages=context.provider_messages(),
                            # 将实际 invocation boundary 的 exact schema 传给 LLM
                            tool_schemas=request_tool_schemas,
                            bus=self._bus,
                            run_id=context.run_id,
                            step=context.step,
                            system=request_system,
                            immutable_system=request_immutable_system,
                            semi_stable_context=request_semi_stable_context,
                            surface_revision=context.surface_state.surface_revision,
                        )
                        context.surface_state.synchronize_epochs(
                            route_epoch=self._provider.route_epoch,
                            prefix_epoch=self._provider.prefix_epoch,
                        )
                        break
                    except ContextAdmissionError as admission_error:
                        if admission_error.origin == "provider":
                            if provider_overflow_retries >= 1:
                                context.mark_failed("CONTEXT_WINDOW_EXCEEDED")
                                pending_terminal = "CONTEXT_WINDOW_EXCEEDED"
                                break
                            provider_overflow_retries += 1
                        elif admission_reductions >= 2:
                            context.mark_failed("CONTEXT_WINDOW_EXCEEDED")
                            pending_terminal = "CONTEXT_WINDOW_EXCEEDED"
                            break
                        if admission_error.origin != "provider":
                            admission_reductions += 1
                        if self._compactor is None:
                            context.mark_failed("CONTEXT_WINDOW_EXCEEDED")
                            pending_terminal = "CONTEXT_WINDOW_EXCEEDED"
                            break
                        try:
                            reduced = await self._compact_context(
                                context,
                                system=request_system,
                                immutable_system=request_immutable_system,
                                semi_stable_context=request_semi_stable_context,
                                tool_schemas=request_tool_schemas,
                            )
                        except CompactionConflict:
                            compaction_conflicts += 1
                            if compaction_conflicts >= 2:
                                context.mark_failed("CONTEXT_WINDOW_EXCEEDED")
                                pending_terminal = "CONTEXT_WINDOW_EXCEEDED"
                                break
                            # 重新从当前 surface 运行完整 admission，而不是复用旧摘要
                            if admission_error.origin != "provider":
                                admission_reductions -= 1
                            continue
                        if reduced is None:
                            context.mark_failed("CONTEXT_WINDOW_EXCEEDED")
                            pending_terminal = "CONTEXT_WINDOW_EXCEEDED"
                            break
                    except asyncio.CancelledError as exc: # 异常策略处理
                        primary_failure = exc
                        context.mark_failed("cancelled")
                        raise
                    except Exception as exc:
                        primary_failure = exc
                        log.error(
                            "LLM call failed run_id=%s step=%d "
                            "failure_role=primary failure_category=llm_error",
                            context.run_id,
                            context.step,
                        )
                        context.mark_failed("llm_error")
                        break

                if context.is_done():
                    continue

                if response.stop_reason == "max_tokens" and response.tool_calls:
                    # A truncated tool_use block is not an executable or
                    # replayable inference unit.  Drop the whole partial
                    # response and terminate without synthetic tool results.
                    context.mark_failed("max_tokens")
                    pending_terminal = "max_tokens"
                    continue

                # [observe] append assistant content blocks to context
                # thinking blocks必须置前，并在extended thinking模式保持原样
                # 从 LLM 响应中提取思考块，并添加到结果列表
                blocks: list[dict[str, object]] = list(response.thinking_blocks)
                if response.text:
                    blocks.append({"type": "text", "text": response.text})
                for tc in response.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                    )
                # 严格保持 think blocks → text → tool_use 顺序，思考块原样置于最前
                context.add_assistant_message(
                    blocks,
                    continuation_state=response.continuation_state,
                )

                # [act] execute each requested tool; errors become tool results so loop continues
                if response.stop_reason == "tool_use":
                    for tc in response.tool_calls:
                        result = await self._tool_invoker.invoke(tc)
                        tool_content = result.content
                        if result.terminal_receipt and result.terminal_outcome is not None:
                            render_receipt = getattr(
                                result.terminal_outcome,
                                "to_parent_receipt",
                                None,
                            )
                            if callable(render_receipt):
                                tool_content = str(render_receipt())
                        if any(
                            (
                                result.evidence_ref,
                                result.raw_truncated,
                                result.original_size is not None,
                                result.captured_size is not None,
                            )
                        ):
                            context.add_tool_result(
                                tc.id,
                                tool_content,
                                is_error=result.is_error,
                                evidence_ref=result.evidence_ref,
                                raw_truncated=result.raw_truncated,
                                original_size=result.original_size,
                                captured_size=result.captured_size,
                            )
                        else:
                            # Preserve the compact legacy call shape for
                            # custom context doubles that only accept the
                            # original tool-result arguments.
                            context.add_tool_result(
                                tc.id,
                                tool_content,
                                is_error=result.is_error,
                            )
                        terminal_reason = self._tool_invoker.terminal_reason()
                        if terminal_reason is not None:
                            context.mark_failed(terminal_reason)
                            break
                # Termination check — end_turn wins over max_steps if both hit on same step
                if response.stop_reason == "end_turn": #end_turn 优先于 max_steps。
                    pending_result = response.text or ""
                    pending_terminal = "success"
                elif context.step >= context.max_steps:
                    pending_terminal = "exceeded_max_steps"

            except asyncio.CancelledError as exc:
                if primary_failure is None:
                    primary_failure = exc
                    context.mark_failed("cancelled")
                raise
            except Exception as exc:
                if primary_failure is None:
                    primary_failure = exc
                raise
            finally:
                try:
                    await _publish_step_finished_once(
                        self._bus,
                        StepFinishedEvent(
                            run_id=context.run_id,
                            step=context.step,
                            ts=_now(),
                        ),
                    )
                except asyncio.CancelledError:
                    if primary_failure is None:
                        context.mark_failed("cancelled")
                        raise
                    primary_category = _primary_failure_category(
                        primary_failure,
                        context,
                    )
                    log.error(
                        "step terminal cancellation treated as secondary "
                        "run_id=%s step=%d failure_role=secondary "
                        "primary_category=%s",
                        context.run_id,
                        context.step,
                        primary_category,
                    )
                except Exception:
                    if primary_failure is None:
                        raise
                    primary_category = _primary_failure_category(
                        primary_failure,
                        context,
                    )
                    log.error(
                        "step terminal delivery failure treated as secondary "
                        "run_id=%s step=%d failure_role=secondary "
                        "primary_category=%s",
                        context.run_id,
                        context.step,
                        primary_category,
                    )

            if pending_terminal == "success":
                context.result = pending_result
                context.mark_success()
            elif pending_terminal == "exceeded_max_steps":
                context.mark_failed("exceeded_max_steps")
