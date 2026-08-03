from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.bus.events import (
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.runs import new_run_id
from kama_claude.core.session.model import Session, SessionMode
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader

if TYPE_CHECKING:
    from kama_claude.core.events.journal import EventJournalCoordinator
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.runner import AgentRunner

logger = logging.getLogger(__name__)

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    # 初始化会话管理器，接入文件存储、runner 工厂、事件总线和可选的 LLM provider（用于手动压缩）
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[Path], AgentRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
        journal: EventJournalCoordinator | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._provider = provider
        self._journal = journal
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 每个 session 至多一个后台 run task；send_message 注册后立即返回
        self._running_runs: dict[str, asyncio.Task[None]] = {}

    # 在 daemon wiring 完成且任何 session 创建前注入 shared journal coordinator
    def attach_journal(self, journal: EventJournalCoordinator) -> None:
        if self._sessions:
            raise RuntimeError("journal must be attached before session creation")
        self._journal = journal

    # 创建新 session 并写入 meta.json
    async def create(
        self,
        mode: SessionMode,
        title: str = "",
        *,
        workspace_root: Path,
    ) -> Session:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        ts = _now()
        session = Session(
            id=sid,
            mode=mode,
            status="active",
            title=title,
            created_at=ts,
            updated_at=ts,
            workspace_root=workspace_root,
            run_ids=[],
        )
        session_path = self._store.ensure_session_dir(sid)
        if self._journal is not None:
            await self._journal.register_session(sid, session_path)
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=ts))
        return session

    # 处理用户消息，追加 thread 并启动一次 agent run
    async def send_message(
        self,
        sid: str,
        content: str,
        *,
        run_id: str | None = None,
        run_registered: asyncio.Event | None = None,
    ) -> str:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked() or sid in self._running_runs:
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            if sid in self._running_runs:
                raise HandlerError(SESSION_BUSY, "session busy")
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )

            if not session.title:
                session.title = content[:40]

            run_id = run_id or new_run_id()
            run_path = self._store.runs_dir(sid) / run_id
            run_path.mkdir(parents=True, exist_ok=True)
            if self._journal is not None:
                await self._journal.register_run(
                    run_id,
                    run_path,
                    session_id=sid,
                )
            if run_registered is not None:
                run_registered.set()
            session.run_ids.append(run_id)
            session.updated_at = _now()
            self._store.write_meta(session)

            # Skill 解析：检测 "/" 前缀，展开为系统提示覆盖和工具白名单
            goal = content
            system_prompt_override: str | None = None
            tool_whitelist: list[str] | None = None
            if content.startswith("/"):
                skill_loader = SkillLoader(session.workspace_root)
                parts = content[1:].split(None, 1)
                skill_name = parts[0]
                arguments = parts[1] if len(parts) > 1 else ""
                skill = skill_loader.resolve(skill_name)
                if skill is not None:
                    goal = skill_loader.render_prompt(skill, arguments)
                    system_prompt_override = skill.system_prompt_template
                    tool_whitelist = skill.allowed_tools or None
                    await self._bus.publish(
                        SkillInvokedEvent(
                            skill_name=skill_name,
                            arguments=arguments,
                            run_id=run_id,
                            ts=_now(),
                        )
                    )

            # run 在后台 task 中执行：send_message 立即返回，CLI 主循环不再被阻塞
            task = asyncio.create_task(
                self._run_and_finalize(
                    sid,
                    run_id,
                    goal,
                    system_prompt_override=system_prompt_override,
                    tool_whitelist=tool_whitelist,
                )
            )
            self._running_runs[sid] = task
            return run_id

    # 在后台执行 agent run；完成后在 lock 内收敛 session 状态并发布事件
    async def _run_and_finalize(
        self,
        sid: str,
        run_id: str,
        goal: str,
        *,
        system_prompt_override: str | None,
        tool_whitelist: list[str] | None,
    ) -> None:
        try:
            try:
                session = self._get_session(sid)
                runner = self._runner_factory(session.workspace_root)
                await runner.run_and_capture(
                    goal,
                    run_id=run_id,
                    session=session,
                    store=self._store,
                    system_prompt_override=system_prompt_override,
                    tool_whitelist=tool_whitelist,
                )
            except asyncio.CancelledError:
                # close()/shutdown 已收敛状态，不再发布 waiting_for_input
                raise
            except Exception:
                logger.exception("run failed sid=%s run_id=%s", sid, run_id)
                return
            session = self._get_session(sid)
            lock = self._locks[sid]
            async with lock:
                if session.status == "closed":
                    return
                session.updated_at = _now()
                if session.mode == "one_shot":
                    session.status = "closed"
                    await self._bus.publish(
                        SessionClosedEvent(session_id=sid, ts=session.updated_at)
                    )
                else:
                    session.status = "waiting_for_input"
                    await self._bus.publish(
                        SessionWaitingForInputEvent(
                            session_id=sid,
                            last_run_id=run_id,
                            ts=session.updated_at,
                        )
                    )
                self._store.write_meta(session)
        finally:
            self._running_runs.pop(sid, None)

    # 关闭指定 session 并更新 meta.json；取消仍在执行的后台 run
    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))
        run_task = self._running_runs.get(sid)
        if run_task is not None and not run_task.done():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)

    # 当前仍有未完成 run 的 task 列表（测试与 shutdown 使用）
    def active_run_tasks(self) -> list[asyncio.Task[None]]:
        return [t for t in self._running_runs.values() if not t.done()]

    # 取消全部后台 run（daemon shutdown 路径）
    async def cancel_active_runs(self) -> None:
        tasks = self.active_run_tasks()
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # 手动压缩指定 session 的 thread，将摘要持久化写入 thread.jsonl
    async def compact(self, sid: str, focus: str = "") -> Any:
        self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(-32020, "provider not available for compaction")
        async with lock:
            from kama_claude.core.bus.commands import SessionCompactResult
            from kama_claude.core.compact.compactor import Compactor
            messages = self._store.read_messages(sid)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(self._bus, session_dir, sid)
            result = await compactor.compact_messages(messages, self._provider, focus=focus)
            if result is None:
                raise HandlerError(-32021, "compaction failed or not beneficial")
            self._store.write_compacted(sid, [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant", "content": "Understood, I'll continue from this summary."},
            ])
            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    # 读取指定 session 的完整 thread 历史
    async def get_history(self, sid: str) -> list[dict[str, Any]]:
        self._get_session(sid)
        return self._store.read_messages(sid)

    # 从内存索引取 session，不存在时抛 JSON-RPC 结构化错误
    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "session not found")
        return session
