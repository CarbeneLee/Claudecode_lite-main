from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from kama_claude.core.bus.envelope import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    HandlerError,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)
from kama_claude.core.trace.record import TraceRecord
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.core.transport.connection import ConnectionContext
from kama_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster

logger = logging.getLogger(__name__)

type CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]

_connection_var: ContextVar[ConnectionContext] = ContextVar("_connection_var")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 返回当前 handler 调用所属连接的 lifecycle context
def get_connection_context() -> ConnectionContext:
    return _connection_var.get()

_MAX_LINE_BYTES = 64 * 1024 * 1024  # 64 MB per frame，兼容 MCP 大文件工具结果


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        broadcaster: IpcEventBroadcaster | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._handlers: dict[str, CommandHandler] = {}
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster = broadcaster
        self._trace = trace
        self._active_connections: set[ConnectionContext] = set()

    # 注册一个方法名对应的命令处理函数
    def register(self, method: str, handler: CommandHandler) -> None:
        self._handlers[method] = handler

    # 启动 TCP 服务器；若端口已被占用则退出进程
    async def start(self) -> str:
        try:
            _r, w = await asyncio.open_connection(self._host, self._port)
            w.close()
            await w.wait_closed()
            raise SystemExit(f"core already running at {self._host}:{self._port}")
        except (ConnectionRefusedError, OSError):
            pass

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
            limit=_MAX_LINE_BYTES,
        )
        return f"{self._host}:{self._port}"

    # 关闭服务器：先断开所有活跃连接，再等待服务器完全关闭（最多 2 秒）
    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        connections = list(self._active_connections)
        if connections:
            await asyncio.gather(
                *(context.close("server stop") for context in connections),
                return_exceptions=True,
            )
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

    # 处理单个客户端连接，完成后关闭写流
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        connection_id = f"conn-{uuid.uuid4().hex[:8]}"
        context = ConnectionContext(
            writer,
            connection_id=connection_id,
            on_close=(
                self._broadcaster.unsubscribe_all
                if self._broadcaster is not None
                else None
            ),
        )
        context.start()
        logger.debug("client connected connection_id=%s peer=%s", connection_id, peer)
        self._active_connections.add(context)
        try:
            await self._read_loop(reader, context)
        finally:
            await context.close("peer EOF")
            self._active_connections.discard(context)
            logger.debug(
                "client disconnected connection_id=%s peer=%s",
                connection_id,
                peer,
            )

    # 持续读取换行分隔的 JSON 行并逐行分发处理
    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        context: ConnectionContext,
    ) -> None:
        while True:
            try:
                line = await reader.readline()
            except asyncio.LimitOverrunError:
                await self._send(
                    context,
                    make_error(None, INVALID_REQUEST, "Request too large"),
                )
                return

            if not line:
                return

            # 每条命令独立作为 task 执行，避免长时间运行的 handler（如 session.send_message）
            # 阻塞读循环，使 permission.respond 等并发命令能被及时处理
            context.create_request_task(self._handle_line(line, context))

    # 解析单行 JSON-RPC 请求并调用对应 handler，将结果或错误写回客户端
    async def _handle_line(
        self,
        line: bytes,
        context: ConnectionContext,
    ) -> None:
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as e:
            await self._send(
                context,
                make_error(None, PARSE_ERROR, f"Parse error: {e}"),
            )
            return

        try:
            req = JsonRpcRequest.model_validate(raw)
        except ValidationError as e:
            await self._send(
                context,
                make_error(None, INVALID_REQUEST, "Invalid Request", str(e)),
            )
            return

        if self._trace is not None:
            client_id = str(context.peername)
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CLIENT→CORE",
                    layer="ipc",
                    kind="command",
                    client_id=client_id,
                    data={"method": req.method, "id": req.id, "params": req.params},
                )
            )

        handler = self._handlers.get(req.method)
        if handler is None:
            await self._send(
                context,
                make_error(req.id, METHOD_NOT_FOUND, f"Method not found: {req.method}"),
            )
            return

        connection_token = _connection_var.set(context)
        try:
            result = await handler(req.params)
        except HandlerError as e:
            await self._send(context, make_error(req.id, e.code, str(e), e.data))
            return
        except ValidationError as e:
            await self._send(
                context,
                make_error(req.id, INVALID_REQUEST, "Invalid params", str(e)),
            )
            return
        except Exception as e:
            logger.exception("handler %s raised: %s", req.method, e)
            await self._send(
                context,
                make_error(req.id, INTERNAL_ERROR, "Internal error"),
            )
            return
        finally:
            _connection_var.reset(connection_token)

        result_data: Any = result.model_dump() if isinstance(result, BaseModel) else result
        try:
            await self._send(context, JsonRpcSuccess(id=req.id, result=result_data))
        except (ConnectionError, OSError):
            logger.debug("client disconnected before response for %s", req.method)

    # 将响应放入 control queue 并等待唯一 writer 完成 drain
    async def _send(self, context: ConnectionContext, msg: BaseModel) -> None:
        # 在成功 drain 后记录响应 trace，避免把未交付帧误报为已发送
        def emit_trace() -> None:
            if self._trace is None:
                return
            kind = "error" if isinstance(msg, JsonRpcError) else "response"
            client_id = str(context.peername)
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE→CLIENT",
                    layer="ipc",
                    kind=kind,
                    client_id=client_id,
                    data=msg.model_dump(),
                )
            )

        receipt = context.enqueue_control(msg, on_written=emit_trace)
        await receipt.written
