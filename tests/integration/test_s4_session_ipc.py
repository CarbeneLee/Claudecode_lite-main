from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path


# 发送一条 JSON-RPC 请求并返回响应对象
async def _send_recv(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
    params: dict,
    req_id: str = "1",
) -> dict:
    req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    return json.loads(line)


# 功能：验证 daemon 暴露 session.create、session.get_history、session.close 三个 S4 IPC 命令
# 设计：不触发 session.send_message，避免真实 LLM 依赖；只验证 CoreApp handler 注册、协议序列化和 session 状态持久化
async def test_session_create_history_close_over_ipc(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)

    created = await _send_recv(
        reader,
        writer,
        "session.create",
        {
            "mode": "chat",
            "title": "ipc test",
            "workspace_root": str(tmp_path.resolve()),
        },
        req_id="create",
    )
    assert "result" in created, created
    session_id = created["result"]["session_id"]
    assert created["result"]["status"] == "active"

    history = await _send_recv(
        reader,
        writer,
        "session.get_history",
        {"session_id": session_id},
        req_id="history",
    )
    assert history["result"]["messages"] == []

    closed = await _send_recv(
        reader,
        writer,
        "session.close",
        {"session_id": session_id},
        req_id="close",
    )
    assert closed["result"]["status"] == "closed"

    writer.close()
    await writer.wait_closed()


# 功能：验证真实 daemon 对相对 workspace_root 返回稳定 JSON-RPC domain error
# 设计：发送原始 NDJSON 请求并断言 code、message、data，覆盖 SocketServer 的 HandlerError 转换
async def test_session_create_rejects_relative_workspace_over_ipc(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)

    response = await _send_recv(
        reader,
        writer,
        "session.create",
        {"mode": "chat", "workspace_root": "relative/project"},
        req_id="invalid-workspace",
    )

    assert response["error"] == {
        "code": -32013,
        "message": "invalid workspace_root",
        "data": {"reason": "not_absolute"},
    }
    writer.close()
    await writer.wait_closed()
