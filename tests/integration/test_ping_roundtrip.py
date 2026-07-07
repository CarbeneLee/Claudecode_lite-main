from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys


# 功能：验证真实 daemon 响应 core.ping 命令并返回包含版本、uptime、时间戳的 PongResult
# 设计：通过原始 TCP 连接发送 JSON-RPC 帧（不经过任何 SDK 客户端层），直接验证 wire 协议的端到端正确性
async def test_ping_returns_pong(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "core.ping",
        "params": {"client": "test/0.0.1"},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "test-1"
    assert "result" in resp
    assert resp["result"]["server_version"] == "0.0.1"
    assert resp["result"]["uptime_ms"] >= 0
    assert "received_at" in resp["result"]


# 功能：验证调用未注册方法时 daemon 返回 METHOD_NOT_FOUND 错误码（-32601）
# 设计：检查精确的 JSON-RPC 错误码，确认 SocketServer 的路由失败路径符合 JSON-RPC 2.0 规范
async def test_unknown_method_returns_error(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "test-2",
        "method": "core.nonexistent",
        "params": {},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert "error" in resp
    assert resp["error"]["code"] == -32601  # METHOD_NOT_FOUND


# 功能：验证发送非 JSON 数据时 daemon 返回 PARSE_ERROR（-32700）并不崩溃
# 设计：发送裸文本而非 JSON，检查错误码，确认 daemon 对格式错误输入的健壮性（不因单个坏帧终止服务）
async def test_invalid_json_returns_error(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    writer.write(b"not valid json\n")
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert "error" in resp
    assert resp["error"]["code"] == -32700  # PARSE_ERROR


# 功能：验证真实 daemon 注册了 core.echo，并能通过 TCP JSON-RPC 返回 EchoResult
# 设计：直接发送原始 NDJSON 帧，不经过 CLI/SocketClient，隔离验证 SocketServer 到 CoreApp handler 的链路
async def test_echo_returns_message_over_tcp(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "echo-1",
        "method": "core.echo",
        "params": {"message": "hello integration"},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "echo-1"
    assert "result" in resp
    assert resp["result"]["server_version"] == "0.0.1"
    assert resp["result"]["message"] == "hello integration"
    assert "received_at" in resp["result"]


# 功能：验证缺少 message 的 core.echo 请求会稳定返回 JSON-RPC error
# 设计：发送通过 JSON-RPC envelope 但不满足 EchoCommand 的 params，覆盖 handler 参数校验失败路径
async def test_echo_missing_message_returns_jsonrpc_error(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "echo-2",
        "method": "core.echo",
        "params": {},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "echo-2"
    assert "error" in resp
    assert resp["error"]["code"] == -32600
    assert resp["error"]["message"] == "Invalid params"


# 功能：验证 kama echo hello 通过真实 CLI 子进程完成端到端回显
# 设计：用 python -m kama_claude.cli 走 argparse、CLI、TCP、SocketServer、JsonRpcSuccess、EchoResult 全链路
async def test_echo_cli_positional_message_roundtrip(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    env = os.environ.copy()
    env["KAMA_PORT"] = str(free_port)
    env["KAMA_LOG_FILE"] = ""
    env["KAMA_LOG_LEVEL"] = "WARNING"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "kama_claude.cli",
        "echo",
        "hello",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

    assert proc.returncode == 0
    assert stderr == b""
    assert "server=0.0.1" in stdout.decode()
    assert "message=hello" in stdout.decode()
