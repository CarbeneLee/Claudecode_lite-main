from __future__ import annotations

import asyncio
import json
import sys
import time

import kama_claude
from kama_claude.core.bus.commands import PongResult
from kama_claude.core.bus.envelope import JsonRpcError, JsonRpcSuccess
from kama_claude.core.config import KamaConfig

def cmd_echo(config: KamaConfig, message: str) -> None:
    try:
        asyncio.run(_echo(config, message))
    except (ConnectionRefusedError, OSError):
        print(f"error: core not running ({config.host}:{config.port})", file=sys.stderr)
        sys.exit(1)

async def _echo(config: KamaConfig, message: str) -> None:
    t0 = time.monotonic()
    reader, writer = await asyncio.open_connection(config.host, config.port)

    req = {
        "jsonrpc": "2.0",
        "id": "cli-1",
        "method": "core.echo",
        "params": {"message": message},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=10.0)
    latency_ms = int((time.monotonic() - t0) * 1000)

    writer.close()
    await writer.wait_closed()

    raw = json.loads(line)
    if "error" in raw:
        err = JsonRpcError.model_validate(raw)
        print(f"error: {err.error.code} {err.error.message}", file=sys.stderr)
        sys.exit(1)

    resp = JsonRpcSuccess.model_validate(raw)
    result = PongResult.model_validate(resp.result)
    print(f"echo response={result.server_version} latency={latency_ms}ms")


        