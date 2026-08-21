# AGENTS.md

This file provides guidance to codex when working with code in this repository.

## Commands

```bash
# Install / sync dependencies
uv sync

# Lint
uv run ruff check src tests scripts
uv run mypy src

# Tests
uv run pytest tests/unit -v           # unit only (fast, no daemon)
uv run pytest tests/integration -v    # needs no running daemon; fixture spawns one
uv run pytest tests/ -v               # all

# Single test
uv run pytest tests/unit/test_envelope.py::test_request_roundtrip -v

# Regenerate WIRE_PROTOCOL.md after changing bus models
uv run python scripts/gen_protocol_doc.py

# Verify WIRE_PROTOCOL.md is in sync (used in CI equivalent)
uv run python scripts/gen_protocol_doc.py --check

# Run daemon manually
uv run kama-core                        # foreground; Ctrl+C to stop
KAMA_PORT=8000 uv run kama-core        # override port

# Send a ping
uv run kama ping
uv run kama --version
```

## Architecture

This is a **dual-process** local AI agent system. `kama-core` is a persistent daemon; `kama` and `kama-tui` are clients that connect to it over TCP loopback using JSON-RPC 2.0 NDJSON.

```
kama-core (daemon)
  └─ listens on 127.0.0.1:7437 (TCP)
       ↑ JSON-RPC 2.0 NDJSON
kama (CLI)   kama-tui (TUI, S2+)
```

**`kama-tui` is the primary frontend.** All user-facing work on task management, observability, and interaction should be designed for and validated in the TUI first. The `kama` CLI exists only for quick scripted testing and debugging — it is not a product surface. When implementing features that touch the user interface, invest in the TUI layout, event rendering, and keyboard interactions. Do not shortcut TUI work by pointing to the CLI as an alternative.

### Protocol layer (`src/kama_claude/core/bus/`)

All IPC messages are typed pydantic v2 models with a **discriminated union on the `type` field**. This is the contract boundary — adding a new command or event means adding a new model class to `commands.py` or `events.py` and extending the `Command`/`Event` union.

- `envelope.py` — `JsonRpcRequest`, `JsonRpcSuccess`, `JsonRpcError`, error code constants, `make_error()`
- `commands.py` — `Command` union and command/result models; current protocol surface is defined by this file
- `events.py` — `Event` union and event models; current event surface is defined by this file

`WIRE_PROTOCOL.md` is **generated** from these models by `scripts/gen_protocol_doc.py`. The current command/event protocol is authoritative in `commands.py`, `events.py`, and the generated `WIRE_PROTOCOL.md`. Always regenerate and commit `WIRE_PROTOCOL.md` after changing bus models.

### Transport layer (`src/kama_claude/core/transport/`)

- `socket_server.py` — TCP server (`asyncio.start_server`); reads NDJSON lines, dispatches to registered `CommandHandler`s, handles JSON-RPC error cases. On `start()`, probes `host:port` first — errors if another daemon is already listening. Handlers registered via `server.register("method.name", handler_fn)`.

### Config (`src/kama_claude/core/config.py`)

Priority: **built-in defaults → `~/.kama/config.toml` → `.kama/config.toml` → `.env` → env vars**.

Current config groups include `core` (`host`, `port`), `logging`, `agent`, `llm`, `trace`, `permission`, `compaction`, and `mcp`. Config files are silently skipped if absent; unknown keys cause a hard exit.

Relevant env vars include `KAMA_CONFIG`, `KAMA_HOST`, `KAMA_PORT`, `KAMA_LOG_LEVEL`, `KAMA_LOG_FILE`, `KAMA_LOG_FORMAT`, `KAMA_MAX_STEPS`, `KAMA_LLM_DEFAULT_MODEL`, `KAMA_TRACE_ENABLED`, `KAMA_TRACE_FILE`, `KAMA_TRACE_INCLUDE_LLM_PAYLOAD`, `KAMA_PERMISSION_TIMEOUT_S`, `KAMA_COMPACT_THRESHOLD`, `KAMA_COMPACT_TOOL_LIMIT`, and `KAMA_COMPACT_TOOL_KEEP`.

### Daemon entry (`src/kama_claude/core/app.py`)

`CoreApp.run()` is the single async entry point: loads config, sets up logging, initializes `EventBus`, optional `TraceWriter`, `IpcEventBroadcaster`, `SessionManager`, `PermissionManager`, optional MCP servers, and `AgentRunner` wiring, then creates `SocketServer`, registers handlers, waits for `SIGINT`/`SIGTERM`, and calls `server.stop()`. Adding new handlers: instantiate a handler method on `CoreApp` and call `server.register()`.

### Testing

Integration tests in `tests/conftest.py` spawn a real daemon subprocess using a random free port (via `free_port` fixture). The fixture finds a free port, releases it, passes it to the daemon via `KAMA_PORT`, then polls `asyncio.open_connection` until the daemon is ready.

### Code style

All functions must have a **single-line Chinese comment** immediately above the `def` line explaining what the function does. Example:

```python
# 发送 JSON-RPC 响应并刷新写缓冲区
async def _send(self, writer: asyncio.StreamWriter, msg: BaseModel) -> None:
    ...
```

Do not write multi-line docstrings; one concise Chinese line is enough.

**Test functions** require **two Chinese comment lines** immediately above the `def` line:

```python
# 功能：验证 publish 后订阅者能收到事件对象
# 设计：用内联 handler 收集事件引用，断言 is 而非 ==，排除序列化中间步骤的干扰
async def test_publish_reaches_subscriber() -> None:
    ...
```

- `# 功能：` — 该测试验证的具体行为或不变式，一句话说清楚"测什么"
- `# 设计：` — 为什么选择这种测试方式：覆盖了什么边界条件、为什么用这个 stub/fixture、这种断言方式相比其他方式的优势

两行注释缺一不可。功能行让读者 5 秒内判断测试意图；设计行让读者理解测试背后的决策，而非只看到操作步骤。

### Design docs (outside the repo)

The planning documents live in `../docs/` (sibling of this repo, not committed here):
- `agent_development_plan.md` — staged development roadmap S0–S8
- `s0_implementation_plan.md` — detailed S0 decisions and rationale
- `agent_functional_outline.md` — full feature catalogue

## Learning mode / current milestone

This repository is currently used as a staged learning project.

For S0–S2 learning tasks, prioritize core protocol, daemon, CLI, tests, and event flow. TUI work should not be expanded unless the task explicitly touches user-facing UI.

For product-level features after S2, TUI remains the primary frontend and should be validated when the feature affects user interaction.

## Project goals and learning deliverables

- Build a local Coding Agent Runtime around AgentLoop, daemon/CLI/TUI, workspace-safe tools, sessions/context, Subagent, MCP, and observability.
- Treat Docker reproducible deployment and repo-task evaluation as explicit future gaps, not current capabilities.
- Every feature phase must deliver a reviewed implementation plan, RED/GREEN evidence, focused/full gates, an external diff report, and one learning devlog.
- Every phase devlog must include comprehension questions with complete answers, interview questions with answer frameworks, resume-claim impact, and known limitations.
- Roadmap entries are not current capabilities; never describe unimplemented work as completed or fabricate implementation/tests to match a resume claim.
- README, devlogs, and resume matrices must distinguish `Implemented`, `Tested`, `Benchmarked`, and `Planned` evidence.
- Devlogs must not contain API keys, tokens, private file contents, job-application details, or unredacted absolute paths.
- Never automatically commit or push learning work, and never modify another repository while completing this repository's phase deliverables.
- Use [docs/learning/README.md](docs/learning/README.md) for the evidence matrix and devlog workflow.
- Use [docs/learning/devlog/TEMPLATE.md](docs/learning/devlog/TEMPLATE.md) for every new phase learning record.

## Agentic Harness Safety

Agentic test harnesses are part of the runtime safety boundary.

Any test involving AgentLoop, SpawnAgentTool, subagents, orchestration,
scripted/fake LLM providers, or repeated tool calling MUST be finite by construction.

Required invariants:

1. Scripted/fake providers must implement an explicit finite-state interaction.
2. Every provider interaction path must have a terminal response.
3. Unexpected provider calls must raise immediately; never return another tool call as a fallback.
4. Agentic tests must enforce explicit budgets for:
   - total provider calls;
   - root-agent provider calls;
   - child-agent spawns.
5. A budget violation is a deterministic test/harness failure. Do not retry it and do not increase the budget merely to obtain a passing test.
6. Tests must not rely on "the model should eventually stop".
7. When changing an agentic harness, run the exact affected test before broader focused or full suites.
8. Do not run multiple broad agentic pytest suites concurrently.
9. Resource-safety failures must stop the current execution before further tests are launched.

The safety hierarchy is:

    bounded test harness
    > runtime/test assertions
    > AGENTS.md instructions
    > prompt compliance

Prompt instructions are not a substitute for hard harness bounds.
