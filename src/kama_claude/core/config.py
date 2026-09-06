from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from kama_claude.core.git.config import GitConfig
from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.semantic.config import SemanticConfig

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7437
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILE = "~/.kama/logs/core.log"
_DEFAULT_LOG_FORMAT = "text"
_DEFAULT_CONFIG_PATH = "~/.kama/config.toml"
_DEFAULT_MAX_STEPS = 20
_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TRACE_FILE = "~/.kama/traces/daemon.jsonl"

# [git] TOML 组允许的 key 与类型约束（严格校验，与其他配置组一致）
_GIT_TOML_KEYS = frozenset(
    {
        "enabled",
        "checkpoint_mode",
        "branch_prefix",
        "mode",
        "auto_rollback_on_fail",
        "author",
        "checkpoint_namespace",
        "squash_on_finalize",
        "keep_checkpoint_refs",
        "rollback_strategy",
    }
)
_GIT_TOML_TYPES: dict[str, type] = {
    "enabled": bool,
    "checkpoint_mode": str,
    "branch_prefix": str,
    "mode": str,
    "auto_rollback_on_fail": bool,
    "author": str,
    "checkpoint_namespace": str,
    "squash_on_finalize": bool,
    "keep_checkpoint_refs": bool,
    "rollback_strategy": str,
}

# [semantic] TOML 组允许的 key 与类型约束（严格校验，与其他配置组一致）
_SEMANTIC_TOML_KEYS = frozenset(
    {
        "enabled",
        "strategy",
        "index_dir",
        "chunk_size",
        "min_chunk_lines",
        "ngram_n",
        "default_top_k",
        "similarity_threshold",
        "max_index_files",
        "max_file_bytes",
        "total_index_bytes",
        "degradation",
        "max_query_chars",
    }
)
_SEMANTIC_TOML_TYPES: dict[str, type] = {
    "enabled": bool,
    "strategy": str,
    "index_dir": str,
    "chunk_size": int,
    "min_chunk_lines": int,
    "ngram_n": int,
    "default_top_k": int,
    "similarity_threshold": float,
    "max_index_files": int,
    "max_file_bytes": int,
    "total_index_bytes": int,
    "degradation": str,
    "max_query_chars": int,
}


@dataclass
class LoggingConfig:
    level: str = _DEFAULT_LOG_LEVEL
    file: str = _DEFAULT_LOG_FILE
    format: str = _DEFAULT_LOG_FORMAT  # "text" | "json"


@dataclass
class AgentConfig:
    max_steps: int = _DEFAULT_MAX_STEPS


@dataclass
class LlmConfig:
    default_model: str = _DEFAULT_MODEL
    router: str = "static"  # "static" | "rule_based" (S4) | "cost_budget" (S6)


@dataclass
class TraceConfig:
    enabled: bool = True
    file: str = _DEFAULT_TRACE_FILE
    include_llm_payload: bool = True  # false 时 LLM 记录只保留摘要


@dataclass
class PermissionConfig:
    timeout_s: float = 60.0  # 审批超时秒数；0 表示不超时


@dataclass
class CompactionConfig:
    auto_threshold: float = 0.0    # legacy override；新配置默认使用 soft_trigger_ratio
    auto_compact: bool = True      # 新配置默认启用 model-aware proactive compaction
    soft_trigger_ratio: float = 0.80
    target_ratio: float = 0.60
    recent_tail_ratio: float = 0.10
    recent_tail_max_tokens: int = 64 * 1024
    summary_max_tokens: int = 4_096
    tool_result_limit: int = 8_000  # tool_result 截断触发字符数
    tool_result_keep: int = 4_000   # 截断后保留的前缀字符数

    # 返回是否允许 AgentLoop 执行 proactive compaction
    @property
    def proactive_enabled(self) -> bool:
        return self.auto_compact

    # 解析 legacy threshold 与新 soft trigger 的统一运行时阈值
    @property
    def effective_soft_trigger_ratio(self) -> float:
        if self.auto_threshold > 0:
            return self.auto_threshold
        return self.soft_trigger_ratio

    # 兼容旧调用方，同时明确 auto_compact=false 只关闭 proactive compaction
    @property
    def effective_threshold(self) -> float:
        return self.effective_soft_trigger_ratio if self.proactive_enabled else 0.0


@dataclass
class McpServerConfig:
    name: str
    transport: str = "stdio"       # "stdio" | "tcp"
    command: str = ""              # stdio 专用：可执行文件路径
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    host: str = "localhost"        # tcp 专用
    port: int = 3000               # tcp 专用


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class KamaConfig:
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    git: GitConfig = field(default_factory=GitConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)


# 构建并返回运行时配置：默认值 → 全局 TOML → 项目本地 TOML → .env → 系统环境变量（后者优先级最高）
def get_config() -> KamaConfig:
    config = KamaConfig()

    # .env 必须在读取 KAMA_CONFIG 之前加载，以便 .env 中的 KAMA_CONFIG 能影响 TOML 路径
    load_dotenv(".env", override=False)

    # 若显式指定 KAMA_CONFIG，只读该文件；否则按优先级叠加：全局 → 项目本地
    explicit = os.environ.get("KAMA_CONFIG")
    if explicit:
        config_paths = [Path(explicit).expanduser()]
    else:
        config_paths = [
            Path(_DEFAULT_CONFIG_PATH).expanduser(),
            Path(".kama/config.toml"),
        ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise SystemExit(f"Config parse error ({config_path}): {e}") from e
            _apply_toml(config, data)

    _apply_env(config)
    return config


# 将已解析的 TOML 根表写入 config；未知小节或类型错误时退出进程
def _apply_toml(config: KamaConfig, data: dict[str, Any]) -> None:
    unknown = set(data.keys()) - {
        "core",
        "logging",
        "agent",
        "llm",
        "trace",
        "permission",
        "compaction",
        "mcp",
        "sandbox",
        "git",
        "semantic",
    }
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {', '.join(sorted(unknown))}")

    if "core" in data:
        core = data["core"]
        if not isinstance(core, dict):
            raise SystemExit("Config error: [core] must be a table")
        unknown_core: set[str] = set(core.keys()) - {"host", "port"}
        if unknown_core:
            raise SystemExit(f"Unknown [core] keys: {', '.join(sorted(unknown_core))}")
        if "host" in core:
            val = core["host"]
            if not isinstance(val, str):
                raise SystemExit("Config error: core.host must be a string")
            config.host = val
        if "port" in core:
            val = core["port"]
            if not isinstance(val, int):
                raise SystemExit("Config error: core.port must be an integer")
            config.port = val

    if "logging" in data:
        log = data["logging"]
        if not isinstance(log, dict):
            raise SystemExit("Config error: [logging] must be a table")
        unknown_log: set[str] = set(log.keys()) - {"level", "file", "format"}
        if unknown_log:
            raise SystemExit(f"Unknown [logging] keys: {', '.join(sorted(unknown_log))}")
        for key in ("level", "file", "format"):
            if key in log:
                val = log[key]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: logging.{key} must be a string")
                setattr(config.logging, key, val)

    if "agent" in data:
        agent = data["agent"]
        if not isinstance(agent, dict):
            raise SystemExit("Config error: [agent] must be a table")
        unknown_agent: set[str] = set(agent.keys()) - {"max_steps"}
        if unknown_agent:
            raise SystemExit(f"Unknown [agent] keys: {', '.join(sorted(unknown_agent))}")
        if "max_steps" in agent:
            val = agent["max_steps"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit("Config error: agent.max_steps must be a positive integer")
            config.agent.max_steps = val

    if "llm" in data:
        llm = data["llm"]
        if not isinstance(llm, dict):
            raise SystemExit("Config error: [llm] must be a table")
        unknown_llm: set[str] = set(llm.keys()) - {"default_model", "router"}
        if unknown_llm:
            raise SystemExit(f"Unknown [llm] keys: {', '.join(sorted(unknown_llm))}")
        if "default_model" in llm:
            val = llm["default_model"]
            if not isinstance(val, str):
                raise SystemExit("Config error: llm.default_model must be a string")
            config.llm.default_model = val
        if "router" in llm:
            val = llm["router"]
            if not isinstance(val, str):
                raise SystemExit("Config error: llm.router must be a string")
            config.llm.router = val

    if "trace" in data:
        trace = data["trace"]
        if not isinstance(trace, dict):
            raise SystemExit("Config error: [trace] must be a table")
        unknown_trace: set[str] = set(trace.keys()) - {"enabled", "file", "include_llm_payload"}
        if unknown_trace:
            raise SystemExit(f"Unknown [trace] keys: {', '.join(sorted(unknown_trace))}")
        if "enabled" in trace:
            val = trace["enabled"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.enabled must be a boolean")
            config.trace.enabled = val
        if "file" in trace:
            val = trace["file"]
            if not isinstance(val, str):
                raise SystemExit("Config error: trace.file must be a string")
            config.trace.file = val
        if "include_llm_payload" in trace:
            val = trace["include_llm_payload"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.include_llm_payload must be a boolean")
            config.trace.include_llm_payload = val

    if "permission" in data:
        perm = data["permission"]
        if not isinstance(perm, dict):
            raise SystemExit("Config error: [permission] must be a table")
        unknown_perm: set[str] = set(perm.keys()) - {"timeout_s"}
        if unknown_perm:
            raise SystemExit(f"Unknown [permission] keys: {', '.join(sorted(unknown_perm))}")
        if "timeout_s" in perm:
            val = perm["timeout_s"]
            if not isinstance(val, (int, float)) or val < 0:
                raise SystemExit("Config error: permission.timeout_s must be a non-negative number")
            config.permission.timeout_s = float(val)

    if "compaction" in data:
        comp = data["compaction"]
        if not isinstance(comp, dict):
            raise SystemExit("Config error: [compaction] must be a table")
        unknown_comp: set[str] = set(comp.keys()) - {
            "auto_threshold",
            "auto_compact",
            "soft_trigger_ratio",
            "target_ratio",
            "recent_tail_ratio",
            "recent_tail_max_tokens",
            "summary_max_tokens",
            "tool_result_limit",
            "tool_result_keep",
        }
        if unknown_comp:
            raise SystemExit(f"Unknown [compaction] keys: {', '.join(sorted(unknown_comp))}")
        if "auto_threshold" in comp:
            val = comp["auto_threshold"]
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise SystemExit("Config error: compaction.auto_threshold must be between 0 and 1")
            config.compaction.auto_threshold = float(val)
            # Legacy config files used zero as an explicit opt-out; retain that
            # meaning unless the file also opts back in with auto_compact=true.
            if float(val) == 0.0 and "auto_compact" not in comp:
                config.compaction.auto_compact = False
        if "auto_compact" in comp:
            val = comp["auto_compact"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: compaction.auto_compact must be a boolean")
            config.compaction.auto_compact = val
        for key in ("soft_trigger_ratio", "target_ratio", "recent_tail_ratio"):
            if key in comp:
                val = comp[key]
                if not isinstance(val, (int, float)) or not (0.0 < float(val) < 1.0):
                    raise SystemExit(f"Config error: compaction.{key} must be between 0 and 1")
                setattr(config.compaction, key, float(val))
        if config.compaction.target_ratio >= config.compaction.effective_soft_trigger_ratio:
            raise SystemExit(
                "Config error: compaction.target_ratio must be below soft_trigger_ratio"
            )
        for key in ("recent_tail_max_tokens", "summary_max_tokens"):
            if key in comp:
                val = comp[key]
                if not isinstance(val, int) or val <= 0:
                    raise SystemExit(f"Config error: compaction.{key} must be positive")
                setattr(config.compaction, key, val)
        if "tool_result_limit" in comp:
            val = comp["tool_result_limit"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compaction.tool_result_limit must be a positive integer"
                )
            config.compaction.tool_result_limit = val
        if "tool_result_keep" in comp:
            val = comp["tool_result_keep"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compaction.tool_result_keep must be a positive integer"
                )
            config.compaction.tool_result_keep = val

    if "mcp" in data:
        mcp = data["mcp"]
        if not isinstance(mcp, dict):
            raise SystemExit("Config error: [mcp] must be a table")
        unknown_mcp: set[str] = set(mcp.keys()) - {"servers"}
        if unknown_mcp:
            raise SystemExit(f"Unknown [mcp] keys: {', '.join(sorted(unknown_mcp))}")
        servers_raw = mcp.get("servers", [])
        if not isinstance(servers_raw, list):
            raise SystemExit("Config error: mcp.servers must be an array of tables")
        for i, srv in enumerate(servers_raw):
            if not isinstance(srv, dict):
                raise SystemExit(f"Config error: mcp.servers[{i}] must be a table")
            name = srv.get("name")
            if not isinstance(name, str) or not name:
                raise SystemExit(f"Config error: mcp.servers[{i}].name must be a non-empty string")
            transport = srv.get("transport", "stdio")
            if transport not in ("stdio", "tcp"):
                raise SystemExit(
                    f"Config error: mcp.servers[{i}].transport must be 'stdio' or 'tcp'"
                )
            s = McpServerConfig(name=name, transport=transport)
            if "command" in srv:
                val = srv["command"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].command must be a string")
                s.command = val
            if "args" in srv:
                val = srv["args"]
                if not isinstance(val, list):
                    raise SystemExit(f"Config error: mcp.servers[{i}].args must be an array")
                s.args = [str(a) for a in val]
            if "env" in srv:
                val = srv["env"]
                if not isinstance(val, dict):
                    raise SystemExit(f"Config error: mcp.servers[{i}].env must be a table")
                s.env = {str(k): str(v) for k, v in val.items()}
            if "host" in srv:
                val = srv["host"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].host must be a string")
                s.host = val
            if "port" in srv:
                val = srv["port"]
                if not isinstance(val, int):
                    raise SystemExit(f"Config error: mcp.servers[{i}].port must be an integer")
                s.port = val
            config.mcp.servers.append(s)

    if "sandbox" in data:
        sb = data["sandbox"]
        if not isinstance(sb, dict):
            raise SystemExit("Config error: [sandbox] must be a table")
        unknown_sb: set[str] = set(sb.keys()) - {"enabled", "image", "network", "exec_timeout_s"}
        if unknown_sb:
            raise SystemExit(f"Unknown [sandbox] keys: {', '.join(sorted(unknown_sb))}")
        if "enabled" in sb:
            val = sb["enabled"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: sandbox.enabled must be a boolean")
            config.sandbox = dataclasses.replace(config.sandbox, enabled=val)
        if "image" in sb:
            val = sb["image"]
            if not isinstance(val, str):
                raise SystemExit("Config error: sandbox.image must be a string")
            config.sandbox = dataclasses.replace(config.sandbox, image=val)
        if "network" in sb:
            val = sb["network"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: sandbox.network must be a boolean")
            config.sandbox = dataclasses.replace(config.sandbox, network=val)
        if "exec_timeout_s" in sb:
            val = sb["exec_timeout_s"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: sandbox.exec_timeout_s must be a positive integer"
                )
            config.sandbox = dataclasses.replace(config.sandbox, exec_timeout_s=val)

    if "git" in data:
        git_data = data["git"]
        if not isinstance(git_data, dict):
            raise SystemExit("Config error: [git] must be a table")
        unknown_git: set[str] = set(git_data.keys()) - _GIT_TOML_KEYS
        if unknown_git:
            raise SystemExit(f"Unknown [git] keys: {', '.join(sorted(unknown_git))}")
        for key, expected_type in _GIT_TOML_TYPES.items():
            if key not in git_data:
                continue
            val = git_data[key]
            if not isinstance(val, expected_type):
                type_label = {bool: "boolean", str: "string"}.get(
                    expected_type, expected_type.__name__
                )
                raise SystemExit(
                    f"Config error: git.{key} must be a {type_label}"
                )
            try:
                config.git = dataclasses.replace(
                    config.git, **cast(dict[str, Any], {key: val})
                )
            except ValueError as exc:
                # 枚举约束（checkpoint_mode/mode/rollback_strategy）由 GitConfig 校验
                raise SystemExit(f"Config error: {exc}")

    if "semantic" in data:
        semantic_data = data["semantic"]
        if not isinstance(semantic_data, dict):
            raise SystemExit("Config error: [semantic] must be a table")
        unknown_semantic: set[str] = set(semantic_data.keys()) - _SEMANTIC_TOML_KEYS
        if unknown_semantic:
            raise SystemExit(f"Unknown [semantic] keys: {', '.join(sorted(unknown_semantic))}")
        for key, expected_type in _SEMANTIC_TOML_TYPES.items():
            if key not in semantic_data:
                continue
            val = semantic_data[key]
            # TOML 整数 1 与 1.0 视为同一阈值取值，统一转 float 后校验
            if key == "similarity_threshold" and isinstance(val, int) and not isinstance(val, bool):
                val = float(val)
            if not isinstance(val, expected_type):
                type_label = {
                    bool: "a boolean",
                    str: "a string",
                    int: "an integer",
                    float: "a number",
                }.get(expected_type, expected_type.__name__)
                raise SystemExit(
                    f"Config error: semantic.{key} must be {type_label}"
                )
            try:
                config.semantic = dataclasses.replace(
                    config.semantic, **cast(dict[str, Any], {key: val})
                )
            except ValueError as exc:
                # 枚举与数值约束（strategy/degradation/ngram_n/阈值等）由 SemanticConfig 校验
                raise SystemExit(f"Config error: {exc}")


# 用 KAMA_* 环境变量覆盖 config 中对应字段（若变量已设置）
def _apply_env(config: KamaConfig) -> None:
    host = os.environ.get("KAMA_HOST")
    if host is not None:
        config.host = host

    port_str = os.environ.get("KAMA_PORT")
    if port_str is not None:
        try:
            config.port = int(port_str)
        except ValueError:
            raise SystemExit(f"Config error: KAMA_PORT must be an integer, got: {port_str!r}")

    log_level = os.environ.get("KAMA_LOG_LEVEL")
    if log_level is not None:
        config.logging.level = log_level

    log_file = os.environ.get("KAMA_LOG_FILE")
    if log_file is not None:
        config.logging.file = log_file

    log_format = os.environ.get("KAMA_LOG_FORMAT")
    if log_format is not None:
        config.logging.format = log_format

    max_steps_str = os.environ.get("KAMA_MAX_STEPS")
    if max_steps_str is not None:
        try:
            val = int(max_steps_str)
            if val <= 0:
                raise SystemExit(
                    "Config error: KAMA_MAX_STEPS must be a positive integer,"
                    f" got: {max_steps_str!r}"
                )
            config.agent.max_steps = val
        except ValueError:
            raise SystemExit(
                f"Config error: KAMA_MAX_STEPS must be an integer, got: {max_steps_str!r}"
            )

    default_model = os.environ.get("KAMA_LLM_DEFAULT_MODEL")
    if default_model is not None:
        config.llm.default_model = default_model

    trace_enabled = os.environ.get("KAMA_TRACE_ENABLED")
    if trace_enabled is not None:
        config.trace.enabled = trace_enabled.lower() not in ("0", "false", "no")

    trace_file = os.environ.get("KAMA_TRACE_FILE")
    if trace_file is not None:
        config.trace.file = trace_file

    trace_payload = os.environ.get("KAMA_TRACE_INCLUDE_LLM_PAYLOAD")
    if trace_payload is not None:
        config.trace.include_llm_payload = trace_payload.lower() not in ("0", "false", "no")

    perm_timeout = os.environ.get("KAMA_PERMISSION_TIMEOUT_S")
    if perm_timeout is not None:
        try:
            perm_timeout_val = float(perm_timeout)
            if perm_timeout_val < 0:
                raise SystemExit(
                    f"Config error: KAMA_PERMISSION_TIMEOUT_S must be >= 0, got: {perm_timeout!r}"
                )
            config.permission.timeout_s = perm_timeout_val
        except ValueError:
            raise SystemExit(
                f"Config error: KAMA_PERMISSION_TIMEOUT_S must be a number, got: {perm_timeout!r}"
            )

    compact_threshold = os.environ.get("KAMA_COMPACT_THRESHOLD")
    if compact_threshold is not None:
        try:
            compact_threshold_val = float(compact_threshold)
            if not (0.0 <= compact_threshold_val <= 1.0):
                raise SystemExit(
                    "Config error: KAMA_COMPACT_THRESHOLD must be between 0 and 1,"
                    f" got: {compact_threshold!r}"
                )
            config.compaction.auto_threshold = compact_threshold_val
            if compact_threshold_val == 0.0 and "KAMA_AUTO_COMPACT" not in os.environ:
                config.compaction.auto_compact = False
        except ValueError:
            raise SystemExit(
                f"Config error: KAMA_COMPACT_THRESHOLD must be a number, got: {compact_threshold!r}"
            )

    compact_auto = os.environ.get("KAMA_AUTO_COMPACT")
    if compact_auto is not None:
        config.compaction.auto_compact = compact_auto.lower() not in ("0", "false", "no")

    compact_soft = os.environ.get("KAMA_COMPACT_SOFT_TRIGGER_RATIO")
    compact_soft_value: float | None = None
    if compact_soft is not None:
        try:
            compact_soft_value = float(compact_soft)
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_SOFT_TRIGGER_RATIO must be a number,"
                f" got: {compact_soft!r}"
            )
        if not 0.0 < compact_soft_value <= 1.0:
            raise SystemExit(
                "Config error: KAMA_COMPACT_SOFT_TRIGGER_RATIO must be between 0 and 1"
            )
    effective_soft = (
        compact_soft_value
        if compact_soft_value is not None
        else config.compaction.effective_soft_trigger_ratio
    )

    compact_target = os.environ.get("KAMA_COMPACT_TARGET_RATIO")
    if compact_target is not None:
        try:
            target = float(compact_target)
        except ValueError:
            raise SystemExit(
                f"Config error: KAMA_COMPACT_TARGET_RATIO must be a number, got: {compact_target!r}"
            )
        if not 0.0 < target < effective_soft:
            raise SystemExit(
                "Config error: KAMA_COMPACT_TARGET_RATIO must be below soft trigger"
            )
        config.compaction.target_ratio = target

    if compact_soft_value is not None:
        if config.compaction.target_ratio >= compact_soft_value:
            raise SystemExit(
                "Config error: KAMA_COMPACT_SOFT_TRIGGER_RATIO must be above target"
            )
        config.compaction.soft_trigger_ratio = compact_soft_value

    compact_tail_ratio = os.environ.get("KAMA_COMPACT_RECENT_TAIL_RATIO")
    if compact_tail_ratio is not None:
        try:
            tail_ratio = float(compact_tail_ratio)
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_RECENT_TAIL_RATIO must be a number,"
                f" got: {compact_tail_ratio!r}"
            )
        if not 0.0 < tail_ratio < 1.0:
            raise SystemExit(
                "Config error: KAMA_COMPACT_RECENT_TAIL_RATIO must be between 0 and 1"
            )
        config.compaction.recent_tail_ratio = tail_ratio

    compact_tail_max = os.environ.get("KAMA_COMPACT_RECENT_TAIL_MAX_TOKENS")
    if compact_tail_max is not None:
        try:
            tail_max = int(compact_tail_max)
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_RECENT_TAIL_MAX_TOKENS must be an integer,"
                f" got: {compact_tail_max!r}"
            )
        if tail_max <= 0:
            raise SystemExit(
                "Config error: KAMA_COMPACT_RECENT_TAIL_MAX_TOKENS must be positive"
            )
        config.compaction.recent_tail_max_tokens = tail_max

    compact_summary_max = os.environ.get("KAMA_COMPACT_SUMMARY_MAX_TOKENS")
    if compact_summary_max is not None:
        try:
            summary_max = int(compact_summary_max)
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_SUMMARY_MAX_TOKENS must be an integer,"
                f" got: {compact_summary_max!r}"
            )
        if summary_max <= 0:
            raise SystemExit(
                "Config error: KAMA_COMPACT_SUMMARY_MAX_TOKENS must be positive"
            )
        config.compaction.summary_max_tokens = summary_max

    compact_tool_limit = os.environ.get("KAMA_COMPACT_TOOL_LIMIT")
    if compact_tool_limit is not None:
        try:
            compact_tool_limit_val = int(compact_tool_limit)
            if compact_tool_limit_val <= 0:
                raise SystemExit(
                    "Config error: KAMA_COMPACT_TOOL_LIMIT must be a positive integer,"
                    f" got: {compact_tool_limit!r}"
                )
            config.compaction.tool_result_limit = compact_tool_limit_val
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_TOOL_LIMIT must be an integer,"
                f" got: {compact_tool_limit!r}"
            )

    compact_tool_keep = os.environ.get("KAMA_COMPACT_TOOL_KEEP")
    if compact_tool_keep is not None:
        try:
            compact_tool_keep_val = int(compact_tool_keep)
            if compact_tool_keep_val <= 0:
                raise SystemExit(
                    "Config error: KAMA_COMPACT_TOOL_KEEP must be a positive integer,"
                    f" got: {compact_tool_keep!r}"
                )
            config.compaction.tool_result_keep = compact_tool_keep_val
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_COMPACT_TOOL_KEEP must be an integer,"
                f" got: {compact_tool_keep!r}"
            )

    sandbox_enabled = os.environ.get("KAMA_SANDBOX_ENABLED")
    if sandbox_enabled is not None:
        config.sandbox = dataclasses.replace(
            config.sandbox,
            enabled=sandbox_enabled.lower() not in ("0", "false", "no"),
        )

    sandbox_image = os.environ.get("KAMA_SANDBOX_IMAGE")
    if sandbox_image is not None:
        config.sandbox = dataclasses.replace(config.sandbox, image=sandbox_image)

    sandbox_network = os.environ.get("KAMA_SANDBOX_NETWORK")
    if sandbox_network is not None:
        config.sandbox = dataclasses.replace(
            config.sandbox,
            network=sandbox_network.lower() not in ("0", "false", "no"),
        )

    sandbox_timeout = os.environ.get("KAMA_SANDBOX_EXEC_TIMEOUT_S")
    if sandbox_timeout is not None:
        try:
            sandbox_timeout_val = int(sandbox_timeout)
            if sandbox_timeout_val <= 0:
                raise SystemExit(
                    "Config error: KAMA_SANDBOX_EXEC_TIMEOUT_S must be a positive integer,"
                    f" got: {sandbox_timeout!r}"
                )
            config.sandbox = dataclasses.replace(
                config.sandbox, exec_timeout_s=sandbox_timeout_val
            )
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_SANDBOX_EXEC_TIMEOUT_S must be an integer,"
                f" got: {sandbox_timeout!r}"
            )

    git_enabled = os.environ.get("KAMA_GIT_ENABLED")
    if git_enabled is not None:
        config.git = dataclasses.replace(
            config.git,
            enabled=git_enabled.lower() not in ("0", "false", "no"),
        )

    git_checkpoint_mode = os.environ.get("KAMA_GIT_CHECKPOINT_MODE")
    if git_checkpoint_mode is not None:
        try:
            config.git = dataclasses.replace(
                config.git, checkpoint_mode=git_checkpoint_mode
            )
        except ValueError as exc:
            raise SystemExit(f"Config error: KAMA_GIT_CHECKPOINT_MODE {exc}")

    git_branch_prefix = os.environ.get("KAMA_GIT_BRANCH_PREFIX")
    if git_branch_prefix is not None:
        config.git = dataclasses.replace(config.git, branch_prefix=git_branch_prefix)

    git_mode = os.environ.get("KAMA_GIT_MODE")
    if git_mode is not None:
        try:
            config.git = dataclasses.replace(config.git, mode=git_mode)
        except ValueError as exc:
            raise SystemExit(f"Config error: KAMA_GIT_MODE {exc}")

    git_rollback = os.environ.get("KAMA_GIT_AUTO_ROLLBACK_ON_FAIL")
    if git_rollback is not None:
        config.git = dataclasses.replace(
            config.git,
            auto_rollback_on_fail=git_rollback.lower() not in ("0", "false", "no"),
        )

    semantic_enabled = os.environ.get("KAMA_SEMANTIC_ENABLED")
    if semantic_enabled is not None:
        config.semantic = dataclasses.replace(
            config.semantic,
            enabled=semantic_enabled.lower() not in ("0", "false", "no"),
        )

    semantic_strategy = os.environ.get("KAMA_SEMANTIC_STRATEGY")
    if semantic_strategy is not None:
        try:
            config.semantic = dataclasses.replace(
                config.semantic, strategy=semantic_strategy
            )
        except ValueError as exc:
            raise SystemExit(f"Config error: KAMA_SEMANTIC_STRATEGY {exc}")

    semantic_degradation = os.environ.get("KAMA_SEMANTIC_DEGRADATION")
    if semantic_degradation is not None:
        try:
            config.semantic = dataclasses.replace(
                config.semantic, degradation=semantic_degradation
            )
        except ValueError as exc:
            raise SystemExit(f"Config error: KAMA_SEMANTIC_DEGRADATION {exc}")

    semantic_index_dir = os.environ.get("KAMA_SEMANTIC_INDEX_DIR")
    if semantic_index_dir is not None:
        config.semantic = dataclasses.replace(config.semantic, index_dir=semantic_index_dir)

    semantic_top_k = os.environ.get("KAMA_SEMANTIC_DEFAULT_TOP_K")
    if semantic_top_k is not None:
        try:
            semantic_top_k_val = int(semantic_top_k)
            if semantic_top_k_val <= 0:
                raise SystemExit(
                    "Config error: KAMA_SEMANTIC_DEFAULT_TOP_K must be a positive integer,"
                    f" got: {semantic_top_k!r}"
                )
            config.semantic = dataclasses.replace(
                config.semantic, default_top_k=semantic_top_k_val
            )
        except ValueError:
            raise SystemExit(
                f"Config error: KAMA_SEMANTIC_DEFAULT_TOP_K must be an integer,"
                f" got: {semantic_top_k!r}"
            )

    semantic_threshold = os.environ.get("KAMA_SEMANTIC_SIMILARITY_THRESHOLD")
    if semantic_threshold is not None:
        try:
            semantic_threshold_val = float(semantic_threshold)
        except ValueError:
            raise SystemExit(
                "Config error: KAMA_SEMANTIC_SIMILARITY_THRESHOLD must be a number,"
                f" got: {semantic_threshold!r}"
            )
        try:
            config.semantic = dataclasses.replace(
                config.semantic, similarity_threshold=semantic_threshold_val
            )
        except ValueError as exc:
            raise SystemExit(f"Config error: KAMA_SEMANTIC_SIMILARITY_THRESHOLD {exc}")
