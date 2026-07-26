from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from kama_claude.core.config import get_config


# 加载待实现的 experiment identity 模块，并把缺失模块转换为明确的 RED 失败
def _experiment_module() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.experiment")
    except ModuleNotFoundError:
        pytest.fail("benchmark experiment identity module is missing")


# 创建只含冻结 baseline 行为与公开 identity hash 的合法 experiment profile
def _profile_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "kama-coding-mvp-v1-deepseek-v4-pro",
        "suite": {
            "manifest": "../suites/kama-coding-mvp-v1.json",
            "freeze_manifest": "../suites/kama-coding-mvp-v1.freeze.json",
            "tasks_root": "../tasks",
            "expected_suite_hash": "a" * 64,
        },
        "provider": {
            "service_provider": "deepseek",
            "wire_protocol": "anthropic_messages",
            "endpoint_id": "deepseek-anthropic-compatible",
            "endpoint": "https://api.deepseek.com/anthropic",
            "model_id": "deepseek-v4-pro",
            "sdk_distribution": "anthropic",
            "sdk_version": "0.111.0",
            "credential_env": "ANTHROPIC_API_KEY",
        },
        "runtime": {
            "max_steps": 20,
            "router": "static",
            "compaction_threshold": 0.0,
            "tool_result_limit": 8000,
            "tool_result_keep": 4000,
            "mcp_enabled": False,
            "trace_enabled": True,
            "include_llm_payload": True,
        },
        "schedule": {
            "repeats": 3,
            "execution_order": "suite_task_then_repeat_ascending",
        },
        "artifacts": {
            "output_root_must_be_new": True,
            "output_root_must_be_outside_repository": True,
            "retain_all_attempts": True,
            "raw_trace_visibility": "private",
        },
        "expected_identity": {
            "prompt_hash": "b" * 64,
            "tool_schema_hash": "c" * 64,
        },
    }


# 用独立标准库实现计算测试 fixture 的 path-content hash
def _fixture_path_content_hash(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# 将 profile 与其相对引用写入临时 benchmark tree
def _write_profile_tree(tmp_path: Path) -> Path:
    benchmark_root = tmp_path / "benchmarks"
    experiments = benchmark_root / "experiments"
    suites = benchmark_root / "suites"
    tasks = benchmark_root / "tasks"
    experiments.mkdir(parents=True)
    suites.mkdir()
    tasks.mkdir()
    (suites / "kama-coding-mvp-v1.json").write_text("{}\n", encoding="utf-8")
    (suites / "kama-coding-mvp-v1.freeze.json").write_text("{}\n", encoding="utf-8")
    profile_path = experiments / "profile.json"
    profile_path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
    return profile_path


# 创建含 suite/freeze/task/dependency 的最小 declaration 输入树
def _write_declaration_tree(tmp_path: Path) -> Path:
    profile_path = _write_profile_tree(tmp_path)
    benchmark_root = tmp_path / "benchmarks"
    task_dir = benchmark_root / "tasks" / "task-a"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Fix the task.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {"id": "tests-pass", "kind": "file_exists", "path": "module.py"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task-a",
                "task_version": 1,
                "category": "bug_fixing",
                "criterion_groups": {
                    "target_behavior": ["tests-pass"],
                    "regression": ["regression-pass"],
                },
            }
        ),
        encoding="utf-8",
    )
    grader = json.loads((private / "grader.json").read_text(encoding="utf-8"))
    grader["criteria"].append(
        {"id": "regression-pass", "kind": "file_exists", "path": "module.py"}
    )
    (private / "grader.json").write_text(json.dumps(grader), encoding="utf-8")
    suite_payload = {
        "schema_version": 1,
        "suite_id": "kama-test-suite",
        "suite_version": 1,
        "task_ids": ["task-a"],
    }
    suite_bytes = json.dumps(
        suite_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    (benchmark_root / "suites" / "kama-coding-mvp-v1.json").write_text(
        json.dumps(suite_payload),
        encoding="utf-8",
    )
    freeze_payload = {
        "schema_version": 1,
        "suite_id": "kama-test-suite",
        "suite_version": 1,
        "suite_hash": suite_hash,
        "tasks": [
            {
                "task_id": "task-a",
                "task_hash": _fixture_path_content_hash(
                    task_dir,
                    [path for path in task_dir.rglob("*") if path.is_file()],
                ),
                "grader_hash": _fixture_path_content_hash(
                    task_dir,
                    [private / "grader.json"],
                ),
                "reference_hash": "f" * 64,
            }
        ],
    }
    (benchmark_root / "suites" / "kama-coding-mvp-v1.freeze.json").write_text(
        json.dumps(freeze_payload),
        encoding="utf-8",
    )
    profile_payload = _profile_payload()
    suite_profile = dict(profile_payload["suite"])  # type: ignore[arg-type]
    suite_profile["expected_suite_hash"] = suite_hash
    profile_payload["suite"] = suite_profile
    profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return profile_path


# 功能：验证 experiment profile 严格冻结 provider、runtime、repeat、trace 与 artifact policy
# 设计：加载真实 JSON 和相对路径，断言行为字段来自 profile 而非调用方或 ambient 配置
def test_load_experiment_profile_freezes_baseline_behavior(tmp_path: Path) -> None:
    experiment = _experiment_module()
    loaded = experiment.load_experiment_profile(_write_profile_tree(tmp_path))

    assert loaded.profile.provider.service_provider == "deepseek"
    assert loaded.profile.provider.wire_protocol == "anthropic_messages"
    assert loaded.profile.provider.endpoint_id == "deepseek-anthropic-compatible"
    assert loaded.profile.provider.model_id == "deepseek-v4-pro"
    assert loaded.profile.provider.sdk_version == "0.111.0"
    assert loaded.profile.runtime.max_steps == 20
    assert loaded.profile.runtime.mcp_enabled is False
    assert loaded.profile.runtime.include_llm_payload is True
    assert loaded.profile.schedule.repeats == 3
    assert loaded.profile.schedule.execution_order == "suite_task_then_repeat_ascending"
    assert loaded.profile.artifacts.raw_trace_visibility == "private"
    assert loaded.tasks_root == tmp_path / "benchmarks" / "tasks"


# 功能：验证 profile 拒绝 secret、未知行为字段和不安全 endpoint
# 设计：逐个破坏合法 payload，防止 schema 扩张成第二套 runtime config 或序列化 credential
@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("provider", "api_key", "secret-value"),
        ("runtime", "permission_mode", "allow_all"),
        ("provider", "endpoint", "http://api.deepseek.com/anthropic"),
        ("provider", "endpoint", "https://user:pass@api.deepseek.com/anthropic"),
        ("provider", "endpoint", "https://api.deepseek.com/anthropic?secret=value"),
        ("provider", "endpoint", "https://other.example/anthropic"),
    ],
)
def test_profile_rejects_secret_unknown_and_unsafe_fields(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    experiment = _experiment_module()
    profile_path = _write_profile_tree(tmp_path)
    payload = _profile_payload()
    section_payload = dict(payload[section])  # type: ignore[arg-type]
    section_payload[field] = value
    payload[section] = section_payload
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid experiment profile"):
        experiment.load_experiment_profile(profile_path)


# 功能：验证 canonical identity hash 对 object key 顺序稳定但保留 list 顺序语义
# 设计：使用手写等价和非等价 payload，避免用被测函数生成期望值造成镜像断言
def test_canonical_identity_hash_is_reproducible_and_order_sensitive() -> None:
    experiment = _experiment_module()

    first = {"provider": {"model": "deepseek", "sdk": "anthropic"}, "tools": ["a", "b"]}
    same = {"tools": ["a", "b"], "provider": {"sdk": "anthropic", "model": "deepseek"}}
    reordered = {
        "provider": {"model": "deepseek", "sdk": "anthropic"},
        "tools": ["b", "a"],
    }

    assert experiment.canonical_hash(first) == experiment.canonical_hash(same)
    assert experiment.canonical_hash(first) != experiment.canonical_hash(reordered)


# 功能：验证 declared identity 在运行前绑定 Git、suite/task/grader、provider、prompt/tools 与依赖
# 设计：用最小真实 suite/freeze 和手写 repository identity 构建声明，并检查公开 JSON 无 secret/绝对路径
def test_capture_and_write_declared_identity_is_complete_and_redacted(
    tmp_path: Path,
) -> None:
    experiment = _experiment_module()
    loaded = experiment.load_experiment_profile(_write_declaration_tree(tmp_path))
    identity = experiment.capture_declared_identity(
        loaded,
        repository_root=tmp_path,
        repository=experiment.RepositoryIdentity(
            commit="1" * 40,
            dirty=False,
        ),
        installed_sdk_version="0.111.0",
    )
    output = tmp_path / "artifacts"

    experiment.write_declared_experiment(output, identity)

    text = (output / "declared-experiment.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload == identity.model_dump(mode="json")
    assert payload["git"]["commit"] == "1" * 40
    assert payload["suite"]["suite_id"] == "kama-test-suite"
    assert set(payload["suite"]["task_hashes"]) == {"task-a"}
    assert len(payload["suite"]["task_hashes"]["task-a"]) == 64
    assert set(payload["suite"]["grader_hashes"]) == {"task-a"}
    assert len(payload["suite"]["grader_hashes"]["task-a"]) == 64
    assert payload["provider"]["service_provider"] == "deepseek"
    assert payload["provider"]["model_id"] == "deepseek-v4-pro"
    assert payload["prompt_hash"] == "b" * 64
    assert payload["tool_schema_hash"] == "c" * 64
    assert payload["runtime"]["max_steps"] == 20
    assert len(payload["runtime_config_hash"]) == 64
    assert len(payload["dependency"]["dependency_hash"]) == 64
    assert payload["host"]["python_version"]
    assert payload["host"]["os"]
    assert "secret" not in text.lower()
    assert str(tmp_path) not in text


# 功能：验证 declaration 对 dirty Git、suite hash 与 SDK drift 全部 fail closed
# 设计：分别注入三类静态 mismatch，证明任何模型调用前就拒绝不可比较实验
def test_declared_identity_rejects_static_mismatch(tmp_path: Path) -> None:
    experiment = _experiment_module()
    profile_path = _write_declaration_tree(tmp_path)
    loaded = experiment.load_experiment_profile(profile_path)

    with pytest.raises(ValueError, match="clean Git"):
        experiment.capture_declared_identity(
            loaded,
            repository_root=tmp_path,
            repository=experiment.RepositoryIdentity(commit="2" * 40, dirty=True),
            installed_sdk_version="0.111.0",
        )
    with pytest.raises(ValueError, match="SDK"):
        experiment.capture_declared_identity(
            loaded,
            repository_root=tmp_path,
            repository=experiment.RepositoryIdentity(commit="2" * 40, dirty=False),
            installed_sdk_version="0.112.0",
        )

    payload = _profile_payload()
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    mismatched = experiment.load_experiment_profile(profile_path)
    with pytest.raises(ValueError, match="suite hash"):
        experiment.capture_declared_identity(
            mismatched,
            repository_root=tmp_path,
            repository=experiment.RepositoryIdentity(commit="2" * 40, dirty=False),
            installed_sdk_version="0.111.0",
        )


# 功能：验证 ambient 环境与 workspace .env 都不能覆盖 experiment behavior profile
# 设计：同时放入冲突的 process env 和 .env，再调用真实 get_config 证明 scoped 映射拥有最高优先级
def test_scoped_experiment_environment_prevents_dotenv_behavior_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment_module()
    loaded = experiment.load_experiment_profile(_write_profile_tree(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "\n".join(
            [
                "KAMA_MAX_STEPS=77",
                "KAMA_LLM_DEFAULT_MODEL=wrong-dotenv-model",
                "ANTHROPIC_BASE_URL=https://wrong-dotenv.example",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("KAMA_MAX_STEPS", "99")
    monkeypatch.setenv("KAMA_LLM_DEFAULT_MODEL", "wrong-ambient-model")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong-ambient.example")
    monkeypatch.setenv("KAMA_CONFIG", "/private/ambient-config.toml")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "credential-only-secret")

    with experiment.scoped_experiment_environment(loaded.profile):
        config = get_config()
        assert config.agent.max_steps == 20
        assert config.llm.default_model == "deepseek-v4-pro"
        assert config.trace.enabled is True
        assert config.trace.include_llm_payload is True
        assert config.compaction.auto_threshold == 0.0
        assert config.compaction.tool_result_limit == 8000
        assert config.compaction.tool_result_keep == 4000
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
        assert "KAMA_CONFIG" not in os.environ
        assert os.environ["ANTHROPIC_API_KEY"] == "credential-only-secret"

    assert os.environ["KAMA_MAX_STEPS"] == "99"
    assert os.environ["KAMA_CONFIG"] == "/private/ambient-config.toml"


# 构造用于 artifact verifier 的完整 declared identity
def _declared_observation_identity(
    experiment: ModuleType,
    *,
    prompt: str,
    tools: list[dict[str, object]],
) -> object:
    runtime = {
        "max_steps": 20,
        "router": "static",
        "compaction_threshold": 0.0,
        "tool_result_limit": 8000,
        "tool_result_keep": 4000,
        "mcp_enabled": False,
        "trace_enabled": True,
        "include_llm_payload": True,
    }
    return experiment.DeclaredExperimentIdentity(
        profile_id="profile-a",
        profile_hash="1" * 64,
        git={"commit": "2" * 40, "dirty": False},
        suite={
            "suite_id": "suite-a",
            "suite_version": 1,
            "suite_hash": "3" * 64,
            "task_hashes": {"task-a": "4" * 64},
            "grader_hashes": {"task-a": "5" * 64},
        },
        provider={
            "service_provider": "deepseek",
            "wire_protocol": "anthropic_messages",
            "endpoint_id": "deepseek-anthropic-compatible",
            "endpoint": "https://api.deepseek.com/anthropic",
            "model_id": "deepseek-v4-pro",
            "sdk_distribution": "anthropic",
            "sdk_version": "0.111.0",
        },
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        tool_schema_hash=experiment.canonical_hash(tools),
        runtime=runtime,
        runtime_config_hash=experiment.canonical_hash(runtime),
        dependency={
            "pyproject_hash": "6" * 64,
            "uv_lock_hash": "7" * 64,
            "dependency_hash": "8" * 64,
        },
        host={
            "python_version": "3.12.13",
            "os": "Darwin",
            "os_release": "test",
            "architecture": "arm64",
        },
        schedule={
            "repeats": 3,
            "execution_order": "suite_task_then_repeat_ascending",
        },
        artifacts={
            "output_root_must_be_new": True,
            "output_root_must_be_outside_repository": True,
            "retain_all_attempts": True,
            "raw_trace_visibility": "private",
        },
    )


# 写入一份最小但真实形状的 runtime trace 与 journal identity evidence
def _write_observed_identity_artifacts(
    attempt_root: Path,
    *,
    prompt: str,
    tools: list[dict[str, object]],
    model_id: str = "deepseek-v4-pro",
    max_steps: int = 20,
    timeout_partial: bool = False,
) -> None:
    runtime_dir = attempt_root / "runtime"
    runtime_dir.mkdir(parents=True)
    provider = {
        "service_provider": "deepseek",
        "wire_protocol": "anthropic_messages",
        "endpoint_id": "deepseek-anthropic-compatible",
        "endpoint": "https://api.deepseek.com/anthropic",
        "model_id": model_id,
        "sdk_distribution": "anthropic",
        "sdk_version": "0.111.0",
    }
    runtime = {
        "max_steps": max_steps,
        "router": "static",
        "compaction_threshold": 0.0,
        "tool_result_limit": 8000,
        "tool_result_keep": 4000,
        "mcp_enabled": False,
        "trace_enabled": True,
        "include_llm_payload": True,
    }
    trace_records = [
        {
            "ts": "2026-07-26T00:00:00+00:00",
            "direction": "CORE",
            "layer": "event",
            "kind": "runtime_identity",
            "run_id": "run-a",
            "data": {"provider": provider, "runtime": runtime},
        },
        {
            "ts": "2026-07-26T00:00:01+00:00",
            "direction": "CORE→LLM",
            "layer": "llm",
            "kind": "api_call",
            "run_id": "run-a",
            "step": 1,
            "data": {
                "messages": [{"role": "user", "content": "private task input"}],
                "tool_schemas": tools,
                "system": prompt,
            },
        },
    ]
    (runtime_dir / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records),
        encoding="utf-8",
    )
    events: list[dict[str, object]] = [
        {
            "type": "run.started",
            "run_id": "run-a",
            "goal": "Observed identity contract.",
            "ts": "2026-07-26T00:00:00+00:00",
        },
        {
            "type": "step.started",
            "run_id": "run-a",
            "step": 1,
            "ts": "2026-07-26T00:00:00.500000+00:00",
        },
        {
            "type": "llm.model_selected",
            "run_id": "run-a",
            "model": model_id,
            "strategy": "static",
            "ts": "2026-07-26T00:00:01+00:00",
        },
    ]
    if not timeout_partial:
        events.extend(
            [
                {
                    "type": "step.finished",
                    "run_id": "run-a",
                    "step": 1,
                    "ts": "2026-07-26T00:00:02+00:00",
                },
                {
                    "type": "run.finished",
                    "run_id": "run-a",
                    "status": "success",
                    "reason": None,
                    "steps": 1,
                    "ts": "2026-07-26T00:00:03+00:00",
                },
            ]
        )
    rows = [
        {
            "schema_version": 2,
            "event_id": f"event-{index}",
            "stream_id": "run:run-a",
            "seq": index,
            "event": event,
        }
        for index, event in enumerate(events, 1)
    ]
    (runtime_dir / "events.v2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


# 功能：验证 observed identity 仅从 trace/journal 提取 hash，并与 declaration 完整匹配
# 设计：写入含敏感 prompt/messages 的 artifact，断言输出只保留 identity/hash 且 verification valid
def test_collect_observed_identity_matches_declaration_without_payload_leak(
    tmp_path: Path,
) -> None:
    experiment = _experiment_module()
    prompt = "private effective system prompt"
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    declared = _declared_observation_identity(experiment, prompt=prompt, tools=tools)
    attempt_root = tmp_path / "attempt-a"
    _write_observed_identity_artifacts(
        attempt_root,
        prompt=prompt,
        tools=tools,
    )

    observed = experiment.collect_observed_identity(attempt_root)
    verification = experiment.verify_declared_observed(declared, observed)

    assert verification.valid is True
    assert verification.mismatches == []
    assert observed.model_event_ids == ["deepseek-v4-pro"]
    assert observed.prompt_hash == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert observed.tool_schema_hash == experiment.canonical_hash(tools)
    serialized = observed.model_dump_json()
    assert prompt not in serialized
    assert "private task input" not in serialized
    experiment.require_identity_match(declared, observed)


# 功能：验证 timeout_partial 模式可从严格 prefix、runtime、API 与 model 证据收集 identity
# 设计：journal 以 run.started 开头但没有 terminal，模拟被 parent timeout 截断的真实前缀
def test_collect_timeout_partial_identity_from_complete_evidence(
    tmp_path: Path,
) -> None:
    experiment = _experiment_module()
    prompt = "timeout effective prompt"
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    attempt_root = tmp_path / "timeout-attempt"
    _write_observed_identity_artifacts(
        attempt_root,
        prompt=prompt,
        tools=tools,
        timeout_partial=True,
    )

    assert hasattr(experiment, "EvidenceMode")
    observed = experiment.collect_observed_identity(
        attempt_root,
        evidence_mode=experiment.EvidenceMode.TIMEOUT_PARTIAL,
    )

    assert observed.run_id == "run-a"
    assert observed.api_call_count == 1
    assert observed.model_event_ids == ["deepseek-v4-pro"]


# 功能：验证 timeout_partial 缺任一 identity 证据或 journal prefix 非法时全部 fail closed
# 设计：每次只破坏 schema/sequence/lifecycle/runtime/API/model 中一个条件，避免混淆失败归属
@pytest.mark.parametrize(
    "case",
    (
        "wrong_version",
        "sequence_gap",
        "missing_start",
        "terminal_event",
        "missing_runtime",
        "missing_api",
        "missing_model",
    ),
)
def test_collect_timeout_partial_identity_rejects_incomplete_or_invalid_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    experiment = _experiment_module()
    attempt_root = tmp_path / case
    _write_observed_identity_artifacts(
        attempt_root,
        prompt="timeout prompt",
        tools=[{"name": "read_file", "input_schema": {"type": "object"}}],
        timeout_partial=True,
    )
    runtime_dir = attempt_root / "runtime"
    trace_path = runtime_dir / "trace.jsonl"
    journal_path = runtime_dir / "events.v2.jsonl"
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    if case == "wrong_version":
        rows[0]["schema_version"] = 3
    elif case == "sequence_gap":
        rows[2]["seq"] = 4
    elif case == "missing_start":
        rows = rows[1:]
        for index, row in enumerate(rows, 1):
            row["seq"] = index
    elif case == "terminal_event":
        rows.append(
            {
                "schema_version": 2,
                "event_id": "event-terminal",
                "stream_id": "run:run-a",
                "seq": 4,
                "event": {
                    "type": "run.finished",
                    "run_id": "run-a",
                    "status": "failed",
                    "reason": "cancelled",
                    "steps": 1,
                    "ts": "2026-07-26T00:00:02+00:00",
                },
            }
        )
    elif case == "missing_runtime":
        trace = [record for record in trace if record["kind"] != "runtime_identity"]
    elif case == "missing_api":
        trace = [record for record in trace if record["kind"] != "api_call"]
    else:
        rows = rows[:2]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace),
        encoding="utf-8",
    )
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity evidence"):
        experiment.collect_observed_identity(
            attempt_root,
            evidence_mode=experiment.EvidenceMode.TIMEOUT_PARTIAL,
        )


# 功能：验证 model、prompt 或 runtime 任一 behavior mismatch 都使 experiment identity invalid
# 设计：参数化三种独立 artifact drift，并断言稳定 mismatch 字段和 fail-closed exception
@pytest.mark.parametrize(
    ("observed_prompt", "observed_model", "observed_steps", "field"),
    [
        ("different prompt", "deepseek-v4-pro", 20, "prompt_hash"),
        ("expected prompt", "deepseek-v4-flash", 20, "provider"),
        ("expected prompt", "deepseek-v4-pro", 99, "runtime"),
    ],
)
def test_observed_behavior_mismatch_invalidates_experiment(
    tmp_path: Path,
    observed_prompt: str,
    observed_model: str,
    observed_steps: int,
    field: str,
) -> None:
    experiment = _experiment_module()
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    declared = _declared_observation_identity(
        experiment,
        prompt="expected prompt",
        tools=tools,
    )
    attempt_root = tmp_path / field
    _write_observed_identity_artifacts(
        attempt_root,
        prompt=observed_prompt,
        tools=tools,
        model_id=observed_model,
        max_steps=observed_steps,
    )
    observed = experiment.collect_observed_identity(attempt_root)

    verification = experiment.verify_declared_observed(declared, observed)

    assert verification.valid is False
    assert field in verification.mismatches
    with pytest.raises(experiment.ExperimentIdentityMismatch):
        experiment.require_identity_match(declared, observed)


# 功能：验证 canonical experiment identity 只接受全部匹配的 observations 并汇总尝试数量
# 设计：重复同一合法 observation，断言 report-facing summary 不携带 run-local payload 或路径
def test_build_verified_experiment_identity_summarizes_matching_attempts(
    tmp_path: Path,
) -> None:
    experiment = _experiment_module()
    prompt = "expected prompt"
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    declared = _declared_observation_identity(experiment, prompt=prompt, tools=tools)
    attempt_root = tmp_path / "attempt"
    _write_observed_identity_artifacts(
        attempt_root,
        prompt=prompt,
        tools=tools,
    )
    observed = experiment.collect_observed_identity(attempt_root)

    identity = experiment.build_verified_experiment_identity(
        declared,
        [observed, observed],
    )

    assert identity.status == "valid"
    assert identity.verification.verified_attempts == 2
    assert identity.verification.mismatches == []
    assert identity.observed.provider == declared.provider
    assert identity.observed.prompt_hash == declared.prompt_hash
    assert identity.observed.tool_schema_hash == declared.tool_schema_hash
    assert identity.observed.runtime_config_hash == declared.runtime_config_hash


# 功能：验证 credential 可来自 repository .env，但同文件 behavior 键不会进入 process environment
# 设计：清除 ambient credential 后读取真实 dotenv 文件，断言只返回 profile 指定 secret value
def test_resolve_experiment_credential_reads_only_named_dotenv_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _experiment_module()
    loaded = experiment.load_experiment_profile(_write_profile_tree(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAMA_MAX_STEPS", raising=False)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=credential-from-dotenv\nKAMA_MAX_STEPS=999\n",
        encoding="utf-8",
    )

    credential = experiment.resolve_experiment_credential(loaded.profile, tmp_path)

    assert credential == "credential-from-dotenv"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "KAMA_MAX_STEPS" not in os.environ


# 功能：验证 tracked DeepSeek profile 与 frozen 九任务 suite、grader 及 lockfile 身份一致
# 设计：对仓库真实 profile 执行无 API declaration preflight，并注入 clean Git snapshot
def test_tracked_first_baseline_profile_matches_frozen_repository() -> None:
    experiment = _experiment_module()
    repository_root = Path(__file__).resolve().parents[2]
    profile_path = (
        repository_root
        / "benchmarks"
        / "experiments"
        / "kama-coding-mvp-v1-deepseek-v4-pro.json"
    )
    loaded = experiment.load_experiment_profile(profile_path)

    identity = experiment.capture_declared_identity(
        loaded,
        repository_root=repository_root,
        repository=experiment.RepositoryIdentity(commit="a" * 40, dirty=False),
        installed_sdk_version="0.111.0",
    )

    assert identity.suite.suite_id == "kama-coding-mvp"
    assert identity.suite.suite_version == 1
    assert len(identity.suite.task_hashes) == 9
    assert len(identity.suite.grader_hashes) == 9
    assert identity.provider.model_id == "deepseek-v4-pro"
    assert identity.runtime.max_steps == 20
    assert identity.schedule.repeats == 3


# 功能：验证 artifact policy 拒绝 repository 内路径与已存在输出，只接受全新的外部目录
# 设计：对三个 concrete path 调用同一 preflight，防止 baseline artifacts 污染 Git identity
def test_validate_experiment_output_enforces_frozen_artifact_policy(
    tmp_path: Path,
) -> None:
    experiment = _experiment_module()
    loaded = experiment.load_experiment_profile(_write_profile_tree(tmp_path))
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    existing = tmp_path / "existing-output"
    existing.mkdir()

    with pytest.raises(ValueError, match="outside repository"):
        experiment.validate_experiment_output(
            repository_root / "artifacts",
            repository_root,
            loaded.profile.artifacts,
        )
    with pytest.raises(ValueError, match="must be new"):
        experiment.validate_experiment_output(
            existing,
            repository_root,
            loaded.profile.artifacts,
        )

    accepted = experiment.validate_experiment_output(
        tmp_path / "new-output",
        repository_root,
        loaded.profile.artifacts,
    )
    assert accepted == tmp_path / "new-output"
