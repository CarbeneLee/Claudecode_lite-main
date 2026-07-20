from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "compose.yaml"
SMOKE = ROOT / "scripts" / "docker_smoke.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "docker.yml"


# 读取仓库根目录下的 UTF-8 文本合同文件
def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


# 返回 .dockerignore 中忽略注释和空行后的有序规则
def _dockerignore_rules() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


# 提取 Dockerfile 中从指定 stage 开始到下一 stage 之前的文本
def _docker_stage(dockerfile: str, stage: str) -> str:
    match = re.search(
        rf"^FROM\s+\S+\s+AS\s+{re.escape(stage)}\s*$",
        dockerfile,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    assert match is not None, f"missing Docker stage: {stage}"
    remainder = dockerfile[match.start() :]
    next_stage = re.search(r"^FROM\s+", remainder[match.end() - match.start() :], re.MULTILINE)
    if next_stage is None:
        return remainder
    return remainder[: match.end() - match.start() + next_stage.start()]


# 提取 Compose 中一个顶层 service 的缩进文本块
def _compose_service(compose: str, service: str) -> str:
    match = re.search(rf"^  {re.escape(service)}:\s*$", compose, flags=re.MULTILINE)
    assert match is not None, f"missing Compose service: {service}"
    remainder = compose[match.start() :]
    next_service = re.search(r"^  [a-zA-Z0-9_-]+:\s*$", remainder[match.end() - match.start() :], re.MULTILINE)
    if next_service is None:
        return remainder
    return remainder[: match.end() - match.start() + next_service.start()]


# 功能：验证 Dockerfile 使用 digest 固定的 Python/uv 输入并声明 builder、test、runtime 三个阶段
# 设计：只审计 Dockerfile 可直接证明的声明，不用字符串测试替代真实 image inspect
def test_dockerfile_declares_pinned_multistage_inputs() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert re.search(
        r"^ARG PYTHON_IMAGE=python:3\.12[^\s@]*@sha256:[0-9a-f]{64}$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:[^\s@]+@sha256:[0-9a-f]{64}$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert "FROM ${UV_IMAGE} AS uv" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS builder" in dockerfile
    assert "FROM builder AS test" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS runtime" in dockerfile


# 功能：验证 production builder 先安装锁定依赖再复制源码并安装项目
# 设计：比较文本位置以锁定两阶段 uv sync 的缓存边界，同时把真实安装行为留给 Docker build
def test_builder_uses_two_phase_frozen_uv_sync() -> None:
    builder = _docker_stage(_read("Dockerfile"), "builder")

    metadata_copy = builder.index("COPY pyproject.toml uv.lock ./")
    dependency_sync = builder.index("uv sync --frozen --no-dev --no-install-project")
    source_copy = builder.index("COPY src ./src")
    project_sync = builder.index("uv sync --frozen --no-dev --no-editable")

    assert metadata_copy < dependency_sync < source_copy < project_sync
    assert builder.count("uv sync --frozen") == 2
    assert "COPY . " not in builder


# 功能：验证 test stage 独立安装开发依赖并执行完整仓库门禁
# 设计：静态确认测试工具只在 test stage 出现，runtime 是否真的无 pytest 由镜像检查补充
def test_test_stage_is_independent_from_runtime() -> None:
    dockerfile = _read("Dockerfile")
    test_stage = _docker_stage(dockerfile, "test")
    runtime_stage = _docker_stage(dockerfile, "runtime")

    assert "UV_PROJECT_ENVIRONMENT=/opt/test-venv" in test_stage
    assert "uv sync --frozen --all-groups --no-editable" in test_stage
    assert "pytest tests/unit" in test_stage
    assert "pytest tests/integration" in test_stage
    assert "ruff check src tests scripts" in test_stage
    assert "mypy src" in test_stage
    assert "gen_protocol_doc.py --check" in test_stage
    assert "pytest" not in runtime_stage
    assert "/opt/test-venv" not in runtime_stage


# 功能：验证 runtime 只复制生产虚拟环境并声明非 root、工作目录和直接 PID 1 命令
# 设计：文本测试限定声明边界，最终用户、命令和可执行文件仍由 image inspect/runtime smoke 验证
def test_runtime_stage_declares_minimal_non_root_contract() -> None:
    runtime = _docker_stage(_read("Dockerfile"), "runtime")

    assert "COPY --from=builder --chown=kama:kama /opt/venv /opt/venv" in runtime
    assert "ENV PATH=\"/opt/venv/bin:$PATH\"" in runtime
    assert "WORKDIR /workspace" in runtime
    assert re.search(r"^USER kama$", runtime, flags=re.MULTILINE)
    assert re.search(r'^CMD \["kama-core"\]$', runtime, flags=re.MULTILINE)
    assert re.search(r"^STOPSIGNAL SIGTERM$", runtime, flags=re.MULTILINE)
    assert "COPY src" not in runtime
    assert "COPY tests" not in runtime
    assert "COPY --from=uv" not in runtime


# 功能：验证 Dockerfile 不声明 secret build input、全仓复制、远程 ADD 或 shell-form daemon 命令
# 设计：这些是可由文本完整判定的负向规则，镜像 history 扫描另行覆盖构建后的泄漏
def test_dockerfile_avoids_text_auditable_secret_and_context_hazards() -> None:
    dockerfile = _read("Dockerfile")

    assert not re.search(r"^(ARG|ENV)\s+ANTHROPIC_API_KEY(?:=|\s|$)", dockerfile, re.MULTILINE)
    assert not re.search(r"^COPY\s+(?:--\S+\s+)*\.\s", dockerfile, re.MULTILINE)
    assert not re.search(r"^ADD\s+", dockerfile, re.MULTILINE)
    assert "CMD kama-core" not in dockerfile


# 功能：验证 .dockerignore 默认拒绝未知文件且只放行 Dockerfile 实际构建输入
# 设计：从 COPY 指令提取 build-context source 并与冻结例外集合比对，同时用未知根 canary 杀死 broad allowlist
def test_dockerignore_is_default_deny_and_allows_only_build_inputs() -> None:
    rules = _dockerignore_rules()
    allowed_rules = {
        "!Dockerfile",
        "!.dockerignore",
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!WIRE_PROTOCOL.md",
        "!src/",
        "!src/**",
        "!tests/",
        "!tests/**",
        "!scripts/",
        "!scripts/**",
    }
    context_sources: set[str] = set()

    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        tokens = shlex.split(line)
        context_sources.update(tokens[1:-1])

    assert rules[0] == "**"
    assert set(rules[1:]) == allowed_rules
    assert "!**" not in rules
    assert context_sources == {"pyproject.toml", "uv.lock", "src", "tests", "scripts", "WIRE_PROTOCOL.md", "README.md"}
    for source in context_sources:
        if source in {"src", "tests", "scripts"}:
            assert f"!{source}/" in allowed_rules
            assert f"!{source}/**" in allowed_rules
        else:
            assert f"!{source}" in allowed_rules
    assert "!docs/**" not in rules
    assert "!.env*" not in rules
    assert "!.kama/**" not in rules
    assert "!phase6-unlisted-private-canary.txt" not in rules


# 功能：验证 Compose 让 daemon/client 共享 fail-closed /workspace 与持久状态但只给 daemon 注入密钥
# 设计：分别截取 service 并断言相同 long-syntax mount 结构，避免短语命中掩盖 create_host_path 默认行为
def test_compose_declares_shared_path_identity_and_secret_boundary() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")
    daemon = _compose_service(compose, "daemon")
    client = _compose_service(compose, "client")

    for service in (daemon, client):
        assert "image: ${KAMA_IMAGE:-kamaclaude:phase6}" in service
        assert "working_dir: /workspace" in service
        assert """    volumes:
      - type: bind
        source: ${KAMA_WORKSPACE:?set KAMA_WORKSPACE}
        target: /workspace
        bind:
          create_host_path: false
      - type: volume
        source: kama-state
        target: /home/kama/.kama
""" in service
        assert service.count("create_host_path: false") == 1
    assert "build:" not in client
    assert "KAMA_HOST: 0.0.0.0" in daemon
    assert "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}" in daemon
    assert "KAMA_HOST: daemon" in client
    assert "ANTHROPIC_API_KEY" not in client
    assert "kama ping" in daemon


# 功能：验证基础 Compose 不自动重启 daemon 且 smoke 会检查运行时 restart policy 为 no
# 设计：静态合同排除任何 restart 声明并锁定 inspect 证据，真实 policy 值由完整 runtime smoke 决定
def test_compose_has_no_restart_policy_and_smoke_inspects_default() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")
    smoke = SMOKE.read_text(encoding="utf-8")

    assert not re.search(r"^\s*restart:", compose, flags=re.MULTILINE)
    assert "RestartPolicy.Name" in smoke
    assert "restart policy must default to no" in smoke


# 功能：验证 smoke 包含不存在 workspace 的 fail-closed bind 负向验收与独立资源清理
# 设计：锁定独立 project、启动失败、路径不创建及 down 清理四个步骤，真实 Docker 行为留给完整 smoke
def test_smoke_probes_missing_workspace_without_creating_host_path() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    assert "MISSING_WORKSPACE_PROJECT=" in smoke
    assert "MISSING_WORKSPACE_DIR=" in smoke
    assert 'test ! -e "$MISSING_WORKSPACE_DIR"' in smoke
    assert 'KAMA_WORKSPACE="$MISSING_WORKSPACE_DIR"' in smoke
    assert 'docker compose -p "$MISSING_WORKSPACE_PROJECT"' in smoke
    assert "missing workspace unexpectedly started daemon" in smoke
    assert "missing workspace path was created" in smoke
    assert "down --volumes --remove-orphans" in smoke


# 功能：验证 Compose 声明最小权限边界且不把无认证 daemon 发布到主机网络
# 设计：只检查可文本证明的配置，实际 capability、UID 和只读写入面由 runtime smoke 验证
def test_compose_declares_runtime_hardening_without_host_exposure() -> None:
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "read_only: true" in compose
    assert "cap_drop:" in compose
    assert "- ALL" in compose
    assert "no-new-privileges:true" in compose
    assert "tmpfs:" in compose
    assert "/tmp" in compose
    assert "pids_limit:" in compose
    assert "ports:" not in compose
    assert "privileged:" not in compose
    assert "network_mode: host" not in compose
    assert "/var/run/docker.sock" not in compose


# 功能：验证 smoke 脚本为每次运行创建唯一 Compose project 并在所有退出路径清理
# 设计：静态锁定 -p project 传播和 trap，真实资源无残留由运行后 Docker 状态检查证明
def test_smoke_script_uses_unique_project_and_trap_cleanup() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    assert "PROJECT_NAME=" in smoke
    assert "$$" in smoke
    assert re.search(r"^cleanup\(\)", smoke, flags=re.MULTILINE)
    assert re.search(r"^trap cleanup EXIT$", smoke, flags=re.MULTILINE)
    assert re.search(r"^trap 'exit 130' INT$", smoke, flags=re.MULTILINE)
    assert re.search(r"^trap 'exit 143' TERM$", smoke, flags=re.MULTILINE)
    assert smoke.index("trap cleanup EXIT") < smoke.index("mktemp -d")
    assert 'docker compose -p "$PROJECT_NAME"' in smoke
    assert 'down --volumes --remove-orphans' in smoke
    assert "docker compose down" not in smoke


# 功能：验证 smoke 收到 INT 或 TERM 时保留标准非零退出码并清理早期临时资源
# 设计：使用脚本内受控信号探针在接触 Docker 前自发信号，直接覆盖真实 Bash trap 而不依赖 daemon
def test_smoke_script_preserves_signal_status_and_cleans_early_resources(tmp_path: Path) -> None:
    canary_path = ROOT / "phase6-unlisted-private-canary.txt"
    assert not canary_path.exists()

    for signal_name, expected_status in (("INT", 130), ("TERM", 143)):
        signal_tmp = tmp_path / signal_name.lower()
        signal_tmp.mkdir()
        env = os.environ.copy()
        env["PHASE6_SIGNAL_PROBE"] = signal_name
        env["TMPDIR"] = str(signal_tmp)

        completed = subprocess.run(
            ["bash", str(SMOKE)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == expected_status, completed.stderr
        assert list(signal_tmp.iterdir()) == []
        assert not canary_path.exists()

# 功能：验证打包后 runtime smoke 同时拒绝 workspace 中指向外部文件和目录的 symlink
# 设计：静态确认脚本会创建两类真实 symlink 并把两条路径送入 packaged SearchCodeTool 拒绝循环
def test_smoke_script_probes_external_file_and_directory_symlinks() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    assert 'ln -s /etc/passwd "$WORKSPACE_DIR/outside-link"' in smoke
    assert 'ln -s /etc "$WORKSPACE_DIR/outside-dir"' in smoke
    assert '("../etc/passwd", "outside-link", "outside-dir", "outside-dir/passwd")' in smoke


# 功能：验证 runtime smoke 通过公开 loader 从已安装 distribution 加载全部 builtin profile 与一个 skill
# 设计：锁定 RECORD、installed module path、无 /build sys.path 和真实 loader 调用，最终资源可用性由镜像内执行证明
def test_smoke_probes_installed_builtin_package_data() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    assert 'distribution("KamaClaude")' in smoke
    assert "kama_claude/core/agents/builtin/planner.toml" in smoke
    assert "kama_claude/core/agents/builtin/executor.toml" in smoke
    assert "kama_claude/core/agents/builtin/reviewer.toml" in smoke
    assert "kama_claude/core/skills/builtin/review.md" in smoke
    assert "AgentProfileLoader(Path(\"/workspace\"))" in smoke
    assert 'for name in ("planner", "executor", "reviewer")' in smoke
    assert "SkillLoader(Path(\"/workspace\")).resolve(\"review\")" in smoke
    assert "is_relative_to(site_root)" in smoke
    assert 'path.startswith("/build/")' in smoke


# 功能：验证 smoke 在 daemon recreate 前后用同一 SessionStore API 写入并读回应用 artifact
# 设计：比较 writer/recreate/reader 的文本位置，证明测试覆盖 named volume 持久化但不冒充 SessionManager rehydrate
def test_smoke_probes_session_store_persistence_across_recreate() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")

    writer = smoke.index("store.write_meta(Session(")
    recreate = smoke.index("compose rm -f daemon")
    reader = smoke.rindex('store.read_meta("sess-phase6-persistence")')

    assert writer < recreate < reader
    assert "from kama_claude.core.session import Session, SessionStore" in smoke
    assert 'SessionStore(Path("/home/kama/.kama/sessions"))' in smoke
    assert 'store.append_note(sid, marker, "phase6-smoke")' in smoke
    assert 'marker in store.read_notes("sess-phase6-persistence")' in smoke


# 功能：验证 CI 构建独立 test stage、执行 runtime smoke 且从不发布镜像
# 设计：工作流文本只证明命令与触发边界，Docker 语义由 CI 中实际 build/inspect/smoke 给出
def test_workflow_builds_test_stage_and_never_pushes() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "docker build --target test" in workflow
    assert "scripts/docker_smoke.sh" in workflow
    assert "uv lock --check" in workflow
    assert "push: true" not in workflow
    assert "docker push" not in workflow
    assert "ANTHROPIC_API_KEY" not in workflow


# 功能：验证 uv.lock 成为冻结构建输入且不再被仓库 ignore 规则排除
# 设计：直接检查文件存在和精确 ignore 行，Git tracked 状态留给最终 name/status 审计
def test_lockfile_exists_and_is_not_ignored() -> None:
    ignored_lines = {
        line.strip()
        for line in _read(".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert (ROOT / "uv.lock").is_file()
    assert "uv.lock" not in ignored_lines
