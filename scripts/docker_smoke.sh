#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMPOSE_FILE="$ROOT_DIR/compose.yaml"
PROJECT_NAME="kamaclaude-p6-${GITHUB_RUN_ID:-local}-$$-${RANDOM:-0}"
MISSING_WORKSPACE_PROJECT="${PROJECT_NAME}-missing-workspace"
TEMP_ROOT=""
WORKSPACE_DIR=""
MISSING_WORKSPACE_DIR=""
CANARY_PATH="$ROOT_DIR/phase6-unlisted-private-canary.txt"
CANARY_CREATED=0
SECRET_CANARY="phase6-build-context-canary-$$-${RANDOM:-0}"
TEST_IMAGE="$PROJECT_NAME-test"
RUNTIME_IMAGE="$PROJECT_NAME-runtime:latest"
DOCKER_TOUCHED=0

export ANTHROPIC_API_KEY="phase6-ci-dummy-key"
export KAMA_GID="${KAMA_GID:-10001}"
export KAMA_IMAGE="$RUNTIME_IMAGE"
export KAMA_UID="${KAMA_UID:-10001}"
export KAMA_WORKSPACE="$ROOT_DIR"

compose() {
  docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" "$@"
}

missing_workspace_compose() {
  KAMA_WORKSPACE="$MISSING_WORKSPACE_DIR" \
    docker compose -p "$MISSING_WORKSPACE_PROJECT" -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DOCKER_TOUCHED" == "1" ]]; then
    if [[ -n "$MISSING_WORKSPACE_DIR" ]] \
      && ! missing_workspace_compose down --volumes --remove-orphans >/dev/null 2>&1; then
      if [[ $status -eq 0 ]]; then
        status=1
      fi
    fi
    if ! compose down --volumes --remove-orphans >/dev/null 2>&1; then
      if [[ $status -eq 0 ]]; then
        status=1
      fi
    fi
    docker image rm "$TEST_IMAGE" >/dev/null 2>&1 || true
    docker image rm "$RUNTIME_IMAGE" >/dev/null 2>&1 || true
  fi
  if [[ "$CANARY_CREATED" == "1" ]]; then
    rm -f "$CANARY_PATH"
  fi
  if [[ -n "$TEMP_ROOT" ]]; then
    rm -rf "$TEMP_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  printf 'phase6 smoke failed: %s\n' "$1" >&2
  exit 1
}

assert_equals() {
  local expected=$1
  local actual=$2
  local message=$3
  [[ "$actual" == "$expected" ]] || fail "$message (expected=$expected actual=$actual)"
}

assert_project_absent() {
  local project_name=$1
  [[ -z "$(docker ps -aq --filter "label=com.docker.compose.project=$project_name")" ]] \
    || fail "project container leaked: $project_name"
  [[ -z "$(docker volume ls -q --filter "label=com.docker.compose.project=$project_name")" ]] \
    || fail "project volume leaked: $project_name"
  [[ -z "$(docker network ls -q --filter "label=com.docker.compose.project=$project_name")" ]] \
    || fail "project network leaked: $project_name"
  [[ -z "$(docker image ls -q --filter "reference=$project_name*")" ]] \
    || fail "project image leaked: $project_name"
}

probe_missing_workspace() {
  local probe_log="$TEMP_ROOT/missing-workspace.log"
  MISSING_WORKSPACE_DIR="$TEMP_ROOT/does-not-exist"
  test ! -e "$MISSING_WORKSPACE_DIR"
  if KAMA_WORKSPACE="$MISSING_WORKSPACE_DIR" \
    docker compose -p "$MISSING_WORKSPACE_PROJECT" -f "$COMPOSE_FILE" \
      up -d --no-build daemon >"$probe_log" 2>&1; then
    fail "missing workspace unexpectedly started daemon"
  fi
  test ! -e "$MISSING_WORKSPACE_DIR" || fail "missing workspace path was created"
  missing_workspace_compose down --volumes --remove-orphans >/dev/null 2>&1
  assert_project_absent "$MISSING_WORKSPACE_PROJECT"
}

wait_healthy() {
  local container_id
  local health
  local attempt
  container_id="$(compose ps -q daemon)"
  [[ -n "$container_id" ]] || fail "daemon container was not created"
  for attempt in $(seq 1 30); do
    health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "$container_id")"
    if [[ "$health" == "healthy" ]]; then
      return 0
    fi
    if [[ "$health" == "unhealthy" ]]; then
      compose logs daemon >&2
      fail "daemon became unhealthy"
    fi
    sleep 1
  done
  compose logs daemon >&2
  fail "daemon did not become healthy"
}

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/kamaclaude-phase6.XXXXXX")"
WORKSPACE_DIR="$TEMP_ROOT/workspace"
export KAMA_WORKSPACE="$WORKSPACE_DIR"

mkdir -p "$WORKSPACE_DIR"
printf '%s\n' 'phase6-docker-marker' >"$WORKSPACE_DIR/marker.txt"
ln -s /etc/passwd "$WORKSPACE_DIR/outside-link"
ln -s /etc "$WORKSPACE_DIR/outside-dir"
test ! -e "$CANARY_PATH" || fail "build-context canary path already exists"
printf '%s\n' "$SECRET_CANARY" >"$CANARY_PATH"
CANARY_CREATED=1

case "${PHASE6_SIGNAL_PROBE:-}" in
  INT) kill -INT "$$" ;;
  TERM) kill -TERM "$$" ;;
  "") ;;
  *) fail "unsupported PHASE6_SIGNAL_PROBE: $PHASE6_SIGNAL_PROBE" ;;
esac

cd "$ROOT_DIR"
DOCKER_TOUCHED=1
docker info >/dev/null
compose config --quiet

if [[ "${PHASE6_SKIP_TEST_BUILD:-0}" != "1" ]]; then
  docker build --target test --tag "$TEST_IMAGE" .
fi

compose build daemon
docker image inspect "$RUNTIME_IMAGE" >/dev/null

RUNTIME_UID="$(docker run --rm --entrypoint id "$RUNTIME_IMAGE" -u)"
[[ "$RUNTIME_UID" != "0" ]] || fail "runtime user must be non-root"
assert_equals '["kama-core"]' "$(docker image inspect --format '{{json .Config.Cmd}}' "$RUNTIME_IMAGE")" "runtime CMD must be kama-core"
assert_equals '/workspace' "$(docker image inspect --format '{{.Config.WorkingDir}}' "$RUNTIME_IMAGE")" "runtime workdir must be /workspace"
docker run --rm --entrypoint sh "$RUNTIME_IMAGE" -c '
  command -v kama >/dev/null
  command -v kama-core >/dev/null
  command -v kama-tui >/dev/null
  python --version
  ! command -v uv >/dev/null
  ! command -v pytest >/dev/null
  test ! -e /opt/test-venv
  test ! -e /build
'
docker run --rm -i \
  --user "$KAMA_UID:$KAMA_GID" \
  --workdir /workspace \
  --env HOME=/tmp/kama-builtin-audit \
  --entrypoint python \
  "$RUNTIME_IMAGE" - <<'PY'
import sys
from importlib.metadata import distribution
from pathlib import Path

from kama_claude.core.agents import AgentProfileLoader
from kama_claude.core.agents import loader as agents_loader
from kama_claude.core.skills import SkillLoader
from kama_claude.core.skills import loader as skills_loader

dist = distribution("KamaClaude")
site_root = Path(dist.locate_file("")).resolve()
record_paths = {str(path) for path in (dist.files or [])}
expected = {
    "kama_claude/core/agents/builtin/planner.toml",
    "kama_claude/core/agents/builtin/executor.toml",
    "kama_claude/core/agents/builtin/reviewer.toml",
    "kama_claude/core/skills/builtin/review.md",
}
assert expected <= record_paths
assert not Path("/build").exists()
assert not any(path == "/build" or path.startswith("/build/") for path in sys.path)
assert not Path("/workspace/.kama").exists()
assert not Path("/tmp/kama-builtin-audit/.kama").exists()
assert Path(agents_loader.__file__).resolve().parent.joinpath("builtin").is_relative_to(site_root)
assert Path(skills_loader.__file__).resolve().parent.joinpath("builtin").is_relative_to(site_root)

profiles = AgentProfileLoader(Path("/workspace"))
for name in ("planner", "executor", "reviewer"):
    profile = profiles.load(name)
    assert profile is not None and profile.name == name
    assert profile.system_prompt and "search_code" in profile.allowed_tools

review = SkillLoader(Path("/workspace")).resolve("review")
assert review is not None and review.name == "review"
assert review.system_prompt_template and "search_code" in review.allowed_tools
print("installed builtin profiles and review skill verified")
PY

IMAGE_HISTORY="$(docker history --no-trunc "$RUNTIME_IMAGE")"
[[ "$IMAGE_HISTORY" != *"$SECRET_CANARY"* ]] || fail "build-context canary leaked into image history"
docker image save --output "$TEMP_ROOT/runtime-image.tar" "$RUNTIME_IMAGE"
if grep -a -F -q "$SECRET_CANARY" "$TEMP_ROOT/runtime-image.tar"; then
  fail "build-context canary leaked into image layers"
fi

probe_missing_workspace

compose up -d daemon
wait_healthy
DAEMON_ID="$(compose ps -q daemon)"

assert_equals 'true' "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$DAEMON_ID")" "daemon root filesystem must be read-only"
assert_equals '["ALL"]' "$(docker inspect --format '{{json .HostConfig.CapDrop}}' "$DAEMON_ID")" "daemon must drop all capabilities"
assert_equals 'no' "$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "$DAEMON_ID")" "restart policy must default to no"
[[ "$(docker port "$DAEMON_ID")" == "" ]] || fail "daemon must not publish a host port"

compose run --rm -T client kama ping
compose run --rm --no-deps -T client sh -c '
  test "$(pwd -P)" = /workspace
  grep -F phase6-docker-marker /workspace/marker.txt >/dev/null
  test -z "${ANTHROPIC_API_KEY:-}"
  printf "%s\n" container-write > /workspace/container-write.txt
  printf "%s\n" persistent-state > /home/kama/.kama/phase6-state.txt
'
[[ "$(sed -n '1p' "$WORKSPACE_DIR/container-write.txt")" == "container-write" ]] || fail "container write did not reach host workspace"

SEARCH_OUTPUT="$(compose run --rm --no-deps -T client python - <<'PY'
import asyncio
import os
from pathlib import Path

from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.workspace.errors import InvalidWorkspacePathError, WorkspaceEscapeError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

root = Path("/workspace")
assert os.O_NOFOLLOW
assert os.O_DIRECTORY
assert os.open in os.supports_dir_fd
assert os.scandir in os.supports_fd
tool = SearchCodeTool(WorkspacePathResolver(root), WorkspaceAccessPolicy(root))
result = asyncio.run(tool.invoke({"query": "phase6-docker-marker"}))
assert not result.is_error
assert "marker.txt:1: phase6-docker-marker" in result.content
assert "/Users/" not in result.content
assert "/private/" not in result.content
for path in ("../etc/passwd", "outside-link", "outside-dir", "outside-dir/passwd"):
    try:
        asyncio.run(tool.invoke({"query": "root", "path": path}))
    except (InvalidWorkspacePathError, WorkspaceEscapeError):
        pass
    else:
        raise AssertionError(f"unsafe search path was accepted: {path}")
print(result.content)
PY
)"
[[ "$SEARCH_OUTPUT" == *"marker.txt:1: phase6-docker-marker"* ]] || fail "packaged search_code smoke returned no result"
[[ "$SEARCH_OUTPUT" != *"$WORKSPACE_DIR"* ]] || fail "search output leaked host workspace path"

compose run --rm --no-deps -T client python - <<'PY'
from pathlib import Path

from kama_claude.core.session import Session, SessionStore

marker = "phase6-sessionstore-persistence"
sid = "sess-phase6-persistence"
store = SessionStore(Path("/home/kama/.kama/sessions"))
store.write_meta(Session(
    id=sid,
    mode="chat",
    status="waiting_for_input",
    title=marker,
    created_at="2026-01-01T00:00:00+00:00",
    updated_at="2026-01-01T00:00:00+00:00",
    workspace_root=Path("/workspace"),
))
store.append_note(sid, marker, "phase6-smoke")
assert store.read_meta(sid).title == marker
assert marker in store.read_notes(sid)
PY

compose exec -T daemon sh -c 'printf "%s\n" tmp-canary > /tmp/phase6-tmp-canary'
STOP_STARTED="$(date +%s)"
compose stop -t 30 daemon
STOP_ELAPSED="$(( $(date +%s) - STOP_STARTED ))"
assert_equals '0' "$(docker inspect --format '{{.State.ExitCode}}' "$DAEMON_ID")" "daemon SIGTERM exit code must be zero"
[[ "$STOP_ELAPSED" -le 30 ]] || fail "daemon exceeded SIGTERM grace period"

compose run --rm --no-deps -T client python - <<'PY'
import json
from pathlib import Path

trace = Path("/home/kama/.kama/traces/daemon.jsonl")
assert trace.is_file()
for line in trace.read_text(encoding="utf-8").splitlines():
    json.loads(line)
assert Path("/home/kama/.kama/phase6-state.txt").read_text(encoding="utf-8").strip() == "persistent-state"
PY

compose rm -f daemon
compose up -d daemon
wait_healthy
compose exec -T daemon test ! -e /tmp/phase6-tmp-canary
compose run --rm --no-deps -T client test -f /home/kama/.kama/phase6-state.txt
compose run --rm --no-deps -T client python - <<'PY'
from pathlib import Path

from kama_claude.core.session import SessionStore

marker = "phase6-sessionstore-persistence"
store = SessionStore(Path("/home/kama/.kama/sessions"))
assert store.read_meta("sess-phase6-persistence").title == marker
assert marker in store.read_notes("sess-phase6-persistence")
PY

FINAL_DAEMON_ID="$(compose ps -q daemon)"
compose stop -t 30 daemon
assert_equals '0' "$(docker inspect --format '{{.State.ExitCode}}' "$FINAL_DAEMON_ID")" "recreated daemon SIGTERM exit code must be zero"

printf 'phase6 docker smoke passed\n'
printf 'project=%s\n' "$PROJECT_NAME"
printf 'image=%s\n' "$RUNTIME_IMAGE"
printf 'platform=%s/%s\n' \
  "$(docker image inspect --format '{{.Os}}' "$RUNTIME_IMAGE")" \
  "$(docker image inspect --format '{{.Architecture}}' "$RUNTIME_IMAGE")"
printf 'runtime_uid=%s\n' "$RUNTIME_UID"
printf 'image_size_bytes=%s\n' "$(docker image inspect --format '{{.Size}}' "$RUNTIME_IMAGE")"
printf 'sigterm_seconds=%s\n' "$STOP_ELAPSED"
