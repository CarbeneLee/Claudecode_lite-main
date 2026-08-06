ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
    && test -x /opt/venv/bin/kama \
    && test -x /opt/venv/bin/kama-core \
    && test -x /opt/venv/bin/kama-tui

FROM builder AS test

ENV UV_PROJECT_ENVIRONMENT=/opt/test-venv

# slim 基础镜像不含 git；git 模块与 git_head_provider 测试需要真实 CLI
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY tests ./tests
COPY scripts ./scripts
COPY WIRE_PROTOCOL.md README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups --no-editable
RUN /opt/test-venv/bin/pytest tests/unit -q \
    && /opt/test-venv/bin/pytest tests/integration -q \
    && /opt/test-venv/bin/ruff check src tests scripts \
    && /opt/test-venv/bin/mypy src \
    && /opt/test-venv/bin/python scripts/gen_protocol_doc.py --check

FROM ${PYTHON_IMAGE} AS runtime

ARG KAMA_UID=10001
ARG KAMA_GID=10001

ENV HOME=/home/kama \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/.cache
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid "${KAMA_GID}" kama \
    && useradd --uid "${KAMA_UID}" --gid kama --create-home \
        --home-dir /home/kama --shell /usr/sbin/nologin kama \
    && mkdir -p /home/kama/.kama /workspace \
    && chown kama:kama /home/kama /home/kama/.kama /workspace

COPY --from=builder --chown=kama:kama /opt/venv /opt/venv

WORKDIR /workspace
USER kama
STOPSIGNAL SIGTERM
CMD ["kama-core"]
