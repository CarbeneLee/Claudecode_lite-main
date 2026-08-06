from __future__ import annotations

import pytest

from kama_claude.core.sandbox.errors import (
    ContainerNotReadyError,
    SandboxCreationFailedError,
    SandboxError,
    SandboxImageError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    classify_cli_error,
)

_ALL_ERROR_TYPES = [
    SandboxUnavailableError,
    SandboxImageError,
    SandboxCreationFailedError,
    ContainerNotReadyError,
    SandboxTimeoutError,
]


# 功能：验证五个稳定沙箱异常类型都继承 SandboxError 基类并可携带 detail
# 设计：参数化遍历全部异常类型，断言基类归属与 detail 透传，确保分类与调用方契约稳定
@pytest.mark.parametrize("exc_type", _ALL_ERROR_TYPES)
def test_sandbox_errors_share_base_and_detail(exc_type: type[SandboxError]) -> None:
    exc = exc_type("boom", detail="stderr content")
    assert isinstance(exc, SandboxError)
    assert exc.detail == "stderr content"
    assert str(exc) == "boom"


# 功能：验证 docker daemon 不可用的 stderr 关键词被分类为 SandboxUnavailableError
# 设计：参数化不同措辞但同一语义的 stderr 文本，覆盖大小写差异，确保分类不依赖精确匹配
@pytest.mark.parametrize(
    "stderr",
    [
        "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
        "cannot connect to the docker daemon. Is the daemon running?",
        "error during connect: Cannot connect to the Docker daemon",
    ],
)
def test_classify_unavailable(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), SandboxUnavailableError)


# 功能：验证镜像相关 stderr 关键词被分类为 SandboxImageError
# 设计：参数化 manifest unknown / pull access denied / no such image 三类镜像错误，
#       验证镜像类故障与 daemon 故障、其他创建失败区分开
@pytest.mark.parametrize(
    "stderr",
    [
        "manifest for python:3.12-slim not found: manifest unknown",
        "pull access denied for python:3.12-slim, repository does not exist",
        "Error response from daemon: No such image: python:3.12-slim",
    ],
)
def test_classify_image_error(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), SandboxImageError)


# 功能：验证未知 stderr 内容归入 SandboxCreationFailedError
# 设计：用与任何已知关键词无关的文本覆盖兜底路径，防止未分类错误泄漏为其他类型
def test_classify_fallback_to_creation_failed() -> None:
    assert isinstance(
        classify_cli_error("docker: Error response from daemon: OCI runtime create failed"),
        SandboxCreationFailedError,
    )
    assert isinstance(classify_cli_error(""), SandboxCreationFailedError)
