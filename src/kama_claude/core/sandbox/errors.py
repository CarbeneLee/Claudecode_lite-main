from __future__ import annotations


class SandboxError(Exception):
    """沙箱错误基类：携带可选详情，供错误分类与日志使用"""

    def __init__(self, message: str, *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


class SandboxUnavailableError(SandboxError):
    """docker CLI 不存在或 docker daemon 无响应"""


class SandboxImageError(SandboxError):
    """镜像不存在、拉取被拒或网络抖动导致镜像不可用"""


class SandboxCreationFailedError(SandboxError):
    """docker run 失败（资源不足、配置错误等非镜像类原因）"""


class ContainerNotReadyError(SandboxError):
    """容器存在但未处于 running 状态"""


class SandboxTimeoutError(SandboxError):
    """容器内命令执行超时"""


# 将 docker CLI stderr 关键词分类为稳定沙箱异常；未知内容归入 creation_failed
def classify_cli_error(stderr: str) -> SandboxError:
    lowered = stderr.lower()
    if "cannot connect to the docker daemon" in lowered:
        return SandboxUnavailableError(
            "docker daemon unavailable", detail=stderr
        )
    if (
        "manifest unknown" in lowered
        or "pull access denied" in lowered
        or "no such image" in lowered
    ):
        return SandboxImageError("sandbox image unavailable", detail=stderr)
    return SandboxCreationFailedError(
        "sandbox container creation failed", detail=stderr
    )
