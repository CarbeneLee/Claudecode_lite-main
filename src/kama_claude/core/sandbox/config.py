from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_EXEC_TIMEOUT_S = 120


@dataclass(frozen=True)
class SandboxConfig:
    # 是否启用 Docker 沙箱；false 时回宿主执行
    enabled: bool = True
    # 沙箱容器使用的镜像
    image: str = _DEFAULT_IMAGE
    # 是否启用容器网络；false 时以 --network none 启动
    network: bool = True
    # 容器内命令执行超时秒数
    exec_timeout_s: int = _DEFAULT_EXEC_TIMEOUT_S
