from __future__ import annotations

import pytest

from kama_claude.core.sandbox.config import SandboxConfig


# 功能：验证 SandboxConfig 的默认值契约：默认沙箱、默认镜像、默认联网、默认超时
# 设计：参数化 (字段, 期望值) 对，用 getattr 统一断言，避免为纯数据类写重复测试
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enabled", True),
        ("image", "python:3.12-slim"),
        ("network", True),
        ("exec_timeout_s", 120),
    ],
)
def test_sandbox_config_defaults(field: str, expected: object) -> None:
    assert getattr(SandboxConfig(), field) == expected


# 功能：验证 SandboxConfig 支持全字段显式构造覆盖
# 设计：显式传全部字段断言值被保留，验证构造路径与默认路径一致
def test_sandbox_config_explicit_override() -> None:
    cfg = SandboxConfig(
        enabled=False,
        image="alpine:3.20",
        network=False,
        exec_timeout_s=60,
    )
    assert (cfg.enabled, cfg.image, cfg.network, cfg.exec_timeout_s) == (
        False,
        "alpine:3.20",
        False,
        60,
    )
