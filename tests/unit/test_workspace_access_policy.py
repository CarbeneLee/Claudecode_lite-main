from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.workspace.errors import SensitivePathError
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy


# 功能：验证 workspace 根下的 .git 路径被拒绝
# 设计：同时传 logical 和 canonical path，锁定最直接的敏感目录规则
def test_git_directory_rejected(tmp_path: Path) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)
    logical = str(Path(".git") / "config")

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(logical, tmp_path / ".git" / "config")


# 功能：验证任意子目录中的 .git segment 被拒绝
# 设计：将 .git 放在非首段，防止策略只检查字符串前缀
def test_nested_git_directory_rejected(tmp_path: Path) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)
    logical = str(Path("vendor") / "repo" / ".git" / "HEAD")

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(logical, tmp_path / "vendor" / "repo" / ".git" / "HEAD")


# 功能：验证 logical alias 指向 .git 时 canonical parts 仍触发拒绝
# 设计：logical path 不含 .git，resolved path 显式指向 .git，证明两套 parts 都被检查
def test_canonical_git_alias_rejected(tmp_path: Path) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed("alias/config", tmp_path / ".git" / "config")


# 功能：验证 .env 及其环境变体被拒绝
# 设计：参数化 basename，覆盖精确名称和 .env.* 前缀规则
@pytest.mark.parametrize("name", [".env", ".env.local"])
def test_environment_files_rejected(tmp_path: Path, name: str) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(name, tmp_path / name)


# 功能：验证示例环境文件不受敏感策略阻止
# 设计：参数化两个明确 allowlist 例外，防止 .env.* 规则过宽
@pytest.mark.parametrize("name", [".env.example", ".env.template"])
def test_environment_examples_allowed(tmp_path: Path, name: str) -> None:
    WorkspaceAccessPolicy(tmp_path).ensure_allowed(name, tmp_path / name)


# 功能：验证 PEM 和 key 后缀文件被拒绝
# 设计：参数化两种冻结后缀并放入普通子目录
@pytest.mark.parametrize("name", ["server.pem", "client.key"])
def test_private_key_suffixes_rejected(tmp_path: Path, name: str) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(f"certs/{name}", tmp_path / "certs" / name)


# 功能：验证常见私钥 basename 被拒绝
# 设计：参数化 RSA 与 Ed25519 名称，覆盖无扩展名私钥规则
@pytest.mark.parametrize("name", ["id_rsa", "id_ed25519"])
def test_private_key_basenames_rejected(tmp_path: Path, name: str) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(f"keys/{name}", tmp_path / "keys" / name)


# 功能：验证对应公钥文件允许访问
# 设计：参数化 .pub 文件，确保 basename 私钥规则不会误伤公开材料
@pytest.mark.parametrize("name", ["id_rsa.pub", "id_ed25519.pub"])
def test_public_key_files_allowed(tmp_path: Path, name: str) -> None:
    WorkspaceAccessPolicy(tmp_path).ensure_allowed(
        f"keys/{name}",
        tmp_path / "keys" / name,
    )


# 功能：验证常见 credential 文件名被拒绝
# 设计：参数化 .netrc 与两种 credentials 名称，锁定 basename 精确匹配
@pytest.mark.parametrize("name", [".netrc", "credentials", "credentials.json"])
def test_credential_files_rejected(tmp_path: Path, name: str) -> None:
    policy = WorkspaceAccessPolicy(tmp_path)

    with pytest.raises(SensitivePathError):
        policy.ensure_allowed(f"config/{name}", tmp_path / "config" / name)


# 功能：验证普通配置文件仍可访问
# 设计：使用 config.toml 作为相邻非敏感案例，防止 credential 规则模糊匹配
def test_regular_config_allowed(tmp_path: Path) -> None:
    WorkspaceAccessPolicy(tmp_path).ensure_allowed(
        "config/config.toml",
        tmp_path / "config" / "config.toml",
    )
