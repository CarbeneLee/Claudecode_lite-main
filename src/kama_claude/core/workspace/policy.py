from __future__ import annotations

from pathlib import Path

from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)

_ENV_EXCEPTIONS = {".env.example", ".env.template"}
_PRIVATE_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
_CREDENTIAL_NAMES = {".netrc", "credentials", "credentials.json"}


class WorkspaceAccessPolicy:
    # 保存用于解释 canonical path parts 的 workspace root
    def __init__(self, workspace_root: Path) -> None:
        try:
            root = workspace_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidWorkspacePathError(
                "workspace root must be an existing directory"
            ) from exc
        if not root.is_dir():
            raise InvalidWorkspacePathError(
                "workspace root must be an existing directory"
            )
        self._root = root

    @property
    # 返回 canonical workspace root，供 trusted registry 做 identity 校验
    def root(self) -> Path:
        return self._root

    # 检查 logical 与 canonical path parts 是否命中敏感路径规则
    def ensure_allowed(self, logical_path: str, resolved_path: Path) -> None:
        logical_parts = Path(logical_path).parts
        canonical_path = resolved_path.resolve(strict=False)
        try:
            canonical_parts = canonical_path.relative_to(self._root).parts
        except ValueError as exc:
            raise WorkspaceEscapeError("path escapes workspace") from exc

        if any(part.casefold() == ".git" for part in (*logical_parts, *canonical_parts)):
            raise SensitivePathError("sensitive workspace path is blocked")

        basenames = {
            parts[-1].casefold()
            for parts in (logical_parts, canonical_parts)
            if parts
        }
        if any(self._is_sensitive_basename(name) for name in basenames):
            raise SensitivePathError("sensitive workspace path is blocked")

    # 判断 basename 是否属于冻结的敏感文件规则
    def _is_sensitive_basename(self, basename: str) -> bool:
        if basename in _ENV_EXCEPTIONS:
            return False
        if basename == ".env" or basename.startswith(".env."):
            return True
        if basename.endswith((".pem", ".key")):
            return True
        return basename in _PRIVATE_KEY_NAMES or basename in _CREDENTIAL_NAMES
