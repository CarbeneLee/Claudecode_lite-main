from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


SafeIdentifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
NonEmptyString = Annotated[str, Field(min_length=1)]


# 校验来自 JSON 的路径是无盘符、无父目录跳转的 POSIX 相对路径
def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("path must be a safe relative path")
    return value


class PublicTaskSpec(_FrozenModel):
    id: SafeIdentifier
    goal: NonEmptyString
    workspace_fixture: NonEmptyString
    timeout_s: Annotated[float, Field(gt=0)]

    @field_validator("workspace_fixture")
    @classmethod
    # 校验公开 workspace fixture 只能使用安全相对路径
    def _workspace_fixture_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class FileExistsCriterion(_FrozenModel):
    id: SafeIdentifier
    kind: Literal["file_exists"]
    path: NonEmptyString

    @field_validator("path")
    @classmethod
    # 校验文件存在条件只能引用最终 workspace 内的相对路径
    def _path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class FileContainsCriterion(_FrozenModel):
    id: SafeIdentifier
    kind: Literal["file_contains"]
    path: NonEmptyString
    text: NonEmptyString

    @field_validator("path")
    @classmethod
    # 校验文件包含条件只能引用最终 workspace 内的相对路径
    def _path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class FileNotContainsCriterion(_FrozenModel):
    id: SafeIdentifier
    kind: Literal["file_not_contains"]
    path: NonEmptyString
    text: NonEmptyString

    @field_validator("path")
    @classmethod
    # 校验文件排除条件只能引用最终 workspace 内的相对路径
    def _path_is_safe(cls, value: str) -> str:
        return _safe_relative_path(value)


class CommandExitCriterion(_FrozenModel):
    id: SafeIdentifier
    kind: Literal["command_exit"]
    argv: Annotated[list[NonEmptyString], Field(min_length=1)]
    expected_exit_code: int = 0
    timeout_s: Annotated[float, Field(gt=0)] = 60.0


Criterion = Annotated[
    FileExistsCriterion
    | FileContainsCriterion
    | FileNotContainsCriterion
    | CommandExitCriterion,
    Field(discriminator="kind"),
]


class PrivateGraderSpec(_FrozenModel):
    criteria: Annotated[list[Criterion], Field(min_length=1)]
