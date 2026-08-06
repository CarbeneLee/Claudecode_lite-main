from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WorkerRequest(_StrictModel):
    task_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    goal: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    runs_dir: str = Field(min_length=1)
    trace_path: str = Field(min_length=1)

    @model_validator(mode="after")
    # 校验 worker 的可写路径均为绝对路径且彼此不存在包含关系
    def _paths_are_absolute_and_distinct(self) -> WorkerRequest:
        paths = tuple(
            Path(value).resolve(strict=False)
            for value in (self.workspace, self.runs_dir, self.trace_path)
        )
        raw_paths = (self.workspace, self.runs_dir, self.trace_path)
        if not all(Path(value).is_absolute() for value in raw_paths):
            raise ValueError("worker paths must be absolute")
        for index, left in enumerate(paths):
            for right in paths[index + 1 :]:
                if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError("worker paths must be distinct")
        return self


class WorkerResult(_StrictModel):
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    runtime_status: Literal["success", "failed", "infra_error"]
    result: str | None = None
    reason: str | None = None
    infra_error: str | None = None
