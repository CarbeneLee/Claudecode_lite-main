from __future__ import annotations

from dataclasses import dataclass

# checkpoint 种类：pre_run 快照 / internal 中间态 / user 最终产物
KIND_PRE_RUN = "pre_run"
KIND_INTERNAL = "internal"
KIND_USER = "user"


# 快照记录：一次 checkpoint 的完整元数据，ref 为对象库中的登记地址
@dataclass(frozen=True)
class GitCheckpoint:
    run_id: str
    step: int  # 0 = baseline / pre-run；>0 = 逐步 checkpoint
    label: str
    kind: str
    commit_sha: str
    ref: str
    short_sha: str
    diff_stat: str  # --stat 摘要（供 review）
    ts: str  # ISO 8601
    dirty: bool  # checkpoint 前工作树是否非干净


# 生成 checkpoint ref 路径：refs/<ns>/checkpoints/<run_id>/<step>
def checkpoint_ref(namespace: str, run_id: str, step: int) -> str:
    return f"{namespace}/checkpoints/{run_id}/{step}"


# 生成 pre-run 快照 ref 路径：refs/<ns>/<run_id>/pre-run（独立于 step 链）
def pre_run_ref(namespace: str, run_id: str) -> str:
    return f"{namespace}/{run_id}/pre-run"
