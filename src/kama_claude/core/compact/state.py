from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompactionState(StrEnum):
    """Hard admission 与 soft/target hysteresis 的独立状态。"""

    HARD_UNSAFE = "HARD_UNSAFE"
    SOFT_PRESSURED = "SOFT_PRESSURED"
    SAFE_ABOVE_TARGET = "SAFE_ABOVE_TARGET"
    AT_TARGET = "AT_TARGET"


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """可按模型/工作负载覆盖的 trigger、target 与 bounded attempt policy。"""

    soft_trigger_ratio: float = 0.80
    target_ratio: float = 0.60
    max_reduction_attempts: int = 2
    min_compactable_tokens: int = 256

    # 校验 policy 数值范围，避免错误配置削弱 hard admission
    def __post_init__(self) -> None:
        if not 0 < self.target_ratio < 1:
            raise ValueError("target_ratio must be between 0 and 1")
        if not 0 < self.soft_trigger_ratio <= 1:
            raise ValueError("soft_trigger_ratio must be between 0 and 1")
        if self.target_ratio >= self.soft_trigger_ratio:
            raise ValueError("target_ratio must be below soft_trigger_ratio")
        if self.max_reduction_attempts < 1:
            raise ValueError("max_reduction_attempts must be positive")


# 将 occupancy/capacity 映射为 hard、soft、target 状态
def classify_compaction_state(
    occupancy_tokens: int,
    capacity_tokens: int,
    *,
    policy: CompactionPolicy = CompactionPolicy(),
) -> CompactionState:
    if capacity_tokens <= 0 or occupancy_tokens > capacity_tokens:
        return CompactionState.HARD_UNSAFE
    ratio = occupancy_tokens / capacity_tokens
    if ratio >= policy.soft_trigger_ratio:
        return CompactionState.SOFT_PRESSURED
    if ratio > policy.target_ratio:
        return CompactionState.SAFE_ABOVE_TARGET
    return CompactionState.AT_TARGET


# 判断请求是否因 hard safety 失败；safe-above-target 永远不是 fatal
def hard_admission_required(state: CompactionState) -> bool:
    return state is CompactionState.HARD_UNSAFE


# 判断 proactive pressure 是否仍可 fail-soft
def proactive_compaction_may_fail_soft(state: CompactionState) -> bool:
    return state in {
        CompactionState.SOFT_PRESSURED,
        CompactionState.SAFE_ABOVE_TARGET,
        CompactionState.AT_TARGET,
    }
