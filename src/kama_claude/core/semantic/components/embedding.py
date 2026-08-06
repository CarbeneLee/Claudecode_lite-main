from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class SparseVector:
    """稀疏向量：非零维的索引与取值；索引严格递增、值非零（L2 归一化后）"""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have equal length")
        if any(a >= b for a, b in itertools.pairwise(self.indices)):
            raise ValueError("indices must be strictly increasing")
        if any(v == 0.0 for v in self.values):
            raise ValueError("values must be non-zero")
