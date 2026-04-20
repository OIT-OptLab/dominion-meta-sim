from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Strategy:
    """重みベクトルで表現されるパラメトリック戦略"""
    weights: NDArray[np.float64]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D vector")

        copied = np.array(self.weights, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "weights", copied)

    @property
    def dim(self) -> int:
        return int(self.weights.shape[0])