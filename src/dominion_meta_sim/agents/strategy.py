from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Strategy:
    """Parametric strategy represented by a weight vector."""
    weights: NDArray[np.float64]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D vector")

    @property
    def dim(self) -> int:
        return int(self.weights.shape[0])