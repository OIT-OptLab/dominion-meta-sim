from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray

from dominion_meta_sim.agents.strategy import Strategy


@dataclass
class Population:
    strategies: list[Strategy]
    probs: NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.strategies) == 0:
            raise ValueError("Population must contain at least one strategy")
        if len(self.strategies) != len(self.probs):
            raise ValueError("strategies and probs must have the same length")

        total = float(np.sum(self.probs))
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")

        self.probs = self.probs / total

    def sample_strategy(self, rng: random.Random) -> Strategy:
        idx = rng.choices(
            population=list(range(len(self.strategies))),
            weights=self.probs.tolist(),
            k=1,
        )[0]
        return self.strategies[idx]