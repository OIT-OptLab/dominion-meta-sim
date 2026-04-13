# src/dominion_meta_sim/agents/parametric_bot.py
from __future__ import annotations

from typing import Any
import numpy as np

from pyminion.bots.optimized_bot import OptimizedBot, OptimizedBotDecider

from dominion_meta_sim.agents.strategy import Strategy


def extract_card_features(card: Any) -> np.ndarray:
    # Pyminion cards have get_cost(player, game), and card.type is used internally
    # in player.py for phase decisions.
    return np.array(
        [
            1.0,  # bias
            float(getattr(card, "money", 0)),  # many treasure/action cards expose money
            1.0 if "Action" in {str(t) for t in getattr(card, "type", [])} else 0.0,
            1.0 if "Treasure" in {str(t) for t in getattr(card, "type", [])} else 0.0,
            1.0 if "Victory" in {str(t) for t in getattr(card, "type", [])} else 0.0,
        ],
        dtype=np.float64,
    )


class ParametricBotDecider(OptimizedBotDecider):
    def __init__(self, strategy: Strategy) -> None:
        super().__init__()
        self.strategy = strategy

    def _score_card(self, card: Any) -> float:
        phi = extract_card_features(card)
        if phi.shape[0] != self.strategy.dim:
            raise ValueError(
                f"Feature dimension {phi.shape[0]} does not match "
                f"strategy dimension {self.strategy.dim}"
            )
        return float(np.dot(self.strategy.weights, phi))

    def action_phase_decision(self, valid_actions, player, game):
        # 最初は既定ロジックに任せる
        return super().action_phase_decision(valid_actions, player, game)

    def buy_phase_decision(self, valid_cards, player, game):
        if not valid_cards:
            return None

        ranked = sorted(valid_cards, key=self._score_card, reverse=True)
        return ranked[0]


class ParametricBot(OptimizedBot):
    def __init__(self, strategy: Strategy, player_id: str | None = None) -> None:
        super().__init__(
            decider=ParametricBotDecider(strategy),
            player_id=player_id or strategy.name or "parametric_bot",
        )
        self.strategy = strategy