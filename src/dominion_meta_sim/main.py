from __future__ import annotations

import numpy as np

from dominion_meta_sim.agents.bot_factory import BotFactory
from dominion_meta_sim.agents.strategy import Strategy
from dominion_meta_sim.evolution.fitness import FitnessEvaluator
from dominion_meta_sim.evolution.population import Population
from dominion_meta_sim.game.pyminion_adapter import PyminionMatchRunner


def main() -> None:
    # Example strategies for 4-dimensional card features:
    # [cost, is_action, is_treasure, is_victory]
    s1 = Strategy(np.array([0.0, 0.2, 1.0, 0.3, -0.5]), name="action_pref")
    s2 = Strategy(np.array([0.0, 0.5, 0.2, 1.0, -0.3]), name="treasure_pref")
    s3 = Strategy(np.array([0.0, 0.1, 0.1, 0.2, 0.8]), name="victory_pref")

    population = Population(
        strategies=[s1, s2, s3],
        probs=np.array([0.4, 0.4, 0.2], dtype=np.float64),
    )

    bot_factory = BotFactory()
    match_runner = PyminionMatchRunner()
    evaluator = FitnessEvaluator(
        bot_factory=bot_factory,
        match_runner=match_runner,
        rng_seed=123,
    )

    fitness = evaluator.evaluate(
        strategy=s1,
        population=population,
        n_opponents=3,
        n_games_per_opponent=2,
    )

    print(f"Estimated fitness of {s1.name}: {fitness:.3f}")


if __name__ == "__main__":
    main()