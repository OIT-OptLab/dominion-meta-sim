from __future__ import annotations

import random

from dominion_meta_sim.agents.bot_factory import BotFactory
from dominion_meta_sim.agents.strategy import Strategy
from dominion_meta_sim.evolution.population import Population
from dominion_meta_sim.game.pyminion_adapter import PyminionMatchRunner


class FitnessEvaluator:
    """
    Estimate empirical fitness by Monte Carlo simulation.

    \hat{f}(w) = (1/N) \sum_k Win_k
    """

    def __init__(
        self,
        bot_factory: BotFactory,
        match_runner: PyminionMatchRunner,
        rng_seed: int = 42,
    ) -> None:
        self.bot_factory = bot_factory
        self.match_runner = match_runner
        self.rng = random.Random(rng_seed)

    def evaluate(
        self,
        strategy: Strategy,
        population: Population,
        n_opponents: int = 5,
        n_games_per_opponent: int = 3,
    ) -> float:
        total_score = 0.0
        total_games = 0

        for _ in range(n_opponents):
            opponent_strategy = population.sample_strategy(self.rng)

            for _ in range(n_games_per_opponent):
                bot_a = self.bot_factory.create(strategy)
                bot_b = self.bot_factory.create(opponent_strategy)

                score = self.match_runner.play_once(bot_a, bot_b)
                total_score += score
                total_games += 1

        if total_games == 0:
            raise ValueError("No games were played")

        return total_score / total_games