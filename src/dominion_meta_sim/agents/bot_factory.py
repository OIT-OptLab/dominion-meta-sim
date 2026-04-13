from dominion_meta_sim.agents.parametric_bot import ParametricBot
from dominion_meta_sim.agents.strategy import Strategy


class BotFactory:
    def create(self, strategy: Strategy) -> ParametricBot:
        return ParametricBot(strategy=strategy)