# src/dominion_meta_sim/game/pyminion_adapter.py
from __future__ import annotations

from pyminion.expansions.base import base_set, smithy
from pyminion.game import Game

from dominion_meta_sim.agents.parametric_bot import ParametricBot


class PyminionMatchRunner:
    def __init__(self) -> None:
        self.expansions = [base_set]
        self.kingdom_cards = [smithy]

    def play_once(self, bot_a: ParametricBot, bot_b: ParametricBot) -> float:
        game = Game(
            players=[bot_a, bot_b],
            expansions=self.expansions,
            kingdom_cards=self.kingdom_cards,
            log_stdout=False,
        )
        result = game.play()

        a_summary = None
        b_summary = None

        for summary in result.player_summaries:
            if summary.player is bot_a:
                a_summary = summary
            elif summary.player is bot_b:
                b_summary = summary

        if a_summary is None or b_summary is None:
            raise RuntimeError("Failed to find player summaries for both bots.")

        if a_summary.score > b_summary.score:
            return 1.0
        if a_summary.score < b_summary.score:
            return 0.0
        return 0.5