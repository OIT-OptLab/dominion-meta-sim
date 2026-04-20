# src/dominion_meta_sim/evolution/fitness.py

"""
モンテカルロシミュレーションに基づいて戦略の経験的適応度を推定する

適応度は固定された対戦相手に対する性能ではなく，
対戦ネットワーク上で遭遇しやすい相手グループ，およびそのグループ内で流行している戦略
との対戦結果に基づいて定義される．

すなわち，グループに属する戦略の適応度を，
ネットワーク構造と各グループ内分布に従ってサンプリングされた相手との対戦結果の平均として
モンテカルロ近似する．
"""

from __future__ import annotations

import random

from dominion_meta_sim.agents.bot_factory import BotFactory
from dominion_meta_sim.agents.strategy import Strategy
from dominion_meta_sim.evolution.meta_network import MetaNetwork
from dominion_meta_sim.game.pyminion_adapter import PyminionMatchRunner


class FitnessEvaluator:
    """
    モンテカルロシミュレーションにより，戦略の経験的適応度を推定するクラス．

    ここで Win_k は各試合の勝敗結果
    （勝利=1.0，敗北=0.0，引き分け=0.5）である．
    """

    def __init__(
        self,
        bot_factory: BotFactory,
        match_runner: PyminionMatchRunner,
        rng_seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        bot_factory : BotFactory
            戦略から Bot インスタンスを生成するファクトリ
        match_runner : PyminionMatchRunner
            2つの Bot を実際に対戦させる実行器
        rng_seed : int, optional
            再現性確保のための乱数シード
        """
        self.bot_factory = bot_factory
        self.match_runner = match_runner
        self.rng = random.Random(rng_seed)

    def evaluate(
        self,
        strategy: Strategy,
        focal_group: int,
        meta_network: MetaNetwork,
        n_opponents: int = 5,
        n_games_per_opponent: int = 3,
    ) -> float:
        """
        指定したグループ focal_group における strategy の適応度を推定する．

        Parameters
        ----------
        strategy : Strategy
            評価対象となる戦略
        focal_group : int
            この戦略が属しているとみなすグループのインデックス
        meta_network : MetaNetwork
            対戦ネットワークおよび各グループ内部の戦略分布
        n_opponents : int, optional
            対戦相手を何回サンプリングするか
        n_games_per_opponent : int, optional
            1回サンプリングした相手戦略に対して何試合行うか

        Returns
        -------
        float
            経験的平均勝率に基づく適応度推定値

        Notes
        -----
        処理の流れは以下の通りである：

        1. 起点グループ focal_group を固定する
        2. 対戦ネットワーク adjacency に基づいて相手グループを選ぶ
        3. 相手グループ内部の戦略分布に従って相手戦略を選ぶ
        4. strategy と相手戦略を複数回対戦させる
        5. 勝敗結果を平均して適応度とする

        これにより，
        「どのグループに属し，どの相手と当たりやすいか」
        を考慮した局所的な適応度評価が可能となる．
        """
        total_score = 0.0
        total_games = 0

        # 指定回数だけ対戦相手をサンプリングする
        for _ in range(n_opponents):
            # 対戦ネットワークに基づいて，相手グループを選ぶ
            opponent_group = meta_network.sample_opponent_group_index(
                focal_group=focal_group,
                rng=self.rng,
            )

            # 選ばれた相手グループ内部の戦略分布に従って，相手戦略を選ぶ
            opponent_strategy = meta_network.sample_strategy_in_group(
                group_index=opponent_group,
                rng=self.rng,
            )

            # 同じ相手戦略に対して複数試合行い，勝率のばらつきを抑える
            for _ in range(n_games_per_opponent):
                bot_a = self.bot_factory.create(strategy)
                bot_b = self.bot_factory.create(opponent_strategy)

                # bot_a 視点での結果を取得
                # 勝利=1.0, 敗北=0.0, 引き分け=0.5
                score = self.match_runner.play_once(bot_a, bot_b)

                total_score += score
                total_games += 1

        if total_games == 0:
            raise ValueError("No games were played")

        # 平均勝率を適応度の推定値として返す
        return total_score / total_games