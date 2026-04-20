# src/dominion_meta_sim/main.py

"""
「戦略の適応度評価」を確認するためのサンプルコード

このコードは、以下のような流れで戦略の適応度評価を行う：
- 複数の戦略（重みベクトル w）からなる母集団 p_t を用意
- 評価対象戦略 w に対して，分布 p_t から対戦相手をサンプリング
- ドミニオンの対戦シミュレーションを複数回実行
- 平均勝率に基づく適応度をモンテカルロ近似により推定
"""

from __future__ import annotations

import numpy as np

from dominion_meta_sim.agents.bot_factory import BotFactory
from dominion_meta_sim.agents.strategy import Strategy
from dominion_meta_sim.evolution.fitness import FitnessEvaluator
from dominion_meta_sim.evolution.meta_network import MetaNetwork
from dominion_meta_sim.game.pyminion_adapter import PyminionMatchRunner


def main() -> None:
    # =========================================
    # 1. 戦略の定義
    # =========================================
    # 各戦略はカードの特徴量に対する重みベクトルとして表現される．
    #
    # 現在の特徴量：
    # [bias, money, is_action, is_treasure, is_victory]
    #
    # 例：
    # action_pref  : アクションカードを好む
    # treasure_pref: 財宝カードを好む（BigMoneyに近い）
    # victory_pref : 勝利点カードを好む
    #

    s1 = Strategy(np.array([0.0, 0.2, 1.0, 0.3, -0.5]), name="action_pref")
    s2 = Strategy(np.array([0.0, 0.5, 0.2, 1.0, -0.3]), name="treasure_pref")
    s3 = Strategy(np.array([0.0, 0.1, 0.1, 0.2, 0.8]), name="victory_pref")

    # =========================================
    # 2. 母集団（戦略分布）の定義
    # =========================================
    # p_t(w) を離散分布として表現
    #
    # strategies: 戦略の集合
    # probs     : 各戦略の出現確率
    # adjacency  : 戦略間の遭遇頻度を表す隣接行列
    #
    # この分布から対戦相手をサンプリングすることで，
    # 「メタ環境」を表現している

    group_probs = np.array([
        [0.6, 0.3, 0.1],  # group 0
        [0.2, 0.6, 0.2],  # group 1
        [0.1, 0.2, 0.7],  # group 2
    ], dtype=np.float64)

    adjacency = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ], dtype=np.float64)

    meta = MetaNetwork(
        strategies=[s1, s2, s3],
        group_probs=group_probs,
        adjacency=adjacency,
        group_names=["local_A", "ladder_mid", "ladder_high"],
    )

    # =========================================
    # 3. 評価器の構築
    # =========================================
    # BotFactory        : 戦略 → Bot を生成
    # MatchRunner       : 2つのBotを対戦させる
    # FitnessEvaluator  : 勝率に基づく適応度を計算

    bot_factory = BotFactory()
    match_runner = PyminionMatchRunner()
    evaluator = FitnessEvaluator(
        bot_factory=bot_factory,
        match_runner=match_runner,
        rng_seed=123,  # 再現性のため乱数シード固定
    )

    # =========================================
    # 4. 適応度の推定
    # =========================================
    # 評価対象：s1（action_pref）
    #
    # n_opponents         : 対戦相手のサンプル数
    # n_games_per_opponent: 各相手との試合数
    #
    # 合計試合数 = n_opponents × n_games_per_opponent
    # 適応度は、s1 の視点での平均勝率として推定される

    fitness = evaluator.evaluate(
        strategy=s1,
        focal_group=0,
        meta_network=meta,
        n_opponents=3,
        n_games_per_opponent=2,
    )

    # =========================================
    # 5. 結果の出力
    # =========================================
    print(f"Estimated fitness of {s1.name}: {fitness:.3f}")


if __name__ == "__main__":
    main()