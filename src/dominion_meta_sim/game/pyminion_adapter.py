# src/dominion_meta_sim/game/pyminion_adapter.py

"""
Pyminion を用いたドミニオン対戦シミュレーションのラッパ

与えられた2つのエージェント（ParametricBot）を対戦させ，
その勝敗結果（勝利=1.0，敗北=0.0，引き分け=0.5）を返す．
"""

from __future__ import annotations
from typing import cast

from pyminion.expansions.base import base_set, smithy
from pyminion.expansions.base import Card
from pyminion.game import Game

from dominion_meta_sim.agents.parametric_bot import ParametricBot


class PyminionMatchRunner:
    """
    Pyminion を用いて 2体のBotを対戦させるクラス．

    ・ゲーム環境（使用カードセット）を固定する
    ・1試合の勝敗結果を数値として返す

    これにより，戦略の適応度評価に利用できる．
    """

    def __init__(self) -> None:
        # 使用する拡張（ここでは Base セットのみ）
        self.expansions = [base_set]

        # 固定するKingdomカードの指定
        # 残りのカードはランダムに選ばれる
        self.kingdom_cards = cast(list[Card], [smithy])

    def play_once(self, bot_a: ParametricBot, bot_b: ParametricBot) -> float:
        """
        2つのBotを1回対戦させ，結果を返す．

        Parameters
        ----------
        bot_a : ParametricBot
            評価対象のエージェント
        bot_b : ParametricBot
            対戦相手のエージェント

        Returns
        -------
        float
            bot_a の視点での勝敗結果
            1.0 : 勝利
            0.0 : 敗北
            0.5 : 引き分け
        """

        # ゲームインスタンスを生成
        game = Game(
            players=[bot_a, bot_b],     # 対戦する2プレイヤー
            expansions=self.expansions, # 使用するカードセット
            kingdom_cards=self.kingdom_cards,  # Kingdomカード
            log_stdout=False,           # ログ出力を抑制
        )

        # ゲームを実行（1試合）
        result = game.play()

        # 各プレイヤーの結果を格納する変数
        a_summary = None
        b_summary = None

        # game.play() の結果には，各プレイヤーの summary が配列で入っている
        # その中から bot_a, bot_b に対応するものを探す
        for summary in result.player_summaries:
            if summary.player is bot_a:
                a_summary = summary
            elif summary.player is bot_b:
                b_summary = summary

        # 万が一取得できなかった場合はエラー
        if a_summary is None or b_summary is None:
            raise RuntimeError("Failed to find player summaries for both bots.")

        # スコア（勝利点）に基づいて勝敗を判定
        if a_summary.score > b_summary.score:
            return 1.0  # bot_a の勝利
        if a_summary.score < b_summary.score:
            return 0.0  # bot_a の敗北

        # 同点の場合
        return 0.5