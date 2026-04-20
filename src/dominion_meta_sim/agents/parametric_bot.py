# src/dominion_meta_sim/agents/parametric_bot.py

"""
パラメトリック戦略（重みベクトル）に基づいてドミニオンの行動を決定する Bot を定義

各戦略はカード特徴量に対する線形評価関数として表現

- カードから特徴量ベクトルを抽出する関数
- その評価関数に基づいて購入行動を決定する Decider
- Pyminion の Bot として動作する ParametricBot
"""

from __future__ import annotations

from typing import Any
import numpy as np

from pyminion.bots.optimized_bot import OptimizedBot, OptimizedBotDecider

from dominion_meta_sim.agents.strategy import Strategy


def extract_card_features(card: Any) -> np.ndarray:
    """
    カードから特徴量ベクトルを抽出する．

    特徴量ベクトル：
        [bias, money, is_action, is_treasure, is_victory]

    Parameters
    ----------
    card : Any
        Pyminion のカードオブジェクト

    Returns
    -------
    np.ndarray
        カード特徴量ベクトル φ(c)

    Notes
    -----
    - bias は定数項として常に1.0
    - money はカードが生み出す金量（存在しない場合は0）
    - type 属性に基づいてカード種別を判定する
    """
    return np.array(
        [
            1.0,  # バイアス項（定数項）
            float(getattr(card, "money", 0)),  # 金量（Treasureや一部Actionで定義される）
            1.0 if "Action" in {str(t) for t in getattr(card, "type", [])} else 0.0,
            1.0 if "Treasure" in {str(t) for t in getattr(card, "type", [])} else 0.0,
            1.0 if "Victory" in {str(t) for t in getattr(card, "type", [])} else 0.0,
        ],
        dtype=np.float64,
    )


class ParametricBotDecider(OptimizedBotDecider):
    """
    パラメトリック戦略に基づいて行動を決定する Decider．

    主に購入フェーズにおいて，
    各カードを評価関数によりスコアリングし，最もスコアの高いカードを選択
    """

    def __init__(self, strategy: Strategy) -> None:
        super().__init__()
        self.strategy = strategy

    def _score_card(self, card: Any) -> float:
        """
        カードのスコア Q(c) を計算する．

        Parameters
        ----------
        card : Any
            評価対象のカード

        Returns
        -------
        float
            スコア Q(c) = w^T φ(c)
        """
        phi = extract_card_features(card)

        # 特徴量次元と戦略次元が一致しているか確認
        if phi.shape[0] != self.strategy.dim:
            raise ValueError(
                f"Feature dimension {phi.shape[0]} does not match "
                f"strategy dimension {self.strategy.dim}"
            )

        # 内積によりカード価値を計算
        return float(np.dot(self.strategy.weights, phi))

    def action_phase_decision(self, valid_actions, player, game):
        """
        アクションフェーズの意思決定．

        現在はベースクラス（OptimizedBot）の既定ロジックに委ねている．
        """
        return super().action_phase_decision(valid_actions, player, game)

    def buy_phase_decision(self, valid_cards, player, game):
        """
        購入フェーズの意思決定．

        購入可能なカード集合 valid_cards に対して，
        各カードをスコアリングし，最も高スコアのカードを選択する．

        Parameters
        ----------
        valid_cards : list
            現在購入可能なカード集合
        player : Player
            プレイヤー状態（未使用）
        game : Game
            ゲーム状態（未使用）

        Returns
        -------
        Card | None
            購入するカード（購入しない場合は None）
        """
        if not valid_cards:
            return None

        # スコアの高い順に並べる
        ranked = sorted(valid_cards, key=self._score_card, reverse=True)

        # 最もスコアの高いカードを選択
        return ranked[0]


class ParametricBot(OptimizedBot):
    """
    パラメトリック戦略に基づいて行動する Bot クラス．

    Pyminion の OptimizedBot を継承し，
    内部の Decider を ParametricBotDecider に置き換えている．
    """

    def __init__(self, strategy: Strategy, player_id: str | None = None) -> None:
        """
        Parameters
        ----------
        strategy : Strategy
            使用する戦略（重みベクトル）
        player_id : str, optional
            プレイヤー識別子（省略時は戦略名を使用）
        """
        super().__init__(
            decider=ParametricBotDecider(strategy),
            player_id=player_id or strategy.name or "parametric_bot",
        )
        self.strategy = strategy