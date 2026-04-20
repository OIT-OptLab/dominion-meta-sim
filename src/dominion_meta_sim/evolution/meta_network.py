# src/dominion_meta_sim/evolution/meta_network.py

"""
対戦ネットワーク上で形成されるメタ環境を表現

1. ノード間の対戦ネットワーク
   - 各ノードはプレイヤーグループを表す
   - エッジ重みはグループ間の遭遇しやすさを表す

2. 各ノード内部の戦略分布
   - 各グループは複数の戦略の採用率分布を持つ

3. 共通の代表戦略集合
   - すべてのグループは同じ戦略集合を共有
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray

from dominion_meta_sim.agents.strategy import Strategy


@dataclass
class MetaNetwork:
    """
    対戦ネットワークと，各グループ内部の戦略分布を表すクラス．

    Attributes
    ----------
    strategies : list[Strategy]
        共通の代表戦略集合 [w_1, ..., w_K]
    group_probs : NDArray[np.float64]
        各グループ内部の戦略分布
        shape = (num_groups, num_strategies)
        group_probs[g, k] = グループ g における戦略 k の採用率
    adjacency : NDArray[np.float64]
        グループ間の対戦ネットワーク
        shape = (num_groups, num_groups)
        adjacency[g, h] が大きいほど，グループ g はグループ h と遭遇しやすい
    group_names : list[str] | None
        グループ名（省略可）
    """
    strategies: list[Strategy]
    group_probs: NDArray[np.float64]
    adjacency: NDArray[np.float64]
    group_names: list[str] | None = None

    def __post_init__(self) -> None:
        """
        初期化後に以下を確認・処理する：

        - 戦略集合が空でないこと
        - group_probs の列数が戦略数と一致すること
        - adjacency が正方行列であり，グループ数と一致すること
        - 各グループ内の戦略分布を正規化すること
        - adjacency の各行を正規化すること
        - group_names が与えられていなければ自動生成すること
        """
        num_strategies = len(self.strategies)
        if num_strategies == 0:
            raise ValueError("MetaNetwork must contain at least one strategy")

        if self.group_probs.ndim != 2:
            raise ValueError("group_probs must be a 2D array")

        num_groups, num_strategies_in_probs = self.group_probs.shape

        if num_strategies_in_probs != num_strategies:
            raise ValueError(
                "Number of columns in group_probs must match number of strategies: "
                f"{num_strategies_in_probs} != {num_strategies}"
            )

        if self.adjacency.shape != (num_groups, num_groups):
            raise ValueError(
                "adjacency must be a square matrix with shape "
                f"({num_groups}, {num_groups}), got {self.adjacency.shape}"
            )

        if self.group_names is None:
            self.group_names = [f"group_{g}" for g in range(num_groups)]
        elif len(self.group_names) != num_groups:
            raise ValueError(
                "Length of group_names must match number of groups: "
                f"{len(self.group_names)} != {num_groups}"
            )

        self.group_probs = self._normalize_group_probs(self.group_probs)
        self.adjacency = self._normalize_rows(self.adjacency)

    @staticmethod
    def _normalize_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        行列の各行を正規化する．

        各行の和が1になるようにし，
        行ごとに条件付き確率として解釈できるようにする．

        行和が0の行については一様分布に置き換える．
        """
        normalized = matrix.astype(np.float64, copy=True)
        num_rows, num_cols = normalized.shape

        for i in range(num_rows):
            row_sum = float(np.sum(normalized[i]))
            if row_sum > 0:
                normalized[i] /= row_sum
            else:
                normalized[i] = np.full(num_cols, 1.0 / num_cols, dtype=np.float64)

        return normalized

    @staticmethod
    def _normalize_group_probs(group_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        各グループ内部の戦略分布を正規化する．

        各行が1つのグループに対応し，行和が1になるようにする．
        行和が0の場合は一様分布に置き換える．
        """
        normalized = group_probs.astype(np.float64, copy=True)
        num_groups, num_strategies = normalized.shape

        for g in range(num_groups):
            row_sum = float(np.sum(normalized[g]))
            if row_sum > 0:
                normalized[g] /= row_sum
            else:
                normalized[g] = np.full(
                    num_strategies, 1.0 / num_strategies, dtype=np.float64
                )

        return normalized

    @property
    def num_groups(self) -> int:
        """グループ数 G を返す．"""
        return int(self.group_probs.shape[0])

    @property
    def num_strategies(self) -> int:
        """戦略数 K を返す．"""
        return len(self.strategies)

    def sample_group_index(self, rng: random.Random) -> int:
        """
        グループ全体を一様に1つサンプリングする．

        これは，どのグループのプレイヤーを起点にするかを選ぶ簡易操作である．
        必要に応じて，将来的にはグループサイズ重み付きサンプリングに拡張可能である．
        """
        return rng.randrange(self.num_groups)

    def sample_opponent_group_index(self, focal_group: int, rng: random.Random) -> int:
        """
        対戦ネットワークに基づいて，相手グループを1つサンプリングする．

        Parameters
        ----------
        focal_group : int
            起点となるグループのインデックス
        rng : random.Random
            乱数生成器

        Returns
        -------
        int
            相手グループのインデックス

        Notes
        -----
        adjacency[focal_group, :] を条件付き確率とみなし，
        グループ focal_group に属するプレイヤーがどのグループと遭遇しやすいかを表現する．
        """
        if not (0 <= focal_group < self.num_groups):
            raise IndexError(f"focal_group out of range: {focal_group}")

        return rng.choices(
            population=list(range(self.num_groups)),
            weights=self.adjacency[focal_group].tolist(),
            k=1,
        )[0]

    def sample_strategy_index_in_group(self, group_index: int, rng: random.Random) -> int:
        """
        指定グループ内部の戦略分布に従って，戦略インデックスを1つサンプリングする．

        Parameters
        ----------
        group_index : int
            グループのインデックス
        rng : random.Random
            乱数生成器

        Returns
        -------
        int
            サンプリングされた戦略インデックス
        """
        if not (0 <= group_index < self.num_groups):
            raise IndexError(f"group_index out of range: {group_index}")

        return rng.choices(
            population=list(range(self.num_strategies)),
            weights=self.group_probs[group_index].tolist(),
            k=1,
        )[0]

    def sample_strategy_in_group(self, group_index: int, rng: random.Random) -> Strategy:
        """
        指定グループ内部の戦略分布に従って，戦略を1つサンプリングする．
        """
        strategy_index = self.sample_strategy_index_in_group(group_index, rng)
        return self.strategies[strategy_index]

    def sample_matchup(
        self,
        focal_group: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        """
        指定した起点グループ focal_group に対して，
        対戦相手グループをネットワークから選び，
        さらに両グループ内部から戦略を1つずつサンプリングする．

        Parameters
        ----------
        focal_group : int
            起点グループのインデックス
        rng : random.Random
            乱数生成器

        Returns
        -------
        tuple[int, int]
            (focal_strategy_index, opponent_strategy_index)

        Notes
        -----
        グループの選択と戦略の選択を分離することで，
        「対戦構造」と「各グループ内のメタ環境」を明示的に区別している．
        """
        opponent_group = self.sample_opponent_group_index(focal_group, rng)

        focal_strategy_index = self.sample_strategy_index_in_group(focal_group, rng)
        opponent_strategy_index = self.sample_strategy_index_in_group(opponent_group, rng)

        return focal_strategy_index, opponent_strategy_index

    def update_group_probs(
        self,
        fitness: NDArray[np.float64],
        eta: float,
    ) -> None:
        """
        各グループ内部の戦略分布を，適応度に基づいて更新する．

        Parameters
        ----------
        fitness : NDArray[np.float64]
            各グループ・各戦略に対する適応度
            shape = (num_groups, num_strategies)
            fitness[g, k] = グループ g における戦略 k の適応度
        eta : float
            選択圧を表すパラメータ
            大きいほど高適応度戦略が急速に増加する
        """
        if fitness.shape != (self.num_groups, self.num_strategies):
            raise ValueError(
                "fitness must have shape "
                f"({self.num_groups}, {self.num_strategies}), got {fitness.shape}"
            )

        if eta < 0:
            raise ValueError("eta must be non-negative")

        new_group_probs = np.zeros_like(self.group_probs)

        for g in range(self.num_groups):
            # 数値安定性のため最大値を引く
            shifted = fitness[g] - np.max(fitness[g])
            updated = self.group_probs[g] * np.exp(eta * shifted)

            total = float(np.sum(updated))
            if total <= 0:
                raise RuntimeError(
                    f"Updated probabilities in group {g} sum to zero"
                )

            new_group_probs[g] = updated / total

        self.group_probs = new_group_probs

    def dominant_strategy_index(self, group_index: int) -> int:
        """
        指定グループ内で最も採用率の高い戦略のインデックスを返す．
        """
        if not (0 <= group_index < self.num_groups):
            raise IndexError(f"group_index out of range: {group_index}")

        return int(np.argmax(self.group_probs[group_index]))

    def copy(self) -> "MetaNetwork":
        """
        MetaNetwork のコピーを返す．
        """
        return MetaNetwork(
            strategies=list(self.strategies),
            group_probs=self.group_probs.copy(),
            adjacency=self.adjacency.copy(),
            group_names=list(self.group_names) if self.group_names is not None else None,
        )