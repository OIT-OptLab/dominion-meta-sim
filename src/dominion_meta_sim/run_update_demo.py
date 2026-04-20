from __future__ import annotations

import numpy as np

from dominion_meta_sim.agents.strategy import Strategy
from dominion_meta_sim.evolution.meta_network import MetaNetwork


def entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def l1_shift(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.sum(np.abs(after - before)))


def print_group_summary(meta: MetaNetwork, fitness: np.ndarray, before: np.ndarray) -> None:
    strategy_names = [s.name or f"strategy_{i}" for i, s in enumerate(meta.strategies)]

    print("=" * 72)
    print("Update result")
    print("=" * 72)

    for g in range(meta.num_groups):
        after = meta.group_probs[g]

        print(f"\nGroup {g}: {meta.group_names[g]}")
        print("-" * 72)
        print(f"{'Strategy':<20} {'Before':>10} {'Fitness':>10} {'After':>10} {'Delta':>10}")

        for k, name in enumerate(strategy_names):
            delta = after[k] - before[g, k]
            print(
                f"{name:<20} "
                f"{before[g, k]:>10.3f} "
                f"{fitness[g, k]:>10.3f} "
                f"{after[k]:>10.3f} "
                f"{delta:>10.3f}"
            )

        before_dom = int(np.argmax(before[g]))
        after_dom = int(np.argmax(after))

        print()
        print(
            f"Dominant: {strategy_names[before_dom]} -> {strategy_names[after_dom]}"
        )
        print(
            f"Entropy:  {entropy(before[g]):.3f} -> {entropy(after):.3f}"
        )
        print(
            f"Shift(L1): {l1_shift(before[g], after):.3f}"
        )


def main() -> None:
    # 代表戦略
    s1 = Strategy(np.array([0.0, 0.2, 1.0, 0.3, -0.5]), name="action_pref")
    s2 = Strategy(np.array([0.0, 0.5, 0.2, 1.0, -0.3]), name="treasure_pref")
    s3 = Strategy(np.array([0.0, 0.1, 0.1, 0.2, 0.8]), name="victory_pref")

    # 各グループの初期戦略分布
    group_probs = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ], dtype=np.float64)

    # グループ間対戦ネットワーク
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

    # 仮の fitness 行列
    # fitness[g, k] = グループ g における戦略 k の適応度
    fitness = np.array([
        [0.55, 0.40, 0.20],
        [0.35, 0.60, 0.30],
        [0.25, 0.45, 0.70],
    ], dtype=np.float64)

    before = meta.group_probs.copy()

    # 分布更新
    meta.update_group_probs(fitness=fitness, eta=2.0)

    # 結果表示
    print_group_summary(meta, fitness, before)


if __name__ == "__main__":
    main()