#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


METRICS_TO_PLOT = [
    "makespan_sim",
    "real_time_total",
    "compute_ratio",
    "total_distance",
    "mean_distance",
    "min_dist_agent",
    "min_dist_obstacle",
    "num_agent_collisions",
    "num_obstacle_collisions",
    "mean_speed",
    "mean_smoothness_ldj",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument(
        "--output",
        default="results/plots/methods_comparison.png"
    )
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.comparison)

    metrics = [m for m in METRICS_TO_PLOT if m in df.columns]
    cols = args.cols
    rows = math.ceil(len(metrics) / cols)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_df = df[["algorithm", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])
        color_map = {
            "gbp": "blue",
            "orca_plus": "orange",
            "dwa": "green"
        }

        colors = [color_map.get(a, "gray") for a in plot_df["algorithm"]]
        ax.bar(plot_df["algorithm"], plot_df[metric], color=colors)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Methods comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"PNG generado: {args.output}")


if __name__ == "__main__":
    main()