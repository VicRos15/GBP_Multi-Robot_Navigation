#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METRICS_TO_PLOT = [
    "makespan_sim",
    "real_time_total",
    "total_distance",
    "mean_distance",
    "max_distance",
    "min_dist_obstacle",
    "min_dist_agent",
    "mean_speed",
    "max_speed",
    "mean_smoothness_ldj",
    "mean_path_curvature",
]


METRIC_TITLES = {
    "makespan_sim": "Makespan [s]",
    "real_time_total": "Real Execution Time [s]",
    "total_distance": "Total Distance [cm]",
    "mean_distance": "Mean Distance [cm]",
    "max_distance": "Max Distance [cm]",
    "min_dist_obstacle": "Minimum Obstacle Clearance [cm]",
    "min_dist_agent": "Minimum Inter-Agent Clearance [cm]",
    "mean_speed": "Mean Speed [cm/s]",
    "max_speed": "Max Speed [cm/s]",
    "mean_smoothness_ldj": "Smoothness LDJ",
    "mean_path_curvature": "Path Curvature [rad/cm]",
}


ALGORITHM_LABELS = {
    "gbp": "GBP",
    "orca_plus": "ORCA+",
    "pred_dwa": "Predictive DWA",
}


COLOR_MAP = {
    "gbp": "#1f77b4",
    "orca_plus": "#ff7f0e",
    "pred_dwa": "#2ca02c",
}


def normalize_success_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "success" not in df.columns:
        df["success"] = False
        return df

    df["success"] = (
        df["success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        required=True,
        help="CSV con todas las runs, por ejemplo results/analysis/env1_all_runs.csv"
    )
    parser.add_argument(
        "--output",
        default="results/plots/env1_methods_boxplots.png"
    )
    parser.add_argument("--cols", type=int, default=3)

    args = parser.parse_args()

    df = pd.read_csv(args.comparison)
    df = normalize_success_column(df)

    df_success = df[df["success"] == True].copy()

    if df_success.empty:
        raise ValueError("No hay ejecuciones con success=True para representar.")

    metrics = [
        m for m in METRICS_TO_PLOT
        if m in df_success.columns
    ]

    if not metrics:
        raise ValueError("No hay métricas válidas para representar.")

    cols = args.cols
    rows = math.ceil((len(metrics) + 1) / cols)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    preferred_order = ["gbp", "orca_plus", "pred_dwa"]
    available_algorithms = list(df_success["algorithm"].dropna().unique())

    algorithms = [
        alg for alg in preferred_order
        if alg in available_algorithms
    ]

    for alg in available_algorithms:
        if alg not in algorithms:
            algorithms.append(alg)

    metric_axes = axes[:len(metrics)]

    for ax, metric in zip(metric_axes, metrics):
        for pos, algorithm in enumerate(algorithms, start=1):
            values = df_success[df_success["algorithm"] == algorithm][metric]
            values = pd.to_numeric(values, errors="coerce").dropna().tolist()

            if len(values) == 0:
                continue

            bp = ax.boxplot(
                values,
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(
                    marker="^",
                    markerfacecolor="green",
                    markeredgecolor="green",
                    markersize=8
                ),
                medianprops=dict(
                    color="black",
                    linewidth=2
                ),
                flierprops=dict(
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=5,
                    linestyle="none"
                )
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(COLOR_MAP.get(algorithm, "gray"))
                patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(algorithms) + 1))
        ax.set_xticklabels(
            [ALGORITHM_LABELS.get(a, a) for a in algorithms],
            rotation=25
        )

        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLOR_MAP["gbp"], alpha=0.75, label="GBP"),
        Patch(facecolor=COLOR_MAP["orca_plus"], alpha=0.75, label="ORCA+"),
        Patch(facecolor=COLOR_MAP["pred_dwa"], alpha=0.75, label="Predictive DWA"),
        Line2D([0], [0], color="black", lw=2, label="Median"),
        Line2D(
            [0], [0],
            marker="^",
            color="green",
            markerfacecolor="green",
            markersize=8,
            linestyle="None",
            label="Mean"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            markersize=5,
            linestyle="None",
            label="Outlier"
        ),
    ]

    legend_ax_index = len(metrics)

    if legend_ax_index < len(axes):
        legend_ax = axes[legend_ax_index]
        legend_ax.axis("off")

        legend_ax.legend(
            handles=legend_elements,
            loc="center",
            fontsize=11,
            frameon=True,
            title="Boxplot Legend",
            title_fontsize=12,
            ncol=1
        )

        for ax in axes[legend_ax_index + 1:]:
            ax.axis("off")

    if "env_type" in df_success.columns:
        env_name = str(df_success["env_type"].iloc[0]).lower()
    else:
        env_name = "ENVIRONMENT 1"

    fig.suptitle(
        f"Methods Comparison for {env_name}",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Boxplot generado: {args.output}")


if __name__ == "__main__":
    main()