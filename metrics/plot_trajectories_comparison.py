#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from matplotlib.lines import Line2D


AGENT_COLORS = {
    "a0": "#e41a1c",
    "a1": "#377eb8",
    "a2": "#4daf4a",
    "a3": "#ff7f00",
    "a4": "#984ea3",
    "a5": "#00bfc4",
    "a6": "#999999",
    "a7": "#a65628",
    "a8": "#f781bf",
    "a9": "#66c2a5",
}

OBSTACLE_FACE = "#555555"
OBSTACLE_EDGE = "#222222"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trajectory(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = ["algorithm", "t_sim", "agent", "x", "y"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en {path}")

    df["t_sim"] = pd.to_numeric(df["t_sim"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    return df.dropna(subset=["t_sim", "x", "y"])


def draw_obstacles(ax, cfg: dict):
    for obs in cfg.get("obstacles", []):
        if obs["type"] == "rectangle":
            cx, cy = obs["center"]
            w, h = obs["size"]

            ax.add_patch(
                Rectangle(
                    (cx - w / 2.0, cy - h / 2.0),
                    w,
                    h,
                    facecolor=OBSTACLE_FACE,
                    edgecolor=OBSTACLE_EDGE,
                    linewidth=1.2,
                    alpha=0.85,
                )
            )

        elif obs["type"] == "circle":
            x, y = obs["position"]
            r = obs["radius"]

            ax.add_patch(
                Circle(
                    (x, y),
                    r,
                    facecolor=OBSTACLE_FACE,
                    edgecolor=OBSTACLE_EDGE,
                    linewidth=1.2,
                    alpha=0.85,
                )
            )


def setup_axis(ax, cfg: dict):
    world = cfg.get("world", {})

    screen_w = float(world.get("screen_w", 1200))
    screen_h = float(world.get("screen_h", 1200))

    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.2)

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")


def get_agent_radius(cfg: dict, agent_name: str) -> float:
    for agent in cfg.get("agents", []):
        if agent.get("name") == agent_name:
            return float(agent.get("radius", 25.0))
    return 25.0


def get_agent_goal(cfg: dict, agent_name: str):
    for agent in cfg.get("agents", []):
        if agent.get("name") == agent_name:
            return agent.get("goal", None)
    return None


def get_agent_start(cfg: dict, agent_name: str):
    for agent in cfg.get("agents", []):
        if agent.get("name") == agent_name:
            return agent.get("start", None)
    return None


def nearest_rows_at_time(df: pd.DataFrame, target_time: float) -> pd.DataFrame:
    rows = []

    for _, group in df.groupby("agent"):
        group = group.copy()
        idx = (group["t_sim"] - target_time).abs().idxmin()
        rows.append(group.loc[idx])

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def draw_trajectory_panel(ax, cfg: dict, df: pd.DataFrame, snapshot_time: float):
    setup_axis(ax, cfg)
    draw_obstacles(ax, cfg)

    agents = sorted(df["agent"].dropna().unique())

    for agent in agents:
        color = AGENT_COLORS.get(agent, "#000000")

        group = (
            df[(df["agent"] == agent) & (df["t_sim"] <= snapshot_time)]
            .sort_values("t_sim")
        )

        if len(group) >= 2:
            ax.plot(
                group["x"].to_numpy(),
                group["y"].to_numpy(),
                color=color,
                linewidth=1.8,
                alpha=0.65,
            )

        start = get_agent_start(cfg, agent)
        goal = get_agent_goal(cfg, agent)

        if start is not None:
            ax.scatter(
                [start[0]],
                [start[1]],
                marker="o",
                s=35,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

        if goal is not None:
            ax.scatter(
                [goal[0]],
                [goal[1]],
                marker="x",
                s=55,
                color=color,
                linewidth=1.8,
                zorder=5,
            )

    snapshot = nearest_rows_at_time(df, snapshot_time)

    for _, row in snapshot.iterrows():
        agent = row["agent"]
        color = AGENT_COLORS.get(agent, "#000000")
        radius = get_agent_radius(cfg, agent)

        ax.add_patch(
            Circle(
                (float(row["x"]), float(row["y"])),
                radius,
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.85,
                zorder=6,
            )
        )


def get_snapshot_times(df: pd.DataFrame):
    t_min = float(df["t_sim"].min())
    t_max = float(df["t_sim"].max())
    t_mid = 0.5 * (t_min + t_max)

    return [t_min, t_mid, t_max]


def label_from_path(path: str) -> str:
    name = os.path.basename(path).lower()

    if "gbp" in name:
        return "GBP"
    if "orca" in name:
        return "ORCA+"
    if "dwa" in name:
        return "Predictive DWA"

    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="YAML del entorno")
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="CSV de trayectorias. Idealmente GBP, ORCA+ y DWA."
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Etiquetas de los algoritmos. Si no se indican, se deducen del nombre del fichero."
    )
    parser.add_argument(
        "--output",
        default="results/plots/trajectories_comparison.png",
        help="PNG de salida"
    )
    parser.add_argument(
        "--title",
        default="Trajectory Evolution Comparison",
        help="Título principal de la figura"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.env)

    trajectory_paths = args.trajectories
    num_algorithms = len(trajectory_paths)

    if args.labels is not None and len(args.labels) > 0:
        if len(args.labels) != num_algorithms:
            raise ValueError("El número de --labels debe coincidir con --trajectories.")
        labels = args.labels
    else:
        labels = [label_from_path(p) for p in trajectory_paths]

    dfs = [load_trajectory(path) for path in trajectory_paths]

    rows = num_algorithms
    cols = 3

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 5.0 * rows))

    gs = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[0.05, 1.0, 1.0, 1.0],
        wspace=0.20,
        hspace=0.35,
    )

    label_axes = []
    axes = []

    for r in range(rows):
        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis("off")
        label_axes.append(label_ax)

        row_axes = []
        for c in range(cols):
            row_axes.append(fig.add_subplot(gs[r, c + 1]))
        axes.append(row_axes)

    column_titles = ["Initial state", "Intermediate state", "Final state"]

    for row_idx, (df, label) in enumerate(zip(dfs, labels)):
        snapshot_times = get_snapshot_times(df)

        label_axes[row_idx].text(
            0.5,
            0.5,
            label,
            fontsize=14,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

        for col_idx, snapshot_time in enumerate(snapshot_times):
            ax = axes[row_idx][col_idx]

            draw_trajectory_panel(
                ax=ax,
                cfg=cfg,
                df=df,
                snapshot_time=snapshot_time,
            )

            if row_idx == 0:
                ax.set_title(
                    column_titles[col_idx],
                    fontsize=12,
                    fontweight="bold",
                    pad=10,
                )

    legend_elements = [
        Patch(
            facecolor=OBSTACLE_FACE,
            edgecolor=OBSTACLE_EDGE,
            label="Obstacles",
            alpha=0.85,
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            label="Agent trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            linestyle="None",
            markersize=7,
            label="Start position",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=8,
            label="Goal position",
        ),
        Patch(
            facecolor="#999999",
            edgecolor="black",
            label="Agent at snapshot",
            alpha=0.85,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.015),
    )

    fig.suptitle(args.title, fontsize=18)

    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )

    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Figura de trayectorias generada: {args.output}")


if __name__ == "__main__":
    main()