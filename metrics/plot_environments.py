#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch


AGENT_COLOR = "#1f77b4"
AGENT_EDGE = "navy"
OBSTACLE_FACE = "lightcoral"
OBSTACLE_EDGE = "darkred"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_title_from_path(path: str, mode: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]

    if mode == "config":
        if "_" in name:
            return name.split("_")[-1]
        return name

    if "_" in name:
        env_base = name.split("_")[0]
    else:
        env_base = name

    env_number = env_base.replace("env", "")
    return f"Environment {env_number}"


def draw_environment(ax, cfg: dict, title: str):
    world = cfg.get("world", {})

    screen_w = float(world.get("screen_w", 1200))
    screen_h = float(world.get("screen_h", 1200))

    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.grid(True, alpha=0.25)

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
                    linewidth=1.5,
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
                    linewidth=1.5,
                    alpha=0.85,
                )
            )

    for agent in cfg.get("agents", []):
        radius = float(agent.get("radius", 25))
        sx, sy = agent["start"]

        ax.add_patch(
            Circle(
                (sx, sy),
                radius,
                facecolor=AGENT_COLOR,
                edgecolor=AGENT_EDGE,
                linewidth=1.2,
                alpha=0.9,
            )
        )


def add_legend(fig):
    legend_elements = [
        Patch(
            facecolor=AGENT_COLOR,
            edgecolor=AGENT_EDGE,
            label="Agents / robots",
            alpha=0.9,
        ),
        Patch(
            facecolor=OBSTACLE_FACE,
            edgecolor=OBSTACLE_EDGE,
            label="Obstacles / static humans",
            alpha=0.85,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.015),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        required=True,
        help="Lista de YAMLs de entornos"
    )
    parser.add_argument(
        "--output",
        default="results/plots/benchmark_environments.png",
        help="PNG de salida"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Número de columnas de la figura"
    )
    parser.add_argument(
        "--title",
        default="Benchmark Environments for Experimental Evaluation",
        help="Título principal de la figura"
    )
    parser.add_argument(
        "--title_mode",
        choices=["environment", "config"],
        default="environment",
        help="environment: Environment 1, Environment 2... | config: 01, 02..."
    )

    args = parser.parse_args()

    env_paths = args.envs
    num_envs = len(env_paths)

    cols = args.cols
    rows = math.ceil(num_envs / cols)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 5 * rows)
    )

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, env_path in zip(axes, env_paths):
        cfg = load_yaml(env_path)
        title = env_title_from_path(env_path, args.title_mode)

        draw_environment(
            ax=ax,
            cfg=cfg,
            title=title
        )

    for ax in axes[num_envs:]:
        ax.axis("off")

    fig.suptitle(
        args.title,
        fontsize=18
    )

    add_legend(fig)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figura de entornos generada: {args.output}")


if __name__ == "__main__":
    main()