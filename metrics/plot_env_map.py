#!/usr/bin/env python3
from __future__ import annotations

import os
import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_environment(cfg: dict, output: str, show_agents: bool = False):
    world = cfg.get("world", {})

    screen_w = float(world.get("screen_w", 1200))
    screen_h = float(world.get("screen_h", 1200))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Environment Layout")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    # Obstáculos
    for obs in cfg.get("obstacles", []):
        if obs["type"] == "rectangle":
            cx, cy = obs["center"]
            w, h = obs["size"]

            rect = Rectangle(
                (cx - w / 2.0, cy - h / 2.0),
                w,
                h,
                facecolor="lightcoral",
                edgecolor="darkred",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

        elif obs["type"] == "circle":
            x, y = obs["position"]
            r = obs["radius"]

            circ = Circle(
                (x, y),
                r,
                facecolor="lightcoral",
                edgecolor="darkred",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(circ)

    # Agentes opcionales
    if show_agents:
        for agent in cfg.get("agents", []):
            color = [c / 255.0 for c in agent.get("color", [50, 50, 220])]
            radius = float(agent.get("radius", 25))

            sx, sy = agent["start"]
            gx, gy = agent["goal"]

            ax.add_patch(
                Circle(
                    (sx, sy),
                    radius,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=2,
                )
            )

            ax.plot(
                [sx, gx],
                [sy, gy],
                linestyle="--",
                linewidth=1.5,
                color=color,
                alpha=0.8,
            )

            ax.scatter([gx], [gy], marker="x", color=color, s=80)

            ax.text(sx + 10, sy - 10, agent["name"], color=color, fontsize=9)

    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(output), exist_ok=True)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Mapa generado: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="YAML del entorno")
    parser.add_argument(
        "--output",
        default="results/plots/environment_map.png",
        help="PNG de salida"
    )
    parser.add_argument(
        "--show_agents",
        action="store_true",
        help="Dibuja agentes, goals y trayectorias iniciales"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.env)
    plot_environment(
        cfg=cfg,
        output=args.output,
        show_agents=args.show_agents,
    )


if __name__ == "__main__":
    main()