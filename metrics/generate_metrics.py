#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import yaml
import math
import argparse
import numpy as np
from collections import defaultdict


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trajectory(path: str):
    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append({
                "algorithm": r["algorithm"],
                "env": r["env"],
                "env_type": r.get("env_type", r["env"].rsplit("_", 1)[0]),
                "test_id": r.get("test_id", r["env"].rsplit("_", 1)[-1]),
                "scenario_id": r.get("scenario_id", r["env"]),
                "run_id": r["run_id"],
                "t_sim": float(r["t_sim"]),
                "t_real": float(r["t_real"]),
                "agent": r["agent"],
                "x": float(r["x"]),
                "y": float(r["y"]),
                "vx": float(r["vx"]),
                "vy": float(r["vy"]),
                "reached": str(r["reached"]).lower() == "true",
            })

    return rows


def point_to_rect_distance(x, y, rect):
    dx = max(rect["x_min"] - x, 0.0, x - rect["x_max"])
    dy = max(rect["y_min"] - y, 0.0, y - rect["y_max"])

    return math.hypot(dx, dy)


def build_agent_radii(cfg):
    return {
        a["name"]: float(a["radius"])
        for a in cfg.get("agents", [])
    }


def build_obstacles(cfg):
    obstacles = []

    for obs in cfg.get("obstacles", []):
        if obs["type"] == "circle":
            x, y = obs["position"]

            obstacles.append({
                "type": "circle",
                "name": obs["name"],
                "cx": float(x),
                "cy": float(y),
                "r": float(obs["radius"]),
            })

        elif obs["type"] == "rectangle":
            cx, cy = obs["center"]
            w, h = obs["size"]

            obstacles.append({
                "type": "rectangle",
                "name": obs["name"],
                "x_min": float(cx - w / 2.0),
                "x_max": float(cx + w / 2.0),
                "y_min": float(cy - h / 2.0),
                "y_max": float(cy + h / 2.0),
            })

    return obstacles


def distance_to_obstacles(x, y, agent_radius, obstacles):
    if not obstacles:
        return float("inf")

    min_clearance = float("inf")

    for obs in obstacles:
        if obs["type"] == "circle":
            d = math.hypot(x - obs["cx"], y - obs["cy"]) - obs["r"]

        elif obs["type"] == "rectangle":
            d = point_to_rect_distance(x, y, obs)

        else:
            continue

        clearance = d - agent_radius
        min_clearance = min(min_clearance, clearance)

    return min_clearance


def compute_ldj(rows):
    """
    LDJ simplificado a partir de la trayectoria temporal.

    Menor suele indicar movimiento dinámicamente más suave.
    """
    velocities = []
    accelerations = []
    jerks = []

    prev_t = None

    for r in rows:
        vel = np.array([r["vx"], r["vy"]], dtype=float)
        velocities.append(vel)

        if len(velocities) >= 2 and prev_t is not None:
            dt = r["t_sim"] - prev_t

            if dt > 1e-9:
                acc = (velocities[-1] - velocities[-2]) / dt
                accelerations.append(acc)

        if len(accelerations) >= 2 and prev_t is not None:
            dt = r["t_sim"] - prev_t

            if dt > 1e-9:
                jerk = (accelerations[-1] - accelerations[-2]) / dt
                jerks.append(jerk)

        prev_t = r["t_sim"]

    if not jerks:
        return 0.0

    jerk_norms_sq = [
        np.linalg.norm(j) ** 2
        for j in jerks
    ]

    mean_jerk = np.mean(jerk_norms_sq)

    return float(np.log(mean_jerk + 1e-9))


def compute_path_curvature(points):
    """
    Curvatura media geométrica del path.

    Menor = trayectoria geométricamente más suave.
    0 = línea recta perfecta.
    """
    if len(points) < 3:
        return 0.0

    total_turn = 0.0
    total_length = 0.0

    for i in range(1, len(points) - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)

        if l1 < 1e-9 or l2 < 1e-9:
            continue

        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])

        da = math.atan2(
            math.sin(a2 - a1),
            math.cos(a2 - a1)
        )

        total_turn += abs(da)
        total_length += l1

    if total_length < 1e-9:
        return 0.0

    return float(total_turn / total_length)


def compute_metrics(traj_rows, cfg):
    if not traj_rows:
        raise ValueError("Trajectory CSV vacío.")

    algorithm = traj_rows[0]["algorithm"]
    env_name = traj_rows[0]["env"]
    run_id = traj_rows[0]["run_id"]
    env_type = traj_rows[0]["env_type"]
    test_id = traj_rows[0]["test_id"]
    scenario_id = traj_rows[0]["scenario_id"]

    radii = build_agent_radii(cfg)
    obstacles = build_obstacles(cfg)

    by_agent = defaultdict(list)
    by_time = defaultdict(list)

    for r in traj_rows:
        by_agent[r["agent"]].append(r)
        by_time[r["t_sim"]].append(r)

    for agent in by_agent:
        by_agent[agent].sort(key=lambda r: r["t_sim"])

    total_distance_by_agent = {}
    arrival_time_by_agent = {}
    min_obstacle_by_agent = {}
    mean_speed_by_agent = {}
    max_speed_by_agent = {}
    smoothness_ldj_by_agent = {}
    path_curvature_by_agent = {}

    obstacle_clearances = []
    agent_clearances = []

    for agent, rows in by_agent.items():
        dist = 0.0
        speeds = []
        points = []

        arrival_time = None
        min_obs = float("inf")
        prev_pos = None

        radius = radii.get(agent, 0.0)

        for r in rows:
            pos = np.array([r["x"], r["y"]], dtype=float)
            vel = np.array([r["vx"], r["vy"]], dtype=float)

            points.append(pos)

            speed = float(np.linalg.norm(vel))
            speeds.append(speed)

            if prev_pos is not None:
                dist += float(np.linalg.norm(pos - prev_pos))

            obs_clearance = distance_to_obstacles(
                r["x"],
                r["y"],
                radius,
                obstacles
            )

            min_obs = min(min_obs, obs_clearance)

            if math.isfinite(obs_clearance):
                obstacle_clearances.append(obs_clearance)

            if r["reached"] and arrival_time is None:
                arrival_time = r["t_sim"]

            prev_pos = pos

        total_distance_by_agent[agent] = dist
        arrival_time_by_agent[agent] = arrival_time
        min_obstacle_by_agent[agent] = min_obs
        mean_speed_by_agent[agent] = float(np.mean(speeds)) if speeds else 0.0
        max_speed_by_agent[agent] = max(speeds) if speeds else 0.0
        smoothness_ldj_by_agent[agent] = compute_ldj(rows)
        path_curvature_by_agent[agent] = compute_path_curvature(points)

    min_agent_clearance = float("inf")

    for _, rows in by_time.items():
        rows = sorted(rows, key=lambda r: r["agent"])

        for i in range(len(rows)):
            ai = rows[i]
            ri = radii.get(ai["agent"], 0.0)

            for j in range(i + 1, len(rows)):
                aj = rows[j]
                rj = radii.get(aj["agent"], 0.0)

                d = math.hypot(ai["x"] - aj["x"], ai["y"] - aj["y"])
                clearance = d - (ri + rj)

                min_agent_clearance = min(min_agent_clearance, clearance)
                agent_clearances.append(clearance)

    arrival_values = [
        t for t in arrival_time_by_agent.values()
        if t is not None
    ]

    success = len(arrival_values) == len(by_agent)

    makespan_sim = max(arrival_values) if success else None
    real_time_total = max(r["t_real"] for r in traj_rows)

    total_distance = sum(total_distance_by_agent.values())

    mean_obstacle_clearance = (
        float(np.mean(obstacle_clearances))
        if obstacle_clearances else None
    )

    mean_agent_clearance = (
        float(np.mean(agent_clearances))
        if agent_clearances else None
    )

    all_clearances = obstacle_clearances + agent_clearances

    mean_safety_clearance = (
        float(np.mean(all_clearances))
        if all_clearances else None
    )

    min_dist_obstacle = (
        float(min(obstacle_clearances))
        if obstacle_clearances else None
    )

    min_dist_agent = (
        float(min(agent_clearances))
        if agent_clearances else None
    )

    summary = {
        "algorithm": algorithm,
        "env": env_name,
        "env_type": env_type,
        "test_id": test_id,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "success": success,
        "num_agents": len(by_agent),
        "makespan_sim": makespan_sim,
        "real_time_total": real_time_total,
        "total_distance": total_distance,
        "mean_distance": float(np.mean(list(total_distance_by_agent.values()))),
        "max_distance": max(total_distance_by_agent.values()),
        "min_dist_obstacle": min_dist_obstacle,
        "min_dist_agent": min_dist_agent,
        "mean_obstacle_clearance": mean_obstacle_clearance,
        "mean_agent_clearance": mean_agent_clearance,
        "mean_safety_clearance": mean_safety_clearance,
        "mean_speed": float(np.mean(list(mean_speed_by_agent.values()))),
        "max_speed": max(max_speed_by_agent.values()),
        "mean_smoothness_ldj": float(np.mean(list(smoothness_ldj_by_agent.values()))),
        "mean_path_curvature": float(np.mean(list(path_curvature_by_agent.values()))),
    }

    per_agent = {
        agent: {
            "distance": total_distance_by_agent[agent],
            "arrival_time": arrival_time_by_agent[agent],
            "min_dist_obstacle": min_obstacle_by_agent[agent],
            "mean_speed": mean_speed_by_agent[agent],
            "max_speed": max_speed_by_agent[agent],
            "smoothness_ldj": smoothness_ldj_by_agent[agent],
            "path_curvature": path_curvature_by_agent[agent],
        }
        for agent in by_agent.keys()
    }

    return summary, per_agent


def print_metrics(summary, per_agent):
    print("\n========== GENERATED METRICS ==========")
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Env:       {summary['env']}")
    print(f"Run:       {summary['run_id']}")
    print("--------------------------------------")

    for agent, m in per_agent.items():
        print(
            f"{agent}: "
            f"arrival={m['arrival_time']}, "
            f"distance={m['distance']:.2f}, "
            f"min_obs={m['min_dist_obstacle']:.2f}, "
            f"mean_speed={m['mean_speed']:.2f}, "
            f"smoothness_ldj={m['smoothness_ldj']:.2f}, "
            f"path_curvature={m['path_curvature']:.5f}"
        )

    print("--------------------------------------")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("======================================\n")


def write_summary(summary, output_csv):
    output_dir = os.path.dirname(output_csv)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.exists(output_csv)
    file_empty = (not file_exists) or os.path.getsize(output_csv) == 0

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))

        if file_empty:
            writer.writeheader()

        writer.writerow(summary)


def default_metrics_output_path(summary):
    algorithm = summary["algorithm"]
    env_type = summary["env_type"]

    return os.path.join(
        "results",
        "metrics",
        f"{env_type}_{algorithm}_metrics.csv"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, help="CSV de trayectoria")
    parser.add_argument("--env", required=True, help="YAML del entorno")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV resumen. Si no se indica, se usa results/metrics/{env}_{algorithm}_metrics.csv"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.env)
    rows = load_trajectory(args.trajectory)

    summary, per_agent = compute_metrics(rows, cfg)

    output_csv = args.output
    if output_csv is None:
        output_csv = default_metrics_output_path(summary)

    print_metrics(summary, per_agent)
    write_summary(summary, output_csv)


if __name__ == "__main__":
    main()