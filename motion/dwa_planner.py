#!/usr/bin/env python3
from __future__ import annotations

import math
import yaml
import numpy as np


class DWAAgent:
    def __init__(self, name, start_xy, goal_xy, radius, color):
        self.name = name
        self.x = float(start_xy[0])
        self.y = float(start_xy[1])
        self.goal = np.array(goal_xy, dtype=float)
        self.radius = float(radius)
        self.color = tuple(color)

        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        self.theta = math.atan2(dy, dx)

        self.v = 0.0
        self.w = 0.0
        self.trail = [(self.x, self.y)]
        self.reached = False

    def pos(self):
        return np.array([self.x, self.y], dtype=float)

    def velocity_xy(self):
        return np.array([
            self.v * math.cos(self.theta),
            self.v * math.sin(self.theta)
        ], dtype=float)


class DWAPlanner:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = self._load_yaml(yaml_path)

        world = self.cfg.get("world", {})
        self.fps = int(world.get("fps", 60))
        self.dt = 1.0 / self.fps
        self.time = 0.0

        self.agents = []
        self.obstacles = []

        self._build_obstacles()
        self._build_agents()

        # =========================
        # DWA SIMPLE + LIGHT PREDICTION
        # =========================
        self.max_v = 80.0
        self.max_w = 1.8

        self.v_samples = 7
        self.w_samples = 11

        self.horizon = 1.2
        self.sim_dt = 0.1

        self.goal_tolerance = 18.0

        # Costes
        self.w_goal = 1.0
        self.w_heading = 15.0
        self.w_obs = 35.0
        self.w_speed = 0.15

        self.static_influence_dist = 70.0
        self.dynamic_influence_dist = 110.0

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_agents(self):
        for a in self.cfg.get("agents", []):
            self.agents.append(
                DWAAgent(
                    name=a["name"],
                    start_xy=a["start"],
                    goal_xy=a["goal"],
                    radius=float(a["radius"]),
                    color=tuple(a.get("color", [0, 0, 220]))
                )
            )

    def _build_obstacles(self):
        for obs in self.cfg.get("obstacles", []):
            if obs["type"] == "circle":
                self.obstacles.append({
                    "type": "circle",
                    "name": obs["name"],
                    "cx": float(obs["position"][0]),
                    "cy": float(obs["position"][1]),
                    "r": float(obs["radius"]),
                })

            elif obs["type"] == "rectangle":
                cx, cy = obs["center"]
                w, h = obs["size"]

                self.obstacles.append({
                    "type": "rectangle",
                    "name": obs["name"],
                    "x_min": float(cx - w / 2.0),
                    "x_max": float(cx + w / 2.0),
                    "y_min": float(cy - h / 2.0),
                    "y_max": float(cy + h / 2.0),
                })

            else:
                raise ValueError(f"Tipo de obstáculo no soportado: {obs['type']}")

    # =========================
    # GEOMETRY
    # =========================
    def point_to_rect_distance(self, x, y, rect):
        dx = max(rect["x_min"] - x, 0.0, x - rect["x_max"])
        dy = max(rect["y_min"] - y, 0.0, y - rect["y_max"])
        return math.hypot(dx, dy)

    def static_clearance(self, x, y, agent_radius):
        min_clearance = float("inf")

        for obs in self.obstacles:
            if obs["type"] == "circle":
                d = math.hypot(x - obs["cx"], y - obs["cy"]) - obs["r"]
            else:
                d = self.point_to_rect_distance(x, y, obs)

            clearance = d - agent_radius
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    def dynamic_clearance(self, x, y, agent, dynamic_obstacles, t_future):
        min_clearance = float("inf")

        for other in dynamic_obstacles:
            ox = other["x"] + other["vx"] * t_future
            oy = other["y"] + other["vy"] * t_future

            d = math.hypot(x - ox, y - oy)
            clearance = d - (agent.radius + other["radius"])
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    # =========================
    # DWA CORE
    # =========================
    def simulate_trajectory(self, x, y, theta, v, w):
        traj = []
        t = 0.0

        while t < self.horizon:
            x += v * math.cos(theta) * self.sim_dt
            y += v * math.sin(theta) * self.sim_dt
            theta += w * self.sim_dt
            t += self.sim_dt

            traj.append((x, y, theta, t))

        return traj

    def compute_obstacle_cost(self, traj, agent, dynamic_obstacles):
        total_penalty = 0.0

        for x, y, _, t_future in traj:
            c_static = self.static_clearance(x, y, agent.radius)

            if c_static <= 0.0:
                return float("inf")

            if c_static < self.static_influence_dist:
                total_penalty += 1.0 / max(c_static, 1e-3)

            c_dyn = self.dynamic_clearance(
                x,
                y,
                agent,
                dynamic_obstacles,
                t_future
            )

            if c_dyn <= 0.0:
                return float("inf")

            if c_dyn < self.dynamic_influence_dist:
                total_penalty += 1.0 / max(c_dyn, 1e-3)

        return total_penalty

    def compute_goal_cost(self, traj, goal):
        x_end, y_end, _, _ = traj[-1]
        return math.hypot(goal[0] - x_end, goal[1] - y_end)

    def compute_heading_cost(self, traj, goal):
        x_end, y_end, theta_end, _ = traj[-1]

        desired = math.atan2(goal[1] - y_end, goal[0] - x_end)
        err = math.atan2(
            math.sin(desired - theta_end),
            math.cos(desired - theta_end)
        )

        return abs(err)

    def dwa_step(self, agent, dynamic_obstacles):
        if agent.reached:
            return 0.0, 0.0

        dist_to_goal = np.linalg.norm(agent.goal - agent.pos())

        if dist_to_goal <= self.goal_tolerance:
            agent.reached = True
            return 0.0, 0.0

        best_score = float("inf")
        best_v = 0.0
        best_w = 0.0

        for v in np.linspace(0.0, self.max_v, self.v_samples):
            for w in np.linspace(-self.max_w, self.max_w, self.w_samples):
                traj = self.simulate_trajectory(
                    agent.x,
                    agent.y,
                    agent.theta,
                    v,
                    w
                )

                obs_cost = self.compute_obstacle_cost(
                    traj,
                    agent,
                    dynamic_obstacles
                )

                if math.isinf(obs_cost):
                    continue

                goal_cost = self.compute_goal_cost(traj, agent.goal)
                heading_cost = self.compute_heading_cost(traj, agent.goal)
                speed_cost = self.max_v - v

                total_cost = (
                    self.w_goal * goal_cost +
                    self.w_heading * heading_cost +
                    self.w_obs * obs_cost +
                    self.w_speed * speed_cost
                )

                if total_cost < best_score:
                    best_score = total_cost
                    best_v = float(v)
                    best_w = float(w)

        if math.isinf(best_score):
            return 0.0, 0.0

        return best_v, best_w

    def step(self):
        controls = []

        snapshot = []
        for a in self.agents:
            vx, vy = a.velocity_xy()
            snapshot.append({
                "x": a.x,
                "y": a.y,
                "vx": vx,
                "vy": vy,
                "radius": a.radius,
                "name": a.name,
            })

        # Calcular controles con snapshot congelado
        for agent in self.agents:
            dynamic_obstacles = [
                o for o in snapshot if o["name"] != agent.name
            ]

            v_cmd, w_cmd = self.dwa_step(agent, dynamic_obstacles)
            controls.append((v_cmd, w_cmd))

        # Aplicar controles simultáneamente
        for agent, (v_cmd, w_cmd) in zip(self.agents, controls):
            if agent.reached:
                agent.v = 0.0
                agent.w = 0.0
                continue

            agent.v = v_cmd
            agent.w = w_cmd

            agent.x += agent.v * math.cos(agent.theta) * self.dt
            agent.y += agent.v * math.sin(agent.theta) * self.dt
            agent.theta += agent.w * self.dt

            agent.trail.append((agent.x, agent.y))

            if len(agent.trail) > 5000:
                agent.trail.pop(0)

            if np.linalg.norm(agent.goal - agent.pos()) <= self.goal_tolerance:
                agent.reached = True
                agent.v = 0.0
                agent.w = 0.0

        self.time += self.dt

    def all_reached(self):
        return all(a.reached for a in self.agents)

    def get_agent_position(self, agent: DWAAgent):
        return agent.x, agent.y

    def get_agent_velocity(self, agent: DWAAgent):
        return (
            agent.v * math.cos(agent.theta),
            agent.v * math.sin(agent.theta)
        )