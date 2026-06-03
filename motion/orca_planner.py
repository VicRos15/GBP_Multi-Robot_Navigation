#!/usr/bin/env python3
from __future__ import annotations

import yaml
import numpy as np
import rvo2


class ORCAAgent:
    def __init__(self, name, agent_id, start, goal, radius, color):
        self.name = name
        self.id = agent_id
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.radius = float(radius)
        self.color = tuple(color)

        self.reached = False
        self.trail = [tuple(start)]

    @property
    def r(self):
        return self.radius


class ORCAPlanner:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = self._load_yaml(yaml_path)

        world = self.cfg.get("world", {})
        self.fps = int(world.get("fps", 60))
        self.dt = 1.0 / self.fps
        self.time = 0.0

        # ORCA+
        self.max_speed = 80.0
        self.goal_tolerance = 20.0
        self.slowdown_dist = 180.0
        self.lookahead_dist = 220.0
        self.alpha_smooth = 0.65

        self.neighbor_dist = 350.0
        self.max_neighbors = 15
        self.time_horizon = 10.0
        self.time_horizon_obst = 8.0

        self.sim = rvo2.PyRVOSimulator(
            self.dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            25.0,
            self.max_speed
        )

        self.agents: list[ORCAAgent] = []
        self.obstacles = []

        self._build_obstacles()
        self.sim.processObstacles()
        self._build_agents()

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_agents(self):
        for a in self.cfg.get("agents", []):
            name = a["name"]
            start = a["start"]
            goal = a["goal"]
            radius = float(a["radius"])
            color = tuple(a.get("color", [0, 0, 220]))

            agent_id = self.sim.addAgent(
                (float(start[0]), float(start[1])),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                radius,
                self.max_speed,
                (0.0, 0.0)
            )

            self.agents.append(
                ORCAAgent(name, agent_id, start, goal, radius, color)
            )

    def _build_obstacles(self):
        for obs in self.cfg.get("obstacles", []):
            if obs["type"] == "rectangle":
                cx, cy = obs["center"]
                w, h = obs["size"]

                x0 = cx - w / 2.0
                x1 = cx + w / 2.0
                y0 = cy - h / 2.0
                y1 = cy + h / 2.0

                self.obstacles.append({
                    "type": "rectangle",
                    "name": obs["name"],
                    "x_min": float(x0),
                    "x_max": float(x1),
                    "y_min": float(y0),
                    "y_max": float(y1),
                })

                self.sim.addObstacle([
                    (x0, y0),
                    (x1, y0),
                    (x1, y1),
                    (x0, y1),
                ])

            elif obs["type"] == "circle":
                cx, cy = obs["position"]
                r = obs["radius"]

                self.obstacles.append({
                    "type": "circle",
                    "name": obs["name"],
                    "cx": float(cx),
                    "cy": float(cy),
                    "r": float(r),
                })

                vertices = []
                n = 16

                for i in range(n):
                    theta = 2.0 * np.pi * i / n
                    vertices.append((
                        cx + r * np.cos(theta),
                        cy + r * np.sin(theta)
                    ))

                self.sim.addObstacle(vertices)

            else:
                raise ValueError(f"Tipo de obstáculo no soportado: {obs['type']}")

    def _point_to_rect_distance(self, x, y, obs):
        dx = max(obs["x_min"] - x, 0.0, x - obs["x_max"])
        dy = max(obs["y_min"] - y, 0.0, y - obs["y_max"])
        return float(np.hypot(dx, dy))

    def _distance_to_obstacle(self, x, y, agent_radius):
        if not self.obstacles:
            return float("inf")

        min_clearance = float("inf")

        for obs in self.obstacles:
            if obs["type"] == "circle":
                d = np.hypot(x - obs["cx"], y - obs["cy"]) - obs["r"]
            else:
                d = self._point_to_rect_distance(x, y, obs)

            clearance = d - agent_radius
            min_clearance = min(min_clearance, clearance)

        return min_clearance

    def _compute_pref_velocity(self, agent: ORCAAgent):
        pos = np.array(self.sim.getAgentPosition(agent.id), dtype=float)
        vel = np.array(self.sim.getAgentVelocity(agent.id), dtype=float)

        to_goal = agent.goal - pos
        dist = np.linalg.norm(to_goal)

        if dist <= self.goal_tolerance:
            agent.reached = True
            return (0.0, 0.0)

        direction = to_goal / max(dist, 1e-6)

        lookahead = min(dist, self.lookahead_dist)
        local_target = pos + direction * lookahead

        local_dir = local_target - pos
        local_dist = np.linalg.norm(local_dir)

        if local_dist > 1e-6:
            local_dir = local_dir / local_dist
        else:
            local_dir = direction

        speed_scale = min(1.0, dist / self.slowdown_dist)
        desired_speed = self.max_speed * speed_scale

        # ORCA+ extra: reducción preventiva cerca de obstáculos.
        d_obs = self._distance_to_obstacle(pos[0], pos[1], agent.radius)

        if d_obs < 60.0:
            safety_scale = max(0.25, d_obs / 60.0)
            desired_speed *= safety_scale

        pref_vel = local_dir * desired_speed

        # ORCA+ extra: suavizado temporal de la velocidad preferida.
        smooth_vel = (
            self.alpha_smooth * pref_vel +
            (1.0 - self.alpha_smooth) * vel
        )

        return tuple(smooth_vel)

    def step(self):
        for agent in self.agents:
            pref_vel = self._compute_pref_velocity(agent)
            self.sim.setAgentPrefVelocity(agent.id, pref_vel)

        self.sim.doStep()
        self.time += self.dt

        for agent in self.agents:
            pos = np.array(self.sim.getAgentPosition(agent.id), dtype=float)

            agent.trail.append(tuple(pos))
            if len(agent.trail) > 5000:
                agent.trail.pop(0)

            if not agent.reached:
                if np.linalg.norm(agent.goal - pos) <= self.goal_tolerance:
                    agent.reached = True

    def all_reached(self):
        return all(agent.reached for agent in self.agents)

    def get_agent_position(self, agent: ORCAAgent):
        return self.sim.getAgentPosition(agent.id)

    def get_agent_velocity(self, agent: ORCAAgent):
        return self.sim.getAgentVelocity(agent.id)