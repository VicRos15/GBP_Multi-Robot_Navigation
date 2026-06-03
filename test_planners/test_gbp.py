#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import pygame as pg
import pygame.locals as pgl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from motion.obstacle import ObstacleMap
from motion.agent_FSM import Agent, Env
import time
from metrics.metrics_logger import MetricsLogger
import subprocess


def pairwise(seq):
    return zip(seq[:-1], seq[1:])


class GBPPlanner:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = self._load_yaml(yaml_path)

        self.world_cfg = self.cfg.get("world", {})
        self.px_per_cm = float(self.world_cfg.get("px_per_cm", 1.2))
        self.screen_w = int(self.world_cfg.get("screen_w", 1920))
        self.screen_h = int(self.world_cfg.get("screen_h", 1080))
        self.bg_color = tuple(self.world_cfg.get("bg_color", [245, 245, 245]))
        self.fps = int(self.world_cfg.get("fps", 60))
        self.steps = int(self.world_cfg.get("steps", 6))
        self.step_plan_iters = int(self.world_cfg.get("step_plan_iters", 12))

        self.omap = ObstacleMap()
        self.env = Env()
        self.agent_colors = {}
        self.agent_real_radius = {}
        self.agent_goals = {}
        self.t_sim = 0.0
        self.dt = 1.0 / self.fps
        self.max_time = 120.0
        self.goal_tolerance = 20.0
        self.auto_stop_when_done = True
        self.finish_grace_time = 0.5
        self.all_reached_since = None

        self.real_t0 = None
        self.reached = {}
        self.trails = {}

        # Obstáculos dinámicos
        self.dynamic_obstacles = {}
        self.sim_time = 0.0

        self._build_obstacles()
        self._build_agents()

        ## Add logger for comparison
        env_name = os.path.splitext(os.path.basename(yaml_path))[0]
        self.logger = MetricsLogger(
            algorithm="gbp",
            env_name=env_name,
            run_id=f"{env_name}_gbp",
            results_dir="results"
        )
        for agent in self.env._agents:
            self.reached[agent.name] = False

            st = agent.get_state()[0]
            if st is not None:
                self.trails[agent.name] = [(float(st[0]), float(st[1]))]
            else:
                self.trails[agent.name] = []

        pg.init()
        self.screen = pg.display.set_mode((self.screen_w, self.screen_h))
        pg.display.set_caption("GBP Planner")
        self.clock = pg.time.Clock()

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def w2s(self, x, y):
        return int(x * self.px_per_cm), int(y * self.px_per_cm)

    def _generate_straight_path(self, start, goal, num_points=50):
        x0, y0, _, _ = start
        x1, y1, _, _ = goal
        path = []

        for t in np.linspace(0.0, 1.0, num_points):
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            path.append((x, y))

        return path

    def _generate_curved_path(self, start, goal, num_points=50, arc_height=80):
        x0, y0, _, _ = start
        x1, y1, _, _ = goal

        cx = (x0 + x1) / 2.0
        cy = min(y0, y1) - arc_height

        path = []
        for t in np.linspace(0.0, 1.0, num_points):
            x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
            y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
            path.append((x, y))

        return path

    def _build_path(self, start, goal, path_cfg: dict):
        path_type = path_cfg.get("type", "straight")
        num_points = int(path_cfg.get("num_points", 50))

        if path_type == "straight":
            return self._generate_straight_path(start, goal, num_points=num_points)

        if path_type == "curved":
            arc_height = float(path_cfg.get("arc_height", 80))
            return self._generate_curved_path(
                start,
                goal,
                num_points=num_points,
                arc_height=arc_height
            )

        raise ValueError(f"Tipo de path no soportado: {path_type}")

    def _build_obstacles(self):
        for obs in self.cfg.get("obstacles", []):
            name = obs["name"]
            otype = obs["type"]

            if otype == "circle":
                x, y = obs["position"]
                r = obs["radius"]

                self.omap.set_circle(name, x, y, r)

                if "dynamic" in obs:
                    self.dynamic_obstacles[name] = {
                        "type": "circle",
                        "initial_position": [float(x), float(y)],
                        "radius": float(r),
                        "dynamic": obs["dynamic"],
                    }

            elif otype == "rectangle":
                cx, cy = obs["center"]
                w, h = obs["size"]

                x_min = cx - w / 2.0
                x_max = cx + w / 2.0
                y_min = cy - h / 2.0
                y_max = cy + h / 2.0

                self.omap.set_rectangle(name, x_min, y_min, x_max, y_max)

                if "dynamic" in obs:
                    self.dynamic_obstacles[name] = {
                        "type": "rectangle",
                        "initial_center": [float(cx), float(cy)],
                        "size": [float(w), float(h)],
                        "dynamic": obs["dynamic"],
                    }

            else:
                raise ValueError(f"Tipo de obstáculo no soportado: {otype}")

    def _build_agents(self):
        for a in self.cfg.get("agents", []):
            name = a["name"]
            radius = float(a["radius"])
            color = tuple(a.get("color", [0, 0, 220]))

            start_xy = a["start"]
            goal_xy = a["goal"]

            start = [start_xy[0], start_xy[1], 0.0, 0.0]
            goal = [goal_xy[0], goal_xy[1], 0.0, 0.0]

            path_cfg = a.get("path", {"type": "straight", "num_points": 50})
            path = self._build_path(start, goal, path_cfg)

            security_radius = radius * 1.2

            agent = Agent(
                name=name,
                state=start,
                target=goal,
                steps=self.steps,
                radius=security_radius,
                omap=self.omap,
                path=path
            )

            self.env.add_agent(agent)
            self.agent_colors[name] = color
            self.agent_real_radius[name] = radius
            self.agent_goals[name] = goal_xy

    # ============================================================
    # OBSTÁCULOS DINÁMICOS
    # ============================================================
    def _update_dynamic_obstacles(self):
        dt = 1.0 / self.fps
        self.sim_time += dt

        for name, dob in self.dynamic_obstacles.items():
            dyn = dob["dynamic"]
            motion_type = dyn.get("type", "linear")

            if motion_type == "linear":
                self._update_linear_obstacle(name, dob)

            elif motion_type == "pingpong":
                self._update_pingpong_obstacle(name, dob)

            else:
                raise ValueError(f"Movimiento dinámico no soportado: {motion_type}")

    def _update_linear_obstacle(self, name: str, dob: dict):
        dyn = dob["dynamic"]
        vx, vy = dyn.get("velocity", [0.0, 0.0])

        if dob["type"] == "circle":
            x0, y0 = dob["initial_position"]
            x = x0 + vx * self.sim_time
            y = y0 + vy * self.sim_time

            o = self.omap.objects[name]
            o["centerx"] = x
            o["centery"] = y

        elif dob["type"] == "rectangle":
            cx0, cy0 = dob["initial_center"]
            w, h = dob["size"]

            cx = cx0 + vx * self.sim_time
            cy = cy0 + vy * self.sim_time

            o = self.omap.objects[name]
            o["x_min"] = cx - w / 2.0
            o["x_max"] = cx + w / 2.0
            o["y_min"] = cy - h / 2.0
            o["y_max"] = cy + h / 2.0

    def _update_pingpong_obstacle(self, name: str, dob: dict):
        dyn = dob["dynamic"]

        axis = dyn.get("axis", "x")
        amplitude = float(dyn.get("amplitude", 100.0))
        speed = float(dyn.get("speed", 30.0))

        # Movimiento sinusoidal suave:
        # amplitude en cm, speed en cm/s aproximadamente
        omega = speed / max(amplitude, 1e-6)
        offset = amplitude * np.sin(omega * self.sim_time)

        if dob["type"] == "circle":
            x0, y0 = dob["initial_position"]

            x = x0 + offset if axis == "x" else x0
            y = y0 + offset if axis == "y" else y0

            o = self.omap.objects[name]
            o["centerx"] = x
            o["centery"] = y

        elif dob["type"] == "rectangle":
            cx0, cy0 = dob["initial_center"]
            w, h = dob["size"]

            cx = cx0 + offset if axis == "x" else cx0
            cy = cy0 + offset if axis == "y" else cy0

            o = self.omap.objects[name]
            o["x_min"] = cx - w / 2.0
            o["x_max"] = cx + w / 2.0
            o["y_min"] = cy - h / 2.0
            o["y_max"] = cy + h / 2.0

    # ============================================================
    # LOGGER COMPARISON
    # ============================================================
    def _log_current_state(self):
        t_real = time.perf_counter() - self.real_t0

        for agent in self.env._agents:
            state = agent.get_state()
            if not state or state[0] is None:
                continue

            x, y, vx, vy = state[0]
            self.trails[agent.name].append((float(x), float(y)))

            if len(self.trails[agent.name]) > 5000:
                self.trails[agent.name].pop(0)

            goal = np.array(self.agent_goals[agent.name], dtype=float)
            pos = np.array([x, y], dtype=float)

            if not self.reached[agent.name]:
                if np.linalg.norm(goal - pos) <= self.goal_tolerance:
                    self.reached[agent.name] = True

            self.logger.log_state(
                t_sim=self.t_sim,
                t_real=t_real,
                agent=agent.name,
                x=float(x),
                y=float(y),
                vx=float(vx),
                vy=float(vy),
                reached=self.reached[agent.name],
            )

    # ============================================================
    # DRAW
    # ============================================================
    def _draw_obstacles(self):
        obstacle_color = (180, 60, 60)
        dynamic_color = (220, 120, 40)

        for name, o in self.omap.objects.items():
            color = dynamic_color if name in self.dynamic_obstacles else obstacle_color

            if o["type"] == "rectangle":
                x0, y0 = self.w2s(o["x_min"], o["y_min"])
                w = int((o["x_max"] - o["x_min"]) * self.px_per_cm)
                h = int((o["y_max"] - o["y_min"]) * self.px_per_cm)

                pg.draw.rect(self.screen, color, (x0, y0, w, h), 0)
                pg.draw.rect(self.screen, color, (x0, y0, w, h), 2)

            elif o["type"] == "circle":
                sx, sy = self.w2s(o["centerx"], o["centery"])
                rr = int(o["radius"] * self.px_per_cm)

                pg.draw.circle(self.screen, color, (sx, sy), rr, 0)
                pg.draw.circle(self.screen, color, (sx, sy), rr, 2)

    def _draw_trails(self):
        for name, trail in self.trails.items():
            if len(trail) < 2:
                continue

            color = self.agent_colors.get(name, (0, 0, 220))
            pts = [self.w2s(x, y) for x, y in trail]

            pg.draw.lines(self.screen, color, False, pts, 2)

    def _draw_agents_and_paths(self):
        for agent in self.env._agents:
            state = agent.get_state()

            if not state or state[0] is None:
                continue

            color = self.agent_colors.get(agent.name, (0, 0, 220))

            x, y, _, _ = state[0]
            sx, sy = self.w2s(x, y)
            real_radius = self.agent_real_radius.get(agent.name, agent.r)
            security_radius = agent.r

            # Radio de seguridad: círculo exterior fino
            pg.draw.circle(
                self.screen,
                color,
                (sx, sy),
                int(security_radius * self.px_per_cm),
                1
            )

            # Radio real: círculo interior más marcado
            pg.draw.circle(
                self.screen,
                color,
                (sx, sy),
                int(real_radius * self.px_per_cm),
                3
            )

            for s0, s1 in pairwise(state):
                if s0 is None or s1 is None:
                    break

                x0, y0, _, _ = s0
                x1, y1, _, _ = s1

                pg.draw.line(
                    self.screen,
                    color,
                    self.w2s(x0, y0),
                    self.w2s(x1, y1),
                    2
                )

    def _draw_goals(self):
        for name, goal in self.agent_goals.items():
            color = self.agent_colors.get(name, (0, 0, 0))

            gx, gy = goal
            sx, sy = self.w2s(gx, gy)

            size = 6
            pg.draw.line(self.screen, color, (sx - size, sy), (sx + size, sy), 2)
            pg.draw.line(self.screen, color, (sx, sy - size), (sx, sy + size), 2)

    def all_reached(self) -> bool:

        if not self.reached:
            return False

        return all(self.reached.values())
    
    def run(self):
        running = True
        self.real_t0 = time.perf_counter()

        while running:

            # =====================================
            # Events
            # =====================================
            for event in pg.event.get():

                if event.type == pgl.QUIT:
                    running = False

                elif event.type == pgl.KEYDOWN:

                    if event.key == pgl.K_ESCAPE:
                        running = False

            # =====================================
            # Timeout safety
            # =====================================
            if self.t_sim >= self.max_time:
                running = False

            # =====================================
            # Planner update
            # =====================================
            self._update_dynamic_obstacles()

            self.env.step_plan(
                iters=self.step_plan_iters
            )

            # =====================================
            # Draw
            # =====================================
            self.screen.fill(self.bg_color)

            self._draw_obstacles()
            self._draw_goals()
            self._draw_trails()
            self._draw_agents_and_paths()

            # =====================================
            # Move simulation
            # =====================================
            self.env.step_move()

            self.t_sim += self.dt

            # Actualiza reached[]
            self._log_current_state()

            # =====================================
            # Auto stop when goals reached
            # =====================================
            if self.auto_stop_when_done:

                if self.all_reached():

                    if self.all_reached_since is None:

                        self.all_reached_since = self.t_sim

                    elif (
                        self.t_sim
                        - self.all_reached_since
                        >= self.finish_grace_time
                    ):

                        print(
                            f"All agents reached goals "
                            f"(t={self.t_sim:.2f}s)"
                        )

                        running = False

                else:

                    self.all_reached_since = None

            pg.display.flip()

            self.clock.tick(self.fps)

        # =====================================
        # Save metrics
        # =====================================
        self.logger.close()

        trajectory_path = self.logger.trajectory_path

        subprocess.run([
            sys.executable,
            os.path.join(
                ROOT,
                "metrics",
                "generate_metrics.py"
            ),
            "--trajectory",
            trajectory_path,
            "--env",
            self.yaml_path,
        ])

        pg.quit()

        sys.exit()



def main():
    if len(sys.argv) < 2:
        print("Uso: python gbp_planner.py /ruta/al/env.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    planner = GBPPlanner(yaml_path)
    planner.run()


if __name__ == "__main__":
    main()