#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import yaml
import time
import math
import subprocess
import pygame as pg
import pygame.locals as pgl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from motion.dwa_planner import DWAPlanner
from metrics.metrics_logger import MetricsLogger


class PredDWATest:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = self._load_yaml(yaml_path)

        world = self.cfg.get("world", {})
        self.px_per_cm = float(world.get("px_per_cm", 1.2))
        self.screen_w = int(world.get("screen_w", 1920))
        self.screen_h = int(world.get("screen_h", 1080))
        self.bg_color = tuple(world.get("bg_color", [245, 245, 245]))
        self.fps = int(world.get("fps", 60))

        self.max_time = 120.0

        # Auto stop
        self.auto_stop_when_done = True
        self.finish_grace_time = 0.5
        self.all_reached_since = None

        self.real_t0 = None

        self.planner = DWAPlanner(yaml_path)

        env_name = os.path.splitext(os.path.basename(yaml_path))[0]

        self.logger = MetricsLogger(
            algorithm="pred_dwa",
            env_name=env_name,
            run_id=f"{env_name}_pred_dwa",
            results_dir="results"
        )

        pg.init()
        self.screen = pg.display.set_mode((self.screen_w, self.screen_h))
        pg.display.set_caption("Predictive DWA Planner")
        self.clock = pg.time.Clock()

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def w2s(self, x, y):
        return int(x * self.px_per_cm), int(y * self.px_per_cm)

    def _draw_obstacles(self):
        color = (180, 60, 60)

        for obs in self.planner.obstacles:
            if obs["type"] == "rectangle":
                x0, y0 = self.w2s(obs["x_min"], obs["y_min"])
                w_px = int((obs["x_max"] - obs["x_min"]) * self.px_per_cm)
                h_px = int((obs["y_max"] - obs["y_min"]) * self.px_per_cm)

                pg.draw.rect(self.screen, color, (x0, y0, w_px, h_px), 0)
                pg.draw.rect(self.screen, color, (x0, y0, w_px, h_px), 2)

            elif obs["type"] == "circle":
                sx, sy = self.w2s(obs["cx"], obs["cy"])
                rr = int(obs["r"] * self.px_per_cm)

                pg.draw.circle(self.screen, color, (sx, sy), rr, 0)
                pg.draw.circle(self.screen, color, (sx, sy), rr, 2)

    def _draw_goals(self):
        for agent in self.planner.agents:
            sx, sy = self.w2s(agent.goal[0], agent.goal[1])
            size = 6

            pg.draw.line(
                self.screen,
                agent.color,
                (sx - size, sy),
                (sx + size, sy),
                2
            )

            pg.draw.line(
                self.screen,
                agent.color,
                (sx, sy - size),
                (sx, sy + size),
                2
            )

    def _draw_trails(self):
        for agent in self.planner.agents:
            if len(agent.trail) < 2:
                continue

            pts = [self.w2s(x, y) for x, y in agent.trail]
            pg.draw.lines(self.screen, agent.color, False, pts, 2)

    def _draw_agents(self):
        for agent in self.planner.agents:
            sx, sy = self.w2s(agent.x, agent.y)
            rr = int(agent.radius * self.px_per_cm)

            pg.draw.circle(
                self.screen,
                agent.color,
                (sx, sy),
                rr,
                2
            )

            hx = agent.x + agent.radius * math.cos(agent.theta)
            hy = agent.y + agent.radius * math.sin(agent.theta)

            pg.draw.line(
                self.screen,
                agent.color,
                (sx, sy),
                self.w2s(hx, hy),
                2
            )

    def _log_current_state(self):
        t_real = time.perf_counter() - self.real_t0

        for agent in self.planner.agents:
            x, y = self.planner.get_agent_position(agent)
            vx, vy = self.planner.get_agent_velocity(agent)

            self.logger.log_state(
                t_sim=self.planner.time,
                t_real=t_real,
                agent=agent.name,
                x=float(x),
                y=float(y),
                vx=float(vx),
                vy=float(vy),
                reached=agent.reached
            )

    def _generate_metrics(self):
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
            if self.planner.time >= self.max_time:
                running = False

            # =====================================
            # Planner update
            # =====================================
            self.planner.step()

            # =====================================
            # Log state
            # =====================================
            self._log_current_state()

            # =====================================
            # Auto stop when goals reached
            # =====================================
            if self.auto_stop_when_done:

                if self.planner.all_reached():

                    if self.all_reached_since is None:

                        self.all_reached_since = self.planner.time

                    elif (
                        self.planner.time
                        - self.all_reached_since
                        >= self.finish_grace_time
                    ):

                        print(
                            f"All agents reached goals "
                            f"(t={self.planner.time:.2f}s)"
                        )

                        running = False

                else:

                    self.all_reached_since = None

            # =====================================
            # Draw
            # =====================================
            self.screen.fill(self.bg_color)

            self._draw_obstacles()
            self._draw_goals()
            self._draw_trails()
            self._draw_agents()

            pg.display.flip()
            self.clock.tick(self.fps)

        # =====================================
        # Save metrics
        # =====================================
        self.logger.close()
        self._generate_metrics()

        pg.quit()
        sys.exit()


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_pred_dwa.py /ruta/al/env.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    test = PredDWATest(yaml_path)
    test.run()


if __name__ == "__main__":
    main()