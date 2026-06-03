#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import yaml
import subprocess
import numpy as np
import pygame as pg
import pygame.locals as pgl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from motion.obstacle import ObstacleMap
from motion.agent_FSM import Agent, Env
from metrics.metrics_logger import MetricsLogger


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
        self.agent_starts = {}
        self.agent_reference_paths = {}

        self.t_sim = 0.0
        self.dt = 1.0 / self.fps
        self.max_time = float(self.world_cfg.get("max_time", 120.0))
        self.goal_tolerance = float(self.world_cfg.get("goal_tolerance", 20.0))
        self.auto_stop_when_done = bool(self.world_cfg.get("auto_stop_when_done", True))
        self.finish_grace_time = float(self.world_cfg.get("finish_grace_time", 0.5))
        self.all_reached_since = None

        self.real_t0 = None
        self.reached = {}
        self.trails = {}

        # Obstáculos dinámicos
        self.dynamic_obstacles = {}
        self.sim_time = 0.0

        self._build_obstacles()
        self._build_agents()

        env_name = os.path.splitext(os.path.basename(yaml_path))[0]
        self.logger = MetricsLogger(
            algorithm="gbp",
            env_name=env_name,
            run_id=f"{env_name}_gbp",
            results_dir="results",
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
        pg.display.set_caption("GBP FSM Demo")
        self.clock = pg.time.Clock()
        # Fuentes intermedias para capturas del TFM
        self.font_title = pg.font.SysFont("arial", 24, bold=True)
        self.font = pg.font.SysFont("arial", 18, bold=True)
        self.font_small = pg.font.SysFont("arial", 15)
        self.font_tiny = pg.font.SysFont("arial", 13)
        # Paleta moderna para visualización y capturas
        self.ui = {
            "background": tuple(self.world_cfg.get("bg_color", [248, 250, 252])),
            "grid": (226, 232, 240),
            "grid_major": (203, 213, 225),
            "text": (15, 23, 42),
            "muted": (100, 116, 139),
            "panel": (255, 255, 255),
            "panel_border": (226, 232, 240),

            "obstacle": (239, 68, 68),
            "obstacle_border": (153, 27, 27),
            "dynamic_obstacle": (249, 115, 22),

            "path": (148, 163, 184),
            "trail": (37, 99, 235),
            "start": (16, 185, 129),
            "goal": (220, 38, 38),

            "follow": (34, 197, 94),
            "reconfigure": (249, 115, 22),
            "recovery": (139, 92, 246),
        }
    # ============================================================
    # LOAD / CONVERSIONS
    # ============================================================

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def w2s(self, x, y):
        return int(x * self.px_per_cm), int(y * self.px_per_cm)

    # ============================================================
    # PATH GENERATION
    # ============================================================

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
            return self._generate_straight_path(
                start,
                goal,
                num_points=num_points,
            )

        if path_type == "curved":
            arc_height = float(path_cfg.get("arc_height", 80))
            return self._generate_curved_path(
                start,
                goal,
                num_points=num_points,
                arc_height=arc_height,
            )

        raise ValueError(f"Tipo de path no soportado: {path_type}")

    # ============================================================
    # BUILD OBSTACLES / AGENTS
    # ============================================================

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
                path=path,
            )

            self.env.add_agent(agent)

            self.agent_colors[name] = color
            self.agent_real_radius[name] = radius
            self.agent_goals[name] = goal_xy
            self.agent_starts[name] = start_xy
            self.agent_reference_paths[name] = path

    # ============================================================
    # DYNAMIC OBSTACLES
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
    # LOGGER
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
    # DRAW HELPERS
    # ============================================================

    def _draw_grid(self):
        """
        Dibuja una cuadrícula ligera para dar referencia espacial sin
        cargar visualmente la escena.
        """
        spacing = int(50 * self.px_per_cm)
        if spacing <= 0:
            return

        for x in range(0, self.screen_w, spacing):
            color = self.ui["grid_major"] if x % (spacing * 2) == 0 else self.ui["grid"]
            pg.draw.line(self.screen, color, (x, 0), (x, self.screen_h), 1)

        for y in range(0, self.screen_h, spacing):
            color = self.ui["grid_major"] if y % (spacing * 2) == 0 else self.ui["grid"]
            pg.draw.line(self.screen, color, (0, y), (self.screen_w, y), 1)
            

    def _draw_obstacles(self):
        obstacle_color = self.ui["obstacle"]
        dynamic_color = self.ui["dynamic_obstacle"]

        for name, o in self.omap.objects.items():
            color = dynamic_color if name in self.dynamic_obstacles else obstacle_color
            border = self.ui["obstacle_border"]

            if o["type"] == "rectangle":
                x0, y0 = self.w2s(o["x_min"], o["y_min"])
                w = int((o["x_max"] - o["x_min"]) * self.px_per_cm)
                h = int((o["y_max"] - o["y_min"]) * self.px_per_cm)

                shadow = pg.Surface((w + 10, h + 10), pg.SRCALPHA)
                pg.draw.rect(shadow, (15, 23, 42, 35), (6, 6, w, h), border_radius=8)
                self.screen.blit(shadow, (x0 - 5, y0 - 5))

                surf = pg.Surface((w, h), pg.SRCALPHA)
                pg.draw.rect(surf, (*color, 190), (0, 0, w, h), border_radius=8)
                pg.draw.rect(surf, (*border, 230), (0, 0, w, h), 2, border_radius=8)
                self.screen.blit(surf, (x0, y0))

            elif o["type"] == "circle":
                sx, sy = self.w2s(o["centerx"], o["centery"])
                rr = int(o["radius"] * self.px_per_cm)

                shadow = pg.Surface((2 * rr + 14, 2 * rr + 14), pg.SRCALPHA)
                pg.draw.circle(shadow, (15, 23, 42, 35), (rr + 8, rr + 8), rr)
                self.screen.blit(shadow, (sx - rr - 7, sy - rr - 7))

                surf = pg.Surface((2 * rr + 4, 2 * rr + 4), pg.SRCALPHA)
                pg.draw.circle(surf, (*color, 185), (rr + 2, rr + 2), rr)
                pg.draw.circle(surf, (*border, 230), (rr + 2, rr + 2), rr, 2)

                self.screen.blit(surf, (sx - rr - 2, sy - rr - 2))

    def _draw_reference_paths(self):
        """
        Dibuja el path nominal inicial como puntos discontinuos.
        """
        for name, path in self.agent_reference_paths.items():
            if not path:
                continue

            if len(path) > 1:
                pts = [self.w2s(x, y) for x, y in path]
                pg.draw.lines(self.screen, self.ui["grid_major"], False, pts, 1)

            for idx, (x, y) in enumerate(path):
                if idx % 3 != 0:
                    continue

                sx, sy = self.w2s(x, y)
                pg.draw.circle(self.screen, self.ui["path"], (sx, sy), 3, 0)
                pg.draw.circle(self.screen, (255, 255, 255), (sx, sy), 3, 1)

    def _draw_start_points(self):
        """
        Dibuja el punto inicial como marcador con anillo.
        """
        for name, start in self.agent_starts.items():
            sx, sy = self.w2s(start[0], start[1])

            pg.draw.circle(self.screen, (255, 255, 255), (sx, sy), 13, 0)
            pg.draw.circle(self.screen, self.ui["start"], (sx, sy), 10, 0)
            pg.draw.circle(self.screen, self.ui["text"], (sx, sy), 10, 2)
            pg.draw.circle(self.screen, (255, 255, 255), (sx, sy), 4, 0)

            label = self.font_small.render("START", True, self.ui["muted"])
            self.screen.blit(label, (sx - 28, sy + 18))

    def _draw_goals(self):
        """
        Dibuja el goal como una X clara.
        """
        for name, goal in self.agent_goals.items():
            gx, gy = goal
            sx, sy = self.w2s(gx, gy)

            size = 11

            pg.draw.line(
                self.screen,
                (255, 255, 255),
                (sx - size - 2, sy - size - 2),
                (sx + size + 2, sy + size + 2),
                6,
            )
            pg.draw.line(
                self.screen,
                (255, 255, 255),
                (sx - size - 2, sy + size + 2),
                (sx + size + 2, sy - size - 2),
                6,
            )

            pg.draw.line(
                self.screen,
                self.ui["goal"],
                (sx - size, sy - size),
                (sx + size, sy + size),
                4,
            )
            pg.draw.line(
                self.screen,
                self.ui["goal"],
                (sx - size, sy + size),
                (sx + size, sy - size),
                4,
            )
            label = self.font_small.render("GOAL", True, self.ui["muted"])
            self.screen.blit(label, (sx - 24, sy + 18))

    def _draw_trails(self):
        """
        Dibuja la trayectoria realmente recorrida por cada agente
        usando el mismo color asignado al agente.
        """
        for name, trail in self.trails.items():
            if len(trail) < 2:
                continue

            pts = [self.w2s(x, y) for x, y in trail]

            agent_color = self.agent_colors.get(name, self.ui["trail"])

            # Línea blanca inferior para mejorar contraste
            pg.draw.lines(self.screen, (255, 255, 255), False, pts, 7)

            # Trayectoria recorrida con el color del agente
            pg.draw.lines(self.screen, agent_color, False, pts, 4)
    def _get_agent_state_name(self, agent: Agent) -> str:
        state_names = {
            1: "FOLLOW_PATH",
            2: "RECOVERY",
            3: "RECONFIGURE",
        }
        return state_names.get(agent._change, f"UNKNOWN({agent._change})")

    def _get_state_color(self, state_name: str):
        if state_name == "FOLLOW_PATH":
            return self.ui["follow"]
        if state_name == "RECONFIGURE":
            return self.ui["reconfigure"]
        if state_name == "RECOVERY":
            return self.ui["recovery"]
        return self.ui["text"]


    def _draw_label_badge(self, text: str, x: int, y: int, color):
        """
        Dibuja una etiqueta tipo badge con fondo blanco.
        Pensada para ser legible en capturas del TFM.
        """
        padding_x = 12
        padding_y = 7

        label = self.font_small.render(text, True, color)
        w, h = label.get_size()

        rect = pg.Rect(x, y, w + 2 * padding_x, h + 2 * padding_y)
        shadow_rect = rect.move(3, 3)

        shadow = pg.Surface((shadow_rect.w, shadow_rect.h), pg.SRCALPHA)
        pg.draw.rect(shadow, (15, 23, 42, 45), shadow.get_rect(), border_radius=10)
        self.screen.blit(shadow, shadow_rect.topleft)

        pg.draw.rect(self.screen, self.ui["panel"], rect, border_radius=10)
        pg.draw.rect(self.screen, color, rect, 3, border_radius=10)
        self.screen.blit(label, (x + padding_x, y + padding_y))

    def _draw_agents_and_paths(self):
        """
        Dibuja:
        - círculo del agente;
        - radio real y radio de seguridad;
        - horizonte local planificado por GBP;
        - etiqueta del estado FSM.
        """
        for agent in self.env._agents:
            state = agent.get_state()

            if not state or state[0] is None:
                continue

            base_color = self.agent_colors.get(agent.name, (37, 99, 235))

            x, y, _, _ = state[0]
            sx, sy = self.w2s(x, y)

            real_radius = self.agent_real_radius.get(agent.name, agent.r)
            security_radius = agent.r

            state_name = self._get_agent_state_name(agent)
            state_color = self._get_state_color(state_name)

            horizon_points = []
            for s in state:
                if s is None:
                    break
                hx, hy, _, _ = s
                horizon_points.append(self.w2s(hx, hy))

            if len(horizon_points) > 1:
                pg.draw.lines(self.screen, (255, 255, 255), False, horizon_points, 6)
                pg.draw.lines(self.screen, state_color, False, horizon_points, 3)

                for p in horizon_points[1:]:
                    pg.draw.circle(self.screen, (255, 255, 255), p, 5, 0)
                    pg.draw.circle(self.screen, state_color, p, 4, 0)

            shadow_radius = int(real_radius * self.px_per_cm) + 3
            shadow = pg.Surface((2 * shadow_radius + 8, 2 * shadow_radius + 8), pg.SRCALPHA)
            pg.draw.circle(
                shadow,
                (15, 23, 42, 45),
                (shadow_radius + 5, shadow_radius + 5),
                shadow_radius,
            )
            self.screen.blit(shadow, (sx - shadow_radius - 4, sy - shadow_radius - 4))

            pg.draw.circle(
                self.screen,
                state_color,
                (sx, sy),
                int(security_radius * self.px_per_cm),
                2,
            )

            pg.draw.circle(
                self.screen,
                base_color,
                (sx, sy),
                int(real_radius * self.px_per_cm),
                0,
            )
            pg.draw.circle(
                self.screen,
                (255, 255, 255),
                (sx, sy),
                int(real_radius * self.px_per_cm),
                3,
            )
            pg.draw.circle(
                self.screen,
                self.ui["text"],
                (sx, sy),
                int(real_radius * self.px_per_cm),
                1,
            )

            pg.draw.circle(self.screen, (255, 255, 255), (sx, sy), 4, 0)

            name_label = self.font_small.render(agent.name, True, self.ui["text"])
            self.screen.blit(name_label, (sx - 8, sy - 35))
            self._draw_label_badge(state_name, sx - 78, sy - 82, state_color)

    def _draw_hud(self):
        """
        Panel superior para capturas del TFM.
        """
        panel_w = 820
        panel_h = 122
        x0, y0 = 18, 16

        shadow = pg.Surface((panel_w, panel_h), pg.SRCALPHA)
        pg.draw.rect(shadow, (15, 23, 42, 40), shadow.get_rect(), border_radius=16)
        self.screen.blit(shadow, (x0 + 4, y0 + 4))

        panel_rect = pg.Rect(x0, y0, panel_w, panel_h)
        pg.draw.rect(self.screen, self.ui["panel"], panel_rect, border_radius=16)
        pg.draw.rect(self.screen, self.ui["panel_border"], panel_rect, 2, border_radius=16)

        title = self.font_title.render("GBP FSM DEMO", True, self.ui["text"])
        self.screen.blit(title, (x0 + 20, y0 + 14))

        subtitle = self.font.render(
            "FOLLOW_PATH  →  RECONFIGURE  →  RECOVERY  →  FOLLOW_PATH",
            True,
            self.ui["muted"],
        )
        self.screen.blit(subtitle, (x0 + 20, y0 + 54))

        legend_items = [
            ("reference path", self.ui["path"]),
            ("executed trajectory", self.ui["trail"]),
            ("local GBP horizon", self.ui["reconfigure"]),
        ]

        lx = x0 + 22
        ly = y0 + 92

        for text, color in legend_items:
            pg.draw.circle(self.screen, color, (lx, ly + 8), 7, 0)
            label = self.font_tiny.render(text, True, self.ui["muted"])
            self.screen.blit(label, (lx + 15, ly))
            lx += 245

    # ============================================================
    # STOP CONDITIONS
    # ============================================================

    def all_reached(self) -> bool:
        if not self.reached:
            return False

        return all(self.reached.values())

    # ============================================================
    # MAIN LOOP
    # ============================================================

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

                    # Guardar captura manual con S.
                    elif event.key == pgl.K_s:
                        filename = f"fsm_capture_{self.t_sim:.2f}.png"
                        pg.image.save(self.screen, filename)
                        print(f"Saved screenshot: {filename}")

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
            self.screen.fill(self.ui["background"])

            self._draw_grid()
            self._draw_reference_paths()
            self._draw_obstacles()
            self._draw_goals()
            self._draw_start_points()
            self._draw_trails()
            self._draw_agents_and_paths()
            self._draw_hud()

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
                "generate_metrics.py",
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
        print("Uso: python test_gbp01.py /ruta/al/env.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    planner = GBPPlanner(yaml_path)
    planner.run()


if __name__ == "__main__":
    main()