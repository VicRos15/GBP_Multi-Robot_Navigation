import pygame as pg
import pygame.locals as pgl
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import itertools
import numpy as np

from motion.obstacle import ObstacleMap
from motion.agent_FSM import Agent, Env

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
PX_PER_CM = 1.2
SCREEN_W, SCREEN_H = 1400, 900
BG_COLOR = (245, 245, 245)

COLORS = {
    'tb': (0, 180, 0),
    'adam': (0, 0, 220),
    'a2': (0, 0, 220),
    'wall': (200, 50, 50),
    'obstacle': (180, 60, 60),
}

def w2s(x, y):
    """World (cm) → Screen (px)"""
    return int(x * PX_PER_CM), int(y * PX_PER_CM)

# ============================================================
# PATH CURVO SIMPLE
# ============================================================
def generate_curved_path(initial, final, num_points=50):
    x0, y0, _, _ = initial
    x1, y1, _, _ = final
    cx = (x0 + x1) / 2
    cy = min(y0, y1) - 80
    path = []
    for t in np.linspace(0, 1, num_points):
        x = (1 - t)**2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
        y = (1 - t)**2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
        path.append((x, y))
    return path

# ============================================================
# MAIN
# ============================================================
def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption("GBP Demo - TurtleBot vs ADAM")
    clock = pg.time.Clock()

    # ========================================================
    # MAPA Y OBSTÁCULOS
    # ========================================================
    omap = ObstacleMap()

    # --- Pasillo ---
    corridor_gap = 440        # cm
    wall_thickness = 20
    wall_height = 600

    x_left = 300
    x_right = x_left + corridor_gap
    y_top = 100
    y_bottom = y_top + wall_height

    # omap.set_rectangle('wall_left',
    #     x_left, y_top, x_left + wall_thickness, y_bottom)
    # omap.set_rectangle('wall_right',
    #     x_right, y_top, x_right + wall_thickness, y_bottom)

    # --- Obstáculos laterales ---
    path_x_center = (x_left + x_right) / 2

    # Cubo izquierdo
    cube_size = 40
    cube_x = path_x_center - 80
    cube_y = y_top + 200
    omap.set_rectangle(
        "cube_left",
        cube_x - cube_size/2,
        cube_y - cube_size/2,
        cube_x + cube_size/2,
        cube_y + cube_size/2
    )

    # Cilindro derecho
    cyl_radius = 25
    cyl_x = path_x_center + 110
    cyl_y = y_top + 350
    omap.set_circle(
        "cyl_right",
        cyl_x,
        cyl_y,
        cyl_radius
    )

    # ========================================================
    # AGENTES
    # ========================================================
    tb_start   = [path_x_center, y_top + 40, 0, 0]
    tb_goal    = [path_x_center, y_bottom - 40, 0, 0]

    adam_start = [path_x_center, y_bottom - 40, 0, 0]
    adam_goal  = [path_x_center, y_top + 40, 0, 0]

    a2_start = [path_x_center - 400, y_bottom - 40, 0, 0]
    a2_goal  = [path_x_center + 300, y_top +40, 0, 0]

    tb_agent = Agent(
        'tb', tb_start, tb_goal,
        steps=6, radius=20,
        omap=omap, path=generate_curved_path(tb_start, tb_goal)
    )

    adam_agent = Agent(
        'adam', adam_start, adam_goal,
        steps=6, radius=71.8,
        omap=omap, path=generate_curved_path(adam_start, adam_goal)
    )

    a2_agent = Agent(
        'a2', a2_start, a2_goal,
        steps=6, radius=20,
        omap=omap, path=generate_curved_path(a2_start, a2_goal)
    )



    env = Env()
    env.add_agent(tb_agent)
    env.add_agent(adam_agent)
    env.add_agent(a2_agent)

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pgl.QUIT:
                running = False

        # --- PLANIFICACIÓN GBP ---
        env.step_plan()

        # --- DIBUJO ---
        screen.fill(BG_COLOR)

        # Dibujar obstáculos
        for o in omap.objects.values():
            if o['type'] == 'rectangle':
                x0, y0 = w2s(o['x_min'], o['y_min'])
                w = int((o['x_max'] - o['x_min']) * PX_PER_CM)
                h = int((o['y_max'] - o['y_min']) * PX_PER_CM)
                pg.draw.rect(screen, COLORS['obstacle'], (x0, y0, w, h), 3)
            elif o['type'] == 'circle':
                sx, sy = w2s(o['centerx'], o['centery'])
                rr = int(o['radius'] * PX_PER_CM)
                pg.draw.circle(screen, COLORS['obstacle'], (sx, sy), rr, 3)

        # Dibujar agentes y trayectorias GBP
        for agent in env._agents:
            state = agent.get_state()
            if state[0] is None:
                continue

            color = COLORS[agent.name]
            x, y, _, _ = state[0]
            sx, sy = w2s(x, y)
            pg.draw.circle(screen, color, (sx, sy), int(agent._radius * PX_PER_CM), 2)

            # Trayectoria GBP
            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                pg.draw.line(screen, color, w2s(x0, y0), w2s(x1, y1), 2)

        # --- MOVIMIENTO ---
        env.step_move()

        pg.display.flip()
        clock.tick(60)

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    main()
