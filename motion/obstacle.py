from typing import Tuple
import numpy as np


class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}

    def set_circle(self, name: str, centerx: float, centery: float, radius: float):
        self.objects[name] = {
            "type": "circle",
            "name": name,
            "centerx": centerx,
            "centery": centery,
            "radius": radius,
        }

    def set_rectangle(self, name: str, x_min: float, y_min: float, x_max: float, y_max: float):
        self.objects[name] = {
            "type": "rectangle",
            "name": name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    def _circle_d_grad(self, x: float, y: float, o: dict):
        ox = o["centerx"]
        oy = o["centery"]
        r = o["radius"]

        vx = x - ox
        vy = y - oy

        dist_center = np.hypot(vx, vy)
        d = dist_center - r

        if dist_center < 1e-9:
            return d, 1.0, 0.0

        return d, vx / dist_center, vy / dist_center

    def _rectangle_d_grad(self, x: float, y: float, o: dict):
        x_min = o["x_min"]
        y_min = o["y_min"]
        x_max = o["x_max"]
        y_max = o["y_max"]

        closest_x = min(max(x, x_min), x_max)
        closest_y = min(max(y, y_min), y_max)

        vx = x - closest_x
        vy = y - closest_y

        outside_dist = np.hypot(vx, vy)

        # ======================================================
        # Fuera del rectángulo:
        # gradiente = desde el rectángulo hacia el agente
        # ======================================================
        if outside_dist > 1e-9:
            return outside_dist, vx / outside_dist, vy / outside_dist

        # ======================================================
        # Dentro del rectángulo:
        # distancia negativa y gradiente hacia la salida más cercana
        # ======================================================
        dist_left = x - x_min
        dist_right = x_max - x
        dist_top = y - y_min
        dist_bottom = y_max - y

        distances = [
            dist_left,
            dist_right,
            dist_top,
            dist_bottom,
        ]

        min_idx = int(np.argmin(distances))
        penetration = distances[min_idx]

        if min_idx == 0:
            return -penetration, -1.0, 0.0

        if min_idx == 1:
            return -penetration, 1.0, 0.0

        if min_idx == 2:
            return -penetration, 0.0, -1.0

        return -penetration, 0.0, 1.0

    def get_d_grad(self, x: float, y: float) -> Tuple[float, float, float]:
        mindist = np.inf
        best_gradx = 0.0
        best_grady = 0.0

        for o in self.objects.values():

            if o["type"] == "circle":
                d, gx, gy = self._circle_d_grad(x, y, o)

            elif o["type"] == "rectangle":
                d, gx, gy = self._rectangle_d_grad(x, y, o)

            else:
                continue

            if d < mindist:
                mindist = d
                best_gradx = gx
                best_grady = gy

        if not np.isfinite(mindist):
            return np.inf, 0.0, 0.0

        return mindist, best_gradx, best_grady