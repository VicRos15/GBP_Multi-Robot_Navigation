from typing import Tuple, List, Dict
import numpy as np

class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}

    # -----------------------------
    # Agregar un círculo
    # -----------------------------
    def set_circle(self, name: str, centerx: float, centery: float, radius: float):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    # -----------------------------
    # Agregar un rectángulo
    # -----------------------------
    def set_rectangle(self, name: str, x_min: float, y_min: float, x_max: float, y_max: float):
        o = {'type':'rectangle', 'name': name, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
        self.objects[name] = o

    # -----------------------------
    # Calcular distancia mínima y gradiente hacia el obstáculo más cercano
    # -----------------------------
    def get_d_grad(self, x: float, y: float) -> Tuple[float, float, float]:
        mindist = np.inf
        mino = None

        # Buscar obstáculo más cercano
        for o in self.objects.values():
            if o['type'] == 'circle':
                ox, oy, r = o['centerx'], o['centery'], o['radius']
                d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
            elif o['type'] == 'rectangle':
                x_min, y_min, x_max, y_max = o['x_min'], o['y_min'], o['x_max'], o['y_max']
                # Distancia al rectángulo (0 si está dentro)
                dx = max(x_min - x, 0, x - x_max)
                dy = max(y_min - y, 0, y - y_max)
                d = np.sqrt(dx**2 + dy**2)
            else:
                continue  # ignorar tipos desconocidos

            if d < mindist:
                mindist = d
                mino = o

        # Si no hay obstáculos
        if mino is None:
            return np.inf, 0.0, 0.0

        # Calcular gradiente
        if mino['type'] == 'circle':
            ox, oy = mino['centerx'], mino['centery']
            dx, dy = x - ox, y - oy
            mag = np.linalg.norm([dx, dy])
            if mag == 0:
                return mindist, 0.0, 0.0
            return mindist, dx/mag, dy/mag

        elif mino['type'] == 'rectangle':
            x_min, y_min, x_max, y_max = mino['x_min'], mino['y_min'], mino['x_max'], mino['y_max']
            dx = max(x_min - x, 0, x - x_max)
            dy = max(y_min - y, 0, y - y_max)
            d = np.sqrt(dx**2 + dy**2)
            if d == 0:
                # dentro del rectángulo: apuntar hacia el borde más cercano
                distances = [x - x_min, x_max - x, y - y_min, y_max - y]
                min_idx = np.argmin(np.abs(distances))
                if min_idx == 0:
                    return 0.0, -1.0, 0.0
                elif min_idx == 1:
                    return 0.0, 1.0, 0.0
                elif min_idx == 2:
                    return 0.0, 0.0, -1.0
                else:
                    return 0.0, 0.0, 1.0
            else:
                return d, dx/d, dy/d

        # fallback
        return mindist, 0.0, 0.0

