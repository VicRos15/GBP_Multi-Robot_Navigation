from typing import Tuple, List, Dict
import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

# Importaciones de concurrencia eliminadas (ya no son necesarias)

from .obstacle import ObstacleMap
from .nodes import DynaFNode, ObstacleFNode, DistFNode, RemoteVNode


class Agent:
    """
    Representa un agente autónomo que planifica su trayectoria usando un Grafo de Factores
    y propagación de creencias Gaussianas (Gaussian Belief Propagation - GBP).
    
    El agente mantiene un grafo de factores que representa su trayectoria futura
    a lo largo de `steps` pasos de tiempo. Este grafo incluye:
    - VNodes (Nodos Variables): Representan el estado (posición y velocidad) en cada paso.
    - FNodes (Nodos Factores): Imponen restricciones o "deseos" sobre los VNodes.
        - fnode_start: Fija el estado inicial (prior).
        - fnode_end: Fija el estado objetivo (target).
        - fnodes_dyna: Imponen la dinámica del movimiento (física) entre pasos.
        - fnodes_obst: Penalizan estados demasiado cercanos a obstáculos estáticos.
        - DistFNode (creados en setup_com): Penalizan estados demasiado cercanos a otros agentes.
    
    El agente opera en una máquina de estados (controlada por `self._change`) para
    gestionar si está siguiendo un camino precalculado (1), moviéndose a un
    punto de recuperación (2), o planificando activamente para evitar colisiones (3).
    """
    def __init__(
            self, name: str, state, target = None, steps: int = 8, radius: int = 5, omap: ObstacleMap = None, env: 'Env' = None,
            start_position_precision = 100,
            start_velocity_precision = 100,
            target_position_precision = 5,
            target_velocity_precision = 5,
            dynamic_position_precision = 10,
            dynamic_velocity_precision = 2,
            obstacle_precision = 50,
            distance_precision = 1, #! Modificado de 100 a 1
            dt: float = 0.1,
            path = None
        ) -> None:
        assert steps > 1, "El número de pasos debe ser mayor que 1"
        
        # --- Normalización de Entradas ---
        # Asegura que 'state' y 'target' sean vectores columna (shape [N, 1])
        if np.shape(state) == ():
            state = np.array([[state]])
        elif len(np.shape(state)) == 1:
            state = np.array(state)[:, None]
        if target is not None:
            if np.shape(target) == ():
                target = np.array([[target]])
            elif len(np.shape(target)) == 1:
                target = np.array(target)[:, None]

        # --- Parámetros Internos ---
        self._steps = steps
        self._name = name
        self._state = np.array(state) # Estado actual *real* (para simulación)
        self._omap = omap
        self._radius = radius
        self._env = env
        self.path = path if path is not None else []
        self.recovery_point= None # Punto en el 'path' al que volver tras evitar
        self.specific_point = None # No usado en el código provisto
        self._dt = dt

        # --- Precisiones (Inversa de la Covarianza) para los Factores ---
        self._startFNode_pos_prec = start_position_precision
        self._startFNode_vel_prec = start_velocity_precision
        self._targetFNode_pos_prec = target_position_precision
        self._targetFNode_vel_prec = target_velocity_precision
        self._distFNode_prec = distance_precision

        # --- Máquina de Estados del Agente ---
        self._change = 1
        self._last_propagation_failed = False

        # Distancia para considerar que el objetivo final ha sido alcanzado.
        self._final_goal_tolerance = 10.0

        # Rango de detección de otros agentes.
        self._near_agent_range = 500.0

        # Distancia mínima a obstáculo para considerar una zona segura.
        self._obstacle_safe_dist = 65.0

        # Distancia mínima a obstáculo que debe tener un punto candidato
        # para poder usarse como punto de recuperación.
        self._recovery_candidate_safe_dist = 20.0

        # Número de índices que se adelanta el recovery point respecto
        # al primer punto válido encontrado en el path.
        self._recovery_index_lookahead = 20

        # Tolerancia para considerar alcanzado el recovery point.
        self._recovery_goal_tolerance = 8.0

        # Tolerancia para considerar alcanzado el final del path.
        self._path_goal_tolerance = 8.0

        # Covarianza de alta confianza usada al forzar seguimiento de path.
        self._path_follow_cov_value = 1e-6

        # Factor aplicado a la covarianza del nodo inicial al entrar en RECONFIGURE.
        self._reconfigure_start_cov_scale = 0.1

        # Ruido usado en la extrapolación del último nodo en RECOVERY.
        self._recovery_tail_noise_scale = 0.01

        # Ruido usado en la extrapolación del último nodo en RECONFIGURE.
        self._reconfigure_tail_noise_scale = 1.0
        
        # 1. Nodos Variables (VNodes)
        # Cada VNode representa el estado [x, y, vx, vy] en un tiempo i
        self._vnodes = [VNode(f'v{i}', [f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy']) for i in range(steps)]

        # 2. Factores de Inicio y Fin (Priors)
        self._fnode_start = FNode('fstart', [self._vnodes[0]]) # Fija el estado en t=0
        self._fnode_end = FNode('fend', [self._vnodes[-1]])   # Fija el estado objetivo en t=steps
        
        # Inicializa los factores y creencias con el estado y target
        self.set_state(state)
        self.set_target(target)
        self._current_index = 0 # Índice para recorrer self.path
        self._target = target     # Almacena el objetivo final (recovery_point es temporal)

        # 3. Factores Dinámicos (Física)
        # Conectan v[i] y v[i+1] usando un modelo de movimiento (velocidad constante)
        self._fnodes_dyna = [DynaFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], dt=self._dt,
            pos_prec=dynamic_position_precision, vel_prec=dynamic_velocity_precision
        ) for i in range(steps-1)]
        
        # 4. Factores de Obstáculos Estáticos
        # Conectan v[i] (para i > 0) y penalizan la cercanía a obstáculos en omap
        self._fnodes_obst = [ObstacleFNode(
            f'fo{i}', [self._vnodes[i]], omap=omap, safe_dist=self.r*1.5,
            z_precision=obstacle_precision
        ) for i in range(1, steps)]

        # 5. Creación y Conexión del Grafo
        self._graph = FactorGraph()

        # Conectar nodo inicial
        self._graph.connect(self._vnodes[0], self._fnode_start)
        
        # Conectar cadena de nodos dinámicos
        for v, f in zip(self._vnodes[:-1], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_dyna):
            self._graph.connect(v, f)
            
        # Conectar factores de obstáculos (a partir de v[1])
        for v, f in zip(self._vnodes[1:], self._fnodes_obst):
            self._graph.connect(v, f)
            
        # Conectar nodo final
        self._graph.connect(self._vnodes[-1], self._fnode_end)

        # Diccionario para gestionar conexiones con otros agentes
        # 'agent_name': {'a': Agent, 'v': List[RemoteVNode], 'f': List[DistFNode]}
        self._others = {}

        # Locks y caches de concurrencia han sido eliminados

    def __str__(self) -> str:
        return f'({self._name} s={self._state})'
    
    def _clamp_path_index(self):
        if not self.path:
            self._current_index = 0
            return
        self._current_index = max(0, min(self._current_index, len(self.path) - 1))

    def _snapshot_beliefs(self):
        snapshot = []
        for v in self._vnodes:
            try:
                snapshot.append(v.belief.copy() if v.belief is not None else None)
            except Exception:
                snapshot.append(None)
        return snapshot

    def _restore_beliefs(self, snapshot):
        for v, b in zip(self._vnodes, snapshot):
            v._belief = b.copy() if b is not None else None

    def _hold_position(self):
        """
        Mantiene al agente quieto durante un ciclo si el paso de GBP
        anterior falló numéricamente. Así evitamos que la FSM use
        beliefs dudosas o parcialmente inconsistentes.
        """
        st = self.get_state()[0]
        if st is None:
            st = self._state[:, 0]

        s = np.array(st).reshape(4, 1).copy()
        s[2:] = 0.0

        safe_cov = np.diag([1e-3, 1e-3, 1e-2, 1e-2])
        for v in self._vnodes:
            v._belief = Gaussian(v.dims, s.copy(), safe_cov.copy())

    @property
    def name(self) -> str:
        """Nombre identificador del agente."""
        return self._name
    @property
    def x(self) -> float:
        """Posición x actual (del estado de simulación)."""
        return self._state[0, 0]
    @property
    def y(self) -> float:
        """Posición y actual (del estado de simulación)."""
        return self._state[1, 0]
    @property
    def r(self) -> float:
        """Radio del agente (usado para colisiones)."""
        return self._radius

    def get_state(self) -> List[np.ndarray]:
        """
        Obtiene la trayectoria planificada (media de las creencias)
        para todos los VNodes.
        """
        poss = []
        for v in self._vnodes:
            if v.belief is None:
                poss.append(None)
            else:
                # Retorna la media de la creencia (el estado más probable)
                poss.append(v.belief.mean[:, 0])
        return poss

    def get_target(self) -> np.ndarray:
        """Obtiene el 'target' actual del factor final (el objetivo)."""
        return self._fnode_end._factor.mean

    def step_connect(self):
        """
        Actualiza las conexiones del grafo de factores con agentes cercanos.
        Este paso gestiona la topología dinámica del grafo.
        """
        # 1. Encontrar agentes cercanos
        others = self._env.find_near(self)
        
        # 2. Establecer nuevas conexiones
        for o in others:
            self.setup_com(o) # Añade DistFNodes y RemoteVNodes si no existen

        # 3. Eliminar conexiones antiguas
        for on in list(self._others.keys()): # Usar list() para poder modificar el dict
            other_agent = self._others[on]['a']
            if other_agent not in others:
                self.end_com(on) # Elimina los nodos y factores del grafo

    def step_com(self):
        """
        Envía los mensajes (creencias y mensajes de factores) a todos
        los agentes conectados.
        """
        for o in list(self._others.keys()):
            # enviar mensajes
            try:
                self.send(o)
            except Exception as e:
                # No queremos que falle todo por un envío
                print(f"[{self._name}] send error to {o}: {e}")

    def step_propagate(self):
        """
        Ejecuta una iteración de propagación de creencias (GBP).

        Si falla numéricamente, restaura el estado anterior para evitar
        que el agente quede con creencias corruptas y marca el ciclo
        como fallido para que step_move no haga maniobras raras justo después.
        """
        if self._change == 1:
            self._last_propagation_failed = False
            return

        snapshot = self._snapshot_beliefs()

        try:
            self._graph.loopy_propagate()

            # Validación post-propagación
            for v in self._vnodes:
                if v.belief is None:
                    continue
                m = v.belief.mean
                c = v.belief.cov
                if not np.all(np.isfinite(m)) or not np.all(np.isfinite(c)):
                    raise FloatingPointError("Non-finite belief detected after GBP")

            self._last_propagation_failed = False

        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as e:
            self._restore_beliefs(snapshot)
            self._last_propagation_failed = True
            print(f"[{self._name}] Warning: GBP failed ({e}). Restoring previous beliefs.")


    def step_move(self):
        """
        Controlador de alto nivel (Máquina de Estados Finitos) que decide
        el comportamiento del agente y actualiza su estado.
        """
        path = self.path

        state_names = {
            1: "FOLLOW_PATH",
            2: "RECOVERY",
            3: "RECONFIGURE",
        }
        current_state_name = state_names.get(self._change, f"UNKNOWN({self._change})")
        pos = self.get_state()[0][:2] if self.get_state()[0] is not None else [None, None]
        path_last_idx = max(0, len(path) - 1)
        debug_idx = min(self._current_index, path_last_idx) if path else 0

        print(
            f"\n [Agent {self._name}] STATE={current_state_name} "
            f"| INDEX={debug_idx}/{path_last_idx} "
            f"| POS=({pos[0]:.2f}, {pos[1]:.2f})"
        )

        # ============================================================
        # 1. Comprobación de llegada al objetivo final
        # ============================================================
        if self._target is not None:
            final_target = np.array(self._target[:2]).flatten()
            current_pos = np.array(pos[:2]).flatten()

            dist_to_final = np.linalg.norm(current_pos - final_target)

            if dist_to_final <= self._final_goal_tolerance:
                print(
                    f"[Agent {self._name}] Final goal reached "
                    f"(dist={dist_to_final:.2f}) -> stopping FSM."
                )

                self._change = 1
                self._current_index = len(path) - 1 if path else self._current_index
                self.set_target(self._target)
                return

        # ============================================================
        # 2. Protección ante fallo numérico previo de GBP
        # ============================================================
        if self._last_propagation_failed and self._change in (2, 3):
            print(
                f"[Agent {self._name}] Previous GBP step failed "
                f"-> holding position this cycle."
            )
            self._hold_position()
            return

        # ============================================================
        # 3. Detección del entorno
        # ============================================================
        other_agents = self._env.find_near(self, range=self._near_agent_range)
        distObs, _, _ = self._omap.get_d_grad(x=pos[0], y=pos[1])

        # ============================================================
        # 4. Lógica de la Máquina de Estados
        # ============================================================

        if not other_agents:

            # --------------------------------------------------------
            # Estado 3: RECONFIGURE -> buscar punto de recuperación
            # --------------------------------------------------------
            if self._change == 3 and distObs > self._obstacle_safe_dist:
                print("Searching recovery point...")

                d_agent = np.linalg.norm(
                    self.get_state()[0][:2]
                    - np.array(self._fnode_end._factor.mean[:2]).flatten()
                )

                # Busca un punto en el path restante que esté:
                # 1. Más cerca del objetivo final que el agente.
                # 2. Lejos de obstáculos estáticos.
                for i in range(len(path) - self._current_index):
                    path_idx = self._current_index + i

                    d = np.linalg.norm(
                        np.array(self._fnode_end._factor.mean[:2]).flatten()
                        - np.array(path[path_idx][:2])
                    )

                    distRecov, _, _ = self._omap.get_d_grad(
                        x=path[path_idx][0],
                        y=path[path_idx][1],
                    )

                    if (
                        distRecov > self._recovery_candidate_safe_dist
                        and d < d_agent
                    ):
                        print(f"Found recovery candidate at path index {path_idx}")

                        recovery_idx = min(
                            path_idx + self._recovery_index_lookahead,
                            len(path) - 1,
                        )
                        target_position = path[recovery_idx]

                        if len(target_position) == 2:
                            target_position = (*target_position, 0, 0)

                        # Establece el punto de recuperación como objetivo TEMPORAL
                        self.recovery_point = np.array(target_position).reshape(4, 1)
                        self.set_target(self.recovery_point)
                        self._current_index = recovery_idx
                        self._change = 2
                        break

                if self._change != 2:
                    # Si no encontró un punto, sigue en modo RECOVERY.
                    self._change = 2

            # --------------------------------------------------------
            # Estado 2: RECOVERY -> moverse hacia recovery_point
            # --------------------------------------------------------
            elif self._change == 2 and distObs > self._obstacle_safe_dist:
                print("Moving to recovery point...")

                # Roll-forward: el estado planificado para t=1 se convierte
                # en el prior para t=0.
                self._fnode_start._factor = Gaussian(
                    self._vnodes[0].dims,
                    self._vnodes[1].mean,
                    self._vnodes[1].belief.cov,
                )

                # Desplaza todas las creencias de la trayectoria un paso.
                for i in range(0, self._steps - 1):
                    v, v_ = self._vnodes[i], self._vnodes[i + 1]
                    v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)

                # Extrapolación simple para el último nodo.
                s = v_.mean + np.random.rand(4, 1) * self._recovery_tail_noise_scale
                s[:2] += s[2:] * self._dt
                v._belief = Gaussian(
                    self._vnodes[-1].dims,
                    s,
                    self._vnodes[-1].belief.cov,
                )

                # Comprobar si ha llegado al punto de recuperación.
                dist_to_recovery = np.linalg.norm(
                    self.get_state()[0][:2]
                    - np.array(self.recovery_point[:2]).flatten()
                )

                if dist_to_recovery <= self._recovery_goal_tolerance:
                    print(f"Reached recovery point at distance {dist_to_recovery:.2f}")
                    self._change = 1
                    self.set_target(self._target)
                else:
                    print(f"Not yet at recovery point (dist={dist_to_recovery:.2f})")

            # --------------------------------------------------------
            # Estado 1: FOLLOW_PATH -> seguir path nominal
            # --------------------------------------------------------
            elif (
                self._change == 1
                and distObs > self._obstacle_safe_dist
                and not other_agents
            ):
                if not path:
                    print("Path is empty.")
                    return

                self._clamp_path_index()
                target_position = path[self._current_index]

                current_pos = self.get_state()[0][:2]
                final_pos = np.array(path[-1][:2])

                if (
                    self._current_index == len(path) - 1
                    and np.linalg.norm(current_pos - final_pos)
                    <= self._path_goal_tolerance
                ):
                    print("End of path reached.")
                    return

                if len(target_position) == 2:
                    target_position = (*target_position, 0, 0)

                s = np.array(target_position).reshape(4, 1)

                high_conf_cov = np.diag(
                    [
                        self._path_follow_cov_value,
                        self._path_follow_cov_value,
                        self._path_follow_cov_value,
                        self._path_follow_cov_value,
                    ]
                )

                self._vnodes[0]._belief = Gaussian(
                    self._vnodes[0].dims,
                    s,
                    high_conf_cov,
                )

                for i in range(1, self._steps):
                    self._vnodes[i]._belief = Gaussian(
                        self._vnodes[i].dims,
                        s,
                        high_conf_cov,
                    )

                if self._current_index < len(path) - 1:
                    self._current_index += 1

            # --------------------------------------------------------
            # Obstáculo estático detectado -> RECONFIGURE
            # --------------------------------------------------------
            else:
                print("Obstacle detected during path/recovery, switching to RECONFIGURE")

                self._fnode_start._factor = Gaussian(
                    self._vnodes[0].dims,
                    self._vnodes[1].mean,
                    self._vnodes[1].belief.cov * self._reconfigure_start_cov_scale,
                )

                for i in range(0, self._steps - 1):
                    v, v_ = self._vnodes[i], self._vnodes[i + 1]
                    v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)

                s = v_.mean + np.random.rand(4, 1) * self._reconfigure_tail_noise_scale
                s[:2] += s[2:] * self._dt
                v._belief = Gaussian(
                    self._vnodes[-1].dims,
                    s,
                    self._vnodes[-1].belief.cov,
                )

                # En el próximo ciclo, step_propagate() ejecutará GBP,
                # y los ObstacleFNode empujarán la trayectoria.
                self._change = 3

        else:
            # --------------------------------------------------------
            # Agentes cercanos detectados -> RECONFIGURE
            # --------------------------------------------------------
            print("Nearby agent detected → AVOID (RECONFIGURE)")
            print("Other agents detected!!!!!!!!!1")

            self._fnode_start._factor = Gaussian(
                self._vnodes[0].dims,
                self._vnodes[1].mean,
                self._vnodes[1].belief.cov * self._reconfigure_start_cov_scale,
            )

            for i in range(0, self._steps - 1):
                v, v_ = self._vnodes[i], self._vnodes[i + 1]
                v._belief = Gaussian(v.dims, v_.mean, v_.belief.cov)

            s = v_.mean + np.random.rand(4, 1) * self._reconfigure_tail_noise_scale
            s[:2] += s[2:] * self._dt
            v._belief = Gaussian(
                self._vnodes[-1].dims,
                s,
                self._vnodes[-1].belief.cov,
            )

            # En el próximo ciclo, step_propagate() ejecutará GBP,
            # y los DistFNode empujarán la trayectoria para evitar agentes.
            self._change = 3
 

    def set_state(self, state):
        """
        Actualiza el estado inicial del agente en el grafo de factores.
        Esto "fija" el inicio de la trayectoria (vnode[0]) a la posición actual.
        """
        self._state = np.array(state)
        v0 = self._vnodes[0]
        # Covarianza inicial (alta precisión = baja covarianza)
        cov = np.diag([1/self._startFNode_pos_prec, 1/self._startFNode_pos_prec, 1/self._startFNode_vel_prec, 1/self._startFNode_vel_prec])
        
        # Actualiza el factor de inicio (el prior)
        self._fnode_start._factor = Gaussian(v0.dims, state, cov)
        # Actualiza la creencia inicial
        v0._belief = Gaussian(v0.dims, state, cov.copy())

        #* Inicializa las creencias del resto de la trayectoria
        # Esto es crucial para que los DistFNode (evitar agentes) tengan
        # una estimación inicial y no fallen por distancias nulas.
        if self.path !=[]:
            # Si hay un path, inicializa la creencia siguiendo el path
            for i in range(1, self._steps):
                # Asegura que i esté dentro de los límites del path
                path_index = min(i, len(self.path) - 1) 
                state_at_i = self.path[path_index]
                if len(state_at_i) == 2:
                    state_at_i = (*state_at_i, 10, 10) # Asume una velocidad si no está
                s = np.array(state_at_i).reshape(4, 1)
                v = self._vnodes[i]
                v._belief = Gaussian(v.dims, s, cov.copy())
        else:
            # Si no hay path, inicializa con una simple extrapolación
            s_prev = state
            for i in range(1, self._steps):
                s = s_prev + np.random.rand(4, 1) * 1
                s[:2] += s[2:] * self._dt # s = s + v*dt
                v = self._vnodes[i]
                v._belief = Gaussian(v.dims, s, cov.copy())
                s_prev = s

    def set_target(self, target):
        """
        Actualiza el objetivo (target) del agente en el grafo de factores.
        Esto "fija" el final de la trayectoria (vnode[-1]).
        """
        if target is not None:
            # Establece el factor final para "atraer" la trayectoria hacia el target
            # Nota: La covarianza aquí es baja (precisión 1), permitiendo flexibilidad.
            self._fnode_end._factor = Gaussian(self._vnodes[-1].dims, target, np.diag([1]*4))
        else:
            # Si no hay target, usa un factor no informativo (identidad)
            self._fnode_end._factor = Gaussian.identity(self._vnodes[-1].dims)

    def push_msg(self, msg):
        """
        "Inbox" del agente. Es llamado por OTROS agentes (via `other.send()`)
        para entregar mensajes (creencias, f2v, v2f).
        """
        _type, aname, vname, p = msg
        if p is None:
            return
        
        # Busca el RemoteVNode correspondiente al mensaje
        if aname not in self._others:
            # Mensaje de un agente ya desconectado. Ignorar.
            return
        vnodes: List[RemoteVNode] = self._others[aname]['v']
        vnode: RemoteVNode = None
        for v in vnodes:
            if v.name == vname:
                vnode = v
                break
        if vnode is None:
            return

        p: Gaussian
        p._dims = vnode.dims
        
        # Actualiza la creencia local del RemoteVNode o los mensajes
        if _type == 'belief':
            vnode._belief = p
        elif _type == 'f2v': # Mensaje del DistFNode al RemoteVNode
            e = vnode.edges[0]
            e.set_message_from(e.get_other(vnode), p)
        elif _type == 'v2f': # Mensaje del RemoteVNode al DistFNode
            e = vnode.edges[0]
            vnode._msgs[e] = p

    def setup_com(self, other: 'Agent'):
        """
        Establece la conexión de grafo de factores con otro agente.
        Crea los RemoteVNodes (para representar la trayectoria del otro)
        y los DistFNodes (para imponer la restricción de distancia).
        """
        on = other._name
        if on in self._others:
            return # Ya conectado

        # 1. Crear VNodes remotos (representan la trayectoria de 'other')
        vnodes = [RemoteVNode(f'{on}.v{i}', [f'{on}.v{i}.x', f'{on}.v{i}.y', f'{on}.v{i}.vx', f'{on}.v{i}.vy']) for i in range(1, self._steps)]
        
        # 2. Crear Factores de Distancia
        # Conectan self._vnodes[i] con remote_vnodes[i-1]
        fnodes = [DistFNode(
            f'{on}.f{i}', [vnodes[i-1], self._vnodes[i]], safe_dist=(self.r+other.r) * 2,
            z_precision=self._distFNode_prec
        ) for i in range(1, self._steps)]

        # 3. Conectar los nuevos nodos y factores al grafo local
        for i in range(1, self._steps):
            self._graph.connect(self._vnodes[i], fnodes[i-1])
            self._graph.connect(vnodes[i-1], fnodes[i-1])
            
        # 4. Registrar la conexión
        self._others[on] = {'a': other, 'v': vnodes, 'f': fnodes}

        # 5. Asegurar que el otro agente también se conecte (reciprocidad)
        if self._name not in other._others:
            other.setup_com(self)

        # 6. Enviar el estado inicial
        self.send(on)

    def send(self, name: str):
        """
        "Outbox" del agente. Envía la información de este agente
        (creencias y mensajes) al agente 'name'.
        """
        other = self._others[name]['a']
        
        for i in range(1, self._steps):
            vname = f'{self._name}.v{i}'
            v = self._vnodes[i]
            f: FNode = self._others[name]['f'][i-1] # El DistFNode

            # Determinar qué creencia (belief) enviar:
            try:
                v_mean = v.belief.mean.copy()
            except Exception:
                v_mean = np.zeros((v.dims, 1))

            if self._change == 1:
                # ESTADO 1 (FOLLOW_PATH): Envía una creencia de *alta confianza*
                # (covarianza baja) que sigue el path.
                # Esto hace que el agente se comporte como un obstáculo predecible
                # para los demás.
                small_cov = np.diag([0.01, 0.01, 0.1, 0.1])
                belief = Gaussian(v.dims, v_mean, small_cov)
            else:
                # ESTADO 2 o 3 (RECOVERY/RECONFIGURE): Envía la creencia *real*
                # (con incertidumbre). Esto permite la "negociación"
                # mutua de trayectorias.
                try:
                    belief = v.belief.copy()
                except Exception:
                    belief = Gaussian(v.dims, v_mean, np.diag([1]*v.dims))

            # Enviar la creencia de este VNode
            other.push_msg(('belief', self._name, vname, belief))

            # --- Enviar mensajes f2v y v2f del DistFNode ---
            # Esto es necesario para que el GBP funcione correctamente
            # a través de la conexión entre los dos agentes.
            
            # f2v: mensaje del DistFNode (f) al RemoteVNode (v)
            f2v = None
            try:
                # [0] es el edge al RemoteVNode, [1] es al VNode local
                f2v = f.edges[0].get_message_to(v) 
                if f2v is not None:
                    f2v = f2v.copy()
            except Exception:
                f2v = None
            other.push_msg(('f2v', self._name, vname, f2v))

            # v2f: mensaje del VNode local (v) al DistFNode (f)
            v2f = None
            try:
                v2f = f.edges[0].get_message_to(f)
                if v2f is not None:
                    v2f = v2f.copy()
            except Exception:
                v2f = None
            other.push_msg(('v2f', self._name, vname, v2f))

            # Lógica de caché eliminada

    def end_com(self, name: str):
        """
        Termina la conexión con el agente 'name'.
        Elimina los RemoteVNodes y DistFNodes asociados del grafo.
        """
        if name not in self._others:
            return

        vnodes = self._others[name]['v']
        fnodes = self._others[name]['f']
        
        # Eliminar nodos y factores del grafo
        for v in vnodes:
            try:
                self._graph.remove_node(v)
            except Exception:
                pass
        for f in fnodes:
            try:
                self._graph.remove_node(f)
            except Exception:
                pass
                
        # Eliminar del registro de conexiones
        other_dict = self._others.pop(name)
        
        # Notificar al otro agente para que también cierre la conexión
        try:
            other_dict['a'].end_com(self._name)
        except Exception:
            pass

    def export_simple_belief(self, vnode_index=0, pos_cov=0.01, vel_cov=0.1) -> Gaussian:
        """
        Función de utilidad para exportar la creencia de un VNode con
        una covarianza simplificada (útil para monitoreo o testing).
        """
        v = self._vnodes[vnode_index]
        m = v.belief.mean
        cov = np.diag([pos_cov, pos_cov, vel_cov, vel_cov])
        return Gaussian(v.dims, m, cov)

    def debug_state(self):
        """Imprime el estado de debugging del agente."""
        st = self.get_state()[0]
        mean_str = "None"
        if st is not None:
            mean_str = f"x={st[0]:.2f}, y={st[1]:.2f}, vx={st[2]:.2f}, vy={st[3]:.2f}"
        print(f"[DEBUG] Agent={self._name} state={self._change} idx={self._current_index} pos={mean_str}")


class Env:
    """
    Clase contenedora que gestiona múltiples agentes y orquesta
    los pasos de simulación (planificación y movimiento).
    """
    def __init__(self) -> None:
        self._agents: List['Agent'] = []

    def add_agent(self, a: 'Agent'):
        """Añade un agente al entorno."""
        if a not in self._agents:
            a._env = self
            self._agents.append(a)

    def find_near(self, this: 'Agent', range: float = 1000, max_num: int = -1) -> List['Agent']:
        """
        Encuentra agentes cercanos a un agente 'this' dentro de un 'range'.
        """
        agent_ds = []
        st_this = this.get_state()[0] # Estado planificado en t=0
        if st_this is None:
            return []

        for a in self._agents:
            if a is this:
                continue
            st_a = a.get_state()[0]
            if st_a is None:
                continue
            try:
                # Calcula distancia euclidiana 2D
                d = np.linalg.norm(st_a[:2] - st_this[:2])
            except Exception:
                continue
            if d < range:
                agent_ds.append((a, d))
        
        # Ordena por distancia
        agent_ds.sort(key=lambda ad: ad[1])
        if max_num > 0:
            agent_ds = agent_ds[:max_num]
        return [a for a, _ in agent_ds]

    def step_plan(self, iters=12):
        """
        Ejecuta el paso de planificación (GBP) para todos los agentes.
        Esto se hace de forma secuencial (sin concurrencia).
        """
        
        # 1. Actualizar Conexiones (Topología del Grafo)
        # Todos los agentes determinan quién está cerca y actualizan
        # sus grafos (añadiendo/quitando DistFNodes).
        for a in self._agents:
            try:
                a.step_connect()
            except Exception as e:
                print(f"[Env] step_connect error for {a.name}: {e}")

        # 2. Bucle de Propagación de Creencias (GBP)
        # Se ejecutan 'iters' iteraciones del algoritmo GBP.
        for i in range(iters):
            
            # 2a. Fase de Comunicación (Envío)
            # Todos los agentes envían sus creencias y mensajes actuales
            # (basados en el estado de la iteración i-1).
            for a in self._agents:
                try:
                    a.step_com()
                except Exception as e:
                    print(f"[Env] step_com error for {a.name}: {e}")
            
            # 2b. Fase de Propagación (Cálculo)
            # Todos los agentes calculan sus nuevas creencias (para la
            # iteración i) usando los mensajes recibidos en 2a.
            # (Nota: step_propagate() internamente omite el cálculo
            # si el agente está en modo FOLLOW_PATH).
            for a in self._agents:
                try:
                    a.step_propagate()
                except Exception as e:
                    print(f"[Env] step_propagate error for {a.name}: {e}")

    def step_move(self):
        """
        Ejecuta el paso de movimiento (la FSM) para todos los agentes.
        Esto actualiza el estado interno del agente y su máquina de estados.
        """
        for a in self._agents:
            try:
                a.step_move()
            except Exception as e:
                print(f"[Env] step_move error for {a.name}: {e}")