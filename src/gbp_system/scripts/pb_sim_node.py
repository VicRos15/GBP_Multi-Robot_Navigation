#!/usr/bin/env python3
import os
import math
import yaml
import rospy
import rospkg
import pybullet as p
import pybullet_data

from std_msgs.msg import Bool
from gbp_system.msg import RobotsState, RobotsTrajectory


class PyBulletSimNode:
    def __init__(self):
        rospy.init_node("pb_sim_node")

        self.yaml_path = rospy.get_param("~env_yaml", "")
        self.state_topic = rospy.get_param("~state_topic", "/robots_state")
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "/robots_trajectories")
        self.use_gui = rospy.get_param("~gui", True)

        if not self.yaml_path:
            raise ValueError("Falta el parámetro ~env_yaml para pb_sim_node")

        self.cfg = self._load_yaml(self.yaml_path)

        self.ready_pub = rospy.Publisher("/sim_ready", Bool, queue_size=1, latch=True)

        rospy.loginfo("[PB] Lanzando PyBullet...")
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")

        self.obstacles = []
        self.obstacle_models = {}
        self.robots = {}
        self.agent_colors = {}
        self.debug_trajectory_items = []
        self.debug_node_items = []

        self.default_turtlebot_urdf = self._get_default_turtlebot_urdf()
        self.default_chair_urdf = self._get_default_obstacle_urdf("chair.urdf")
        self.default_table_urdf = self._get_default_obstacle_urdf("table.urdf")

        self._build_obstacles_from_yaml()
        self._build_agents_from_yaml()
        self._build_agent_colors_from_yaml()

        rospy.Subscriber(self.state_topic, RobotsState, self.robots_state_callback, queue_size=1)
        rospy.Subscriber(
            self.trajectory_topic,
            RobotsTrajectory,
            self.robots_trajectory_callback,
            queue_size=1
        )

        rospy.loginfo(f"[PB] Nodo listo. Escuchando {self.state_topic}")


        rospy.sleep(5.0)
        self.ready_pub.publish(Bool(data=True))
        rospy.loginfo("[PB] Simulador listo. Publicado /sim_ready=True")

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def cm_to_m(self, value_cm: float) -> float:
        return float(value_cm) / 100.0

    def _get_pkg_path(self) -> str:
        rospack = rospkg.RosPack()
        return rospack.get_path("gbp_system")

    def _get_default_turtlebot_urdf(self) -> str:
        pkg_path = self._get_pkg_path()

        urdf_path = os.path.join(
            pkg_path,
            "src", "gbp_system", "AdamSim", "models", "robot",
            "rb1_base_description", "robots", "turtlebot.urdf"
        )

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"No encuentro el URDF por defecto del Turtlebot en: {urdf_path}")

        return urdf_path

    def _get_default_obstacle_urdf(self, filename: str) -> str:
        pkg_path = self._get_pkg_path()

        urdf_path = os.path.join(
            pkg_path,
            "src", "gbp_system", "AdamSim", "data", "models", filename
        )

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"No encuentro el URDF del obstáculo en: {urdf_path}")

        return urdf_path

    def _resolve_path(self, path_str: str) -> str:
        if not path_str:
            return ""

        if os.path.isabs(path_str):
            return path_str

        pkg_path = self._get_pkg_path()
        candidate = os.path.join(pkg_path, path_str)

        if os.path.exists(candidate):
            return candidate

        return path_str

    def _resolve_obstacle_model_path(self, obs: dict):
        name = obs["name"]

        urdf_path = self._resolve_path(obs.get("urdf_path", ""))
        if urdf_path:
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"[PB] No encuentro el URDF para '{name}' en: {urdf_path}")
            return "urdf", urdf_path

        sdf_path = self._resolve_path(obs.get("sdf_path", ""))
        if sdf_path:
            if not os.path.exists(sdf_path):
                raise FileNotFoundError(f"[PB] No encuentro el SDF para '{name}' en: {sdf_path}")
            return "sdf", sdf_path

        if name.startswith("office_chair"):
            return "urdf", self.default_chair_urdf

        if name.startswith("office_table"):
            return "urdf", self.default_table_urdf

        return None, ""

    def _build_obstacles_from_yaml(self):
        for obs in self.cfg.get("obstacles", []):
            name = obs["name"]
            otype = obs["type"]

            model_type, model_path = self._resolve_obstacle_model_path(obs)

            if model_path:
                obs_copy = dict(obs)

                if model_type == "urdf":
                    body_id = self._load_obstacle_urdf(obs_copy, model_path)
                elif model_type == "sdf":
                    body_id = self._load_obstacle_sdf(obs_copy, model_path)
                else:
                    raise ValueError(f"[PB] Tipo de modelo no soportado: {model_type}")

                self.obstacles.append(body_id)
                self.obstacle_models[name] = {
                    "body_id": body_id,
                    "model_type": model_type,
                    "model_path": model_path,
                }
                continue

            if otype == "circle":
                x_cm, y_cm = obs["position"]
                radius_cm = obs["radius"]
                body_id = self._add_circle_obstacle(name, x_cm, y_cm, radius_cm)
                self.obstacles.append(body_id)

            elif otype == "rectangle":
                cx_cm, cy_cm = obs["center"]
                width_cm, height_cm = obs["size"]
                body_id = self._add_rectangle_obstacle(name, cx_cm, cy_cm, width_cm, height_cm)
                self.obstacles.append(body_id)

            else:
                raise ValueError(f"Tipo de obstáculo no soportado en PyBullet: {otype}")

    def _build_agents_from_yaml(self):
        for agent_cfg in self.cfg.get("agents", []):
            name = agent_cfg["name"]
            start_xy = agent_cfg["start"]

            x0 = self.cm_to_m(start_xy[0])
            y0 = self.cm_to_m(start_xy[1])

            urdf_path = self._resolve_path(agent_cfg.get("urdf_path", ""))
            if not urdf_path or not os.path.exists(urdf_path):
                urdf_path = self.default_turtlebot_urdf

            yaw_offset_deg = float(agent_cfg.get("yaw_offset_deg", 0.0))
            yaw_offset_rad = math.radians(yaw_offset_deg)

            body_id = p.loadURDF(
                urdf_path,
                [x0, y0, 0.05],
                p.getQuaternionFromEuler([0.0, 0.0, yaw_offset_rad]),
                useFixedBase=False
            )

            self.robots[name] = {
                "body_id": body_id,
                "urdf_path": urdf_path,
                "yaw_offset_rad": yaw_offset_rad,
            }

            rospy.loginfo(
                f"[PB] Robot '{name}' cargado con URDF: {urdf_path} "
                f"| yaw_offset={yaw_offset_deg} deg"
            )

    def _get_obstacle_xy_cm(self, obs: dict):
        otype = obs["type"]
        if otype == "circle":
            return obs["position"]
        if otype == "rectangle":
            return obs["center"]
        raise ValueError(f"Tipo de obstáculo no soportado para modelo: {otype}")

    def _load_obstacle_urdf(self, obs: dict, urdf_path: str) -> int:
        name = obs["name"]

        x_cm, y_cm = self._get_obstacle_xy_cm(obs)
        x_m = self.cm_to_m(x_cm)
        y_m = self.cm_to_m(y_cm)

        z_m = float(obs.get("z", 0.0))
        yaw_deg = float(obs.get("yaw_deg", 0.0))
        yaw_rad = math.radians(yaw_deg)
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])

        body_id = p.loadURDF(
            urdf_path,
            [x_m, y_m, z_m],
            quat,
            useFixedBase=True
        )

        rospy.loginfo(
            f"[PB] Obstáculo URDF '{name}' cargado desde {urdf_path} "
            f"en ({x_cm}, {y_cm}) cm, z={z_m} m, yaw={yaw_deg} deg"
        )
        return body_id

    def _load_obstacle_sdf(self, obs: dict, sdf_path: str) -> int:
        name = obs["name"]

        x_cm, y_cm = self._get_obstacle_xy_cm(obs)
        x_m = self.cm_to_m(x_cm)
        y_m = self.cm_to_m(y_cm)

        z_m = float(obs.get("z", 0.0))
        yaw_deg = float(obs.get("yaw_deg", 0.0))
        yaw_rad = math.radians(yaw_deg)
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])

        # Muy importante para que las rutas relativas del SDF funcionen mejor
        sdf_dir = os.path.dirname(sdf_path)
        p.setAdditionalSearchPath(sdf_dir)

        body_ids = p.loadSDF(sdf_path)
        if not body_ids:
            raise RuntimeError(f"[PB] loadSDF no devolvió ningún body para: {sdf_path}")

        body_id = body_ids[0]

        p.resetBasePositionAndOrientation(
            body_id,
            [x_m, y_m, z_m],
            quat
        )

        rospy.loginfo(
            f"[PB] Obstáculo SDF '{name}' cargado desde {sdf_path} "
            f"en ({x_cm}, {y_cm}) cm, z={z_m} m, yaw={yaw_deg} deg"
        )
        return body_id

    def _add_circle_obstacle(
        self,
        name: str,
        x_cm: float,
        y_cm: float,
        radius_cm: float,
        height_m: float = 1.0,
        z_offset_m: float = 0.0,
    ) -> int:
        radius_m = self.cm_to_m(radius_cm)
        x_m = self.cm_to_m(x_cm)
        y_m = self.cm_to_m(y_cm)
        z_center = height_m / 2.0 + z_offset_m

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius_m,
            height=height_m
        )

        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius_m,
            length=height_m,
            rgbaColor=[0.75, 0.25, 0.25, 1.0]
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x_m, y_m, z_center]
        )

        rospy.loginfo(f"[PB] Obstáculo circular '{name}' en ({x_cm},{y_cm}) cm, r={radius_cm} cm")
        return body_id

    def _add_rectangle_obstacle(
        self,
        name: str,
        cx_cm: float,
        cy_cm: float,
        width_cm: float,
        height_cm: float,
        wall_height_m: float = 2.0,
        z_offset_m: float = 0.0,
    ) -> int:
        half_x = self.cm_to_m(width_cm) / 2.0
        half_y = self.cm_to_m(height_cm) / 2.0
        half_z = wall_height_m / 2.0

        x_m = self.cm_to_m(cx_cm)
        y_m = self.cm_to_m(cy_cm)
        z_m = half_z + z_offset_m

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z]
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z],
            rgbaColor=[0.7, 0.7, 0.7, 1.0]
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x_m, y_m, z_m]
        )

        rospy.loginfo(
            f"[PB] Obstáculo rectangular '{name}' centro=({cx_cm},{cy_cm}) cm, "
            f"size=({width_cm},{height_cm}) cm"
        )
        return body_id
    

    def _build_agent_colors_from_yaml(self):
        for agent_cfg in self.cfg.get("agents", []):
            name = agent_cfg["name"]
            color = agent_cfg.get("color", [50, 50, 220])
            self.agent_colors[name] = self._rgb255_to_rgb01(color)


    def _rgb255_to_rgb01(self, color):
        return [
            float(color[0]) / 255.0,
            float(color[1]) / 255.0,
            float(color[2]) / 255.0
        ]


    def _clear_debug_trajectories(self):
        for item_id in self.debug_trajectory_items:
            try:
                p.removeUserDebugItem(item_id)
            except Exception:
                pass

        for item_id in self.debug_node_items:
            try:
                p.removeUserDebugItem(item_id)
            except Exception:
                pass

        self.debug_trajectory_items = []
        self.debug_node_items = []

    def _add_debug_cross(self, x_m, y_m, z_m, color, size=0.035, width=2.0, life_time=0.08):
        id1 = p.addUserDebugLine(
            [x_m - size, y_m, z_m],
            [x_m + size, y_m, z_m],
            color,
            lineWidth=width,
            lifeTime=life_time
        )

        id2 = p.addUserDebugLine(
            [x_m, y_m - size, z_m],
            [x_m, y_m + size, z_m],
            color,
            lineWidth=width,
            lifeTime=life_time
        )

        self.debug_node_items.append(id1)
        self.debug_node_items.append(id2)

    def robots_trajectory_callback(self, msg: RobotsTrajectory):
        """
        Dibuja en PyBullet el horizonte completo generado por GBP.

        Cada robot se dibuja con su color del YAML.
        Las líneas representan los segmentos entre VNodes.
        Las cruces pequeñas representan los nodos del horizonte.
        """
        self._clear_debug_trajectories()

        z_m = 0.08

        for robot_traj in msg.robots:
            name = robot_traj.name
            color = self.agent_colors.get(name, [0.0, 0.0, 1.0])

            xs = list(robot_traj.x)
            ys = list(robot_traj.y)

            if len(xs) < 2 or len(ys) < 2:
                continue

            n = min(len(xs), len(ys))

            # Dibujar segmentos del horizonte GBP
            for i in range(n - 1):
                x0 = self.cm_to_m(xs[i])
                y0 = self.cm_to_m(ys[i])
                x1 = self.cm_to_m(xs[i + 1])
                y1 = self.cm_to_m(ys[i + 1])

                line_id = p.addUserDebugLine(
                    [x0, y0, z_m],
                    [x1, y1, z_m],
                    color,
                    lineWidth=3.0,
                    lifeTime=3.5
                )

                self.debug_trajectory_items.append(line_id)

            # Dibujar VNodes como cruces pequeñas
            for i in range(n):
                x_m = self.cm_to_m(xs[i])
                y_m = self.cm_to_m(ys[i])

                self._add_debug_cross(
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m + 0.015,
                    color=color,
                    size=0.025,
                    width=1.5,
                    life_time=0
                )

    def robots_state_callback(self, msg: RobotsState):
        for robot_state in msg.robots:
            name = robot_state.name

            if name not in self.robots:
                rospy.logwarn_throttle(2.0, f"[PB] Robot '{name}' no existe en simulación")
                continue

            body_id = self.robots[name]["body_id"]

            x_m = self.cm_to_m(robot_state.x)
            y_m = self.cm_to_m(robot_state.y)
            yaw = math.atan2(robot_state.vy, robot_state.vx)

            if abs(robot_state.vx) < 1e-6 and abs(robot_state.vy) < 1e-6:
                yaw = 0.0

            yaw_offset = self.robots[name].get("yaw_offset_rad", 0.0)

            quat = p.getQuaternionFromEuler(
                [0.0, 0.0, yaw]
            )

            p.resetBasePositionAndOrientation(
                body_id,
                [x_m, y_m, 0.05],
                quat
            )


        p.stepSimulation()

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = PyBulletSimNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass