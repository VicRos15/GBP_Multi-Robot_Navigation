#!/usr/bin/env python3
import rospy
import os
import yaml
import numpy as np
import pygame as pg
import pygame.locals as pgl

from std_msgs.msg import Bool
from gbp_system.motion.obstacle import ObstacleMap
from gbp_system.motion.agent_FSM import Agent, Env
from gbp_system.msg import RobotState, RobotsState, RobotTrajectory, RobotsTrajectory


def pairwise(seq):
    return zip(seq[:-1], seq[1:])


class GBPPlannerNode:
    def __init__(self):
        rospy.init_node("gbp_planner_node")

        self.yaml_path = rospy.get_param("~env_yaml", "")
        self.sim = rospy.get_param("~sim", "pg").lower()
        self.state_topic = rospy.get_param("~state_topic", "/robots_state")
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "/robots_trajectories")

        if not self.yaml_path:
            raise ValueError("Falta el parámetro ~env_yaml")

        self.cfg = self._load_yaml(self.yaml_path)

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

        self._build_obstacles()
        self._build_agents()

        self.pub_states = rospy.Publisher(self.state_topic, RobotsState, queue_size=1)
        self.pub_trajectories = rospy.Publisher(
            self.trajectory_topic,
            RobotsTrajectory,
            queue_size=1
        )

        self.use_pygame = (self.sim == "pg")
        self.screen = None
        self.clock = None

        self.sim_ready = False

        if self.sim == "pb":
            rospy.Subscriber("/sim_ready", Bool, self._sim_ready_cb, queue_size=1)

        if self.use_pygame:
            pg.init()
            self.screen = pg.display.set_mode((self.screen_w, self.screen_h))
            pg.display.set_caption("GBP Planner Node")
            self.clock = pg.time.Clock()

        rospy.loginfo(f"[GBP] Nodo iniciado | sim={self.sim} | env={self.yaml_path}")

        if self.sim == "pb":
            self._wait_for_sim_ready()

    def _sim_ready_cb(self, msg: Bool):
        self.sim_ready = bool(msg.data)

    def _wait_for_sim_ready(self):
        rospy.loginfo("[GBP] Esperando a que PyBullet esté listo...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.sim_ready:
            rate.sleep()
        if not rospy.is_shutdown():
            rospy.loginfo("[GBP] PyBullet listo. Arrancando planner.")

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
            return self._generate_curved_path(start, goal, num_points=num_points, arc_height=arc_height)

        raise ValueError(f"Tipo de path no soportado: {path_type}")

    def _build_obstacles(self):
        for obs in self.cfg.get("obstacles", []):
            name = obs["name"]
            otype = obs["type"]

            if otype == "circle":
                x, y = obs["position"]
                r = obs["radius"]
                self.omap.set_circle(name, x, y, r)

            elif otype == "rectangle":
                cx, cy = obs["center"]
                w, h = obs["size"]
                x_min = cx - w / 2.0
                x_max = cx + w / 2.0
                y_min = cy - h / 2.0
                y_max = cy + h / 2.0
                self.omap.set_rectangle(name, x_min, y_min, x_max, y_max)

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

    def _draw_obstacles(self):
        obstacle_color = (180, 60, 60)

        for o in self.omap.objects.values():
            if o["type"] == "rectangle":
                x0, y0 = self.w2s(o["x_min"], o["y_min"])
                w = int((o["x_max"] - o["x_min"]) * self.px_per_cm)
                h = int((o["y_max"] - o["y_min"]) * self.px_per_cm)
                pg.draw.rect(self.screen, obstacle_color, (x0, y0, w, h), 0)
                pg.draw.rect(self.screen, obstacle_color, (x0, y0, w, h), 2)

            elif o["type"] == "circle":
                sx, sy = self.w2s(o["centerx"], o["centery"])
                rr = int(o["radius"] * self.px_per_cm)
                pg.draw.circle(self.screen, obstacle_color, (sx, sy), rr, 0)
                pg.draw.circle(self.screen, obstacle_color, (sx, sy), rr, 2)

    def _draw_agents_and_paths(self):
        for agent in self.env._agents:
            state = agent.get_state()
            if not state or state[0] is None:
                continue

            color = self.agent_colors.get(agent.name, (0, 0, 220))

            x, y, _, _ = state[0]
            sx, sy = self.w2s(x, y)
            real_radius = self.agent_real_radius.get(agent.name, agent.r)
            pg.draw.circle(self.screen, color, (sx, sy), int(real_radius * self.px_per_cm), 2)

            for s0, s1 in pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                pg.draw.line(self.screen, color, self.w2s(x0, y0), self.w2s(x1, y1), 2)

    def _draw_goals(self):
        for name, goal in self.agent_goals.items():
            color = self.agent_colors.get(name, (0, 0, 0))

            gx, gy = goal
            sx, sy = self.w2s(gx, gy)

            size = 6
            pg.draw.line(self.screen, color, (sx - size, sy), (sx + size, sy), 2)
            pg.draw.line(self.screen, color, (sx, sy - size), (sx, sy + size), 2)


    def _publish_robot_trajectories(self):
        msg = RobotsTrajectory()
        msg.robots = []

        for agent in self.env._agents:
            state = agent.get_state()

            if not state:
                continue

            traj_msg = RobotTrajectory()
            traj_msg.name = agent.name
            traj_msg.x = []
            traj_msg.y = []

            for s in state:
                if s is None:
                    continue

                x, y, _, _ = s
                traj_msg.x.append(float(x))
                traj_msg.y.append(float(y))

            msg.robots.append(traj_msg)

        self.pub_trajectories.publish(msg)

    def _publish_robot_states(self):
        msg = RobotsState()
        msg.robots = []

        for agent in self.env._agents:
            state = agent.get_state()
            if not state or state[0] is None:
                continue

            x, y, vx, vy = state[0]

            robot_msg = RobotState()
            robot_msg.name = agent.name
            robot_msg.x = float(x)
            robot_msg.y = float(y)
            robot_msg.vx = float(vx)
            robot_msg.vy = float(vy)

            msg.robots.append(robot_msg)

        self.pub_states.publish(msg)

    def _handle_pygame_events(self):
        if not self.use_pygame:
            return True

        for event in pg.event.get():
            if event.type == pgl.QUIT:
                return False
            if event.type == pgl.KEYDOWN and event.key == pgl.K_ESCAPE:
                return False

        return True

    def _output_pg(self):
        self.screen.fill(self.bg_color)
        self._draw_obstacles()
        self._draw_goals()
        self._draw_agents_and_paths()
        pg.display.flip()
        self.clock.tick(self.fps)

    def _publish_pb_plan(self):
        self._publish_robot_trajectories()


    def _publish_pb_state(self):
        self._publish_robot_states()

    def run(self):
        rate = rospy.Rate(self.fps)

        while not rospy.is_shutdown():
            if not self._handle_pygame_events():
                break

            self.env.step_plan(iters=self.step_plan_iters)

            if self.sim == "pg":
                self._output_pg()
                self.env.step_move()

            elif self.sim == "pb":
                # Igual que Pygame: primero se muestra el estado/horizonte actual
                self._publish_robot_states()
                self._publish_robot_trajectories()

                # Después se avanza la FSM para el siguiente ciclo
                self.env.step_move()

            else:
                rospy.logwarn_throttle(
                    5.0,
                    f"[GBP] sim desconocido: {self.sim}. Publicando por ROS."
                )

                self.env.step_move()
                self._publish_robot_states()

            if self.sim != "pg":
                rate.sleep()

        if self.use_pygame:
            pg.quit()


def main():
    try:
        node = GBPPlannerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()