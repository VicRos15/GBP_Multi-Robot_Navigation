#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from typing import Dict, Any


def split_env_name(env_name: str):
    """
    env1_01 -> env_type=env1, test_id=01, scenario_id=env1_01
    env2_07 -> env_type=env2, test_id=07, scenario_id=env2_07
    """
    match = re.match(r"^(.*)_(\d+)$", env_name)

    if match:
        env_type = match.group(1)
        test_id = match.group(2)
    else:
        env_type = env_name
        test_id = "01"

    scenario_id = f"{env_type}_{test_id}"

    return env_type, test_id, scenario_id


class MetricsLogger:
    def __init__(
        self,
        algorithm: str,
        env_name: str,
        run_id: str = "run_001",
        results_dir: str = "results",
    ):
        self.algorithm = algorithm
        self.env_name = env_name
        self.env_type, self.test_id, self.scenario_id = split_env_name(env_name)
        self.run_id = run_id
        self.results_dir = results_dir

        self.raw_dir = os.path.join(
            results_dir,
            "raw",
            algorithm,
            self.env_type
        )
        os.makedirs(self.raw_dir, exist_ok=True)

        self.trajectory_path = os.path.join(
            self.raw_dir,
            f"{self.run_id}_trajectory.csv"
        )

        self._trajectory_file = open(self.trajectory_path, "w", newline="")
        self._trajectory_writer = csv.DictWriter(
            self._trajectory_file,
            fieldnames=[
                "algorithm",
                "env",
                "env_type",
                "test_id",
                "scenario_id",
                "run_id",
                "t_sim",
                "t_real",
                "agent",
                "x",
                "y",
                "vx",
                "vy",
                "reached",
            ],
        )

        self._trajectory_writer.writeheader()

    def log_state(
        self,
        t_sim: float,
        t_real: float,
        agent: str,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        reached: bool = False,
    ):
        self._trajectory_writer.writerow({
            "algorithm": self.algorithm,
            "env": self.env_name,
            "env_type": self.env_type,
            "test_id": self.test_id,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "t_sim": float(t_sim),
            "t_real": float(t_real),
            "agent": agent,
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "reached": bool(reached),
        })

    def close(self):
        self._trajectory_file.close()