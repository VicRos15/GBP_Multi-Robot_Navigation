# Multi-Robot Navigation using Gaussian Belief Propagation

Standalone Python/Pygame implementation of a multi-robot navigation framework based on Gaussian Belief Propagation.

This branch contains the core implementation used for algorithm development, testing, benchmark execution, metric generation and comparison against ORCA+ and Predictive DWA.

## Features

- Multi-agent navigation in 2D environments.
- Gaussian Belief Propagation planner.
- Finite State Machine for path following, reconfiguration and recovery.
- Static obstacle avoidance.
- Inter-agent collision avoidance.
- Pygame visualisation.
- YAML benchmark environments.
- Metric generation and plotting tools.

## Requirements

- Ubuntu 20.04.6 LTS
- Python 3
- numpy
- pygame
- matplotlib
- pandas
- pyyaml

## Running an example

```bash
python test_planners/test_gbp.py config/env1/env1_01.yaml
```

## Branches

- pygame-core: standalone Python/Pygame implementation.
- ros-pybullet: ROS Noetic and PyBullet implementation.
