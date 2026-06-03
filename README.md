# Multi-Robot Navigation using Gaussian Belief Propagation

Standalone Python/Pygame implementation of a multi-robot navigation framework based on **Gaussian Belief Propagation (GBP)**.

This branch contains the core implementation used for algorithm development, testing, benchmark execution, metric generation and comparison against alternative local planners such as **ORCA+** and **Predictive DWA**.

---

## Overview

The system uses a local factor graph representation to plan future robot states. Each agent switches between path following, GBP-based reconfiguration and recovery behaviour using a finite state machine.

![Agent navigation FSM](docs/images/fsm_gbp.png)

---

## Benchmark Environments

The benchmark includes several multi-robot navigation scenarios with static obstacles and different interaction patterns between agents.

![Benchmark environments](docs/images/all_environments.png)

---

## Pygame Demonstration

The standalone implementation includes a Pygame visualisation of the GBP planner, showing path following, reconfiguration and recovery stages.

![Pygame GBP demonstration](docs/images/pg_demonstration.png)

---

## Features

- Multi-agent navigation in 2D environments.
- Gaussian Belief Propagation planner.
- Finite State Machine for path following, reconfiguration and recovery.
- Static obstacle avoidance.
- Inter-agent collision avoidance.
- Pygame-based visualisation.
- YAML benchmark environments.
- Trajectory logging.
- Metric generation and plotting tools.
- Comparison against ORCA+ and Predictive DWA.

---

## Requirements

Tested on:

- Ubuntu 20.04.6 LTS
- Python 3

Main Python dependencies:

```bash
pip install numpy pygame matplotlib pandas pyyaml