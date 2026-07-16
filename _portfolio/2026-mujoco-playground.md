---
title: "MuJoCo Playground -- Vision-Based RL for Panda Arm Grasping"
excerpt: "GPU-parallelized reinforcement learning for vision-based Panda arm pick-and-place."
collection: portfolio
date: 2026-06-01
---

- Reproduced Google DeepMind's MuJoCo Playground vision-based grasping task, training a GPU-parallelized PPO policy for a Franka Panda arm to pick and place objects from pixel observations.
- Configured the JAX/Brax training pipeline with Madrona-MJX rendering, scaling to thousands of parallel environments on a single GPU to reduce wall-clock training time from hours to minutes.
- Tuned reward shaping (reach, grasp, lift, and place terms) and domain randomization (object pose, lighting, and camera viewpoint) to improve policy robustness and sim-to-real transfer readiness.
