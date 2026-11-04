"""Entornos Gymnasium del proyecto."""

import gymnasium as gym

from .drone_env import ENVIRONMENT_VERSION, DroneEnv

if "DroneRescue-v0" not in gym.registry:
    gym.register(id="DroneRescue-v0", entry_point="envs.drone_env:DroneEnv")

__all__ = ["DroneEnv", "ENVIRONMENT_VERSION"]
