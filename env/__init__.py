"""Compatibilidad: importa :mod:`envs` para registrar DroneRescue-v0."""

from envs import ENVIRONMENT_VERSION, DroneEnv

__all__ = ["DroneEnv", "ENVIRONMENT_VERSION"]
