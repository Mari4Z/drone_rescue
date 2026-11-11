"""Política PPO (Stable-Baselines3) para las familias drone-rescue-v2 y drone-rescue-v3."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agents.dqn import FEATURES_VERSION, N_FEATURES, make_features
from envs.drone_env import DroneEnv
from envs.fast_drone_env import FastDroneEnv
from envs.stabilized_drone_env import ACTION_NAMES, FAMILY as STABILIZED_FAMILY, StabilizedDroneEnv

FORCE_ACTIONS = ["+x", "-x", "+y", "-y", "subir", "bajar", "soltar"]


class FeatureObservation(gym.ObservationWrapper):
    """Entrega a la red la observación escalada por ``make_features``."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (N_FEATURES,), np.float32)

    def observation(self, observation):
        return make_features(observation)


def make_env(config_path: str | Path, family: str, *, fast: bool = False) -> gym.Env:
    """Ambiente de una familia. ``fast`` solo para entrenar; da las mismas transiciones (ver tests)."""
    base = FastDroneEnv(config_path) if fast else DroneEnv(config_path)
    if family == STABILIZED_FAMILY:
        base = StabilizedDroneEnv(base)
    elif family != "drone-rescue-v2":
        raise ValueError(f"Familia no soportada: {family}")
    return FeatureObservation(base)


def input_contract(family: str) -> dict:
    """Contrato de entrada/salida que acompaña a cada modelo (sección 11 de la guía)."""
    stabilized = family == STABILIZED_FAMILY
    return {
        "family": family,
        "method": "make_features",
        "features_version": FEATURES_VERSION,
        "observation_dimensions": 19,
        "state_dimensions": N_FEATURES,
        "action_dimensions": len(ACTION_NAMES) if stabilized else len(FORCE_ACTIONS),
        "actions": ACTION_NAMES if stabilized else FORCE_ACTIONS,
        "action_unit": "velocidad objetivo (m/s), con estabilizador" if stabilized else "fuerza por eje (N)",
        "features": [
            {"indices": "0:3", "from": "vector al objetivo (m)", "scale": "/100"},
            {"indices": "3:6", "from": "vector al objetivo (m)", "scale": "/10, clip [-1, 1]"},
            {"indices": "6:9", "from": "velocidad (m/s)", "scale": "/5"},
            {"indices": "9:12", "from": "velocidad (m/s)", "scale": "clip [-1, 1]"},
            {"indices": "12", "from": "altura (m)", "scale": "/60"},
            {"indices": "13", "from": "altura (m)", "scale": "/5, min 1"},
            {"indices": "14:20", "from": "sensores (m)", "scale": "/20"},
            {"indices": "20:26", "from": "batería, carga y etapa", "scale": "sin cambio"},
        ],
    }


class PPOPolicy:
    """Modelo congelado, sin exploración. Misma interfaz que ``pipelines.evaluate``."""

    def __init__(self, model_path: str | Path):
        self.model = PPO.load(str(model_path), device="cpu")

    def act(self, features, config: dict | None = None) -> int:
        del config
        return int(self.model.predict(features, deterministic=True)[0])
