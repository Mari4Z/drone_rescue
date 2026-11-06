"""Familia drone-rescue-v3: el agente ordena velocidades objetivo y un estabilizador aplica fuerzas.

Es la opción que sugiere la guía del proyecto ("usar estabilización convencional"). La física, las
reglas, la recompensa y la observación son las de ``DroneEnv``; cambian las acciones, la frecuencia
de decisión y que la entrega se activa por proximidad y estabilidad, en vez de ser una acción propia.
Por eso los modelos no son compatibles con drone-rescue-v2.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

FAMILY = "drone-rescue-v3"
STABILIZER_VERSION = "1.0.0"
SUBSTEPS = 2          # pasos físicos por decisión: decide cada 1 s con dt = 0.5 s
FAST_MPS = 4.0
SLOW_MPS = 1.0
VERTICAL_MPS = 2.0
HOLD_GAIN = 0.5       # velocidad (m/s) ordenada por metro de error al sostener posición

# Acciones: velocidad objetivo (vx, vy, vz) en m/s, o None para HOLD.
VELOCITY_COMMANDS = (
    (FAST_MPS, 0, 0), (-FAST_MPS, 0, 0), (0, FAST_MPS, 0), (0, -FAST_MPS, 0),
    (SLOW_MPS, 0, 0), (-SLOW_MPS, 0, 0), (0, SLOW_MPS, 0), (0, -SLOW_MPS, 0),
    (0, 0, VERTICAL_MPS), (0, 0, -VERTICAL_MPS),
)
HOLD = len(VELOCITY_COMMANDS)
N_ACTIONS = HOLD + 1
ACTION_NAMES = [
    "+x rápido", "-x rápido", "+y rápido", "-y rápido", "+x lento", "-x lento", "+y lento", "-y lento",
    "subir", "bajar", "sostener",
]


def thrust_for(velocity: np.ndarray, target: np.ndarray) -> int:
    """Acción de fuerza (0..5 de DroneEnv) que más acerca la velocidad a la objetivo."""
    error = target - velocity
    axis = int(np.argmax(np.abs(error)))
    return axis * 2 + (0 if error[axis] >= 0 else 1)


class StabilizedDroneEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._anchor = None

    def reset(self, **kwargs):
        self._anchor = None
        return self.env.reset(**kwargs)

    def _target(self, action: int) -> np.ndarray:
        if action == HOLD:
            if self._anchor is None:
                self._anchor = self.unwrapped.position.copy()
            return HOLD_GAIN * (self._anchor - self.unwrapped.position)
        self._anchor = None
        return np.asarray(VELOCITY_COMMANDS[action], dtype=np.float64)

    def _can_deliver(self) -> bool:
        drone = self.unwrapped
        return (
            drone.stage == "delivery"
            and drone.has_supply
            and drone._inside_delivery_zone(drone.position)
            and np.linalg.norm(drone.velocity) <= float(drone._world["delivery_max_speed_mps"])
        )

    def step(self, action):
        action = int(action)
        drone = self.unwrapped
        total = 0.0
        for _ in range(SUBSTEPS):
            # Entrega por proximidad y estabilidad: solo al sostener posición, dentro de la zona y lento
            inner = drone._release_action if action == HOLD and self._can_deliver() else thrust_for(drone.velocity, self._target(action))
            observation, reward, terminated, truncated, info = self.env.step(inner)
            total += float(reward)
            if terminated or truncated:
                break
        return observation, total, terminated, truncated, info
