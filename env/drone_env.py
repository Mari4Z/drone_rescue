"""Entorno 3D del dron: cuadrícula de 5 m (20 x 20 x 12 celdas = 100 x 100 x 60 m)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

CELL = 5  # metros por celda
NX, NY, NZ = 20, 20, 12

START = (0, 0, 0)    # lugar A
GOAL = (19, 19, 0)   # lugar B (esquina opuesta)

# Columnas fijas: (x_min, x_max, y_min, y_max, alto_en_metros); rangos inclusivos en celdas
COLUMNS = [
    (4, 6, 2, 4, 55),
    (10, 12, 6, 8, 50),
    (3, 5, 12, 14, 55),
    (14, 16, 10, 12, 45),
    (9, 11, 15, 17, 55),
]

# 0..5: +x, -x, +y, -y, subir, bajar
ACTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

R_GOAL, R_CRASH = 100.0, -100.0
R_STEP = -1.0  # penalización por cada paso
R_PROGRESS = 1.0  # bonus por celda que se acerca a B (distancia Manhattan); 0 lo desactiva


def dist_to_goal(p):
    return sum(abs(a - b) for a, b in zip(p, GOAL))


def build_obstacles():
    grid = np.zeros((NX, NY, NZ), dtype=bool)
    for x0, x1, y0, y1, h in COLUMNS:
        grid[x0:x1 + 1, y0:y1 + 1, : h // CELL] = True  # 55 m -> 11 celdas (z=0..10)
    return grid


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=400):
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.MultiDiscrete([NX, NY, NZ])
        self.obstacles = build_obstacles()
        self.max_steps = max_steps
        self.pos = START
        self.steps = 0

    def _obs(self):
        return np.array(self.pos, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        """Tras un choque (o éxito) el dron vuelve siempre a A."""
        super().reset(seed=seed)
        self.pos = START
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        dx, dy, dz = ACTIONS[action]
        x, y, z = self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz
        self.steps += 1

        out = not (0 <= x < NX and 0 <= y < NY and 0 <= z < NZ)
        # El suelo (z=0) solo es transitable en A y B: hay que despegar en vertical
        # desde A y aterrizar en vertical sobre B.
        on_ground = not out and z == 0 and (x, y) not in (START[:2], GOAL[:2])
        if out or on_ground or self.obstacles[x, y, z]:
            return self._obs(), R_CRASH, True, False, {"result": "crash"}

        reward = R_STEP + R_PROGRESS * (dist_to_goal(self.pos) - dist_to_goal((x, y, z)))
        self.pos = (x, y, z)
        if self.pos == GOAL:
            return self._obs(), R_GOAL, True, False, {"result": "goal"}
        if self.steps >= self.max_steps:
            return self._obs(), reward, False, True, {"result": "timeout"}
        return self._obs(), reward, False, False, {"result": "flying"}
