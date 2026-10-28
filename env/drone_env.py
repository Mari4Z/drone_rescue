"""Entorno 3D del dron: cuadrícula de 5 m (20 x 20 x 12 celdas = 100 x 100 x 60 m)."""
from collections import Counter

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
R_LOW = -0.5  # penalización por paso volando por debajo de la altura de crucero (fuera de A y B)


def dist_to_goal(p):
    return sum(abs(a - b) for a, b in zip(p, GOAL))


def build_obstacles(columns=COLUMNS):
    grid = np.zeros((NX, NY, NZ), dtype=bool)
    for x0, x1, y0, y1, h in columns:
        grid[x0:x1 + 1, y0:y1 + 1, : h // CELL] = True  # 55 m -> 11 celdas (z=0..10)
    return grid


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=400, columns=COLUMNS, min_cruise_m=0):
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.MultiDiscrete([NX, NY, NZ])
        self.columns = columns
        self.obstacles = build_obstacles(columns)
        self.max_steps = max_steps
        self.min_cruise_z = min_cruise_m // CELL  # 0 = sin penalización por volar bajo
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

    def blocked(self, x, y, z):
        """Celda prohibida: fuera del mapa, columna, o suelo (z=0) salvo en A y B.
        Así el dron tiene que despegar en vertical desde A y aterrizar en vertical sobre B."""
        if not (0 <= x < NX and 0 <= y < NY and 0 <= z < NZ):
            return True
        if z == 0 and (x, y) not in (START[:2], GOAL[:2]):
            return True
        return bool(self.obstacles[x, y, z])

    def step(self, action):
        dx, dy, dz = ACTIONS[action]
        x, y, z = self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz
        self.steps += 1

        if self.blocked(x, y, z):
            return self._obs(), R_CRASH, True, False, {"result": "crash"}

        reward = R_STEP + R_PROGRESS * (dist_to_goal(self.pos) - dist_to_goal((x, y, z)))
        if z < self.min_cruise_z and (x, y) not in (START[:2], GOAL[:2]):
            reward += R_LOW
        self.pos = (x, y, z)
        if self.pos == GOAL:
            return self._obs(), R_GOAL, True, False, {"result": "goal"}
        if self.steps >= self.max_steps:
            return self._obs(), reward, False, True, {"result": "timeout"}
        return self._obs(), reward, False, False, {"result": "flying"}


# V1 "dummy": mismo mapa, A, B y reglas, pero con una sola columna en el centro.
# Sirve como entorno base para validar el pipeline antes de usar las 5 columnas.
DUMMY_COLUMNS = [(8, 11, 8, 11, 55)]


# V2: mapa aleatorio en cada intento + sensores, para que el agente generalice a mapas nuevos.
SENSOR_RANGE = 8  # celdas (40 m) que alcanza cada sensor de distancia
SAFE_ZONE = 2     # celdas libres alrededor de A y B donde no se generan columnas


def random_columns(rng, n=(5, 25)):
    """Entre n[0] y n[1] columnas: desde mapas abiertos hasta mapas densos tipo ciudad.
    En los densos rodear deja de ser gratis y compensa subir para sobrevolar las columnas bajas."""
    count = int(rng.integers(n[0], n[1] + 1))
    cols = []
    while len(cols) < count:
        w, d = rng.integers(2, 6, size=2)  # 2..5 celdas (10..25 m) de lado
        x0, y0 = int(rng.integers(0, NX - w + 1)), int(rng.integers(0, NY - d + 1))
        x1, y1 = x0 + int(w) - 1, y0 + int(d) - 1
        near = lambda p: x0 - SAFE_ZONE <= p[0] <= x1 + SAFE_ZONE and y0 - SAFE_ZONE <= p[1] <= y1 + SAFE_ZONE
        if near(START) or near(GOAL):
            continue
        h = int(rng.integers(3, 12)) * CELL  # 15..55 m
        cols.append((x0, x1, y0, y1, h))
    return cols


class DroneSensorEnv(DroneEnv):
    """Observación continua (46 valores, todo en [-1, 1]):
      - posición normalizada (3)
      - vector hacia B normalizado (3)
      - 6 sensores de distancia (+x, -x, +y, -y, arriba, abajo), 1 = libre hasta SENSOR_RANGE (6)
      - ocupación del cubo 3x3x3 alrededor del dron, 1 = bloqueado (27)
      - memoria: visitas a la celda actual y a las 6 vecinas en este intento, 1 = 3 o más (7).
        Sin ella la política es determinista y puede quedarse oscilando entre dos celdas.
    Con columns=None genera un mapa aleatorio en cada reset; con una lista usa ese mapa fijo."""

    def __init__(self, max_steps=250, columns=None, n_columns=(5, 25), min_cruise_m=20):
        super().__init__(max_steps=max_steps, columns=columns or [], min_cruise_m=min_cruise_m)
        self.random_map = columns is None
        self.n_columns = n_columns
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(46,), dtype=np.float32)
        self.visits = Counter()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_map:
            self.columns = random_columns(self.np_random, self.n_columns)
            self.obstacles = build_obstacles(self.columns)
        self.visits = Counter({self.pos: 1})
        return self._obs(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self.visits[self.pos] += 1
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        x, y, z = self.pos
        size = np.array([NX - 1, NY - 1, NZ - 1], dtype=np.float32)
        pos = np.array(self.pos, dtype=np.float32) / size
        to_goal = (np.array(GOAL, dtype=np.float32) - self.pos) / size
        rays = []
        for dx, dy, dz in ACTIONS:
            d = 1
            while d <= SENSOR_RANGE and not self.blocked(x + dx * d, y + dy * d, z + dz * d):
                d += 1
            rays.append((d - 1) / SENSOR_RANGE)
        cube = [float(self.blocked(x + i, y + j, z + k))
                for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        memory = [min(self.visits[self.pos], 3) / 3] + [
            min(self.visits[(x + dx, y + dy, z + dz)], 3) / 3 for dx, dy, dz in ACTIONS]
        return np.concatenate([pos, to_goal, rays, cube, memory]).astype(np.float32)
