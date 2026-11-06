"""Evaluación de un agente en mapas que no ha visto, comparando con la ruta óptima de A*."""
from collections import Counter

import gymnasium as gym
import numpy as np

from agents.grid.astar import astar
from envs.grid_drone_env import CELL

EVAL_SEED = 100_000  # semillas de evaluación, lejos de las de entrenamiento


def run_episode(e, policy, seed=None):
    """Ejecuta un intento. Devuelve (ruta de celdas, resultado)."""
    obs, _ = e.reset(seed=seed)
    path, done = [e.unwrapped.pos], False
    while not done:
        obs, _, terminated, truncated, info = e.step(policy(obs))
        path.append(e.unwrapped.pos)
        done = terminated or truncated
    return path, info["result"]


def evaluate(policy, env_id="DroneRescueGridRandom-v2", episodes=100, seed=EVAL_SEED):
    """policy: función obs -> acción, o un agente con .act(obs)."""
    policy = getattr(policy, "act", policy)
    e = gym.make(env_id)
    results, ratios, cruise = Counter(), [], []
    for i in range(episodes):
        path, result = run_episode(e, policy, seed=seed + i)
        results[result] += 1
        if result == "goal":
            best = astar(e.unwrapped)
            ratios.append((len(path) - 1) / (len(best) - 1))
            cruise.append(np.median([p[2] for p in path]) * CELL)
    return {
        "success": results["goal"] / episodes,
        "results": dict(results),
        "len_ratio": float(np.mean(ratios)) if ratios else None,  # 1.0 = igual que A*
        "cruise_m": float(np.mean(cruise)) if cruise else None,
    }


def fmt(m):
    s = f"éxito {m['success']:.0%}"
    if m["len_ratio"]:
        s += f" | ruta {m['len_ratio']:.2f}x A* | altura mediana {m['cruise_m']:.0f} m"
    return s


def score(m):
    """Para elegir el mejor modelo: primero la tasa de éxito, después rutas más cortas."""
    return (m["success"], -(m["len_ratio"] or 9))
