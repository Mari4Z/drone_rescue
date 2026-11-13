"""Evaluación de un agente en mapas que no ha visto, comparando con la ruta óptima de A*."""
from collections import Counter

import gymnasium as gym
import numpy as np

from agents.grid.astar import astar
from envs.grid_drone_env import CELL

EVAL_SEED = 100_000  # semillas de evaluación, lejos de las de entrenamiento


def run_episode(e, policy, seed=None, trace=None):
    """Ejecuta un intento. Devuelve (ruta de celdas, resultado).

    Si se pasa una lista en trace, añade el info de cada paso (batería, ráfagas en grid-drone-v2)."""
    obs, _ = e.reset(seed=seed)
    path, done = [e.unwrapped.pos], False
    while not done:
        obs, _, terminated, truncated, info = e.step(policy(obs))
        path.append(e.unwrapped.pos)
        if trace is not None:
            trace.append(info)
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


def storm_episode(e, policy, seed):
    """Un intento en DroneStormEnv con todo lo que se reporta por episodio."""
    trace = []
    path, result = run_episode(e, policy, seed=seed, trace=trace)
    env = e.unwrapped
    used = env.capacity - env.battery
    best = env.route_energy(env.best_route)
    return {
        "seed": int(seed), "result": result, "cause": trace[-1].get("cause") if trace else None,
        "steps": len(path) - 1, "columns": len(env.columns), "goal": list(env.goal),
        "wind": round(env.wind, 4), "wind_dir": [round(float(v), 4) for v in env.wind_dir],
        "capacity": round(env.capacity, 3), "energy_used": round(used, 3),
        "battery_left_frac": round(max(env.battery, 0.0) / env.capacity, 4),
        "best_route_energy": round(best, 3), "pushes": env.pushes,
        "max_altitude_m": max(p[2] for p in path) * CELL,
        **({"birds": len(getattr(env, "bird_history", [[]])[0]), "lines": len(env.lines)} if hasattr(env, "lines") else {}),
    }


def evaluate_storm(policy, env_id="DroneRescueGridStorm-v3", episodes=100, seed=EVAL_SEED, make_policy=None):
    """Métricas de misión en grid-drone-v2. make_policy(env) crea una política por ambiente (A*)."""
    e = gym.make(env_id)
    act = make_policy(e).act if make_policy else getattr(policy, "act", policy)
    rows = [storm_episode(e, act, seed + i) for i in range(episodes)]
    return {"episodes": rows, "metrics": storm_metrics(rows)}


def storm_metrics(rows):
    causes = Counter(r["result"] + (f"_{r['cause']}" if r["cause"] in ("viento", "ave") else "") for r in rows)
    goals = [r for r in rows if r["result"] == "goal"]
    n = len(rows)
    return {
        "episodes": n,
        "success_rate": causes["goal"] / n,
        "cause_counts": dict(causes),
        "crash_rate": (causes["crash"] + causes["crash_viento"] + causes["crash_ave"]) / n,
        "wind_crash_rate": causes["crash_viento"] / n,
        "bird_crash_rate": causes["crash_ave"] / n,
        "battery_rate": causes["battery"] / n,
        "timeout_rate": causes["timeout"] / n,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        # Solo en los intentos que llegan: 1.0 = la energía de la mejor ruta sin ráfagas
        "energy_ratio": float(np.mean([r["energy_used"] / r["best_route_energy"] for r in goals])) if goals else None,
        "battery_left_frac": float(np.mean([r["battery_left_frac"] for r in goals])) if goals else None,
        "mean_pushes": float(np.mean([r["pushes"] for r in rows])),
    }


def fmt(m):
    s = f"éxito {m['success']:.0%}"
    if m["len_ratio"]:
        s += f" | ruta {m['len_ratio']:.2f}x A* | altura mediana {m['cruise_m']:.0f} m"
    return s


def fmt_storm(m):
    s = f"éxito {m['success_rate']:.0%} | choque {m['crash_rate']:.0%} (viento {m['wind_crash_rate']:.0%}" + (f", aves {m['bird_crash_rate']:.0%}" if m.get("bird_crash_rate") else "") + f") | batería {m['battery_rate']:.0%}"
    if m["energy_ratio"]:
        s += f" | energía {m['energy_ratio']:.2f}x"
    return s


def score(m):
    """Para elegir el mejor modelo: primero la tasa de éxito, después rutas más cortas."""
    return (m["success"], -(m["len_ratio"] or 9))


def score_storm(m):
    """Primero el éxito; a igual éxito, menos energía."""
    return (m["success_rate"], -(m["energy_ratio"] or 9))
