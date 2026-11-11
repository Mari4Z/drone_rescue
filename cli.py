"""CLI de la familia de cuadrícula (grid-drone-v1): operar el entorno y entrenar/evaluar agentes.

No usa el ambiente continuo drone-rescue-v2; para ese están pipelines/ y la web.

Ejemplos:
  python cli.py envs
  python cli.py rules --env DroneRescueGridRandom-v2
  python cli.py play  --env DroneRescueGridRandom-v2 --seed 7
  python cli.py train --algo dqn --steps 1000000 --out models/dqn.pt
  python cli.py eval  --model models/dqn.pt --episodes 100
  python cli.py run   --model models/dqn.pt --seed 7 --web
  python cli.py train --algo qlearning --env DroneRescueGridDummy-v1 --episodes 6000 --out models/dummy.npy
  python cli.py run   --env DroneRescueGridStorm-v3 --model artifacts/models/<run_id>/model.pt --seed 600000 --web

Los runs formales de grid-drone-v2 se entrenan con pipelines/train_grid.py (MLflow, manifiesto y DVC).
"""
import argparse
import webbrowser
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

import envs.grid_drone_env  # noqa: F401  (registra los entornos)
from agents.grid.astar import astar
from agents.grid.evaluate import EVAL_SEED, evaluate, evaluate_storm, fmt, fmt_storm, run_episode, score, score_storm
from envs.grid_drone_env import (CELL, GOAL, NX, NY, NZ, R_CRASH, R_GOAL, R_LOW, R_PROGRESS, R_STEP,
                           SENSOR_RANGE, START)
from viewer.text_view import show

ENVS = {
    "DroneRescueGrid-v0": "mapa fijo de 5 columnas, observación = posición (Q-learning tabular)",
    "DroneRescueGridDummy-v1": "mapa fijo de 1 columna, observación = posición (Q-learning tabular)",
    "DroneRescueGridRandom-v2": "mapa aleatorio en cada intento + sensores + altura de crucero (DQN)",
    "DroneRescueGridFixed-v2": "mapa fijo de 5 columnas con la observación de v2 (prueba del DQN)",
    "DroneRescueGridStorm-v3": "grid-drone-v2: 12 a 35 edificios, B aleatorio, viento con ráfagas y batería (DQN)",
}
STORM = "DroneRescueGridStorm-v3"
DEFAULT_ENV = {"dqn": "DroneRescueGridRandom-v2", "qlearning": "DroneRescueGridDummy-v1"}
KEYS = {"d": 0, "a": 1, "w": 2, "s": 3, "r": 4, "f": 5}  # +x, -x, +y, -y, subir, bajar


def algo_of(model):
    return "dqn" if str(model).endswith(".pt") else "qlearning"


def env_for(args, algo):
    env_id = args.env or DEFAULT_ENV[algo]
    tabular = isinstance(gym.make(env_id).observation_space, gym.spaces.MultiDiscrete)
    if (algo == "qlearning") != tabular:
        need = "DroneRescueGrid-v0 o DroneRescueGridDummy-v1" if algo == "qlearning" else "un entorno v2"
        raise SystemExit(f"{algo} necesita {need} (pediste {env_id})")
    return env_id


def load_policy(model, device="auto", env_id="DroneRescueGridRandom-v2"):
    """Devuelve una función obs -> acción para un modelo .pt (DQN) o .npy (tabla Q)."""
    if algo_of(model) == "dqn":
        from agents.grid.dqn import DQNAgent
        agent = DQNAgent.load(model, device)
        n_obs = gym.make(env_id).observation_space.shape[0]
        if agent.q.net[0].in_features != n_obs:
            raise SystemExit(f"{model} espera {agent.q.net[0].in_features} observaciones y el entorno da {n_obs}: "
                             "es de otra versión o familia del entorno (ver docs/familia_cuadricula.md)")
        return agent, agent.act
    from agents.grid.qlearning import QAgent
    agent = QAgent((NX, NY, NZ), 6)
    agent.q = np.load(model)
    return agent, lambda obs: agent.act(tuple(obs), greedy=True)


def cmd_envs(args):
    for k, v in ENVS.items():
        print(f"  {k:26s} {v}")


def cmd_rules(args):
    env_id = args.env or "DroneRescueGridRandom-v2"
    e = gym.make(env_id).unwrapped
    e.reset(seed=args.seed)
    v2 = hasattr(e, "random_map")
    print(f"""Entorno {env_id}: {ENVS[env_id]}
  Mapa: 100 x 100 x 60 m, celdas de {CELL} m ({NX} x {NY} x {NZ})
  A (inicio): {START}   B (meta): {GOAL}
  Columnas: {len(e.columns)}{' (aleatorias en cada intento; ejemplo:)' if v2 and e.random_map else ''}
    {e.columns}
  Acciones: 0 +x | 1 -x | 2 +y | 3 -y | 4 subir | 5 bajar
  Reglas:
    - El suelo solo es transitable en A y B: despegue y aterrizaje verticales.
    - Chocar con columna, suelo o borde termina el intento y el dron vuelve a A.
    - Límite de {e.max_steps} pasos por intento.""")
    if e.min_cruise_z:
        print(f"    - Volar por debajo de {e.min_cruise_z * CELL} m (fuera de A y B) penaliza {R_LOW:+}/paso.")
    print(f"  Recompensas: meta {R_GOAL:+} | choque {R_CRASH:+} | paso {R_STEP:+} | acercarse a B {R_PROGRESS:+}/celda")
    if v2 and env_id != STORM:
        print(f"  Observación (46): posición, vector hacia B, 6 sensores de distancia (hasta {SENSOR_RANGE * CELL} m),"
              " ocupación del cubo 3x3x3 alrededor del dron y visitas a la celda y sus vecinas")
    if env_id == STORM:
        import envs.grid_drone_env as g
        print(f"""  B cambia en cada mapa (a {g.GOAL_MIN_DIST} celdas o más de A). Este mapa: B = {e.goal}
  Viento: dirección y fuerza por intento (hasta {g.WIND_MAX}). Tras cada movimiento en el aire, una ráfaga empuja
    una celda a favor del viento con probabilidad fuerza x (0.4 a 1.0 según la altura). Contra un edificio: choque.
    Este mapa: fuerza {e.wind:.2f}, dirección ({e.wind_dir[0]:+.2f}, {e.wind_dir[1]:+.2f}).
  Batería: horizontal {g.ENERGY_H}, subir {g.ENERGY_UP}, bajar {g.ENERGY_DOWN} por celda; contra el viento hasta
    +{g.HEADWIND * g.WIND_MAX:.1f}, a favor hasta -{g.HEADWIND * g.WIND_MAX:.1f}; cada ráfaga +{g.GUST_ENERGY}.
    Capacidad = mejor ruta x {g.BATTERY_MARGIN[0]} a {g.BATTERY_MARGIN[1]} + {g.BATTERY_RESERVE}. Este mapa: {e.capacity:.1f}.
    Sin batería: {g.R_BATTERY:+} y termina. Recompensa por paso: -energía (en vez de {R_STEP:+}).
  Observación (51): las 46 de v2, viento (2), probabilidad de ráfaga a esta altura, batería y margen hasta B""")


def cmd_train(args):
    env_id = env_for(args, args.algo)
    out = Path(args.out or f"models/{args.algo}.{'pt' if args.algo == 'dqn' else 'npy'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.algo == "dqn":
        from agents.grid.dqn import train_dqn

        def eval_fn(agent):
            if env_id == STORM:
                m = evaluate_storm(agent, env_id, episodes=args.eval_episodes)["metrics"]
                return "eval " + fmt_storm(m), score_storm(m)
            m = evaluate(agent, env_id, episodes=args.eval_episodes)
            return "eval " + fmt(m), score(m)
        train_dqn(env_id, steps=args.steps, device=args.device, seed=args.seed,
                  log_every=args.eval_every, eval_fn=eval_fn, save_path=out)
        print("mejor modelo guardado en", out)
    else:
        from agents.grid.train import train
        _, agent, history = train(episodes=args.episodes, seed=args.seed, env_id=env_id)
        first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
        print("primer éxito en el intento:", first, "| total:", dict(Counter(history)))
        np.save(out, agent.q)
        print("modelo guardado en", out)


def cmd_eval(args):
    algo = algo_of(args.model)
    env_id = env_for(args, algo)
    _, policy = load_policy(args.model, args.device, env_id)
    print(f"{args.model} en {env_id}, {args.episodes} mapas (semillas {args.seed}..{args.seed + args.episodes - 1})")
    if env_id == STORM:
        m = evaluate_storm(policy, env_id, episodes=args.episodes, seed=args.seed)["metrics"]
        print(" ", fmt_storm(m), "|", m["cause_counts"])
        return
    m = evaluate(policy, env_id, episodes=args.episodes, seed=args.seed)
    print(" ", fmt(m), "|", m["results"])


def mid_flight_climbs(path):
    """Cambios de altura fuera de la vertical de A y B (sobrevuelos)."""
    ends = (START[:2], GOAL[:2])
    return sum(1 for a, b in zip(path, path[1:]) if a[2] != b[2] and (b[0], b[1]) not in ends)


def cmd_run(args):
    algo = algo_of(args.model)
    env_id = env_for(args, algo)
    _, policy = load_policy(args.model, args.device, env_id)
    e = gym.make(env_id)
    random_map = getattr(e.unwrapped, "random_map", False)
    first = args.seed if args.seed is not None else int(np.random.default_rng().integers(1_000_000))
    seeds = [first + i for i in range(args.maps if random_map else 1)]
    episodes = []
    for seed in seeds:
        trace = []
        path, result = run_episode(e, policy, seed=seed, trace=trace)
        env = e.unwrapped
        best = astar(env)
        episode = {"seed": seed, "columns": env.columns, "path": path, "result": result, "astar": len(best) - 1}
        line = (f"semilla {seed:7d} | {len(env.columns):2d} columnas | {result:7s} | "
                f"{len(path) - 1:3d} pasos (A*: {len(best) - 1}) | altura máx. {max(p[2] for p in path) * CELL} m"
                f" | {mid_flight_climbs(path)} cambios de altura en vuelo")
        if env_id == STORM:
            episode.update({
                "start": list(env.start), "goal": list(env.goal), "cause": trace[-1].get("cause"),
                "wind": round(env.wind, 4), "wind_dir": [round(float(v), 4) for v in env.wind_dir],
                "capacity": round(env.capacity, 2), "battery": [round(env.capacity, 2)] + [round(max(i["battery"], 0), 2) for i in trace],
                "pushes": [k + 1 for k, i in enumerate(trace) if i.get("pushed")],
            })
            line += (f" | viento {env.wind:.2f} | {env.pushes} ráfagas | batería {max(env.battery, 0) / env.capacity:4.0%}"
                     + (" | choque por el viento" if trace[-1].get("cause") == "viento" else ""))
        episodes.append(episode)
        print(line)
    if len(episodes) == 1:
        show(e.unwrapped, episodes[0]["path"])
    if args.plot:
        from viewer.grid_plot import draw
        draw(episodes[0]["path"], episodes[0]["columns"], out=args.plot_out)
    if args.web:
        from viewer.export import export_html
        out = Path(export_html(episodes, env_id, args.web_out)).resolve()
        print("vista 3D (three.js) en", out)
        if not args.no_open:
            webbrowser.open(out.as_uri())


def cmd_play(args):
    """Modo manual: tú pilotas el dron para comprobar las reglas."""
    env_id = args.env or "DroneRescueGridRandom-v2"
    e = gym.make(env_id)
    seed = args.seed
    e.reset(seed=seed)
    path, total = [e.unwrapped.pos], 0.0
    print("Controles: w/s = ±y, d/a = ±x, r = subir, f = bajar, m = mapa, q = salir")
    while True:
        p = e.unwrapped.pos
        key = input(f"pos {p} ({p[2] * CELL} m) | recompensa {total:+.1f} > ").strip().lower()
        if key == "q":
            break
        if key == "m":
            show(e.unwrapped, path)
            continue
        if key not in KEYS:
            print("tecla no válida")
            continue
        _, r, terminated, truncated, info = e.step(KEYS[key])
        total += r
        path.append(e.unwrapped.pos)
        if terminated or truncated:
            print(f"*** {info['result'].upper()} *** recompensa total {total:+.1f}")
            show(e.unwrapped, path)
            e.reset(seed=seed)  # mismo mapa para volver a intentarlo
            path, total = [e.unwrapped.pos], 0.0
            print("\nNuevo intento desde A")


def main():
    p = argparse.ArgumentParser(description="Drone Rescue CLI", formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n", 2)[2])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_env(sp):
        sp.add_argument("--env", choices=ENVS, help="por defecto según el algoritmo")

    def add_device(sp):
        sp.add_argument("--device", default="auto", help="auto (GPU si hay: cuda/mps), cpu, mps o cuda")

    sp = sub.add_parser("envs", help="lista los entornos disponibles")
    sp.set_defaults(fn=cmd_envs)

    sp = sub.add_parser("rules", help="muestra las reglas, recompensas y observación de un entorno")
    add_env(sp)
    sp.add_argument("--seed", type=int, default=0, help="mapa de ejemplo (entornos aleatorios)")
    sp.set_defaults(fn=cmd_rules)

    sp = sub.add_parser("train", help="entrena un agente")
    sp.add_argument("--algo", choices=["dqn", "qlearning"], default="dqn")
    add_env(sp)
    add_device(sp)
    sp.add_argument("--steps", type=int, default=1_000_000, help="DQN: pasos de entrenamiento")
    sp.add_argument("--eval-every", type=int, default=100_000, help="DQN: evaluar (y guardar el mejor) cada N pasos")
    sp.add_argument("--eval-episodes", type=int, default=100, help="DQN: mapas por evaluación")
    sp.add_argument("--episodes", type=int, default=20000, help="Q-learning: intentos")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", help="por defecto models/<algo>.pt|.npy")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("eval", help="tasa de éxito en mapas no vistos y comparación con A*")
    sp.add_argument("--model", required=True)
    add_env(sp)
    add_device(sp)
    sp.add_argument("--episodes", type=int, default=100)
    sp.add_argument("--seed", type=int, default=EVAL_SEED)
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("run", help="ejecuta un agente en un mapa y muestra su ruta")
    sp.add_argument("--model", required=True)
    add_env(sp)
    add_device(sp)
    sp.add_argument("--seed", type=int, help="semilla del primer mapa (por defecto aleatoria)")
    sp.add_argument("--maps", type=int, default=10, help="número de mapas (entornos aleatorios)")
    sp.add_argument("--plot", action="store_true", help="genera la imagen 3D (matplotlib)")
    sp.add_argument("--plot-out", default="drone_route.png")
    sp.add_argument("--web", action="store_true", help="genera la vista 3D interactiva (three.js)")
    sp.add_argument("--web-out", default="drone_view.html")
    sp.add_argument("--no-open", action="store_true", help="no abrir el navegador")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("play", help="pilota el dron a mano")
    add_env(sp)
    sp.add_argument("--seed", type=int, default=0, help="mapa a usar (entornos aleatorios)")
    sp.set_defaults(fn=cmd_play)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
