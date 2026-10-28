"""CLI de Drone Rescue: operar el entorno y entrenar/evaluar los agentes.

Ejemplos:
  python cli.py envs
  python cli.py rules --env DroneRescueRandom-v2
  python cli.py play  --env DroneRescueRandom-v2 --seed 7
  python cli.py train --algo dqn --steps 1000000 --out models/dqn.pt
  python cli.py eval  --model models/dqn.pt --episodes 100
  python cli.py run   --model models/dqn.pt --seed 7 --web
  python cli.py train --algo qlearning --env DroneRescueDummy-v1 --episodes 6000 --out models/dummy.npy
"""
import argparse
import webbrowser
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

import env  # noqa: F401  (registra los entornos)
from agent.astar import astar
from agent.evaluate import EVAL_SEED, evaluate, fmt, run_episode, score
from env.drone_env import (CELL, GOAL, NX, NY, NZ, R_CRASH, R_GOAL, R_LOW, R_PROGRESS, R_STEP,
                           SENSOR_RANGE, START)
from visualize import show

ENVS = {
    "DroneRescue-v0": "mapa fijo de 5 columnas, observación = posición (Q-learning tabular)",
    "DroneRescueDummy-v1": "mapa fijo de 1 columna, observación = posición (Q-learning tabular)",
    "DroneRescueRandom-v2": "mapa aleatorio en cada intento + sensores + altura de crucero (DQN)",
    "DroneRescueFixed-v2": "mapa fijo de 5 columnas con la observación de v2 (prueba del DQN)",
}
DEFAULT_ENV = {"dqn": "DroneRescueRandom-v2", "qlearning": "DroneRescueDummy-v1"}
KEYS = {"d": 0, "a": 1, "w": 2, "s": 3, "r": 4, "f": 5}  # +x, -x, +y, -y, subir, bajar


def algo_of(model):
    return "dqn" if str(model).endswith(".pt") else "qlearning"


def env_for(args, algo):
    env_id = args.env or DEFAULT_ENV[algo]
    tabular = isinstance(gym.make(env_id).observation_space, gym.spaces.MultiDiscrete)
    if (algo == "qlearning") != tabular:
        need = "DroneRescue-v0 o DroneRescueDummy-v1" if algo == "qlearning" else "un entorno v2"
        raise SystemExit(f"{algo} necesita {need} (pediste {env_id})")
    return env_id


def load_policy(model, device="auto"):
    """Devuelve una función obs -> acción para un modelo .pt (DQN) o .npy (tabla Q)."""
    if algo_of(model) == "dqn":
        from agent.dqn import DQNAgent
        agent = DQNAgent.load(model, device)
        return agent, agent.act
    from agent.qlearning import QAgent
    agent = QAgent((NX, NY, NZ), 6)
    agent.q = np.load(model)
    return agent, lambda obs: agent.act(tuple(obs), greedy=True)


def cmd_envs(args):
    for k, v in ENVS.items():
        print(f"  {k:22s} {v}")


def cmd_rules(args):
    env_id = args.env or "DroneRescueRandom-v2"
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
    if v2:
        print(f"  Observación (39): posición, vector hacia B, 6 sensores de distancia (hasta {SENSOR_RANGE * CELL} m)"
              " y ocupación del cubo 3x3x3 alrededor del dron")


def cmd_train(args):
    env_id = env_for(args, args.algo)
    out = Path(args.out or f"models/{args.algo}.{'pt' if args.algo == 'dqn' else 'npy'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.algo == "dqn":
        from agent.dqn import train_dqn

        def eval_fn(agent):
            m = evaluate(agent, env_id, episodes=args.eval_episodes)
            return "eval " + fmt(m), score(m)
        train_dqn(env_id, steps=args.steps, device=args.device, seed=args.seed,
                  log_every=args.eval_every, eval_fn=eval_fn, save_path=out)
        print("mejor modelo guardado en", out)
    else:
        from train import train
        _, agent, history = train(episodes=args.episodes, seed=args.seed, env_id=env_id)
        first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
        print("primer éxito en el intento:", first, "| total:", dict(Counter(history)))
        np.save(out, agent.q)
        print("modelo guardado en", out)


def cmd_eval(args):
    algo = algo_of(args.model)
    env_id = env_for(args, algo)
    _, policy = load_policy(args.model, args.device)
    m = evaluate(policy, env_id, episodes=args.episodes, seed=args.seed)
    print(f"{args.model} en {env_id}, {args.episodes} mapas (semillas {args.seed}..{args.seed + args.episodes - 1})")
    print(" ", fmt(m), "|", m["results"])


def mid_flight_climbs(path):
    """Cambios de altura fuera de la vertical de A y B (sobrevuelos)."""
    ends = (START[:2], GOAL[:2])
    return sum(1 for a, b in zip(path, path[1:]) if a[2] != b[2] and (b[0], b[1]) not in ends)


def cmd_run(args):
    algo = algo_of(args.model)
    env_id = env_for(args, algo)
    _, policy = load_policy(args.model, args.device)
    e = gym.make(env_id)
    random_map = getattr(e.unwrapped, "random_map", False)
    first = args.seed if args.seed is not None else int(np.random.default_rng().integers(1_000_000))
    seeds = [first + i for i in range(args.maps if random_map else 1)]
    episodes = []
    for seed in seeds:
        path, result = run_episode(e, policy, seed=seed)
        best = astar(e.unwrapped)
        episodes.append({"seed": seed, "columns": e.unwrapped.columns, "path": path,
                         "result": result, "astar": len(best) - 1})
        print(f"semilla {seed:7d} | {len(e.unwrapped.columns):2d} columnas | {result:7s} | "
              f"{len(path) - 1:3d} pasos (A*: {len(best) - 1}) | altura máx. {max(p[2] for p in path) * CELL} m"
              f" | {mid_flight_climbs(path)} cambios de altura en vuelo")
    if len(episodes) == 1:
        show(e.unwrapped, episodes[0]["path"])
    if args.plot:
        from plot3d import draw
        draw(episodes[0]["path"], out=args.plot_out)
    if args.web:
        from viewer.export import export_html
        out = Path(export_html(episodes, env_id, args.web_out)).resolve()
        print("vista 3D (three.js) en", out)
        if not args.no_open:
            webbrowser.open(out.as_uri())


def cmd_play(args):
    """Modo manual: tú pilotas el dron para comprobar las reglas."""
    env_id = args.env or "DroneRescueRandom-v2"
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
