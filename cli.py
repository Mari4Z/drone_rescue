"""CLI de Drone Rescue: entrenar, ejecutar y operar el agente.

Ejemplos:
  python cli.py rules
  python cli.py train --env DroneRescueDummy-v1 --episodes 6000 --out models/dummy.npy
  python cli.py run --model models/dummy.npy --env DroneRescueDummy-v1 --plot
  python cli.py run --model models/dummy.npy --web
  python cli.py play --env DroneRescueDummy-v1
"""
import argparse
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

import env  # noqa: F401  (registra los entornos)
from agent.qlearning import QAgent
from env.drone_env import CELL, GOAL, R_CRASH, R_GOAL, R_PROGRESS, R_STEP, START
from train import greedy_path, train
from visualize import show

ENVS = ["DroneRescue-v0", "DroneRescueDummy-v1"]
KEYS = {"d": 0, "a": 1, "w": 2, "s": 3, "r": 4, "f": 5}  # +x, -x, +y, -y, subir, bajar


def cmd_rules(args):
    e = gym.make(args.env).unwrapped
    print(f"""Entorno {args.env}
  Mapa: 100 x 100 x 60 m, celdas de {CELL} m ({' x '.join(map(str, e.observation_space.nvec))})
  A (inicio): {START}   B (meta): {GOAL}
  Columnas: {len(e.columns)}  ->  {e.columns}
  Acciones: 0 +x | 1 -x | 2 +y | 3 -y | 4 subir | 5 bajar
  Reglas:
    - El suelo solo es transitable en A y B: despegue y aterrizaje verticales.
    - Chocar con columna, suelo o borde termina el intento y el dron vuelve a A.
    - Límite de {e.max_steps} pasos por intento.
  Recompensas: meta {R_GOAL:+} | choque {R_CRASH:+} | paso {R_STEP:+} | acercarse a B {R_PROGRESS:+}/celda""")


def cmd_train(args):
    e, agent, history = train(episodes=args.episodes, seed=args.seed, env_id=args.env)
    first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
    print("primer éxito en el intento:", first, "| total:", dict(Counter(history)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, agent.q)
    print("modelo guardado en", args.out)


def load_agent(e, path):
    agent = QAgent(e.observation_space.nvec, e.action_space.n)
    agent.q = np.load(path)
    if agent.q.shape[:-1] != tuple(e.observation_space.nvec):
        raise SystemExit(f"El modelo {path} no coincide con el entorno")
    return agent


def cmd_run(args):
    e = gym.make(args.env)
    agent = load_agent(e, args.model)
    path, result = greedy_path(e, agent)
    print(f"resultado: {result} | {len(path) - 1} pasos | altura máx. {max(p[2] for p in path) * CELL} m")
    show(e.unwrapped, path)
    if args.plot:
        from plot3d import draw
        draw(path, out=args.plot_out)
    if args.web:
        import webbrowser

        from viewer.export import export_html
        out = Path(export_html(e.unwrapped, path, result, args.env, args.web_out)).resolve()
        print("vista 3D (three.js) en", out)
        if not args.no_open:
            webbrowser.open(out.as_uri())


def cmd_play(args):
    """Modo manual: tú pilotas el dron para comprobar las reglas."""
    e = gym.make(args.env)
    obs, _ = e.reset()
    path, total = [tuple(obs.tolist())], 0.0
    print("Controles: w/s = ±y, d/a = ±x, r = subir, f = bajar, q = salir")
    while True:
        key = input(f"pos {path[-1]} ({path[-1][2] * CELL} m) | recompensa {total:+.0f} > ").strip().lower()
        if key == "q":
            break
        if key not in KEYS:
            print("tecla no válida")
            continue
        obs, r, terminated, truncated, info = e.step(KEYS[key])
        total += r
        path.append(tuple(obs.tolist()))
        if terminated or truncated:
            print(f"*** {info['result'].upper()} *** recompensa total {total:+.0f}")
            show(e.unwrapped, path)
            obs, _ = e.reset()
            path, total = [tuple(obs.tolist())], 0.0
            print("\nNuevo intento desde A")


def main():
    p = argparse.ArgumentParser(description="Drone Rescue CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_env(sp):
        sp.add_argument("--env", choices=ENVS, default="DroneRescueDummy-v1")

    sp = sub.add_parser("rules", help="muestra las reglas del entorno")
    add_env(sp)
    sp.set_defaults(fn=cmd_rules)

    sp = sub.add_parser("train", help="entrena el agente Q-learning y guarda la tabla Q")
    add_env(sp)
    sp.add_argument("--episodes", type=int, default=20000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="models/qtable.npy")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("run", help="ejecuta un agente entrenado y muestra su ruta")
    add_env(sp)
    sp.add_argument("--model", default="models/qtable.npy")
    sp.add_argument("--plot", action="store_true", help="genera además la vista 3D")
    sp.add_argument("--plot-out", default="drone_route.png")
    sp.add_argument("--web", action="store_true", help="genera la vista 3D interactiva con three.js")
    sp.add_argument("--web-out", default="drone_view.html")
    sp.add_argument("--no-open", action="store_true", help="no abrir el navegador")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("play", help="pilota el dron a mano")
    add_env(sp)
    sp.set_defaults(fn=cmd_play)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
