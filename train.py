from collections import Counter

import gymnasium as gym

import env  # noqa: F401  (registra DroneRescue-v0)
from agent.qlearning import QAgent
from env.drone_env import GOAL
from visualize import show


def train(episodes=20000, seed=0):
    e = gym.make("DroneRescue-v0")
    agent = QAgent(e.observation_space.nvec, e.action_space.n, seed=seed)
    history = []
    for ep in range(1, episodes + 1):
        obs, _ = e.reset(seed=seed if ep == 1 else None)
        s, done = tuple(obs), False
        while not done:
            a = agent.act(s)
            obs, r, terminated, truncated, info = e.step(a)
            s2, done = tuple(obs), terminated or truncated
            agent.update(s, a, r, s2, terminated)  # truncated no corta el bootstrap
            s = s2
        agent.decay()
        history.append(info["result"])
        if ep % 2000 == 0:
            print(f"ep {ep:6d}  eps={agent.eps:.3f}  últimos 2000: {dict(Counter(history[-2000:]))}")
    return e, agent, history


def greedy_path(e, agent):
    obs, _ = e.reset()
    path, done = [tuple(obs.tolist())], False
    while not done:
        obs, _, terminated, truncated, info = e.step(agent.act(tuple(obs), greedy=True))
        path.append(tuple(obs.tolist()))
        done = terminated or truncated
    return path, info["result"]


if __name__ == "__main__":
    e, agent, history = train()
    first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
    print("primer éxito en el intento:", first)
    path, result = greedy_path(e, agent)
    print(f"ruta greedy: {result}, {len(path) - 1} pasos, termina en {path[-1]} (meta {GOAL})")
    print("altura máx. de la ruta:", max(p[2] for p in path) * 5, "m")
    show(e.unwrapped, path)
