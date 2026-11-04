from collections import Counter

import gymnasium as gym

import envs  # noqa: F401  (registra DroneRescue-v0)
from agents.qlearning import QAgent
from envs.drone_env import discretize_observation
from visualize import show


def train(episodes=20000, seed=0):
    e = gym.make("DroneRescue-v0")
    agent = QAgent(n_actions=e.action_space.n, seed=seed)
    history = []
    for ep in range(1, episodes + 1):
        obs, _ = e.reset(seed=seed if ep == 1 else None)
        s, done = discretize_observation(obs, e.unwrapped.config), False
        while not done:
            a = agent.act(s)
            obs, r, terminated, truncated, info = e.step(a)
            s2 = discretize_observation(obs, e.unwrapped.config)
            done = terminated or truncated
            agent.update(s, a, r, s2, terminated)  # truncated no corta el bootstrap
            s = s2
        agent.decay()
        history.append(info["result"])
        if ep % 2000 == 0:
            print(f"ep {ep:6d}  eps={agent.eps:.3f}  últimos 2000: {dict(Counter(history[-2000:]))}")
    return e, agent, history


def greedy_path(e, agent):
    obs, _ = e.reset()
    path, done = [tuple(e.unwrapped.position.tolist())], False
    while not done:
        state = discretize_observation(obs, e.unwrapped.config)
        obs, _, terminated, truncated, info = e.step(agent.act(state, greedy=True))
        path.append(tuple(info["position_m"].tolist()))
        done = terminated or truncated
    return path, info["cause"] or info["result"]


if __name__ == "__main__":
    e, agent, history = train()
    first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
    print("primer éxito en el intento:", first)
    path, result = greedy_path(e, agent)
    print(f"ruta greedy: {result}, {len(path) - 1} pasos, termina en {path[-1]} m")
    print("altura máx. de la ruta:", max(p[2] for p in path), "m")
    show(e.unwrapped, path)
