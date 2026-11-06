from collections import Counter

import gymnasium as gym

import envs.grid_drone_env  # noqa: F401  (registra los entornos)
from agents.grid.evaluate import run_episode
from agents.grid.qlearning import QAgent


def train(episodes=20000, seed=0, env_id="DroneRescueGrid-v0"):
    e = gym.make(env_id)
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
    return run_episode(e, lambda obs: agent.act(tuple(obs), greedy=True))
