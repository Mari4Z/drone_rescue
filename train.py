from collections import Counter

from agent.qlearning import QAgent
from env.drone_env import DroneEnv, GOAL
from visualize import show


def train(episodes=20000, seed=0):
    env = DroneEnv()
    agent = QAgent(env.shape, env.n_actions, seed=seed)
    history = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
        agent.decay()
        history.append(info["result"])
        if ep % 2000 == 0:
            print(f"ep {ep:6d}  eps={agent.eps:.3f}  últimos 2000: {dict(Counter(history[-2000:]))}")
    return env, agent, history


def greedy_path(env, agent):
    s, path, done = env.reset(), [], False
    path.append(s)
    while not done:
        s, _, done, info = env.step(agent.act(s, greedy=True))
        path.append(s)
    return path, info["result"]


if __name__ == "__main__":
    env, agent, history = train()
    first = next((i for i, r in enumerate(history, 1) if r == "goal"), None)
    print("primer éxito en el intento:", first)
    path, result = greedy_path(env, agent)
    print(f"ruta greedy: {result}, {len(path) - 1} pasos, termina en {path[-1]} (meta {GOAL})")
    print("altura máx. de la ruta:", max(p[2] for p in path) * 5, "m")
    show(env, path)
