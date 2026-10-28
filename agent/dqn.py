"""Q-learning con red neuronal (Double DQN) en PyTorch."""
import time
from collections import Counter, deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import env  # noqa: F401  (registra los entornos)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # GPU de Apple Silicon
    return torch.device("cpu")


class QNet(nn.Module):
    def __init__(self, n_obs, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, n_obs, size):
        self.obs = np.zeros((size, n_obs), np.float32)
        self.next_obs = np.zeros((size, n_obs), np.float32)
        self.act = np.zeros(size, np.int64)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.i, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        self.obs[self.i], self.act[self.i], self.rew[self.i], self.next_obs[self.i], self.done[self.i] = o, a, r, o2, d
        self.i = (self.i + 1) % self.size
        self.full |= self.i == 0

    def __len__(self):
        return self.size if self.full else self.i

    def sample(self, rng, n, device):
        idx = rng.integers(0, len(self), n)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.done)


class DQNAgent:
    def __init__(self, n_obs, n_actions, device="auto", lr=5e-4, gamma=0.99, hidden=256, seed=0):
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.q = QNet(n_obs, n_actions, hidden).to(self.device)
        self.target = QNet(n_obs, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.n_actions, self.hidden = gamma, n_actions, hidden
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        return int(q.argmax(1).item())

    def learn(self, batch):
        o, a, r, o2, d = batch
        with torch.no_grad():
            # Double DQN: la red online elige la acción, la red objetivo la evalúa
            a2 = self.q(o2).argmax(1, keepdim=True)
            target = r + self.gamma * (1 - d) * self.target(o2).gather(1, a2).squeeze(1)
        pred = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def save(self, path):
        torch.save({"state": self.q.state_dict(), "n_obs": int(self.q.net[0].in_features),
                    "n_actions": int(self.n_actions), "hidden": int(self.hidden)}, path)

    @classmethod
    def load(cls, path, device="auto"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(ck["n_obs"], ck["n_actions"], device=device, hidden=ck["hidden"])
        agent.q.load_state_dict(ck["state"])
        agent.q.eval()
        return agent


def train_dqn(env_id="DroneRescueRandom-v2", steps=300_000, device="auto", seed=0,
              batch_size=128, buffer_size=200_000, learning_starts=5_000, train_freq=2,
              target_every=2_000, eps_start=1.0, eps_end=0.05, eps_fraction=0.4,
              reward_scale=0.1, log_every=10_000, eval_fn=None, save_path=None):
    """Entrena con épsilon decreciente lineal. reward_scale baja ±100 a ±10 para estabilizar la red.
    eval_fn(agent) -> (texto, puntuación): si se da, se guarda en save_path el modelo con mejor puntuación."""
    e = gym.make(env_id)
    n_obs = e.observation_space.shape[0]
    agent = DQNAgent(n_obs, e.action_space.n, device=device, seed=seed)
    buf = ReplayBuffer(n_obs, buffer_size)
    print(f"entrenando DQN en {env_id} | dispositivo: {agent.device} | {steps} pasos")

    obs, _ = e.reset(seed=seed)
    results, losses, t0 = deque(maxlen=500), deque(maxlen=1000), time.time()
    best = None
    for step in range(1, steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / (eps_fraction * steps))
        a = agent.act(obs, eps)
        obs2, r, terminated, truncated, info = e.step(a)
        buf.add(obs, a, r * reward_scale, obs2, float(terminated))  # truncated no corta el bootstrap
        obs = obs2
        if terminated or truncated:
            results.append(info["result"])
            obs, _ = e.reset()

        if step >= learning_starts and step % train_freq == 0:
            losses.append(agent.learn(buf.sample(agent.rng, batch_size, agent.device)))
        if step % target_every == 0:
            agent.target.load_state_dict(agent.q.state_dict())
        if step % log_every == 0:
            c = Counter(results)
            n = max(1, len(results))
            line = (f"paso {step:7d} | eps {eps:.2f} | éxito {c['goal'] / n:5.1%} choque {c['crash'] / n:5.1%}"
                    f" | loss {np.mean(losses) if losses else 0:.3f} | {time.time() - t0:.0f}s")
            if eval_fn:
                text, score = eval_fn(agent)
                line += " | " + text
                if best is None or score > best:
                    best = score
                    if save_path:
                        agent.save(save_path)
                        line += " *guardado*"
            print(line, flush=True)
    if save_path and not eval_fn:
        agent.save(save_path)
    return agent
