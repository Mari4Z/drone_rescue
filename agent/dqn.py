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
    """Experiencias guardadas directamente en el dispositivo (GPU) para no copiar cada lote.
    Las nuevas se acumulan en CPU y se suben en bloques de flush_every (subir de una en una es lento)."""

    def __init__(self, n_obs, size, device, flush_every=256):
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)
        self.obs, self.next_obs = z(size, n_obs), z(size, n_obs)
        self.act, self.rew, self.done = z(size, dtype=torch.int64), z(size), z(size)
        self.size, self.i, self.n, self.device = size, 0, 0, device
        self.pending, self.flush_every = [], flush_every

    def add(self, o, a, r, o2, d):
        self.pending.append((o, a, r, o2, d))

    def flush(self):
        if not self.pending:
            return
        o, a, r, o2, d = map(np.array, zip(*self.pending))
        idx = torch.as_tensor((self.i + np.arange(len(o))) % self.size, device=self.device)
        put = lambda dst, x, dtype=torch.float32: dst.index_copy_(0, idx, torch.as_tensor(x, dtype=dtype).to(self.device))
        put(self.obs, o); put(self.next_obs, o2); put(self.act, a, torch.int64); put(self.rew, r); put(self.done, d)
        self.i = (self.i + len(o)) % self.size
        self.n = min(self.size, self.n + len(o))
        self.pending = []

    def __len__(self):
        return self.n + len(self.pending)

    def sample(self, n):
        if len(self.pending) >= self.flush_every or self.n == 0:
            self.flush()
        idx = torch.randint(0, self.n, (n,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


class DQNAgent:
    def __init__(self, n_obs, n_actions, device="auto", lr=5e-4, gamma=0.99, hidden=256, seed=0):
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.q = QNet(n_obs, n_actions, hidden).to(self.device)
        self.target = QNet(n_obs, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        # Copia en CPU para decidir cada acción: con una red tan pequeña, ir y volver de la GPU
        # en cada paso cuesta más que el cálculo. El aprendizaje (lotes) se hace en self.device.
        self.actor = QNet(n_obs, n_actions, hidden)
        self.sync_actor()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.n_actions, self.hidden = gamma, n_actions, hidden
        self.rng = np.random.default_rng(seed)

    def sync_actor(self):
        self.actor.load_state_dict({k: v.cpu() for k, v in self.q.state_dict().items()})

    @torch.no_grad()
    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(self.actor(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())

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
        return loss.detach()  # sin .item(): evita esperar a la GPU en cada actualización

    def save(self, path):
        torch.save({"state": self.q.state_dict(), "n_obs": int(self.q.net[0].in_features),
                    "n_actions": int(self.n_actions), "hidden": int(self.hidden)}, path)

    @classmethod
    def load(cls, path, device="auto"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(ck["n_obs"], ck["n_actions"], device=device, hidden=ck["hidden"])
        agent.q.load_state_dict(ck["state"])
        agent.sync_actor()
        return agent


def train_dqn(env_id="DroneRescueRandom-v2", steps=300_000, device="auto", seed=0,
              batch_size=128, buffer_size=200_000, learning_starts=5_000, train_freq=2,
              target_every=2_000, actor_every=500, eps_start=1.0, eps_end=0.05, eps_fraction=0.4,
              reward_scale=0.1, log_every=10_000, eval_fn=None, save_path=None):
    """Entrena con épsilon decreciente lineal. reward_scale baja ±100 a ±10 para estabilizar la red.
    eval_fn(agent) -> (texto, puntuación): si se da, se guarda en save_path el modelo con mejor puntuación."""
    e = gym.make(env_id)
    n_obs = e.observation_space.shape[0]
    agent = DQNAgent(n_obs, e.action_space.n, device=device, seed=seed)
    buf = ReplayBuffer(n_obs, buffer_size, agent.device)
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
            losses.append(agent.learn(buf.sample(batch_size)))
        if step % target_every == 0:
            agent.target.load_state_dict(agent.q.state_dict())
        if step % actor_every == 0:
            agent.sync_actor()
        if step % log_every == 0:
            c = Counter(results)
            n = max(1, len(results))
            line = (f"paso {step:7d} | eps {eps:.2f} | éxito {c['goal'] / n:5.1%} choque {c['crash'] / n:5.1%}"
                    f" | loss {torch.stack(list(losses)).mean().item() if losses else 0:.3f} | {time.time() - t0:.0f}s")
            if eval_fn:
                agent.sync_actor()
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
