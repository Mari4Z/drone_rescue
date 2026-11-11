"""Q-learning con red neuronal (Double DQN con retornos a n pasos) para drone-rescue-v2."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

FEATURES_VERSION = "1"
N_ACTIONS = 7
N_FEATURES = 26


def make_features(observation) -> np.ndarray:
    """Convierte la observación de 19 valores en la entrada de la red.

    Escala cada grupo a [-1, 1] aproximadamente y añade objetivo, velocidad y altura a
    escala fina: dividido por 100 m, el radio de 3.5 m de BASE sería casi invisible.
    """
    obs = np.asarray(observation, dtype=np.float32)
    return np.concatenate([
        obs[0:3] / 100.0,                     # vector al objetivo
        np.clip(obs[0:3] / 10.0, -1.0, 1.0),  # vector al objetivo, de cerca
        obs[3:6] / 5.0,                       # velocidad
        np.clip(obs[3:6], -1.0, 1.0),         # velocidad, para frenar con precisión
        [obs[6] / 60.0, min(obs[6] / 5.0, 1.0)],  # altura, global y cerca del suelo
        obs[7:13] / 20.0,                     # sensores
        obs[13:19],                           # batería, carga y etapa
    ], dtype=np.float32)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # GPU de Apple Silicon
    return torch.device("cpu")


class QNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experiencias guardadas en el dispositivo; se suben en bloques porque de una en una es lento."""

    def __init__(self, size: int, device: torch.device, flush_every: int = 256):
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)
        self.obs, self.next_obs = zeros(size, N_FEATURES), zeros(size, N_FEATURES)
        self.act, self.rew = zeros(size, dtype=torch.int64), zeros(size)
        self.discount = zeros(size)  # gamma^n, o 0 si el episodio terminó de verdad
        self.size, self.i, self.n, self.device = size, 0, 0, device
        self.pending, self.flush_every = [], flush_every

    def add(self, obs, action, reward, next_obs, discount) -> None:
        self.pending.append((obs, action, reward, next_obs, discount))

    def flush(self) -> None:
        if not self.pending:
            return
        obs, act, rew, next_obs, discount = map(np.array, zip(*self.pending))
        idx = torch.as_tensor((self.i + np.arange(len(obs))) % self.size, device=self.device)
        put = lambda dst, x, dtype=torch.float32: dst.index_copy_(0, idx, torch.as_tensor(x, dtype=dtype).to(self.device))
        put(self.obs, obs); put(self.next_obs, next_obs); put(self.act, act, torch.int64)
        put(self.rew, rew); put(self.discount, discount)
        self.i = (self.i + len(obs)) % self.size
        self.n = min(self.size, self.n + len(obs))
        self.pending = []

    def __len__(self) -> int:
        return self.n + len(self.pending)

    def sample(self, n: int):
        if len(self.pending) >= self.flush_every or self.n == 0:
            self.flush()
        idx = torch.randint(0, self.n, (n,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.discount[idx]


class NStepCollector:
    """Acumula n pasos y entrega transiciones con la recompensa descontada de esos pasos."""

    def __init__(self, buffer: ReplayBuffer, n: int, gamma: float):
        self.buffer, self.n, self.gamma = buffer, n, gamma
        self.queue: deque = deque()

    def _emit(self, next_obs, terminal: bool) -> None:
        obs, action, _ = self.queue[0]
        total = sum(self.gamma ** k * item[2] for k, item in enumerate(self.queue))
        discount = 0.0 if terminal else self.gamma ** len(self.queue)
        self.buffer.add(obs, action, total, next_obs, discount)
        self.queue.popleft()

    def add(self, obs, action, reward, next_obs, terminated: bool, truncated: bool) -> None:
        self.queue.append((obs, action, reward))
        if terminated or truncated:
            while self.queue:  # truncated conserva el valor del estado siguiente; terminated no
                self._emit(next_obs, terminated)
        elif len(self.queue) == self.n:
            self._emit(next_obs, False)


class DQNAgent:
    def __init__(self, device: str = "auto", lr: float = 3e-4, hidden: int = 256, seed: int = 0):
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.q = QNet(hidden).to(self.device)
        self.target = QNet(hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        # Copia en CPU para decidir cada acción: ir y volver de la GPU en cada paso cuesta
        # más que el cálculo. El aprendizaje por lotes se hace en self.device.
        self.actor = QNet(hidden)
        self.sync_actor()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.hidden = hidden

    def sync_actor(self) -> None:
        self.actor.load_state_dict({k: v.cpu() for k, v in self.q.state_dict().items()})

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def greedy(self, features: np.ndarray) -> int:
        return int(self.actor(torch.as_tensor(features).unsqueeze(0)).argmax(1).item())

    def learn(self, batch) -> torch.Tensor:
        obs, act, rew, next_obs, discount = batch
        with torch.no_grad():
            # Double DQN: la red online elige la acción y la red objetivo la evalúa
            best = self.q(next_obs).argmax(1, keepdim=True)
            target = rew + discount * self.target(next_obs).gather(1, best).squeeze(1)
        pred = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return loss.detach()  # sin .item(): evita esperar a la GPU en cada actualización

    def save(self, path) -> None:
        self.sync_actor()
        torch.save({"state": self.actor.state_dict(), "hidden": int(self.hidden),
                    "features_version": FEATURES_VERSION}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQNAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["features_version"] != FEATURES_VERSION:
            raise ValueError(f"El modelo {path} usa otras entradas; hay que volver a entrenarlo")
        agent = cls(device=device, hidden=checkpoint["hidden"])
        agent.q.load_state_dict(checkpoint["state"])
        agent.sync_target()
        agent.sync_actor()
        return agent


class DQNPolicy:
    """Misma interfaz que las políticas de ``pipelines.evaluate``."""

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, observation, config: dict | None = None) -> int:
        del config
        return self.agent.greedy(make_features(observation))
