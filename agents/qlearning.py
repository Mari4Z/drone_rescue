from collections import defaultdict

import numpy as np


class SparseQTable(defaultdict):
    """Tabla dispersa con lectura compatible por índice en estados 1D."""

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            key = (int(key),)
        return super().__getitem__(key)


class QAgent:
    """Q-learning tabular con tabla dispersa para estados discretizados."""

    def __init__(self, shape=None, n_actions=7, alpha=0.2, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.999, seed=0):
        # ``shape`` se conserva para compatibilidad con el agente denso anterior.
        self.shape = tuple(shape) if shape is not None else None
        self.q = SparseQTable(lambda: np.zeros(n_actions, dtype=np.float64))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    @staticmethod
    def _state_key(state):
        if isinstance(state, np.ndarray):
            return tuple(state.tolist())
        if isinstance(state, list):
            return tuple(state)
        return state

    def act(self, state, greedy=False):
        key = self._state_key(state)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        qs = self.q[key]
        return int(self.rng.choice(np.flatnonzero(qs == qs.max())))

    def update(self, state, action, reward, next_state, done):
        key, next_key = self._state_key(state), self._state_key(next_state)
        target = reward if done else reward + self.gamma * self.q[next_key].max()
        self.q[key][action] += self.alpha * (target - self.q[key][action])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
