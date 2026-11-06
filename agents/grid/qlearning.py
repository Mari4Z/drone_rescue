import numpy as np


class QAgent:
    def __init__(self, shape, n_actions, alpha=0.2, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.999, seed=0):
        self.q = np.zeros((*map(int, shape), n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        qs = self.q[s]
        return int(self.rng.choice(np.flatnonzero(qs == qs.max())))  # desempate aleatorio

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.q[s2].max()
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
