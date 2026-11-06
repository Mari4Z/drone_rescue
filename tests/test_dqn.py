from pathlib import Path

import numpy as np
import pytest

from agents.dqn import N_FEATURES, DQNAgent, DQNPolicy, NStepCollector, ReplayBuffer, make_features, pick_device
from envs.drone_env import DroneEnv
from envs.fast_drone_env import FastDroneEnv

LEVELS = ["configs/drone_rescue.yaml", "configs/levels/n1_mision_base.yaml", "configs/levels/n3_operacion_exigente.yaml"]


@pytest.mark.parametrize("config", LEVELS)
def test_fast_env_matches_official_env(config):
    official, fast = DroneEnv(config), FastDroneEnv(config)
    rng = np.random.default_rng(0)
    for episode in range(5):
        a, _ = official.reset(seed=episode)
        b, _ = fast.reset(seed=episode)
        assert np.array_equal(a, b)
        for _ in range(150):
            action = int(rng.integers(7))
            a, ra, ta, tra, ia = official.step(action)
            b, rb, tb, trb, ib = fast.step(action)
            assert np.array_equal(a, b) and ra == rb and ta == tb and tra == trb and ia["cause"] == ib["cause"]
            if ta or tra:
                break


@pytest.mark.parametrize("config", LEVELS)
def test_fast_sensors_match_at_random_positions(config):
    official, fast = DroneEnv(config), FastDroneEnv(config)
    rng = np.random.default_rng(1)
    for _ in range(300):
        official.position = fast.position = rng.uniform([0, 0, 0], [100, 100, 60])
        assert np.array_equal(official._sensor_readings(), fast._sensor_readings())


def test_features_have_fixed_size_and_scale():
    observation, _ = DroneEnv("configs/levels/n1_mision_base.yaml").reset(seed=0)
    features = make_features(observation)
    assert features.shape == (N_FEATURES,) and features.dtype == np.float32
    assert np.abs(features).max() <= 1.5


def test_n_step_discounts_and_terminal():
    buffer = ReplayBuffer(16, pick_device("cpu"))
    collector = NStepCollector(buffer, n=2, gamma=0.5)
    zero = np.zeros(N_FEATURES, dtype=np.float32)
    collector.add(zero, 0, 1.0, zero, False, False)
    collector.add(zero, 1, 2.0, zero, False, False)  # 1 + 0.5 * 2, sigue con gamma^2
    collector.add(zero, 2, 4.0, zero, True, False)   # 2 + 0.5 * 4 y luego 4; terminal
    buffer.flush()
    assert buffer.rew[:3].tolist() == [2.0, 4.0, 4.0]
    assert buffer.discount[:3].tolist() == [0.25, 0.0, 0.0]
    assert buffer.act[:3].tolist() == [0, 1, 2]


def test_truncation_keeps_bootstrap():
    buffer = ReplayBuffer(16, pick_device("cpu"))
    collector = NStepCollector(buffer, n=3, gamma=0.5)
    zero = np.zeros(N_FEATURES, dtype=np.float32)
    collector.add(zero, 0, 1.0, zero, False, True)
    buffer.flush()
    assert buffer.discount[0].item() == 0.5


def test_checkpoint_roundtrip_gives_same_actions(tmp_path: Path):
    agent = DQNAgent(device="cpu", seed=3)
    path = tmp_path / "dqn.pt"
    agent.save(path)
    loaded = DQNPolicy(DQNAgent.load(path))
    policy = DQNPolicy(agent)
    observation, _ = DroneEnv("configs/levels/n1_mision_base.yaml").reset(seed=0)
    assert policy.act(observation) == loaded.act(observation)
