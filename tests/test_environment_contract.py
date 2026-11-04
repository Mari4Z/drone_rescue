import gymnasium as gym
import numpy as np

import env
import envs
from agent.qlearning import QAgent as LegacyQAgent
from agents.qlearning import QAgent
from env.drone_env import DroneEnv as LegacyDroneEnv
from envs.drone_env import DroneEnv


def test_environment_registration_and_legacy_imports():
    assert env.DroneEnv is envs.DroneEnv is DroneEnv
    assert LegacyDroneEnv is DroneEnv
    assert LegacyQAgent is QAgent
    assert isinstance(gym.make("DroneRescue-v0").unwrapped, DroneEnv)


def test_reset_and_vertical_takeoff_keep_existing_state_rules():
    environment = DroneEnv()
    observation, _ = environment.reset(seed=7)

    assert tuple(observation) == (0, 0, 0)
    next_observation, reward, terminated, truncated, info = environment.step(4)

    assert tuple(next_observation) == (0, 0, 1)
    assert reward == -2
    assert not terminated and not truncated
    assert info["result"] == "flying"


def test_collision_ends_episode_without_moving_the_drone():
    environment = DroneEnv()
    environment.reset()
    environment.pos = (3, 2, 0)

    observation, reward, terminated, truncated, info = environment.step(0)

    assert tuple(observation) == (3, 2, 0)
    assert reward == -100
    assert terminated and not truncated
    assert info["result"] == "crash"


def test_q_agent_compatibility_and_update():
    agent = QAgent(shape=(2,), n_actions=2, alpha=0.5, gamma=0.9, eps=0, seed=0)
    agent.update((0,), 1, 2.0, (1,), done=True)

    assert np.array_equal(agent.q[0], [0.0, 1.0])
    assert agent.act((0,), greedy=True) in (0, 1)
