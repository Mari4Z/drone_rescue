import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
import envs.grid_drone_env as grid
from agents.grid.astar import astar
from agents.grid.dqn import DQNAgent
from agents.grid.evaluate import evaluate, run_episode
from agents.grid.train import train
from envs.drone_env import DroneEnv
from viewer.export import export_html

GRID_IDS = list(grid._GRID_ENVS)


def test_grid_ids_do_not_replace_continuous_environment():
    assert isinstance(gym.make("DroneRescue-v0").unwrapped, DroneEnv)
    assert not any(isinstance(gym.make(env_id).unwrapped, DroneEnv) for env_id in GRID_IDS)


@pytest.mark.parametrize("env_id", GRID_IDS)
def test_grid_environments_follow_gymnasium_contract(env_id):
    check_env(gym.make(env_id).unwrapped, skip_render_check=True)


def test_ground_is_only_passable_at_a_and_b():
    env = gym.make("DroneRescueGrid-v0").unwrapped
    env.reset(seed=0)
    assert not env.blocked(*grid.START) and not env.blocked(*grid.GOAL)
    assert env.blocked(1, 0, 0) and not env.blocked(1, 0, 1)
    _, reward, terminated, _, info = env.step(0)  # horizontal desde A a ras de suelo
    assert terminated and info["result"] == "crash" and reward == grid.R_CRASH


def test_astar_returns_valid_route_on_random_maps():
    env = gym.make("DroneRescueGridRandom-v2").unwrapped
    for seed in range(5):
        env.reset(seed=seed)
        path = astar(env)
        assert path[0] == grid.START and path[-1] == grid.GOAL
        assert not any(env.blocked(*cell) for cell in path)
        assert all(sum(abs(a - b) for a, b in zip(p, q)) == 1 for p, q in zip(path, path[1:]))


def test_memory_observation_counts_visits():
    env = gym.make("DroneRescueGridRandom-v2").unwrapped
    env.reset(seed=0)
    env.step(4)
    env.step(0)
    observation, *_ = env.step(1)
    assert observation.shape == (46,)
    assert observation[39] == pytest.approx(2 / 3)


def test_tabular_training_and_dqn_policy_run(tmp_path):
    env, agent, history = train(episodes=20, env_id="DroneRescueGridDummy-v1")
    assert len(history) == 20 and agent.q.shape == (20, 20, 12, 6)
    dqn = DQNAgent(46, 6, device="cpu")
    dqn.save(tmp_path / "grid.pt")
    loaded = DQNAgent.load(tmp_path / "grid.pt", "cpu")
    metrics = evaluate(loaded, "DroneRescueGridFixed-v2", episodes=1)
    assert set(metrics["results"]) <= {"goal", "crash", "timeout"}


def test_viewer_embeds_real_map_and_route(tmp_path):
    env = gym.make("DroneRescueGridRandom-v2")
    path, result = run_episode(env, lambda _: 4, seed=3)
    out = tmp_path / "vista.html"
    export_html([{"seed": 3, "columns": env.unwrapped.columns, "path": path, "result": result, "astar": 1}], "grid", out)
    html = out.read_text(encoding="utf-8")
    assert "/*DATA*/null" not in html and '"seed": 3' in html


@pytest.mark.parametrize("args", [["envs"], ["rules", "--env", "DroneRescueGridRandom-v2"]])
def test_cli_commands_run(args):
    result = subprocess.run([sys.executable, "cli.py", *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_rejects_incompatible_environment(tmp_path):
    table = tmp_path / "q.npy"
    np.save(table, np.zeros((20, 20, 12, 6)))
    result = subprocess.run([sys.executable, "cli.py", "eval", "--model", str(table), "--env", "DroneRescueGridRandom-v2"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "qlearning necesita" in result.stderr
