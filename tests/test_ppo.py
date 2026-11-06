import json
import os
from pathlib import Path

import numpy as np
import pytest

from agents.dqn import make_features
from agents.ppo import FORCE_ACTIONS, PPOPolicy, input_contract, make_env
from envs.drone_env import DroneEnv
from envs.fast_drone_env import FastDroneEnv
from envs.stabilized_drone_env import ACTION_NAMES, HOLD, N_ACTIONS, StabilizedDroneEnv, thrust_for
from pipelines import train_ppo

LEVEL = "configs/levels/n1_mision_base.yaml"


def _deliver_position(env, height=2.5):
    drone = env.unwrapped
    drone.stage, drone.has_supply = "delivery", True
    drone.position = np.array([95.0, 95.0, height])
    drone.velocity = np.zeros(3)


def test_thrust_for_chooses_axis_with_largest_error():
    assert thrust_for(np.zeros(3), np.array([0.0, -4.0, 1.0])) == 3
    assert thrust_for(np.array([2.0, 0.0, 0.0]), np.zeros(3)) == 1


def test_stabilized_env_keeps_physics_and_decides_every_two_steps():
    env = StabilizedDroneEnv(DroneEnv(LEVEL))
    observation, _ = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS == len(ACTION_NAMES)
    assert observation.shape == (19,)
    env.step(8)
    assert env.unwrapped.steps == 2


def test_hold_delivers_only_inside_zone_and_slow():
    env = StabilizedDroneEnv(DroneEnv(LEVEL))
    env.reset(seed=0)
    _deliver_position(env)
    env.unwrapped.velocity = np.array([2.0, 0.0, 0.0])  # demasiado rápido al entrar: frena primero
    env.step(0)
    assert env.unwrapped.stage == "delivery"
    _deliver_position(env)
    _, reward, *_ = env.step(HOLD)
    assert env.unwrapped.stage == "return" and reward > 20.0


def test_hold_outside_zone_never_delivers_or_penalizes_release():
    env = StabilizedDroneEnv(DroneEnv(LEVEL))
    env.reset(seed=0)
    env.step(8)
    env.unwrapped.position = np.array([50.0, 50.0, 40.0])
    env.unwrapped.velocity = np.zeros(3)
    _, reward, *_ = env.step(HOLD)
    assert env.unwrapped.stage == "delivery" and reward > -0.5


@pytest.mark.parametrize("family", ["drone-rescue-v2", "drone-rescue-v3"])
def test_fast_and_official_families_give_same_transitions(family):
    official, fast = make_env(LEVEL, family), make_env(LEVEL, family, fast=True)
    rng = np.random.default_rng(0)
    a, _ = official.reset(seed=1)
    b, _ = fast.reset(seed=1)
    assert np.array_equal(a, b)
    for _ in range(100):
        action = int(rng.integers(official.action_space.n))
        a, ra, ta, tra, _ = official.step(action)
        b, rb, tb, trb, _ = fast.step(action)
        assert np.array_equal(a, b) and ra == rb and ta == tb and tra == trb
        if ta or tra:
            break


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        make_env(LEVEL, "otra-familia")


def test_contract_declares_actions_and_scaling():
    v2, v3 = input_contract("drone-rescue-v2"), input_contract("drone-rescue-v3")
    assert v2["actions"] == FORCE_ACTIONS and v2["action_dimensions"] == 7
    assert v3["actions"] == ACTION_NAMES and v3["action_dimensions"] == N_ACTIONS
    assert v3["state_dimensions"] == len(make_features(DroneEnv(LEVEL).reset(seed=0)[0]))


def test_mission_score_ignores_return():
    good = [{"success": True, "stage": "complete", "steps": 400, "return": -50.0}]
    bad = [{"success": False, "stage": "delivery", "steps": 100, "return": 500.0}]
    assert train_ppo.mission_score(good) > train_ppo.mission_score(bad)


def test_run_rejected_with_uncommitted_changes(monkeypatch):
    monkeypatch.setattr(train_ppo, "_git", lambda *args: " M algo.py")
    with pytest.raises(RuntimeError, match="sin commit"):
        train_ppo.train_run("configs/training/ppo_v3_n1.yaml", 0)


def test_profile_requires_hypothesis(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("experiment: {id: x}\nlevel: {id: N1, config: a}\nmodel_family: f\ntraining: {timesteps: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hypothesis"):
        train_ppo.load_ppo_profile(path)


def test_complete_run_produces_recoverable_package(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "evaluation_suite_id: s\nversion: 1.0.0\nsplit: public_validation\nepisodes_per_level: 1\nseeds: [101]\n"
        f"levels:\n  - id: N1\n    config: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', LEVEL))}\n"
        "references: [random, rule, q_learning_greedy]\n", encoding="utf-8")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = train_ppo.train_run(
        os.path.join(repo, "configs/training/ppo_v3_n1.yaml"), 3, manifest_dir=tmp_path / "manifests",
        package_dir=tmp_path / "models", suite_path=suite, timesteps_override=200, allow_dirty=True,
    )
    manifest = json.loads(open(result["manifest_path"], encoding="utf-8").read())
    for field in ("run_id", "level_id", "environment_version", "git_commit", "config_sha256", "model_uri", "model_sha256",
                  "normalization_uri", "runtime_ref", "evaluation_suite_id", "seed", "effective_timesteps", "registry_uri"):
        assert manifest[field] not in (None, ""), field
    assert manifest["parent_run_id"] is None and manifest["model_family"] == "drone-rescue-v3"
    model = tmp_path / "models" / manifest["run_id"] / "model.zip"
    assert train_ppo._sha256(model) == manifest["model_sha256"]
    downloaded = mlflow.artifacts.download_artifacts(artifact_uri=manifest["model_uri"])
    assert train_ppo._sha256(Path(downloaded)) == manifest["model_sha256"]
    env = make_env(os.path.join(repo, LEVEL), "drone-rescue-v3")
    features, _ = env.reset(seed=0)
    assert 0 <= PPOPolicy(model).act(features) < N_ACTIONS
