import gymnasium as gym
import numpy as np
import pytest
import yaml

import env
import envs
from pipelines.evaluate import evaluate, load_q_table, load_suite, save_q_table
from agent.qlearning import QAgent as LegacyQAgent
from agents.qlearning import QAgent
from env.drone_env import DroneEnv as LegacyDroneEnv
from envs.drone_env import DEFAULT_CONFIG_PATH, ENVIRONMENT_VERSION, DroneEnv, discretize_observation, load_config


def quiet_environment():
    environment = DroneEnv()
    environment.reset(seed=0)
    environment.wind = np.zeros(3)
    return environment


def test_registration_versions_and_legacy_imports():
    assert env.DroneEnv is envs.DroneEnv is DroneEnv
    assert LegacyDroneEnv is DroneEnv
    assert LegacyQAgent is QAgent
    assert ENVIRONMENT_VERSION == "2.0.0"
    registered = gym.make("DroneRescue-v0")
    assert isinstance(registered.unwrapped, DroneEnv)


def test_reset_is_reproducible_and_observation_is_valid():
    environment = DroneEnv()
    first, first_info = environment.reset(seed=42)
    first_wind = first_info["wind_mps2"]
    second, second_info = environment.reset(seed=42)

    assert np.array_equal(first, second)
    assert np.array_equal(first_wind, second_info["wind_mps2"])
    assert environment.observation_space.contains(second)
    assert environment.position.tolist() == environment.config["world"]["base_center_m"]
    assert environment.velocity.tolist() == [0.0, 0.0, 0.0]
    assert environment.battery_wh == environment.config["battery"]["capacity_wh"]
    assert not environment.has_supply
    assert environment.contacts == 0
    assert environment.stage == "outbound"


def test_physics_and_reward_version_are_loaded_from_yaml(tmp_path):
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["world"]["cell_size_m"] = 10.0
    config["world"]["grid_cells"] = [10, 10, 6]
    config["world"]["base_center_m"] = [5.0, 5.0, 0.0]
    config["world"]["delivery_center_m"] = [95.0, 95.0, 0.0]
    config["physics"]["dt_s"] = 1.0
    config["reward"]["version"] = "test-version"
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    environment = DroneEnv(config_path=config_path)
    environment.wind = np.zeros(3)

    _, reward, _, _, info = environment.step(4)

    assert np.allclose(environment.position, [5.0, 5.0, 2.0])
    assert np.allclose(environment.velocity, [0.0, 0.0, 2.0])
    assert info["reward_version"] == "test-version"
    assert environment.environment_version == ENVIRONMENT_VERSION
    assert reward > config["reward"]["step_penalty"]


def test_transition_1_takeoff_picks_up_supply_and_changes_active_target():
    environment = quiet_environment()
    old_distance = np.linalg.norm(environment._delivery - environment._base)

    observation, reward, terminated, truncated, info = environment.step(4)

    new_distance = np.linalg.norm(environment._delivery - environment.position)
    assert np.allclose(environment.position, [2.5, 2.5, 0.5])
    assert np.allclose(environment.velocity, [0.0, 0.0, 1.0])
    assert environment.has_supply and environment.stage == "delivery"
    assert reward == pytest.approx(-0.1 + 10.0 + max(0.0, old_distance - new_distance) * 0.05)
    assert np.allclose(observation[:3], environment._delivery - environment.position)
    assert observation[14] == 1.0 and observation[16] == 1.0
    assert info["cause"] is None and not terminated and not truncated


def test_transition_2_wind_and_acceleration_update_position_velocity_and_reward():
    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = np.array([50.0, 50.0, 10.0])
    environment.wind = np.array([0.2, 0.0, 0.0])
    environment._best_distance_m = environment._distance_to_target(environment.position)
    old_distance = environment._best_distance_m

    observation, reward, terminated, truncated, info = environment.step(0)

    expected_velocity = np.array([1.1, 0.0, 0.0])
    expected_position = np.array([50.55, 50.0, 10.0])
    new_distance = np.linalg.norm(environment._delivery - expected_position)
    assert np.allclose(environment.velocity, expected_velocity)
    assert np.allclose(environment.position, expected_position)
    assert reward == pytest.approx(-0.1 + 0.05 * (old_distance - new_distance))
    assert np.allclose(observation[:3], environment._delivery - expected_position)
    assert not terminated and not truncated and info["cause"] is None


def test_transition_3_release_outside_zone_has_no_delivery_bonus():
    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = np.array([50.0, 50.0, 10.0])
    environment._best_distance_m = environment._distance_to_target(environment.position)

    observation, reward, terminated, truncated, info = environment.step(6)

    assert reward == pytest.approx(-0.1 - 0.5)
    assert environment.has_supply and environment.stage == "delivery"
    assert observation[14] == 1.0 and observation[16] == 1.0
    assert not environment._delivered
    assert not terminated and not truncated and info["cause"] is None


def test_transition_4_valid_release_is_slow_and_switches_target_to_base():
    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = environment._delivery + np.array([0.0, 0.0, 1.0])
    environment._best_distance_m = environment._distance_to_target(environment.position)

    observation, reward, terminated, truncated, info = environment.step(6)

    assert reward == pytest.approx(-0.1 + 30.0)
    assert not environment.has_supply and environment.stage == "return"
    assert np.allclose(observation[:3], environment._base - environment.position)
    assert observation[14] == 0.0 and observation[17] == 1.0
    assert info["cause"] is None and not terminated and not truncated


def test_transition_5_return_to_base_with_battery_completes_mission():
    environment = quiet_environment()
    environment.stage = "return"
    environment.position = environment._base + np.array([0.2, 0.0, 0.5])
    environment.velocity = np.array([-1.1, 0.0, 0.0])
    environment._best_distance_m = environment._distance_to_target(environment.position)
    old_distance = environment._best_distance_m

    observation, reward, terminated, truncated, info = environment.step(0)

    expected_position = np.array([2.684375, 2.5, 0.5])
    new_distance = np.linalg.norm(expected_position - environment._base)
    assert np.allclose(environment.position, expected_position)
    assert environment.velocity[0] == pytest.approx(-0.03125)
    assert reward == pytest.approx(-0.1 + 0.05 * (old_distance - new_distance) + 100.0)
    assert terminated and not truncated and info["cause"] == "success"
    assert environment.stage == "complete" and observation[18] == 1.0
    assert environment.battery_wh > 0.0


def test_collision_and_out_of_bounds_are_terminated_with_causes():
    environment = quiet_environment()
    environment.position = np.array([19.0, 15.0, 10.0])
    environment.stage = "return"
    _, _, terminated, truncated, info = environment.step(0)
    assert terminated and not truncated and info["cause"] == "collision"
    assert environment.contacts == 1

    environment.reset(seed=1)
    environment.position = np.array([99.8, 10.0, 10.0])
    environment.velocity = np.array([3.0, 0.0, 0.0])
    environment.stage = "return"
    _, _, terminated, truncated, info = environment.step(0)
    assert terminated and not truncated and info["cause"] == "out_of_bounds"


def test_battery_zero_terminates_and_step_limit_truncates():
    environment = quiet_environment()
    environment.battery_wh = 0.0001
    environment.stage = "delivery"
    environment.has_supply = True
    _, _, terminated, truncated, info = environment.step(0)
    assert terminated and not truncated and info["cause"] == "battery_depleted"

    environment = quiet_environment()
    environment.stage = "return"
    environment.position = environment._base + np.array([0.2, 0.0, 0.5])
    environment.velocity = np.array([-1.1, 0.0, 0.0])
    environment.battery_wh = 0.0001
    _, _, terminated, truncated, info = environment.step(0)
    assert terminated and not truncated and info["cause"] == "battery_depleted"
    assert environment.stage == "return"

    environment = quiet_environment()
    environment._mission["max_steps"] = 1
    _, _, terminated, truncated, info = environment.step(6)
    assert not terminated and truncated and info["cause"] == "step_limit"


def test_invalid_and_repeated_release_never_reaward_delivery():
    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = environment._delivery + np.array([0.0, 0.0, 1.0])
    _, first_reward, _, _, first_info = environment.step(6)
    assert first_reward == pytest.approx(29.9)
    assert first_info["stage"] == "return"

    _, repeated_reward, terminated, truncated, info = environment.step(6)
    assert repeated_reward == pytest.approx(-0.6)
    assert not environment.has_supply and not environment._returned
    assert not terminated and not truncated and info["cause"] is None

    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = environment._delivery + np.array([0.0, 0.0, 1.0])
    environment.velocity = np.array([0.8, 0.0, 0.0])
    _, fast_release_reward, _, _, _ = environment.step(6)
    assert fast_release_reward == pytest.approx(-0.6)
    assert environment.has_supply and not environment._delivered


def test_progress_reward_only_sets_new_best_distance_for_active_target():
    environment = quiet_environment()
    environment.stage = "delivery"
    environment.has_supply = True
    environment.position = np.array([50.0, 50.0, 10.0])
    environment._best_distance_m = environment._distance_to_target(environment.position)
    _, forward_reward, _, _, _ = environment.step(0)
    assert forward_reward > environment.config["reward"]["step_penalty"]

    environment.velocity = np.zeros(3)
    environment.wind = np.zeros(3)
    _, backward_reward, _, _, _ = environment.step(1)
    assert backward_reward == pytest.approx(environment.config["reward"]["step_penalty"])


def test_q_learning_discretization_and_legacy_q_agent_api():
    environment = quiet_environment()
    observation, _ = environment.reset(seed=7)
    state = discretize_observation(observation, environment.config)
    agent = QAgent(n_actions=7, eps=0, seed=0)
    agent.update(state, 1, 2.0, state, done=True)

    assert len(state) == 16
    assert agent.q[state][1] == pytest.approx(0.4)
    assert agent.q[0].shape == (7,)
    assert 0 <= agent.act(state, greedy=True) < 7

    legacy_agent = QAgent(shape=(2,), n_actions=2, alpha=0.5, gamma=0.9, eps=0, seed=0)
    legacy_agent.update((0,), 1, 2.0, (1,), done=True)
    assert np.array_equal(legacy_agent.q[0], [0.0, 1.0])


CONFIG_VARIANTS = [
    "configs/levels/n1_mision_base.yaml",
    "configs/levels/n2_mas_decisiones.yaml",
    "configs/levels/n3_operacion_exigente.yaml",
    "configs/experiments/exp-001.yaml",
    "configs/experiments/exp-002.yaml",
    "configs/experiments/exp-003.yaml",
    "configs/experiments/exp-004.yaml",
]


@pytest.mark.parametrize("config_path", CONFIG_VARIANTS)
def test_versioned_level_and_experiment_configs_are_executable(config_path):
    environment = DroneEnv(config_path=config_path)
    observation, info = environment.reset(seed=2026)
    assert environment.observation_space.contains(observation)
    assert environment.action_space.n == 7
    assert info["stage"] == "outbound"


@pytest.mark.parametrize(
    ("experiment_id", "expected_dimension"),
    [
        ("exp-001", "world.obstacles"),
        ("exp-002", "battery.capacity_wh"),
        ("exp-003", "wind.acceleration_bounds"),
        ("exp-004", "physics.mass_kg"),
    ],
)
def test_experiments_declare_one_baseline_dimension(experiment_id, expected_dimension):
    from pathlib import Path

    experiment_path = Path("configs/experiments") / f"{experiment_id}.yaml"
    environment = DroneEnv(config_path=experiment_path)
    experiment = environment.config["experiment"]
    assert experiment["id"] == experiment_id
    assert experiment["changed_dimension"] == expected_dimension
    assert experiment["episode_budget"] == 5000
    assert len(experiment["seeds"]) == 5
    assert experiment["model_family"] == "drone-rescue-v2"
    assert environment.observation_space.shape == (19,)
    assert environment.action_space.n == 7

    baseline = load_config()
    changed_sections = {
        key for key in ("world", "physics", "wind", "battery")
        if environment.config[key] != baseline[key]
    }
    expected_section = expected_dimension.split(".")[0]
    assert changed_sections == {expected_section}
    if expected_section == "world":
        assert {key for key in environment.config["world"] if environment.config["world"][key] != baseline["world"][key]} == {"obstacles"}
    elif expected_section in {"physics", "battery"}:
        section = expected_section
        assert {key for key in environment.config[section] if environment.config[section][key] != baseline[section][key]} == {expected_dimension.split(".")[1]}
    elif expected_section == "wind":
        assert {key for key in environment.config["wind"] if environment.config["wind"][key] != baseline["wind"][key]} == {
            "acceleration_min_mps2", "acceleration_max_mps2"
        }


def test_fixed_evaluation_suite_has_public_seeds_and_three_references():
    suite, _ = load_suite()
    assert suite["evaluation_suite_id"] == "drone-rescue-public-validation"
    assert suite["version"] == "1.0.0"
    assert suite["split"] == "public_validation"
    assert suite["episodes_per_level"] == len(suite["seeds"]) == 10
    assert suite["references"] == ["random", "rule", "q_learning_greedy"]


def test_evaluation_writes_case_metrics_without_cross_level_return_aggregation(tmp_path):
    from pathlib import Path

    config = load_config("configs/levels/n1_mision_base.yaml")
    config["mission"]["max_steps"] = 12
    config_path = tmp_path / "short_n1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    suite = {
        "evaluation_suite_id": "test-public-suite",
        "version": "test-1",
        "split": "public_validation",
        "episodes_per_level": 1,
        "seeds": [42],
        "levels": [{"id": "N1", "config": str(config_path)}],
        "references": ["random", "rule", "q_learning_greedy"],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    q_path = tmp_path / "q_table.npz"
    np.savez_compressed(q_path, states=np.zeros((1, 16), dtype=np.int64), q_values=np.zeros((1, 7)))
    assert len(load_q_table(q_path)) == 1

    output = tmp_path / "evaluacion.json"
    result = evaluate(suite_path, q_path, output)

    assert output.exists()
    assert len(result["cases"]) == 3
    assert "mean_return" not in result
    for case in result["cases"]:
        assert case["metrics"]["episodes"] == 1
        assert "success_rate" in case["metrics"] and "mean_return" in case["metrics"]
        episode = case["episodes"][0]
        assert set(("success", "cause", "steps", "battery_remaining_wh", "delivery_precision_m",
                    "delivery_speed_mps", "collisions", "return", "inference_latency_ms")).issubset(episode)
    with pytest.raises(FileExistsError):
        evaluate(suite_path, q_path, output)


def test_reserved_test_suite_is_rejected(tmp_path):
    suite = {
        "evaluation_suite_id": "private-test",
        "version": "1.0.0",
        "split": "test_reserved",
        "episodes_per_level": 1,
        "seeds": [5],
        "levels": [],
        "references": ["random", "rule", "q_learning_greedy"],
    }
    path = tmp_path / "reserved.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="public_validation"):
        load_suite(path)


def test_q_table_checkpoint_round_trip(tmp_path):
    agent = QAgent(n_actions=7, eps=0, seed=1)
    agent.q[(1,) * 16][4] = 3.0
    checkpoint = save_q_table(agent, tmp_path / "table.npz")
    loaded = load_q_table(checkpoint)
    assert np.array_equal(loaded[(1,) * 16], agent.q[(1,) * 16])
    with pytest.raises(FileExistsError):
        save_q_table(agent, checkpoint)
