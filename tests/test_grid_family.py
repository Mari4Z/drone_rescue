import hashlib
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
import envs.grid_drone_env as grid
from agents.grid.astar import AwareAStarReplanner, astar
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


def _storm(seed):
    env = gym.make("DroneRescueGridStorm-v3").unwrapped
    env.reset(seed=seed)
    return env


def test_storm_observation_and_reproducibility():
    a, b = gym.make("DroneRescueGridStorm-v3").unwrapped, gym.make("DroneRescueGridStorm-v3").unwrapped
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert oa.shape == (grid.STORM_N_OBS,) and np.array_equal(oa, ob)
    assert a.columns == b.columns and a.goal == b.goal and a.wind == b.wind and a.capacity == b.capacity
    assert grid.STORM_COLUMNS[0] <= len(a.columns) <= grid.STORM_COLUMNS[1]
    assert a.goal[0] + a.goal[1] >= grid.GOAL_MIN_DIST and not a.blocked(*a.goal)
    for action in (4, 4, 4, 4, 0, 2, 0):
        ra, rb = a.step(action), b.step(action)
        assert np.array_equal(ra[0], rb[0]) and ra[1:4] == rb[1:4] and a.pos == b.pos


def test_storm_gust_can_crash_into_a_building():
    env = _storm(3)
    env.wind, env.wind_dir = 1.0, np.array([1.0, 0.0])  # a la altura máxima la ráfaga hacia +x es segura
    env.obstacles[:] = False
    env.obstacles[6, 0, :] = True                          # edificio justo a sotavento
    env.pos = (4, 0, grid.NZ - 1)
    _, reward, terminated, _, info = env.step(0)           # avanza a (5, 0, 11) y la ráfaga lo empuja a (6, 0, 11)
    assert terminated and reward == grid.R_CRASH and info["result"] == "crash" and info["cause"] == "viento"


def test_storm_battery_runs_out_and_headwind_costs_more():
    env = _storm(5)
    env.wind, env.wind_dir = 0.3, np.array([1.0, 0.0])
    assert env.move_energy((5, 5, 11), 1) > env.move_energy((5, 5, 1), 1) > grid.ENERGY_H > env.move_energy((5, 5, 11), 0)
    env.wind = 0.0  # sin ráfagas: el resultado depende solo de la batería
    env.pos, env.battery = (0, 0, 0), 1.0
    _, reward, terminated, _, info = env.step(4)
    assert terminated and info["result"] == "battery" and reward == grid.R_BATTERY


def test_storm_planner_matches_python_astar_and_battery_fits_best_route():
    from types import SimpleNamespace

    for seed in range(8):
        env = _storm(seed)
        route = astar(env)  # usa el Dijkstra vectorizado del ambiente
        assert route[0] == env.start and route[-1] == env.goal and not any(env.blocked(*c) for c in route)
        proxy = SimpleNamespace(blocked=env.blocked, plan_cost=env.plan_cost, plan_heuristic=env.plan_heuristic,
                                start=env.start, goal=env.goal, min_cruise_z=env.min_cruise_z)
        reference = astar(proxy)  # el A* en Python, con el mismo coste
        cost = lambda path: sum(env.plan_cost(p, grid.ACTIONS.index(tuple(b - a for a, b in zip(p, q))), q) for p, q in zip(path, path[1:]))
        assert cost(route) == pytest.approx(cost(reference))
        lo, hi = grid.BATTERY_MARGIN
        energy = env.route_energy(env.best_route)
        assert energy * lo + grid.BATTERY_RESERVE <= env.capacity + 1e-9 <= energy * hi + grid.BATTERY_RESERVE + 1e-9


def test_storm_replanner_and_cli_reject_v1_model(tmp_path):
    from agents.grid.astar import AStarReplanner
    from agents.grid.evaluate import evaluate_storm

    result = evaluate_storm(None, episodes=3, seed=600000, make_policy=lambda e: AStarReplanner(e, risk_weight=30))
    assert result["metrics"]["episodes"] == 3 and len(result["episodes"]) == 3
    old = DQNAgent(46, 6, device="cpu")
    old.save(tmp_path / "v1.pt")
    run = subprocess.run([sys.executable, "cli.py", "eval", "--model", str(tmp_path / "v1.pt"), "--env", "DroneRescueGridStorm-v3",
                          "--episodes", "1"], capture_output=True, text=True)
    assert run.returncode != 0 and "espera 46 observaciones" in run.stderr


def _city(seed):
    env = gym.make("DroneRescueGridCity-v4").unwrapped
    env.reset(seed=seed)
    return env


def test_city_map_and_observation():
    a, b = _city(7), _city(7)
    assert a.shape == grid.CITY_SHAPE and a.observation_space.shape == (grid.CITY_N_OBS,)
    assert a.columns == b.columns and a.lines == b.lines and a.birds == b.birds and a.goal == b.goal
    assert grid.CITY_COLUMNS[0] <= len(a.columns) <= grid.CITY_COLUMNS[1] and len(a.birds) <= grid.CITY_BIRDS[1]
    for x, y in a.poles:  # poste: bloqueado hasta 10 m; cable: solo la celda de 5 a 10 m
        assert a.blocked(x, y, 1) and not a.blocked(x, y, grid.CITY_POLE_HEIGHT)
    for x, y in a.cables:
        assert a.blocked(x, y, 1) and not a.blocked(x, y, 2)
    route = astar(a)
    assert route[0] == a.start and route[-1] == a.goal and not any(a.blocked(*c) for c in route)
    for bird in a.birds:
        assert not a.blocked(*bird) and grid.CITY_BIRD_Z[0] <= bird[2] <= grid.CITY_BIRD_Z[1]


def test_city_bird_collision_and_sensing():
    env = _city(2)
    env.wind = 0.0
    env.birds = [(10, 10, 5)]
    env.pos = (10, 8, 5)
    features = env._bird_features()
    assert features[2] == pytest.approx(1 / grid.CITY_BIRD_SENSE)  # +y: el ave está a 2 celdas, 1 libre
    assert features[6] == pytest.approx(1 / grid.CITY_BIRDS[1])    # una ave cerca
    env.pos = (10, 9, 5)
    _, reward, terminated, _, info = env.step(2)                    # +y: entra en la celda del ave
    assert terminated and reward == grid.R_BIRD and info["cause"] == "ave"


def test_city_birds_stay_off_obstacles_and_history_matches_steps():
    env = _city(4)
    env.wind = 0.0
    steps = 0
    done = False
    while not done and steps < 40:
        _, _, terminated, truncated, _ = env.step(4 if steps < 3 else 0)
        done = terminated or truncated
        steps += 1
        assert all(not env._static_blocked(*b) for b in env.birds) and len(set(env.birds)) == len(env.birds)
    assert len(env.bird_history) == steps + 1


# Huella de los mapas 700000 a 700004 de DroneRescueGridCity-v4 (ambiente 3.0.0). Si cambia, se rompió la
# reproducibilidad del experimento grid-city-dqn: el ambiente nuevo debe ir en otra clase y otro identificador.
CITY_V4_DIGEST = "4232ee2b2781c4dba02c17db22d989da26a101b2d2f7b431f0fe8d4a0d844eda"


def test_city_v4_maps_are_frozen():
    env = gym.make("DroneRescueGridCity-v4").unwrapped
    parts = []
    for seed in range(700000, 700005):
        obs, _ = env.reset(seed=seed)
        parts.append([env.columns, env.lines, env.poles, env.cables, env.birds, env.goal, round(float(env.wind), 6),
                      round(float(env.capacity), 4), hashlib.sha256(obs.tobytes()).hexdigest()[:16]])
    assert hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest() == CITY_V4_DIGEST


def _hard(seed, **kwargs):
    env = grid.DroneCityHardEnv(**kwargs)
    env.reset(seed=seed)
    return env


def test_hard_city_layout_battery_and_reproducibility():
    a, b = _hard(5), _hard(5)
    assert a.start == grid.HARD_START and a.columns == b.columns and a.lines == b.lines and a.birds == b.birds and a.goal == b.goal
    assert a.observation_space.shape == (grid.CITY_N_OBS,) and a._obs().shape == (grid.CITY_N_OBS,)
    assert grid.HARD_BIRDS[0] <= len(a.birds) <= grid.HARD_BIRDS[1]
    assert not a.blocked(*a.start) and not a.blocked(*a.goal)
    route = astar(a)
    assert route[0] == a.start and route[-1] == a.goal and not any(a.blocked(*c) for c in route)
    low, high = grid.HARD_BATTERY_MARGIN
    energy = a.route_energy(a.best_route)
    assert energy * low + grid.BATTERY_RESERVE <= a.capacity + 1e-9 <= energy * high + grid.BATTERY_RESERVE + 1e-9


def test_hard_poles_reach_cruise_altitude_and_cables_sit_in_it():
    top, (z0, z1) = grid.HARD_POLE_TOP, grid.HARD_CABLE_Z
    assert z0 * grid.CELL <= 20 < (z1 + 1) * grid.CELL  # la banda del cable contiene la altura de crucero (20 m)
    longest, lines = 0, 0
    for seed in range(12):
        env = _hard(seed)
        lines += len(env.lines)
        for _, _, positions in env.lines:
            longest = max(longest, (positions[-1] - positions[0]) * grid.CELL)
        for x, y in env.poles:  # el poste está en suelo libre y llega a 30 m
            assert not env.obstacles[x, y, 0] and all(env.blocked(x, y, z) for z in range(1, top))
            assert env.blocked(x, y, top) == bool(env.obstacles[x, y, top])
        for x, y in env.cables:  # el cable bloquea su banda, deja pasar por encima y por debajo (salvo edificios)
            assert all(env._static[x, y, z] for z in range(z0, z1 + 1))
            assert env.blocked(x, y, z1 + 1) == bool(env.obstacles[x, y, z1 + 1])
            assert env.blocked(x, y, z0 - 1) == bool(env.obstacles[x, y, z0 - 1])
        keep = [env.start, env.goal]
        assert not any(abs(x - k[0]) <= grid.SAFE_ZONE and abs(y - k[1]) <= grid.SAFE_ZONE for x, y in env.poles + env.cables for k in keep)
    assert lines >= 12 and longest >= 100  # hay tendidos y alguno cruza al menos la mitad del mapa


def test_hard_birds_spawn_near_route_and_avoid_obstacles():
    for seed in range(8):
        env = _hard(seed)
        route = env.best_route[5:-5] or env.best_route
        for bird in env.birds:
            assert not env.blocked(*bird) and grid.HARD_BIRD_SPAWN_Z[0] <= bird[2] <= grid.HARD_BIRD_SPAWN_Z[1]
            assert min(max(abs(bird[0] - c[0]), abs(bird[1] - c[1])) for c in route) <= grid.HARD_BIRD_NEAR_ROUTE


def test_hard_birds_keep_heading_speed_and_danger_zone():
    env = _hard(3, bird_speed=2)
    env.wind = 0.0
    env.obstacles[:] = False
    env.poles, env.cables, env.lines = [], [], []
    env._static = env.static_obstacles()
    env.birds, env._bird_heading, env._bird_left = [(10, 10, 5)], [(1, 0)], [10]
    env._move_birds()
    assert env.birds[0][:2] == (12, 10)  # 2 celdas por paso, con su rumbo
    env.birds, env.bird_speed = [(10, 10, 5)], 1
    env._bird_heading, env._bird_left = [(1, 0)], [10]
    for expected in (11, 12, 13):
        env._move_birds()
        assert env.birds[0][:2] == (expected, 10)  # mantiene el rumbo mientras le quedan pasos
    # sensor: la zona de peligro de 1 celda empieza una celda antes del ave
    env.birds, env.pos = [(10, 10, 5)], (10, 8, 5)
    features = env._bird_features()
    assert features[2] == 0.0 and features[0] == 1.0 and features[6] == pytest.approx(1 / grid.HARD_BIRDS[1])
    # entrar en la zona de peligro es choque; quedarse a 2 celdas no
    env.pos, env.battery, env.steps = (10, 8, 5), env.capacity, 0
    _, reward, terminated, _, info = env.step(2)
    assert terminated and reward == grid.R_BIRD and info["cause"] == "ave"
    env.birds, env._bird_heading, env._bird_left = [(10, 10, 5)], [(1, 0)], [10]
    env.pos, env.battery, env.steps = (10, 7, 5), env.capacity, 0
    _, _, terminated, _, info = env.step(2)
    assert not terminated and info["result"] == "flying"


def test_hard_env_variants_and_aware_planner_run():
    from agents.grid.evaluate import evaluate_storm

    soft = grid.DroneCityHardEnv(start=(0, 0, 0), pole_top=2, cable_z=(1, 1), long_lines=False, bird_mode="walk",
                                 bird_radius=0, n_birds=(0, 6), battery_margin=grid.BATTERY_MARGIN)
    soft.reset(seed=1)
    for _ in range(20):
        _, _, terminated, truncated, _ = soft.step(4)
        if terminated or truncated:
            break
    result = evaluate_storm(None, "DroneRescueGridCityHard-v0", episodes=3, seed=800000,
                            make_policy=lambda e: AwareAStarReplanner(e, risk_weight=30))
    assert result["metrics"]["episodes"] == 3 and len(result["episodes"]) == 3
    env = _hard(2)
    AwareAStarReplanner(env, risk_weight=30).act()
    assert all(env._bird_cost[b] > 0 for b in env.birds)  # las celdas de cada ave cuestan más


def test_cli_exports_hard_city_viewer_with_cable_band_and_bird_zones(tmp_path):
    model, page = tmp_path / "hard.pt", tmp_path / "vista.html"
    DQNAgent(grid.CITY_N_OBS, 6, device="cpu").save(model)
    run = subprocess.run([sys.executable, "cli.py", "run", "--env", "DroneRescueGridCityHard-v0", "--model", str(model), "--seed", "800000",
                          "--maps", "2", "--web", "--web-out", str(page), "--no-open"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    html = page.read_text(encoding="utf-8")
    assert '"cable_z": [4, 5]' in html and '"bird_radius": 1' in html and '"start": [4, 4, 0]' in html
    assert "/*DATA*/null" not in html and '"size": [40, 40, 12]' in html


def test_grid_trainer_contracts_follow_the_environment():
    from pipelines.train_grid import ENVIRONMENTS, input_contract

    for env_id, spec in ENVIRONMENTS.items():
        contract = input_contract(env_id)
        assert contract["environment_id"] == env_id and contract["environment_version"] == spec["version"]
        assert contract["observation_dimensions"] == gym.make(env_id).observation_space.shape[0]
    assert ENVIRONMENTS["DroneRescueGridCity-v4"]["family"] == ENVIRONMENTS["DroneRescueGridCityHard-v0"]["family"]


@pytest.mark.parametrize("args", [["envs"], ["rules", "--env", "DroneRescueGridRandom-v2"], ["rules", "--env", "DroneRescueGridStorm-v3"],
                                  ["rules", "--env", "DroneRescueGridCity-v4"],
                                  ["rules", "--env", "DroneRescueGridCityHard-v0"]])
def test_cli_commands_run(args):
    result = subprocess.run([sys.executable, "cli.py", *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_rejects_incompatible_environment(tmp_path):
    table = tmp_path / "q.npy"
    np.save(table, np.zeros((20, 20, 12, 6)))
    result = subprocess.run([sys.executable, "cli.py", "eval", "--model", str(table), "--env", "DroneRescueGridRandom-v2"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "qlearning necesita" in result.stderr
