"""Continuous 3D Drone Rescue environment, configured from YAML."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

ENVIRONMENT_VERSION = "2.0.0"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "drone_rescue.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    required = {"reward", "world", "physics", "wind", "battery", "sensors", "mission", "actions", "q_learning"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Faltan secciones de configuración: {sorted(missing)}")
    if config["actions"].get("release_index") != len(config["actions"].get("movement_directions", [])):
        raise ValueError("release_index debe seguir inmediatamente a los seis movimientos")
    if len(config["actions"]["movement_directions"]) != 6:
        raise ValueError("Drone Rescue requiere exactamente seis movimientos")
    if config["battery"]["capacity_wh"] <= 0 or config["physics"]["dt_s"] <= 0:
        raise ValueError("La batería y dt_s deben ser mayores que cero")


def discretize_observation(observation, config: dict | None = None) -> tuple[int, ...]:
    """Discretiza la observación continua para Q-learning tabular disperso.

    La salida contiene 3 bins del vector al objetivo, 3 de velocidad, 1 de
    altura, 6 de sensores, 1 de batería, carga binaria y etapa categórica.
    """
    config = config or load_config()
    bins = config["q_learning"]["discretization"]
    obs = np.asarray(observation, dtype=np.float64)

    def quantize(values, width):
        return tuple(np.floor(np.asarray(values) / width + 0.5).astype(int).tolist())

    target = quantize(obs[0:3], bins["target_component_step_m"])
    velocity = quantize(obs[3:6], bins["velocity_component_step_mps"])
    altitude = quantize(obs[6:7], bins["altitude_step_m"])
    sensors = quantize(obs[7:13], bins["sensor_step_m"])
    battery_bins = int(bins["battery_fraction_bins"])
    battery = min(battery_bins - 1, int(np.floor(obs[13] * battery_bins)))
    has_supply = int(obs[14] >= 0.5)
    stage = int(np.argmax(obs[15:]))
    return target + velocity + altitude + sensors + (battery, has_supply, stage)


class DroneEnv(gym.Env):
    """Misión de salida desde BASE, entrega de suministro y regreso a BASE."""

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.environment_version = ENVIRONMENT_VERSION
        self.reward_version = str(self.config["reward"]["version"])
        self._world = self.config["world"]
        self._physics = self.config["physics"]
        self._battery_config = self.config["battery"]
        self._mission = self.config["mission"]
        self._directions = np.asarray(self.config["actions"]["movement_directions"], dtype=np.float64)
        self._release_action = int(self.config["actions"]["release_index"])
        self._world_size = np.asarray(self._world["grid_cells"], dtype=np.float64) * float(self._world["cell_size_m"])
        self._base = np.asarray(self._world["base_center_m"], dtype=np.float64)
        self._delivery = np.asarray(self._world["delivery_center_m"], dtype=np.float64)
        self.action_space = spaces.Discrete(self._release_action + 1)

        max_speed = float(self._physics["max_speed_mps"])
        sensor_range = float(self.config["sensors"]["range_m"])
        low = np.array(
            [-self._world_size[0], -self._world_size[1], -self._world_size[2],
             -max_speed, -max_speed, -max_speed, 0.0,
             *([0.0] * len(self._directions)), 0.0, 0.0,
             *([0.0] * len(self._mission["stages"]))],
            dtype=np.float32,
        )
        high = np.array(
            [self._world_size[0], self._world_size[1], self._world_size[2],
             max_speed, max_speed, max_speed, self._world_size[2],
             *([sensor_range] * len(self._directions)), 1.0, 1.0,
             *([1.0] * len(self._mission["stages"]))],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._obstacles = [
            (np.asarray(obstacle["min_m"], dtype=np.float64), np.asarray(obstacle["max_m"], dtype=np.float64))
            for obstacle in self._world["obstacles"]
        ]
        self.reset()

    @property
    def pos(self):
        """Alias de compatibilidad para código que consultaba ``env.pos``."""
        return self.position

    @pos.setter
    def pos(self, value):
        self.position = np.asarray(value, dtype=np.float64)

    @property
    def current_target(self) -> np.ndarray:
        if self.stage == "outbound":
            return self._base
        if self.stage == "delivery":
            return self._delivery
        return self._base

    def _distance_to_target(self, position: np.ndarray) -> float:
        return float(np.linalg.norm(self.current_target - position))

    def _at_base(self, position: np.ndarray) -> bool:
        horizontal_distance = float(np.linalg.norm(position[:2] - self._base[:2]))
        return (
            horizontal_distance <= float(self._world["base_radius_m"])
            and position[2] <= float(self._world["base_max_altitude_m"])
        )

    def _inside_delivery_zone(self, position: np.ndarray) -> bool:
        horizontal_distance = float(np.linalg.norm(position[:2] - self._delivery[:2]))
        return (
            horizontal_distance <= float(self._world["delivery_radius_m"])
            and float(self._world["delivery_min_altitude_m"]) <= position[2]
            <= float(self._world["delivery_max_altitude_m"])
        )

    def _collides(self, position: np.ndarray) -> bool:
        radius = float(self._physics["drone_radius_m"])
        if position[2] < float(self._physics["floor_clearance_m"]) and not self._at_base(position):
            return True
        for minimum, maximum in self._obstacles:
            if np.all(position >= minimum - radius) and np.all(position <= maximum + radius):
                return True
        return False

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        return bool(np.any(position < 0.0) or np.any(position > self._world_size))

    def _sensor_readings(self) -> np.ndarray:
        sensor_config = self.config["sensors"]
        max_range = float(sensor_config["range_m"])
        ray_step = float(sensor_config["ray_step_m"])
        distances = np.arange(ray_step, max_range + ray_step, ray_step)
        readings = np.full(len(self._directions), max_range, dtype=np.float64)
        for sensor_index, direction in enumerate(self._directions):
            for distance in distances:
                point = self.position + direction * distance
                if self._out_of_bounds(point) or self._collides(point):
                    readings[sensor_index] = min(float(distance), max_range)
                    break
        return readings

    def _observation(self) -> np.ndarray:
        stage_vector = np.zeros(len(self._mission["stages"]), dtype=np.float64)
        stage_vector[self._mission["stages"].index(self.stage)] = 1.0
        return np.asarray(
            [
                *(self.current_target - self.position),
                *self.velocity,
                self.position[2],
                *self._sensor_readings(),
                self.battery_wh / float(self._battery_config["capacity_wh"]),
                float(self.has_supply),
                *stage_vector,
            ],
            dtype=np.float32,
        )

    def _info(self, cause: str | None, result: str) -> dict:
        return {
            "cause": cause,
            "result": result,
            "stage": self.stage,
            "contacts": self.contacts,
            "battery_wh": self.battery_wh,
            "reward_version": self.reward_version,
            "environment_version": self.environment_version,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self._base.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.battery_wh = float(self._battery_config["capacity_wh"])
        wind_min = np.asarray(self.config["wind"]["acceleration_min_mps2"], dtype=np.float64)
        wind_max = np.asarray(self.config["wind"]["acceleration_max_mps2"], dtype=np.float64)
        self.wind = self.np_random.uniform(wind_min, wind_max)
        self.contacts = 0
        self.has_supply = False
        self.stage = self._mission["stages"][0]
        self.steps = 0
        self._picked_up = False
        self._delivered = False
        self._returned = False
        self._best_distance_m = self._distance_to_target(self.position)
        return self._observation(), {"cause": None, "stage": self.stage, "wind_mps2": self.wind.copy()}

    def _battery_cost(self, distance_m: float) -> float:
        idle_wh = float(self._battery_config["idle_power_w"]) * float(self._physics["dt_s"]) / 3600.0
        movement_wh = float(self._battery_config["movement_energy_wh_per_m"]) * distance_m
        return idle_wh + movement_wh

    def _integrate_movement(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = float(self._physics["dt_s"])
        mass = float(self._physics["mass_kg"])
        thrust = direction * float(self._physics["control_force_n"])
        drag = float(self._physics["linear_drag_kg_per_s"]) * self.velocity
        acceleration = thrust / mass + self.wind - drag / mass
        acceleration_norm = float(np.linalg.norm(acceleration))
        max_acceleration = float(self._physics["max_acceleration_mps2"])
        if acceleration_norm > max_acceleration:
            acceleration *= max_acceleration / acceleration_norm
        velocity = self.velocity + acceleration * dt
        speed = float(np.linalg.norm(velocity))
        max_speed = float(self._physics["max_speed_mps"])
        if speed > max_speed:
            velocity *= max_speed / speed
        position = self.position + velocity * dt
        return position, velocity

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        self.steps += 1
        reward_config = self.config["reward"]
        reward = float(reward_config["step_penalty"])
        cause = None
        terminated = False
        truncated = False
        moved_distance = 0.0

        if int(action) == self._release_action:
            if self.stage == "delivery" and self.has_supply and self._inside_delivery_zone(self.position):
                if float(np.linalg.norm(self.velocity)) <= float(self._world["delivery_max_speed_mps"]):
                    self.has_supply = False
                    self.stage = self._mission["stages"][2]
                    if not self._delivered:
                        reward += float(reward_config["delivery_bonus"])
                        self._delivered = True
                    self._best_distance_m = self._distance_to_target(self.position)
                else:
                    reward += float(reward_config["invalid_release_penalty"])
            else:
                reward += float(reward_config["invalid_release_penalty"])
            self.battery_wh = max(0.0, self.battery_wh - self._battery_cost(0.0))
        else:
            if self.stage == "outbound" and not self.has_supply and self._at_base(self.position):
                self.has_supply = True
                self.stage = self._mission["stages"][1]
                if not self._picked_up:
                    reward += float(reward_config["pickup_bonus"])
                    self._picked_up = True
                self._best_distance_m = self._distance_to_target(self.position)

            old_position = self.position.copy()
            candidate_position, candidate_velocity = self._integrate_movement(self._directions[int(action)])
            moved_distance = float(np.linalg.norm(candidate_position - old_position))
            self.battery_wh = max(0.0, self.battery_wh - self._battery_cost(moved_distance))

            if self._out_of_bounds(candidate_position):
                cause = "out_of_bounds"
                terminated = True
                reward += float(reward_config["failure_penalty"])
                self.velocity = np.zeros(3, dtype=np.float64)
            elif self._collides(candidate_position):
                cause = "collision"
                terminated = True
                self.contacts += 1
                reward += float(reward_config["failure_penalty"])
                self.velocity = np.zeros(3, dtype=np.float64)
            else:
                previous_best = self._best_distance_m
                self.position = candidate_position
                self.velocity = candidate_velocity
                new_distance = self._distance_to_target(self.position)
                improvement = max(0.0, previous_best - new_distance)
                reward += improvement * float(reward_config["progress_per_meter"])
                self._best_distance_m = min(previous_best, new_distance)

                if self.battery_wh <= 0.0:
                    cause = "battery_depleted"
                    terminated = True
                    reward += float(reward_config["failure_penalty"])
                elif self.stage == "return" and self._at_base(self.position):
                    if float(np.linalg.norm(self.velocity)) <= float(self._world["base_max_speed_mps"]):
                        self.stage = self._mission["stages"][3]
                        if not self._returned:
                            reward += float(reward_config["return_bonus"])
                            self._returned = True
                        cause = "success"
                        terminated = True

            if not terminated and self.battery_wh <= 0.0:
                cause = "battery_depleted"
                terminated = True
                reward += float(reward_config["failure_penalty"])

        if not terminated and self.battery_wh <= 0.0:
            cause = "battery_depleted"
            terminated = True
            reward += float(reward_config["failure_penalty"])

        if not terminated and self.steps >= int(self._mission["max_steps"]):
            cause = "step_limit"
            truncated = True

        if cause == "success":
            result = "goal"
        elif cause in {"collision", "out_of_bounds", "battery_depleted"}:
            result = "crash"
        elif cause == "step_limit":
            result = "timeout"
        else:
            result = "flying"

        info = self._info(cause, result)
        info["wind_mps2"] = self.wind.copy()
        info["position_m"] = self.position.copy()
        info["velocity_mps"] = self.velocity.copy()
        return self._observation(), reward, terminated, truncated, info
