"""DroneEnv con sensores vectorizados: mismas lecturas, unas 6 veces más rápido.

Solo se usa para entrenar. La evaluación oficial sigue en ``DroneEnv``; una prueba
comprueba que ambos devuelven las mismas observaciones y recompensas.
"""

from __future__ import annotations

import numpy as np

from envs.drone_env import DroneEnv


class FastDroneEnv(DroneEnv):
    def _sensor_readings(self) -> np.ndarray:
        sensors = self.config["sensors"]
        max_range = float(sensors["range_m"])
        ray_step = float(sensors["ray_step_m"])
        distances = np.arange(ray_step, max_range + ray_step, ray_step)
        # Puntos de cada rayo: (sensores, distancias, xyz)
        points = self.position + self._directions[:, None, :] * distances[None, :, None]

        blocked = np.any(points < 0.0, axis=2) | np.any(points > self._world_size, axis=2)
        floor = points[..., 2] < float(self._physics["floor_clearance_m"])
        at_base = (
            np.linalg.norm(points[..., :2] - self._base[:2], axis=2) <= float(self._world["base_radius_m"])
        ) & (points[..., 2] <= float(self._world["base_max_altitude_m"]))
        blocked |= floor & ~at_base
        radius = float(self._physics["drone_radius_m"])
        for minimum, maximum in self._obstacles:
            blocked |= np.all(points >= minimum - radius, axis=2) & np.all(points <= maximum + radius, axis=2)

        hit = blocked.any(axis=1)
        first = blocked.argmax(axis=1)
        return np.where(hit, np.minimum(distances[first], max_range), max_range)
