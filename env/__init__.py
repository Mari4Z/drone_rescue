import gymnasium as gym

from env.drone_env import COLUMNS, DUMMY_COLUMNS

gym.register(id="DroneRescue-v0", entry_point="env.drone_env:DroneEnv")
gym.register(
    id="DroneRescueDummy-v1",
    entry_point="env.drone_env:DroneEnv",
    kwargs={"columns": DUMMY_COLUMNS},
)
# Mapa aleatorio en cada intento, con sensores y penalización por volar bajo (para DQN)
gym.register(id="DroneRescueRandom-v2", entry_point="env.drone_env:DroneSensorEnv")
# El mapa fijo de 5 columnas con la misma observación que v2: sirve como mapa de prueba
gym.register(
    id="DroneRescueFixed-v2",
    entry_point="env.drone_env:DroneSensorEnv",
    kwargs={"columns": COLUMNS},
)
