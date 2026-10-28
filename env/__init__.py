import gymnasium as gym

from env.drone_env import DUMMY_COLUMNS

gym.register(id="DroneRescue-v0", entry_point="env.drone_env:DroneEnv")
gym.register(
    id="DroneRescueDummy-v1",
    entry_point="env.drone_env:DroneEnv",
    kwargs={"columns": DUMMY_COLUMNS},
)
