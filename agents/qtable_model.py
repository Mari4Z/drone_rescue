"""MLflow pyfunc wrapper para una Q-table Drone Rescue versionada."""

import json

import mlflow.pyfunc
import numpy as np


class QTablePyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from envs.drone_env import discretize_observation

        table = np.load(context.artifacts["q_table"], allow_pickle=False)
        if table.dtype.names != ("states", "q_values"):
            raise ValueError("La Q-table debe ser NPY estructurado")
        self.q_table = {
            tuple(state.tolist()): np.asarray(values, dtype=np.float64)
            for state, values in zip(table["states"], table["q_values"])
        }
        with open(context.artifacts["config"], encoding="utf-8") as config_file:
            self.config = json.load(config_file)["environment"]
        self.discretize_observation = discretize_observation

    def predict(self, context, model_input, params=None):
        del context, params
        observations = np.asarray(model_input, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations[None, :]
        actions = []
        for observation in observations:
            state = self.discretize_observation(observation, self.config)
            q_values = self.q_table.get(tuple(state))
            actions.append(0 if q_values is None else int(np.argmax(q_values)))
        return np.asarray(actions, dtype=np.int64)
