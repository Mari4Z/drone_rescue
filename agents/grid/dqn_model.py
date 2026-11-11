"""MLflow pyfunc para un DQN de la familia de cuadrícula: recibe la observación del ambiente."""

import mlflow.pyfunc
import numpy as np


class GridDQNPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from agents.grid.dqn import DQNAgent

        self.agent = DQNAgent.load(context.artifacts["model"], "cpu")

    def predict(self, context, model_input, params=None):
        del context, params
        import torch

        obs = np.asarray(model_input, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        with torch.no_grad():
            return self.agent.actor(torch.as_tensor(obs)).argmax(1).numpy().astype(np.int64)
