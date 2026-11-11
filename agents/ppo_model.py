"""MLflow pyfunc para una política PPO versionada: recibe entradas ya escaladas por make_features."""

import mlflow.pyfunc
import numpy as np


class PPOPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from stable_baselines3 import PPO

        self.model = PPO.load(context.artifacts["model"], device="cpu")

    def predict(self, context, model_input, params=None):
        del context, params
        features = np.asarray(model_input, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        return np.asarray(self.model.predict(features, deterministic=True)[0], dtype=np.int64)
