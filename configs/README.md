# Configuración

`drone_rescue.yaml` es la configuración activa del ambiente: mundo, misión,
dinámica, viento, batería, sensores, acciones, recompensa y discretización de
Q-learning. `ENVIRONMENT_VERSION` se define en `envs/drone_env.py`; cambia esa
versión cuando un cambio altere el contrato observable del entorno.
