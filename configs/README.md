# Configuraciones

- `drone_rescue.yaml` es el baseline del ambiente.
- `levels/` contiene N1, N2 y N3 de progresión, documentados en
  [`docs/niveles.md`](../docs/niveles.md).
- `experiments/` contiene exp-001 a exp-004. Cada uno declara hipótesis,
  dimensión modificada, presupuesto, semillas y familia compatible.
- `training/` contiene los perfiles de entrenamiento: `n1` a `n3` (Q-learning), `ppo_v2_*` y
  `ppo_v3*` (PPO), `dqn_n1` y `grid_*` (DQN de las familias de cuadrícula). Cada uno declara el nivel o
  ambiente, la hipótesis (escrita antes de entrenar), el presupuesto y las semillas.
- `*_evaluation_suite.yaml` son las suites de evaluación, con id, versión y semillas;
  `selection_criteria.yaml` son los criterios de selección del pipeline ([`docs/pipeline.md`](../docs/pipeline.md)).

Los YAML de niveles y experimentos admiten `_extends` con ruta relativa al archivo.
`load_config()` combina mapas recursivamente; listas como obstáculos y vectores se
reemplazan completos. Se pueden pasar directamente a `DroneEnv(config_path=...)`.
Los niveles y experimentos preservan la interfaz `drone-rescue-v2` (19 observaciones y 7 acciones);
los PPO usan `drone-rescue-v3`, con la misma física, observación y recompensa y 11 acciones
([`docs/ficha_ambiente.md`](../docs/ficha_ambiente.md)).
