# Configuraciones

- `drone_rescue.yaml` es el baseline del ambiente.
- `levels/` contiene N1, N2 y N3 de progresión, documentados en
  [`docs/niveles.md`](../docs/niveles.md).
- `experiments/` contiene exp-001 a exp-004. Cada uno declara hipótesis,
  dimensión modificada, presupuesto, semillas y familia compatible.
- `training/` contiene perfiles de ejecución para N1, N2 y N3; seleccionan nivel,
  hipótesis, presupuesto de entrenamiento y semillas para la curva greedy.

Los YAML de niveles y experimentos admiten `_extends` con ruta relativa al archivo.
`load_config()` combina mapas recursivamente; listas como obstáculos y vectores se
reemplazan completos. Se pueden pasar directamente a `DroneEnv(config_path=...)`.
Todos preservan la interfaz `drone-rescue-v2` (19 observaciones y 7 acciones).
