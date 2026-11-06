# Familia de modelos `grid-drone-v1`

Primera versión del proyecto, desarrollada en la rama `1Experiment` e integrada como familia
independiente. Se conserva como intento histórico y como referencia de aprendizaje en un
problema más simple. **No resuelve la misión de `drone-rescue-v2` y sus resultados no son
comparables con los de esa familia.**

| | `grid-drone-v1` | `drone-rescue-v2` |
|---|---|---|
| Dinámica | Cuadrícula: cada acción mueve una celda de 5 m | Continua: masa, fuerza, velocidad y viento |
| Misión | Ir de A a B | Recoger, entregar y volver a base |
| Acciones | 6 movimientos | 6 movimientos y soltar |
| Batería | No | Sí |
| Observación | Posición, o 46 valores en v2 | 19 valores |
| Código | `envs/grid_drone_env.py`, `agents/grid/`, `viewer/`, `cli.py` | `envs/drone_env.py`, `agents/`, `pipelines/`, `api/`, `web/` |

## Ambientes

| ID | Mapa | Observación | Agente |
|---|---|---|---|
| `DroneRescueGrid-v0` | 5 columnas fijas | Posición | Q-learning tabular |
| `DroneRescueGridDummy-v1` | 1 columna fija | Posición | Q-learning tabular |
| `DroneRescueGridRandom-v2` | 5 a 25 columnas aleatorias por intento | 46 valores | DQN |
| `DroneRescueGridFixed-v2` | 5 columnas fijas | 46 valores | DQN (mapa de prueba) |

Observación v2, en este orden: posición normalizada (3), vector hacia B (3), sensores de
distancia en las 6 direcciones (6), ocupación del cubo 3×3×3 (27) y visitas a la celda actual
y a sus 6 vecinas (7). Ver `DroneSensorEnv`.

Reglas: chocar con una columna, con el borde o con el suelo termina el intento; el suelo solo
es transitable en A y B. Recompensas: meta +100, choque -100, paso -1, acercarse a B +1 por
celda y, en v2, -0.5 por paso bajo 20 m fuera de A y B.

## Resultado registrado

DQN con 1 500 000 pasos: 99 % de éxito (496 de 500 mapas, semillas 500000 a 500499) y rutas
1.06 veces más largas que A\*.

## Límites

- El modelo se entrenó antes de esta integración, sin MLflow ni manifiesto, así que no cuenta
  como modelo recuperable y su peso solo existe en el equipo donde se entrenó. Para usarlo
  como evidencia formal hay que volver a entrenarlo con el pipeline versionado.
- La suite de evaluación de `drone-rescue-v2` no aplica a esta familia.
