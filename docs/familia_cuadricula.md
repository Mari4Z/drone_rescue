# Familias de la cuadrícula

Dos familias: `grid-drone-v1` (la primera versión del proyecto) y `grid-drone-v2` (viento,
batería y ciudad densa; al final de este documento).

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

# Familia `grid-drone-v2`: viento, batería y ciudad densa

Segundo experimento de la cuadrícula (12 de noviembre), a partir del DQN con memoria de
`grid-drone-v1`. Cambia el ambiente, no el agente: `DroneRescueGridStorm-v3`
(`DroneStormEnv`, `envs/grid_drone_env.py`), `ENVIRONMENT_VERSION` 2.0.0, `REWARD_VERSION`
2.0.0. Es otra familia: sus modelos no se intercambian con `grid-drone-v1`.

| | `grid-drone-v1` (`…Random-v2`) | `grid-drone-v2` (`…Storm-v3`) |
|---|---|---|
| Edificios por mapa | 5 a 25 | **12 a 35** |
| B | Fija, esquina opuesta | **Aleatoria**, a 22 celdas o más de A |
| Viento | No | **Dirección y fuerza por mapa** (0 a 0.3). Tras cada movimiento en el aire, una ráfaga empuja una celda a sotavento con probabilidad fuerza × (0.4 a 1.0 según la altura). Contra un edificio o el borde: choque |
| Batería | No | **Sí.** Horizontal 1, subir 1.5, bajar 0.6 por celda; contra el viento hasta +0.6 y a favor hasta −0.6; cada ráfaga +0.5. Capacidad = energía de la mejor ruta × (1.15 a 1.5) + 3, así que hay margen para corregir, no para dar vueltas. Sin batería: −100 y termina |
| Observación | 46 valores | **51**: las 46 más viento (2), probabilidad de ráfaga a la altura actual, batería restante y margen hasta B |
| Recompensa por paso | −1 | **−energía gastada** (lo demás igual: meta +100, choque −100, +1 por acercarse, −0.5 bajo 20 m) |
| Límite | 250 pasos | 300 pasos |

**Referencias** (`pipelines/evaluate_grid_references.py`, `artifacts/results/grid_storm_referencias.json`):
`astar` conoce el mapa completo y replanea en cada paso con el viento medio; `astar_viento`
además evita las celdas con un edificio a sotavento (risk_weight 30); `dqn_memoria_v1` es el
DQN de v1 (run `dd7b29c6`) con sus 46 entradas, sin ver viento ni batería. El planificador es
un Dijkstra vectorizado (`DroneStormEnv.plan_route`, scipy) con el mismo coste que A\* (hay
prueba) y 7 veces más rápido, porque cada `reset` lo necesita para fijar la batería.

## Experimento `grid-storm-dqn`

Perfil `configs/training/grid_storm_dqn.yaml`, pipeline `pipelines/train_grid.py` (MLflow,
Model Registry `DroneRescueGridDQN-grid-drone-v2`, paquete y manifiesto), suite
`grid-storm-public` v1.0.0 (500 mapas, semillas 600000 a 600499). Mismos hiperparámetros que
`grid-dqn-memoria`; 2 500 000 pasos y 3 semillas.

**Hipótesis, escrita antes de entrenar:** el DQN mantiene al menos un 70 % de éxito, frente al
82 % de `astar_viento`, y el DQN de v1 cae claramente.

| Política (500 mapas) | Éxito | Choque por viento | Otro choque | Batería | Energía vs. mejor ruta |
|---|---:|---:|---:|---:|---:|
| `astar_viento` (mapa completo, evita sotavento) | 82.0 % | 8 % | 0 % | 10 % | 1.10× |
| **`grid-storm-dqn` semilla 23** (run `3b95307e`) | **61.8 %** | 30 % | 7 % | 1 % | 1.15× |
| `grid-storm-dqn` semilla 37 (run `d4629af8`) | 58.0 % | 33 % | 4 % | 6 % | 1.20× |
| `grid-storm-dqn` semilla 11 (run `33d21d89`) | 55.2 % | 33 % | 5 % | 7 % | 1.20× |
| `astar` (mapa completo, sin margen al viento) | 49.4 % | 50 % | 0 % | 0 % | 1.03× |
| `dqn_memoria_v1` (sin ver viento ni batería) | 0.0 % | 56 % | 0 % | 44 % | — |

**Decisión: la hipótesis no se cumple** (55 a 62 % frente al 70 % pedido) y el experimento se
conserva como está. Lo que sí muestra: aprender el viento y la batería es necesario (el DQN de
v1 pasa del 99 % al 0 %); el agente supera a A\* sin margen al viento, que es una referencia
con información total; la batería casi deja de ser causa de fallo (1 a 7 %); y el fallo
dominante es el viento (30 a 33 %), sobre todo con viento fuerte (41 % de éxito con fuerza
mayor que 0.2, frente a 79 % con menos de 0.1) y en los mapas más densos (52 % con 28 a 35
edificios). Para cerrar la brecha con `astar_viento` la siguiente hipótesis sería penalizar en
la recompensa las celdas a sotavento de un edificio, o darle al agente esa señal.

Las tres semillas quedaron entre el 55 % y el 62 %: a diferencia de `ppo-v3b-n1`, el resultado
no depende de la semilla. Los tiempos de entrenamiento registrados (16.7 h) incluyen unas 15 h
en que el equipo estuvo suspendido; el entrenamiento efectivo fue de unos 60 minutos por
semilla.

Visor: `python cli.py run --env DroneRescueGridStorm-v3 --model artifacts/models/3b95307e890245d3b011bfda923b9977/model.pt --seed 600000 --maps 12 --web --web-out drone_view_storm.html`
muestra la ciudad, las flechas y la manga de viento, la barra de batería y cada ráfaga.
