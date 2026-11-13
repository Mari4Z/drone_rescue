# Familias de la cuadrícula

Tres familias: `grid-drone-v1` (la primera versión del proyecto), `grid-drone-v2` (viento,
batería y ciudad densa) y `grid-drone-v3` (mapa de 200 × 200 m, tendidos eléctricos y aves); las
dos últimas, al final de este documento.

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

# Familia `grid-drone-v3`: mapa grande, tendidos eléctricos y aves

Tercer experimento de la cuadrícula (13 de noviembre), a partir del DQN de `grid-drone-v2`.
Ambiente `DroneRescueGridCity-v4` (`DroneCityEnv`, `envs/grid_drone_env.py`),
`ENVIRONMENT_VERSION` 3.0.0, misma recompensa que v2 (`REWARD_VERSION` 2.0.0). Es otra familia:
sus modelos esperan 59 valores y no se intercambian con `grid-drone-v2`.

| | `grid-drone-v2` | `grid-drone-v3` |
|---|---|---|
| Mapa | 20 × 20 × 12 celdas (100 × 100 × 60 m) | **40 × 40 × 12 celdas (200 × 200 × 60 m)** |
| Edificios | 12 a 35 | 40 a 110 (misma densidad) |
| Tendidos eléctricos | No | **3 a 6 por mapa**: postes cada 20 a 30 m unidos por un cable a 10 m. Postes y cables bloquean y los sensores los ven; no se puede pasar por debajo (el suelo está prohibido fuera de A y B) |
| Aves | No | **0 a 6**, entre 10 y 40 m. En cada paso se quedan (25 %) o se mueven una celda horizontal al azar. Tocar una termina el intento (−100) y el dron las detecta a 4 celdas |
| Observación | 51 valores | **59**: las 51 más el ave más cercana por dirección (6), aves cerca (1) y fracción de ruta restante (1) |
| Límite | 300 pasos | 600 pasos |

**Referencias** (suite `grid-city-public` v1.0.0, 500 mapas, semillas 700000 a 700499;
`pipelines/evaluate_grid_references.py`): `astar_viento` conoce el mapa completo con postes y cables,
evita los edificios a sotavento y no ve las aves; `dqn_storm_v2` es el mejor DQN de v2 (run `3b95307e`)
con sus 51 entradas.

## Experimento `grid-city-dqn`

Perfil `configs/training/grid_city_dqn.yaml`, pipeline `pipelines/train_grid.py`, 4 000 000 pasos y
3 semillas, con los mismos hiperparámetros que `grid-storm-dqn` y un buffer de 300 000. El mejor
checkpoint de cada semilla se elige con 100 mapas de validación (semillas 950000 en adelante),
distintos de la suite.

**Hipótesis, escrita antes de entrenar:** al menos un 50 % de éxito en 500 mapas, y menos del 10 %
de los fallos por aves.

| Política (500 mapas) | Éxito | Viento | Otro choque | Aves | Batería | Energía vs. mejor ruta |
|---|---:|---:|---:|---:|---:|---:|
| `astar_viento` (mapa completo, no ve aves) | 86.2 % | 5.2 % | 0 % | 1.4 % | 7.2 % | 1.10× |
| **`grid-city-dqn` semilla 37** (run `5a8c68f2`) | **63.0 %** | 30.2 % | 3.4 % | 0.8 % | 2.6 % | 1.16× |
| `grid-city-dqn` semilla 11 (run `2b1e7ed3`) | 56.6 % | 28.2 % | 6.4 % | 1.6 % | 7.2 % | 1.25× |
| `grid-city-dqn` semilla 23 (run `f2337e88`) | 54.0 % | 36.8 % | 5.8 % | 1.0 % | 2.4 % | 1.14× |
| `dqn_storm_v2` (sin ver aves, 51 entradas) | 40.6 % | 39.6 % | 16.8 % | 2.0 % | 1.0 % | — |

**Decisión: la hipótesis se cumple** (54 a 63 %, media 58 %; las aves son el 2 a 4 % de los
fallos), pero hay que leerla con los cuatro hallazgos siguientes, que le quitan valor como
prueba de que el agente aprendió a esquivar postes, cables y aves.

1. **Los postes y cables a 10 m no influyen.** En 150 mapas no cambian la ruta óptima de A\* en
   ninguno, y esa ruta nunca baja de 20 m, porque volar por debajo de 20 m se penaliza. El dron
   los ve, pero no tiene motivo para esquivarlos. Corrección propuesta (pendiente): postes de
   30 m y cables entre 20 y 30 m, la altura de crucero, de modo que cruzar un tendido obligue a
   pasar por debajo (volando bajo), por encima (más batería y más viento) o a rodearlo. Sería
   `ENVIRONMENT_VERSION` 3.1.0 y un experimento nuevo.
2. **Las aves casi no estorban.** A\*, que no las ve, choca con ellas en el 1.4 % de los mapas, y
   el DQN, que sí las ve, en el 0.8 a 1.6 %. Con 6 aves como máximo en 200 × 200 m hay muy poca
   densidad.
3. **El viento domina y un tercio de sus choques son del despegue.** Entre el 28 y el 37 % de los
   mapas terminan estrellados por una ráfaga, y del 30 al 38 % de esos choques ocurren en los
   primeros 6 pasos (10 a 12 % de todos los mapas). A está en la esquina (0, 0): una ráfaga hacia
   fuera del mapa estrella al dron antes de que pueda alejarse del borde. Por física se pierde el
   3.4 % de los mapas en el primer ascenso aunque la política sea perfecta, y el 10.5 % si el dron
   sigue pegado a la esquina durante 3 pasos. A\* pierde el 3.8 % porque se aleja del borde pronto;
   el DQN pierde 10 a 12 %, así que unos 7 puntos de éxito son aprendibles. Sin los choques del
   despegue el éxito sería de 61 a 71 %.
4. **El DQN no es estable con 4 millones de pasos.** La semilla 11 llegó al 54 % de validación en
   el paso 2 600 000 y cayó al 0 % en el siguiente corte, sin recuperarse en 1 400 000 pasos. Las
   semillas 23 y 37 oscilan unos 10 puntos entre cortes. Por eso se guarda el mejor checkpoint de
   cada semilla; la cifra de la suite corresponde a ese checkpoint, elegido con la validación.

El entrenamiento duró 1.6 a 1.7 h por semilla (sin suspensiones esta vez). Visor:
`python cli.py run --env DroneRescueGridCity-v4 --model artifacts/models/5a8c68f2d3a449488475142303fee41b/model.pt --seed 700000 --maps 12 --web --web-out drone_view_city.html`
muestra los postes, los cables con su curva, las aves animadas, el viento, la batería y cada ráfaga.
