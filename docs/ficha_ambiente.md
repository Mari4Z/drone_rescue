# Ficha del ambiente Drone Rescue

## Identidad y fuente de parámetros

- `ENVIRONMENT_VERSION = "2.0.0"` se expone desde `envs.drone_env` y cada
  `info` de Gymnasium también incluye `environment_version`.
- La recompensa se identifica con `reward.version` de
  [`configs/drone_rescue.yaml`](../configs/drone_rescue.yaml); actualmente es
  `2.0.0` y se devuelve como `info["reward_version"]`.
- La cuadrícula, zonas, obstáculos, dinámica, masa, fuerzas, viento, batería,
  sensores, límite de pasos, acciones y cuantización viven en ese YAML. El
  código contiene las ecuaciones y reglas del contrato, no valores físicos de
  reserva.

## Misión y etapas

1. **outbound:** el dron aparece en BASE sin carga. Al iniciar una acción de
   movimiento dentro de BASE recoge el suministro automáticamente y despega;
   no se añade una octava acción de recogida.
2. **delivery:** el objetivo activo es el centro de la zona de entrega y el
   dron lleva el suministro.
3. **return:** liberar correctamente el suministro cambia el objetivo activo a
   BASE.
4. **complete:** entrar en el radio de BASE a baja velocidad y con batería
   restante termina la misión con éxito.

La entrega solo es válida dentro del radio horizontal configurado, entre las
alturas configuradas y con rapidez igual o inferior a `delivery_max_speed_mps`.
El comando de liberar representa una espera estacionaria de un `dt`; no integra
traslación, pero consume energía inactiva.

## Estado interno

| Campo | Representación y unidad |
|---|---|
| `position` | vector `(x, y, z)` en metros |
| `velocity` | vector `(vx, vy, vz)` en m/s |
| `battery_wh` | energía restante en Wh |
| `has_supply` | booleano de carga |
| `wind` | aceleración de viento `(ax, ay, az)` en m/s², constante por episodio |
| `contacts` | número de colisiones del episodio |
| `stage` | etapa categórica: outbound, delivery, return o complete |
| `steps` | acciones simuladas en el episodio |

`reset(seed=...)` muestrea el viento de los rangos YAML con el generador
sembrado de Gymnasium; repetir la misma semilla reproduce observación y viento.
Los contactos cuentan las colisiones con obstáculos o suelo; salir del mundo
es una terminación distinta y no incrementa el contador.

## Observación continua

`observation_space` es un `Box` `float32` de 19 componentes, en este orden:

| Índices | Contenido | Unidad / rango |
|---|---|---|
| `0:3` | vector desde el dron al objetivo activo | m; puede ser negativo por eje |
| `3:6` | velocidad `(vx, vy, vz)` | m/s |
| `6` | altura actual | m |
| `7:13` | seis sensores cardinales `+x,-x,+y,-y,+z,-z` | distancia al obstáculo/límite, m, saturada a `range_m` |
| `13` | batería restante / capacidad | fracción `[0,1]` |
| `14` | carga | `0` o `1` |
| `15:19` | one-hot de etapa | cuatro valores binarios en orden de `mission.stages` |

Los sensores lanzan rayos en las seis direcciones de movimiento, muestreados a
`ray_step_m`; detectan cajas de obstáculos expandidas por el radio del dron y
los límites del mundo. Por tanto, son distancias aproximadas: se pierde la
geometría entre los rayos, la forma del obstáculo y cualquier obstáculo que
quede fuera de la dirección cardinal observada.

## Discretización para Q-learning tabular

`discretize_observation()` devuelve una clave de 16 enteros para la tabla
dispersa del agente:

- Vector al objetivo: redondeo por componente a bins de 10 m; pierde hasta 5 m
  por eje y no distingue posiciones dentro del mismo bin.
- Velocidad: bins de 1 m/s por componente; pierde hasta 0,5 m/s por eje.
- Altura: bins de 5 m; pierde hasta 2,5 m.
- Cada sensor: bins de 5 m; pierde hasta 2,5 m adicionales, además de la
  aproximación de muestreo de 1 m del rayo.
- Batería: 5 bins uniformes, cada uno representa 20 puntos porcentuales; se
  pierde el nivel exacto dentro de cada intervalo.
- Carga y etapa se conservan exactamente (binaria y categórica).

La discretización reduce cardinalidad, pero puede agrupar estados físicamente
distintos. La tabla Q es dispersa para no reservar memoria para combinaciones
que nunca se visitan.

## Acciones y física

El paso fijo es `physics.dt_s = 0.5 s`. No depende de la velocidad con que una
interfaz anime el dron. La observación representa metros, m/s y batería en Wh.

| Índice | Acción | Fuerza aplicada |
|---:|---|---|
| 0 | mover `+x` | `(control_force_n, 0, 0)` N |
| 1 | mover `-x` | `(-control_force_n, 0, 0)` N |
| 2 | mover `+y` | `(0, control_force_n, 0)` N |
| 3 | mover `-y` | `(0, -control_force_n, 0)` N |
| 4 | subir | `(0, 0, control_force_n)` N |
| 5 | bajar | `(0, 0, -control_force_n)` N |
| 6 | liberar suministro / mantener posición | sin traslación; aplica las reglas de entrega |

La acción es un entero discreto en `[0,6]`; las seis acciones de movimiento
aplican una fuerza de magnitud `4 N` en el eje indicado, con aceleración
resultante limitada a `3 m/s²` y rapidez limitada a `5 m/s`. Los límites y el
paso `0.5 s` proceden de `physics` en YAML. No hay entrada continua de
acelerador ni un paso variable por acción.

Para las acciones 0–5, cada componente usa la ecuación vectorial
`a = fuerza / masa + viento - (arrastre / masa)`, limitada por
`max_acceleration_mps2`; luego `v_nueva = clip_norm(v + a·dt)` y
`p_nueva = p + v_nueva·dt` (Euler semiimplícito). La rapidez se limita a
`max_speed_mps`. El movimiento del ambiente se integra una vez por acción; el
renderizado no participa en esta ecuación.

La batería consume en cada acción
`idle_power_w·dt/3600 + movement_energy_wh_per_m·distancia_m`. La liberación
consume solo energía inactiva.

## Terminación y causa

- `terminated=True`, `truncated=False`, `cause="success"`: regreso a BASE con
  batería restante y velocidad/altura dentro de límites.
- `terminated=True`, `cause="collision"`: contacto con obstáculo o suelo.
- `terminated=True`, `cause="out_of_bounds"`: posición fuera de la caja del
  mundo.
- `terminated=True`, `cause="battery_depleted"`: la batería llega a cero,
  incluso en la acción que habría completado el regreso.
- `terminated=False`, `truncated=True`, `cause="step_limit"`: se alcanza
  `mission.max_steps`.
- En los demás pasos `cause=None`.

`info` conserva `result` para lecturas antiguas y añade `cause`, `stage`,
`contacts`, energía, versiones, posición y velocidad físicas.

## Recompensa `2.0.0` y anti-explotación

Todos los valores siguientes se configuran en `reward`:

- Penalización por acción: `-0.1`.
- Progreso: `0.05` por metro de mejora hacia el **objetivo activo**, solo cuando
  baja el mejor registro de distancia del episodio para ese objetivo. Retroceder
  y volver a una posición anterior no vuelve a pagar el mismo progreso.
- Recogida, entrega y regreso: `+10`, `+30`, `+100`; cada hito tiene un
  indicador por episodio y solo se paga una vez.
- Colisión, salida o batería agotada: `-100`.
- Liberar fuera de condiciones, sin carga o repetir la liberación: `-0.5` además
  de la penalización de paso; nunca paga el bono de entrega ni cambia de etapa.

Al cambiar el objetivo activo (recogida o entrega), se inicia un nuevo mejor
registro de distancia desde la posición actual. Ninguna distancia al objetivo
anterior aporta progreso después del cambio.

## Cinco transiciones de referencia

Los cinco casos corresponden a
`test_transition_1_...` a `test_transition_5_...` en
[`tests/test_environment_contract.py`](../tests/test_environment_contract.py).
En los casos de entrega y retorno se prepara explícitamente el estado previo
para aislar el contrato de esa transición.

### 1. Despegue y recogida automática

- **Observación previa:** `p=(2.5,2.5,0) m`, `v=(0,0,0) m/s`, carga `0`, etapa
  outbound, batería `30 Wh`; para este caso el viento es cero.
- **Acción:** `4` (subir, 4 N) durante `0.5 s`.
- **Física:** masa `2 kg` da `a_z=2 m/s²`; `v'=(0,0,1)`, `p'=(2.5,2.5,0.5)`.
  Se recoge el suministro y el objetivo activo pasa a entrega.
- **Reward:** `-0.1 + 10 = 9.9`; el movimiento vertical no reduce la distancia
  al centro de entrega y no cobra progreso.
- **Nueva observación:** vector al objetivo `(95,95,-0.5) m`, velocidad
  `(0,0,1) m/s`, altura `0.5 m`, sensores `(20,3,20,3,20,1) m`, carga `1`,
  etapa delivery. Batería: `29.99222 Wh`.

### 2. Avance con viento

- **Observación previa:** `p=(50,50,10) m`, `v=(0,0,0)`, etapa delivery con
  carga; viento fijado en `(0.2,0,0) m/s²` para el caso.
- **Acción:** `0` (`+x`, 4 N) durante `0.5 s`.
- **Física:** `a_x=4/2+0.2=2.2 m/s²`; `v'=(1.1,0,0) m/s`,
  `p'=(50.55,50,10) m`. La distancia al objetivo baja de `67.91539` a
  `67.53186 m`.
- **Reward:** `-0.1 + 0.05·(67.91539-67.53186) = -0.08082`.
- **Nueva observación:** vector al objetivo `(46.95,47.5,-10) m`, velocidad
  `(1.1,0,0) m/s`, altura `10 m`, sensores `(19,20,20,5,20,10) m`, carga `1`,
  etapa delivery. Batería: `29.99172 Wh`.

### 3. Liberación fuera de la zona

- **Observación previa:** `p=(50,50,10) m`, velocidad cero, carga `1`, etapa
  delivery; está fuera del radio de entrega.
- **Acción:** `6` (liberar) durante un paso de `0.5 s`.
- **Física:** no hay traslación; se descuenta energía inactiva.
- **Reward:** `-0.1 - 0.5 = -0.6`; no se paga bono ni se cambia de etapa.
- **Nueva observación:** vector al objetivo sigue en `(47.5,47.5,-10) m`,
  carga `1`, etapa delivery, batería `29.99722 Wh`; `cause=None`.

### 4. Liberación válida a baja velocidad

- **Observación previa:** `p=(97.5,97.5,1) m`, velocidad cero, carga `1`, etapa
  delivery.
- **Acción:** `6` (liberar) durante un paso de `0.5 s`.
- **Física:** la posición queda fija; la distancia horizontal al centro es cero,
  la altura está en el intervalo permitido y la rapidez es `0 m/s`.
- **Reward:** `-0.1 + 30 = 29.9`; el objetivo activo cambia a BASE.
- **Nueva observación:** vector a BASE `(-95,-95,-1) m`, carga `0`, etapa
  return, batería `29.99722 Wh`; `cause=None`.

### 5. Regreso completo a BASE

- **Observación previa:** `p=(2.7,2.5,0.5) m`, `v=(-1.1,0,0) m/s`, etapa return,
  sin carga; viento cero y batería `30 Wh`.
- **Acción:** `0` (`+x`, 4 N) durante `0.5 s`.
- **Física:** `a_x=2.1375 m/s²` por empuje y arrastre; `v'=(-0.03125,0,0) m/s`,
  `p'=(2.684375,2.5,0.5) m`. Queda dentro de BASE a baja velocidad.
- **Reward:** `-0.1 + 100 + 0.05·(0.538516-0.532911) = 99.90028`.
- **Nueva observación:** vector a BASE `(-0.184375,0,-0.5) m`, carga `0`, etapa
  complete, batería `29.99707 Wh`; `terminated=True`, `cause="success"`.

## Registro de validación

- Hipótesis: el ambiente continuo desde YAML cumple el contrato de misión y su
  observación puede discretizarse sin reservar una tabla Q densa.
- MLflow `run_id`: `af559520730e4ace92a78e169c8707d9`.
- Resultado: 13 pruebas aprobadas; checker Gymnasium aprobado; 2 episodios de
  humo ejecutaron el entrenamiento y la ruta greedy. Esa ruta terminó en
  `out_of_bounds`; el smoke valida integración, no demuestra que el agente ya
  haya aprendido a completar la misión.

## Familia drone-rescue-v3

Conserva física, reglas, observación y recompensa de esta ficha. Cambian las acciones: el agente elige entre 11 órdenes cada 1 s (2 pasos físicos, `dt = 0.5 s`), y un estabilizador aplica en cada paso la fuerza del eje con mayor error de velocidad.

| Índice | Orden | Velocidad objetivo |
|---:|---|---|
| 0 a 3 | `+x`, `-x`, `+y`, `-y` rápido | 4 m/s |
| 4 a 7 | `+x`, `-x`, `+y`, `-y` lento | 1 m/s |
| 8 y 9 | Subir y bajar | 2 m/s |
| 10 | Sostener posición | 0.5 m/s por metro de error respecto al punto donde empezó a sostener |

La entrega se activa al sostener posición con el suministro, dentro de la zona y a 0.75 m/s o menos. La acción de soltar no existe como tal, por lo que la penalización por soltar mal no se aplica en v3. El regreso se completa igual que en v2. El estabilizador tiene la versión `STABILIZER_VERSION`.

