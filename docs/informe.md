# Informe breve: decisiones, resultados, límites y contribuciones

Estado al 17 de noviembre de 2026. Complementa el [README](../README.md); cada afirmación remite al archivo que la respalda. Las cifras de evaluación salen de `evaluacion.json` de cada paquete y de los manifiestos, y se pueden repetir con [`scripts/reproduce_evaluation.py`](../scripts/reproduce_evaluation.py).

## 1. Misión y estado

Un dron sale de BASE, recoge un suministro, lo entrega en la zona de entrega y vuelve a BASE ([ficha del ambiente](ficha_ambiente.md)). Python calcula toda la física; la web en Three.js solo dibuja el estado que devuelve la API. Hay tres niveles ([niveles](niveles.md)): N1, misión base y sin viento; N2, con ocho columnas y menos batería; N3, N2 con viento.

**Resultado: un modelo oficial para N1.** `ppo-v3c-n1` (run `cef121d1`, publicación v0004) completa N1 10 de 10 en 199 pasos. Ninguna configuración resuelve N2 ni N3. Todos los intentos se conservan, también los que no mejoraron.

## 2. Decisiones técnicas

| Decisión | Por qué | Evidencia |
|---|---|---|
| Python es la autoridad; la web solo dibuja | Una sola implementación de la física, la misma para entrenar, evaluar y mostrar | [README](../README.md), `api/main.py` |
| PPO (Stable-Baselines3) como agente de referencia; Q-learning queda como línea base | Una red neuronal maneja la observación continua; discretizar 19 valores pierde resolución y Q-learning nunca completó la misión | [decision.md](decision.md) |
| Familia `drone-rescue-v3`: el agente ordena velocidades y un estabilizador aplica la fuerza | En v2 el agente nunca aprendió a entregar: soltar fuera de la zona resta 0.5 y la probabilidad de soltar medida fue 0.0 | [versionamiento.md](versionamiento.md), [ficha del ambiente](ficha_ambiente.md) |
| Criterios de selección fijados antes de ver resultados; el retorno no cuenta | Un candidato puede sumar puntos sin completar la misión | [`configs/selection_criteria.yaml`](../configs/selection_criteria.yaml), [pipeline.md](pipeline.md) |
| Cada run deja un manifiesto con hashes; los modelos se recuperan con Git, DVC (DagsHub) y la release de GitHub; en Hugging Face va solo el modelo oficial | Un modelo recuperable son pesos, normalización y ambiente compatible; Hugging Face solo con el oficial por pedido del profesor | [versionamiento.md](versionamiento.md) |
| Airflow orquesta el pipeline y el rollback | Cada etapa deja estado y evidencia, y el rollback es un DAG verificable | [pipeline.md](pipeline.md) |
| Experimentos de cuadrícula en familias aparte (`grid-drone-v1` a `v3`) | Aislar qué aportan el viento, la batería, el mapa grande y las aves en una misión más simple; sus resultados no son comparables con los de la misión principal | [familia_cuadricula.md](familia_cuadricula.md) |
| Web en Vercel y API en un equipo del grupo publicada con ngrok | Render Free se suspende tras 15 minutos sin tráfico y tiene 512 MB, sin comprobar que quepa PPO con torch; `render.yaml` queda preparado | [despliegue.md](despliegue.md) |

## 3. Resultados

### Misión principal (suite `drone-rescue-public-validation` 1.0.0, 10 episodios por nivel)

| Configuración | Semillas y presupuesto | Resultado en su nivel |
|---|---|---|
| `exp-001` a `exp-004` (Q-learning; una dimensión cada uno: obstáculos, batería, viento, masa) | Finalistas con 3 semillas (11, 23, 37), 5 000 episodios | 0 %. `exp-001` y `exp-002`, los finalistas, rechazados para uso operativo; `exp-003` y `exp-004`, rechazados en el cribado. Todos conservados |
| `ppo-v2-n1` | 1 semilla, 3.0 M pasos | 0 %: nunca entrega (límite de pasos) |
| `ppo-v3-n1` | 3 semillas, 5.0 M pasos | 0 %: la 11 y la 23 entregan y chocan al volver; la 37 no entrega |
| `ppo-v3b-n1` | 3 semillas, 8.0 M pasos | **1 de 3**: la 11 completa 10/10 (201 pasos); la 37 entrega y choca; la 23 no entrega |
| `ppo-v3b-n2` | 3 semillas, 10.0 M pasos | 0 %: la 23 entrega y choca; la 11 y la 37 no entregan |
| `ppo-v3b-n3` | 3 semillas, 10.0 M pasos | 0 %: la 23 y la 37 entregan pero no vuelven a la base; la 11 no entrega |
| `ppo-v3c-n1` | 1 semilla, continúa 1.0 M pasos desde `ppo-v3b-n1` semilla 11 | **10/10 en 199 pasos**: modelo oficial |

### Cuadrícula (otra misión, más simple; 500 mapas por suite)

| Experimento | Éxito | Hipótesis fijada antes de entrenar |
|---|---|---|
| `grid-drone-v1`, DQN con memoria | 99.2 % (496 de 500 mapas) | — (referencia) |
| `grid-storm-dqn` (viento, batería, ciudad densa) | 55.2 a 61.8 % | ≥ 70 %: **no cumplida**, conservada |
| `grid-city-dqn` (mapa de 200 × 200 m, tendidos, aves) | 54.0 a 63.0 % | ≥ 50 %: cumplida, pero los postes de 10 m no influían y las aves casi no estorbaban |
| `grid-city-hard-dqn` (postes de 30 m, aves con zona de peligro) | 65.2 a 70.4 % (media 67.9 %) | Media ≥ 57 % y cada semilla ≥ 50 %: cumplida. Choques con aves ≤ 15 %: **no cumplida** (15.6 a 19.0 %) |

Detalle, referencias (A\*) y hallazgos en [familia_cuadricula.md](familia_cuadricula.md).

## 4. Presupuesto utilizado

| Experimento | Semillas | Pasos por run | Duración registrada por run |
|---|---|---|---|
| `ppo-v2-n1` | 11 | 3.0 M | 439 min |
| `ppo-v3-n1` | 11, 23, 37 | 5.0 M | 23 a 24 min |
| `ppo-v3b-n1` | 11, 23, 37 | 8.0 M | 43 a 44 min |
| `ppo-v3b-n2` | 11, 23, 37 | 10.0 M | 65 a 70 min |
| `ppo-v3b-n3` | 11, 23, 37 | 10.0 M | 56 a 66 min |
| `ppo-v3c-n1` | 11 | 1.0 M más (9.0 M acumulados) | 9 min |
| `grid-storm-dqn` | 11, 23, 37 | 2.5 M | 999 a 1 007 min; incluyen unas 15 h con el equipo suspendido, el entrenamiento efectivo fue de unos 60 min |
| `grid-city-dqn` | 11, 23, 37 | 4.0 M | 97 a 103 min |
| `grid-city-hard-dqn` | 11, 23, 37 | 3.0 M | 78 a 85 min |

Los pasos y duraciones salen de `effective_timesteps` y `training_duration_s` de cada manifiesto.

## 5. Incidente controlado e intervención

Perturbación pertinente al mundo (guía, sección 14): viento con ráfagas, batería y una ciudad más densa. Se aplicó sobre la cuadrícula, que permite aislar la perturbación; no es la misión principal.

| | Qué se midió | Resultado |
|---|---|---|
| Antes | DQN con memoria de `grid-drone-v1` en su ambiente | 99.2 % (496 de 500 mapas) |
| Degradación | El mismo modelo en `DroneRescueGridStorm-v3`, sin ver viento ni batería | **0.0 %**: 56 % de choques por viento y 44 % de batería agotada |
| Intervención | Darle al agente esas señales (de 46 a 51 valores de observación) y reentrenarlo, como familia nueva `grid-drone-v2` | 3 semillas, 2.5 M pasos |
| Después | Mismos 500 mapas | **55.2 a 61.8 %** (media 58.3 %) |
| Decisión | La intervención mejora de 0 a 58 %, pero no alcanza la hipótesis de 70 % | Aceptada como mejora y rechazada como solución; los tres candidatos se conservan. Referencia: A\* con margen al viento, 82 %. La siguiente hipótesis, penalizar las celdas a sotavento, no se ejecutó |

Una segunda perturbación, el mapa de 200 × 200 m con tendidos y aves: el mejor DQN de la ciudad 3.0.0 cae de 63.0 % a 47.4 % al endurecer postes y aves sin reentrenarlo; reentrenado alcanza 65.2 a 70.4 %, aunque los choques con aves siguen entre 15.6 y 19.0 %.

En la misión principal, N3 (N2 más viento) solo muestra la degradación: `ppo-v3b-n3` queda en 0 % con las 3 semillas y no hay una intervención que la resuelva.

## 6. Decisión sobre el modelo oficial

`ppo-v3b-n1` con la semilla 11 fue el primer modelo que completó N1, y el pipeline lo publicó como v0001. Se continuó su entrenamiento 1 M de pasos (`ppo-v3c-n1`, `parent_run_id` `9a188153`): 199 pasos en lugar de 201, y el pipeline lo aceptó como v0002 porque cumplía los criterios y no empeoraba al vigente. Se probó el rollback (v0003 restaura v0001 y la API vuelve a servir su SHA-256) y se volvió a promover `ppo-v3c-n1` (v0004, el oficial). El pipeline también rechazó a `ppo-v3b-n2` semilla 23 (éxito 0 % por debajo del mínimo de 80 %) sin publicar nada. Detalle y evidencia en [pipeline.md](pipeline.md).

## 7. Límites

- Solo 1 de 3 semillas de `ppo-v3b-n1` completa N1, y ninguna configuración resuelve N2 ni N3.
- En N1 y N2, que no tienen viento, las 10 semillas de la suite dan el mismo episodio: 100 % o 0 % reflejan un único vuelo. Solo N3 cambia de episodio con la semilla.
- **No hay un test final reservado.** La selección de los PPO usó la suite pública (10 episodios por nivel). En `grid-city-dqn` y `grid-city-hard-dqn` el checkpoint se eligió con mapas de validación distintos de los de la suite, que solo se usó para reportar. `pipelines/evaluate.py` rechaza suites que no sean públicas: un test reservado, si se crea, debe quedar fuera del repositorio.
- Los pesos originales de `exp-001` a `exp-004` se perdieron y se reentrenaron con el mismo commit, perfil, semilla y suite; las métricas coinciden con [decision.md](decision.md), pero solo `exp-003` coincide además bit a bit ([versionamiento.md](versionamiento.md)).
- Cinco manifiestos históricos no son reproducibles (sin `config_sha256` ni `runtime_ref`) y no se usan para seleccionar. `requirements_sha256` solo está en los manifiestos de los runs más recientes; `requirements.txt` fija todas las versiones.
- En MLflow todos los runs terminaron (`FINISHED`); los rechazos quedan como decisión (`selection_decision`, [decision.md](decision.md), evidencia del pipeline), no como estado del run.
- La API corre en un equipo del grupo y solo está disponible mientras ese equipo esté encendido ([despliegue.md](despliegue.md)). DagsHub exige una cuenta incluso para leer; sin cuenta, los modelos se recuperan de la release de GitHub o de Hugging Face (solo el oficial).
- **La release no incluye la base de MLflow** (351 MB). Con una copia limpia y solo la release, `scripts/verify_packages.py` y `scripts/reproduce_evaluation.py` funcionan, pero la API arranca con el catálogo vacío y `POST /load` falla («Run … not found»): levantar la web en local exige `dvc pull artifacts/mlflow`, es decir, una cuenta de DagsHub.
- Los resultados de la cuadrícula no son comparables con los de la misión principal: son otra misión, más simple.

## 8. Contribuciones

Equipo de dos integrantes (cuentas de GitHub `Mari4Z` y `alomariaDev`). Lo siguiente se deriva del historial de Git: muestra qué quedó versionado y cuándo, no quién diseñó o revisó cada decisión; cada integrante explica sus decisiones en la sustentación.

| Integrante | Commits | Periodo | Trabajo visible en el historial |
|---|---:|---|---|
| `alomariaDev` | 16 | 4 al 8 de noviembre | Estructura base de MLOps; ambiente desde YAML; niveles y experimentos configurables; evaluación reproducible; instrumentación con MLflow y manifiestos; cribado y comparación de finalistas; API FastAPI; consola web; sesiones y despliegue compartido; reintento de migraciones; URL de producción de la API; fusión de los PR #1 a #4 |
| `Mari4Z` | 47 | 23 de octubre al 17 de noviembre | Entorno 3D y Q-learning iniciales; migración a Gymnasium; CLI y visor 3D; DQN en mapas aleatorios; PPO, familia v3 y estabilizador; pipeline con Airflow y rollback; DVC, DagsHub y Hugging Face; releases; familias de cuadrícula v1 a v3 y sus experimentos; despliegue en Vercel y publicación por ngrok |

Se usó Claude Code (Anthropic) como asistente de código; figura como coautor (`Co-Authored-By`) en 46 de los 63 commits que existían al redactar este informe. Lo generado se comprobó con pruebas automáticas, la verificación de hashes de los paquetes y evaluaciones reproducibles, y las decisiones y sus motivos están en las hipótesis de cada perfil (escritas antes de entrenar) y en este informe.

## 9. Dónde verificar

| Qué | Dónde |
|---|---|
| Entrar y reproducir | [README](../README.md) |
| Pipeline, rollback y corridas reales | [pipeline.md](pipeline.md) |
| Qué se versiona y cómo se recupera | [versionamiento.md](versionamiento.md) |
| Experimentos de cuadrícula | [familia_cuadricula.md](familia_cuadricula.md) |
| Cumplimiento punto por punto de la guía | [auditoria.md](auditoria.md) |
