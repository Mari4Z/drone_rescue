# Dataset de Drone Rescue

En aprendizaje por refuerzo no hay un dataset fijo: los datos salen de la interacción con el ambiente. Por eso el dataset versiona lo que la guía llama generador, distribuciones, escenarios y semillas (§10), más los episodios que esos escenarios producen. Vive en `data/`, se versiona con DVC (`data.dvc`) y se sube a DagsHub con `dvc push`.

```bash
dvc pull data.dvc                     # recuperarlo
python -m pipelines.build_dataset     # regenerarlo desde el código y la suite
dvc add data && git add data.dvc      # versionar una regeneración
```

## Contenido (versión 1.0.0)

| Archivo | Qué contiene |
|---|---|
| `escenarios/N1.json`, `N2.json`, `N3.json` | Mundo (límites, base, zona de entrega, columnas), física, batería, distribución del viento y misión de cada nivel, con el SHA-256 de su configuración. Por cada semilla de la suite: el viento sorteado y la observación inicial |
| `episodios/oficial.parquet` | Los 30 episodios de la suite (3 niveles × 10 semillas) del modelo oficial de N1 |
| `episodios/aleatoria.parquet` | Los mismos 30 episodios con la política aleatoria |
| `episodios/regla.parquet` | Los mismos 30 episodios con la regla reactiva |
| `dataset.json` | Versión, commit, versiones del ambiente y del estabilizador, suite, políticas, resumen y SHA-256 de cada archivo |

Cada fila de un episodio es una decisión de la política. Columnas: política, familia del ambiente, nivel, semilla, `paso` (decisión) y `pasos_fisicos` (con estabilizador, 2 por decisión); los 19 valores de la observación antes del escalado (vector al objetivo, velocidad, altura, 6 sensores, batería, carga y etapa); acción y su nombre; recompensa; `terminated`, `truncated` y causa; etapa, posición, batería en Wh y contactos después de la acción.

## Cómo se genera

- Suite `drone-rescue-public-validation` v1.0.0 (`configs/evaluation_suite.yaml`), semillas 101 a 1009.
- El modelo oficial es la publicación vigente de N1 en `manifests/published/` y se comprueba con el SHA-256 de su manifiesto. Usa la familia `drone-rescue-v3`; las líneas base usan `drone-rescue-v2`, con 7 acciones de fuerza.
- N1 y N2 no tienen viento: sus 10 semillas repiten el mismo episodio. En N3 cada semilla sortea un viento distinto.

Al cambiar el ambiente, la recompensa, la suite o el modelo oficial, se regenera y se versiona como una versión nueva.
