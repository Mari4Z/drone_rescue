# Pipelines

## Entrenamiento reproducible

Ejecuta un perfil versionado con una semilla explícita:

```bash
python -m pipelines.train --profile configs/training/n1.yaml --seed 101
```

También están los perfiles `n2.yaml` y `n3.yaml`. Cada run usa el nombre
`<exp>-seed-<n>`, registra el commit completo y las versiones del ambiente,
nivel, hipótesis y run padre. Guarda métricas de cada episodio como
`training/return`, `training/success` y `training/epsilon`. Las evaluaciones
periódicas greedy usan epsilon cero y métricas `evaluation/*` separadas.

La salida incluye checkpoint sparse `q_table.npy`, snapshot resuelto
`config.json`, discretización `normalization.json`, evaluación pública
`evaluacion.json` con ambas curvas y el manifiesto `manifests/<run_id>.json`.
La configuración se guarda como artefacto inmutable dentro del run y su SHA-256
queda en el manifiesto. Los fallos dejan un run MLflow `FAILED`, diagnóstico
`sin checkpoint disponible` y manifiesto de fallo sin URI de modelo.

Para continuar una corrida, se verifica que el run padre terminó, corresponde al
mismo nivel/versión/configuración del ambiente y tenga checkpoint:

```bash
python -m pipelines.train --profile configs/training/n1.yaml --seed 202 \
  --parent-run-id <run_id>
```

Cada Q-table se registra como versión de `DroneRescueQTable-<nivel>` en MLflow
después de la evaluación final. El `model_uri` del manifiesto señala al
checkpoint `.npy` inmutable en el run; `registry_uri` conserva la versión
registrada. Para repetir la evaluación de un modelo desde su manifiesto:

```python
from pipelines.evaluate import evaluate_from_manifest

evaluate_from_manifest(
    "manifests/<run_id>.json",
    "configs/evaluation_suite.yaml",
    "artifacts/results/reproduccion-<run_id>.json",
)
```

## Evaluación pública

[`evaluate.py`](evaluate.py) compara tres referencias sobre la suite pública fija
[`configs/evaluation_suite.yaml`](../configs/evaluation_suite.yaml): política
aleatoria, regla reactiva y Q-learning greedy. Registra 10 episodios por nivel
(N1–N3), usando una semilla pública fija por episodio. La suite tiene id,
versión y `split: public_validation`. El conjunto de test final reservado no se
incluye ni se genera en este repositorio; su acceso y ejecución deben manejarse
fuera del repo.

Uso independiente del pipeline de entrenamiento:

```bash
python pipelines/evaluate.py \
  --q-table artifacts/q_tables/drone_rescue_q.npz \
  --output artifacts/results/evaluacion.json
```

El checkpoint puede ser NPZ con arrays `states` `[n,16]` y `q_values` `[n,7]`,
o NPY estructurado con esos campos. Para exportar una tabla NPZ desde un
`QAgent` entrenado:

```python
from pipelines.evaluate import save_q_table
save_q_table(agent, "artifacts/q_tables/drone_rescue_q.npz")
```

No se sobrescribe una salida de manera silenciosa; usa una ruta/version nueva
para cada ejecución. Cada episodio guarda éxito y causa, pasos, batería,
precisión horizontal y velocidad en el primer intento de liberación, contactos,
retorno y latencia media/total de inferencia. Si no se intenta liberar, precisión
y velocidad son `null`. Cada caso es una combinación de nivel, política y
versión de reward.

La tasa de éxito y el retorno son métricas separadas. `mean_return` solo se calcula
dentro del mismo caso; el JSON no agrega ni compara retornos entre niveles o
versiones de recompensa. Las semillas, suite, políticas, nivel, versión de ambiente
y reward se conservan junto al resultado.
