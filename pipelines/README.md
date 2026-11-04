# Pipelines

## Evaluación pública

[`evaluate.py`](evaluate.py) compara tres referencias sobre la suite pública fija
[`configs/evaluation_suite.yaml`](../configs/evaluation_suite.yaml): política
aleatoria, regla reactiva y Q-learning greedy. Registra 10 episodios por nivel
(N1–N3), usando una semilla pública fija por episodio. La suite tiene id,
versión y `split: public_validation`. El conjunto de test final reservado no se
incluye ni se genera en este repositorio; su acceso y ejecución deben manejarse
fuera del repo.

Uso:

```bash
python pipelines/evaluate.py \\
  --q-table artifacts/q_tables/drone_rescue_q.npz \\
  --output artifacts/results/evaluacion.json
```

El NPZ entrenado debe tener `states` como matriz `[n,16]` de estados discretizados
y `q_values` como `[n,7]` con sus valores Q. Para exportar una tabla desde un
`QAgent` entrenado:

```python
from pipelines.evaluate import save_q_table
save_q_table(agent, "artifacts/q_tables/drone_rescue_q.npz")
```

No se sobrescribe una salida de manera silenciosa; usa un nombre de archivo nuevo
para cada ejecución. Cada episodio guarda éxito y causa, pasos, batería, precisión
horizontal y velocidad en el primer intento de liberación, contactos, retorno y
latencia media/total de inferencia. Si no se intenta liberar, precisión y velocidad
son `null`. Cada caso es una combinación de nivel, política y versión de reward.

La tasa de éxito y el retorno son métricas separadas. `mean_return` solo se calcula
dentro del mismo caso; el JSON no agrega ni compara retornos entre niveles o
versiones de recompensa. Las semillas, suite, políticas, nivel, versión de ambiente
y reward se conservan junto al resultado.
