# Decisión de comparación de finalistas

- Comparación: `finalistas-20261105T151229Z`
- Suite común: `drone-rescue-finalists-public` v`1.0.0` (10 episodios por configuración).
- Presupuesto final: 5000 episodios por run; semillas independientes emparejadas: 11, 23, 37.
- Cribado: 500 episodios por candidato, semilla 7001; se conservaron los cuatro candidatos.

## Decisión: RECHAZAR ambos finalistas para uso operativo; no hay evidencia de éxito que justifique promover o recuperar una versión anterior.

Los dos finalistas registraron 0% de éxito medio en la evaluación común; los candidatos quedan conservados para análisis y nuevos experimentos.
El cribado ordena por tasa de éxito, luego fallos catastróficos, pasos, batería y latencia; el retorno no interviene en la selección. La tabla final reporta variabilidad entre semillas de entrenamiento por separado de la dispersión entre episodios de evaluación.

## Evidencia del cribado

| Rango | Candidato | Éxito | Colisión | Fuera de límites | Batería agotada | Pasos medios | Run cribado | Estado |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `exp-001` | 0.0% | 3 | 7 | 0 | 4.1 | `f60c92943c6b496588603a02fb10190f` | finalista |
| 2 | `exp-002` | 0.0% | 10 | 0 | 0 | 4.6 | `ffb5dc085a654049b1347611db54b568` | finalista |
| 3 | `exp-003` | 0.0% | 9 | 1 | 0 | 5.6 | `d21f0d0528d74cb6b107217230ac0d13` | rechazado en cribado; run conservado |
| 4 | `exp-004` | 0.0% | 4 | 6 | 0 | 21.3 | `d4d0c204a8ef4a7e9ba0dd0c79e1333e` | rechazado en cribado; run conservado |

## Resultados finales: entrenamiento vs. evaluación

Cada celda muestra media ± desviación estándar de las medias por semilla de entrenamiento; `ep σ` es la desviación típica media dentro de cada modelo entre episodios de evaluación. Las tasas de éxito/fallo se expresan en porcentaje. Duración está en pasos, batería en Wh, latencia en ms. Métricas globales agregan los cuatro casos de la misma suite; el JSON conserva además resultados desglosados por caso.

| Métrica | exp-001 (entre semillas; ep σ) | exp-002 (entre semillas; ep σ) |
|---|---|---|
| Éxito (%) | 0.0 ± 0.0; ep σ=0.0 | 0.0 ± 0.0; ep σ=0.0 |
| Colisión (%) | 85.0 ± 10.9; ep σ=34.1 | 75.0 ± 2.5; ep σ=43.8 |
| Fuera de límites (%) | 5.8 ± 3.8; ep σ=22.8 | 9.2 ± 8.0; ep σ=23.2 |
| Batería agotada (%) | 0.0 ± 0.0; ep σ=0.0 | 0.0 ± 0.0; ep σ=0.0 |
| Límite de pasos (%) | 9.2 ± 13.8; ep σ=19.9 | 15.8 ± 6.3; ep σ=36.3 |
| Duración (pasos) | 92.3 ± 105.1; ep σ=159.3 | 147.0 ± 45.8; ep σ=281.9 |
| Batería restante (Wh) | 27.36 ± 0.25; ep σ=3.59 | 27.14 ± 0.12; ep σ=3.61 |
| Latencia inferencia (ms/acción) | 0.125 ± 0.051; ep σ=0.055 | 0.072 ± 0.009; ep σ=0.035 |

## Conservación y trazabilidad

- `exp-003` rechazado en cribado: run `d21f0d0528d74cb6b107217230ac0d13`, perfil/config `configs/experiments/exp-003.yaml`; su run, checkpoint y versión registrada se conservan en MLflow.
- `exp-004` rechazado en cribado: run `d4d0c204a8ef4a7e9ba0dd0c79e1333e`, perfil/config `configs/experiments/exp-004.yaml`; su run, checkpoint y versión registrada se conservan en MLflow.
- `exp-001` semilla 11: run `612347eb239a4989ad5e426db3702fb8`, manifiesto `manifests/612347eb239a4989ad5e426db3702fb8.json`, Registry `models:/DroneRescueQTable-exp-001/2`.
- `exp-001` semilla 23: run `0bf1f13861324ab7bf99e870aa84c86d`, manifiesto `manifests/0bf1f13861324ab7bf99e870aa84c86d.json`, Registry `models:/DroneRescueQTable-exp-001/3`.
- `exp-001` semilla 37: run `ea85481755f9438089c38bdb47851dd9`, manifiesto `manifests/ea85481755f9438089c38bdb47851dd9.json`, Registry `models:/DroneRescueQTable-exp-001/4`.
- `exp-002` semilla 11: run `d43628f87e9a49d9b295d6e59b60f8d2`, manifiesto `manifests/d43628f87e9a49d9b295d6e59b60f8d2.json`, Registry `models:/DroneRescueQTable-exp-002/2`.
- `exp-002` semilla 23: run `cb3a90fac6244a3297a01be8c4997f81`, manifiesto `manifests/cb3a90fac6244a3297a01be8c4997f81.json`, Registry `models:/DroneRescueQTable-exp-002/3`.
- `exp-002` semilla 37: run `12b77008d4474052aaa051c2e9d958e0`, manifiesto `manifests/12b77008d4474052aaa051c2e9d958e0.json`, Registry `models:/DroneRescueQTable-exp-002/4`.

Los retornos se guardan por caso en el JSON de comparación, pero no se promedian ni se usan para ordenar candidatos. Todas las configuraciones conservan `reward_version=2.0.0` y el contrato de observación/acción de la misma familia de modelos.
