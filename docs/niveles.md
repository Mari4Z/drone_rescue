# Niveles y experimentos Drone Rescue

## Configuración versionada y familia de modelos

El baseline ejecutable es [`configs/drone_rescue.yaml`](../configs/drone_rescue.yaml).
Los niveles de progresión están en `configs/levels/` y los experimentos controlados
individuales en `configs/experiments/`. Cada YAML puede pasarse directamente a
`DroneEnv(config_path="configs/levels/n1_mision_base.yaml")`; `_extends` carga y
combina profundamente el archivo padre, por lo que los valores no repetidos se
heredan del baseline o del nivel previo.

N1, N2, N3 y los cuatro experimentos declaran `model_family: drone-rescue-v2`.
La observación sigue teniendo 19 componentes, las acciones conservan los índices
0–5 de movimiento y 6 de liberación, y sus unidades y significado no cambian.
Por ello, la Q-table puede cargarse estructuralmente entre estos escenarios, aunque
la política aprendida puede rendir distinto por el cambio de dinámica. Si una
versión futura cambia dimensiones, orden o significado de observaciones/acciones,
debe declararse una familia nueva y no se deben reutilizar directamente esos
modelos.

## Progresión de niveles

| Nivel | Configuración | Qué decisión nueva debe tomar el agente |
|---|---|---|
| N1 — misión base | [`n1_mision_base.yaml`](../configs/levels/n1_mision_base.yaml) | Aprender la secuencia de despegar/recoger, navegar alrededor de dos columnas, liberar dentro de la zona lentamente y volver a BASE. Sin viento y con 60 Wh, prima aprender la misión y sus etapas. |
| N2 — más decisiones | [`n2_mas_decisiones.yaml`](../configs/levels/n2_mas_decisiones.yaml) | Elegir entre corredores alternativos alrededor de ocho columnas, planear el rodeo y reservar batería para regreso. La capacidad baja a 22 Wh. |
| N3 — operación exigente | [`n3_operacion_exigente.yaml`](../configs/levels/n3_operacion_exigente.yaml) | Mantener la ruta y gestionar deriva con viento horizontal aleatorio de hasta 0.25 m/s² por eje, corrigiendo la trayectoria sin agotar la batería. Respecto de N2, se perturba solo el viento. |

Los archivos anteriores son independientes y siguen ejecutables aunque se agreguen
niveles futuros. Para reproducibilidad, registrar la ruta/versionado de config, la
semilla, `environment_version`, `reward_version` y `model_family` junto con cada run.

## Cómo medir la dificultad

Comparar niveles con igual presupuesto por semilla, misma familia/algoritmo y
configuración de recompensa. En cada run guardar al menos:

- tasa de éxito (`cause=success / episodios`), que mide si resuelve la misión;
- tasa de colisión y de agotamiento de batería por causa terminal;
- pasos y distancia recorrida de episodios exitosos, para medir eficiencia;
- batería restante al completar, como reserva operativa;
- retorno acumulado por episodio, como métrica secundaria dado que mezcla progreso,
  hitos, penalizaciones y duración.

Reportar media y desviación estándar por semilla, además del total agregado. Un
nivel se considera más difícil empíricamente si, bajo el mismo presupuesto de
entrenamiento, baja la tasa de éxito o aumenta la tasa de fallo; pasos, reserva y
retorno ayudan a explicar el cambio, pero no sustituyen la tasa de éxito. No
comparar retornos entre recompensas distintas sin normalizarlos.

## Experimentos controlados

Cada experimento hereda el baseline directamente y cambia una sola dimensión
indicada por `experiment.changed_dimension`. Cada hipótesis se evalúa contra el
baseline con el mismo algoritmo, recompensa, 5 semillas emparejadas y presupuesto
de 5,000 episodios por semilla (25,000 episodios por variante contando todas las
semillas). Las semillas están explícitas en cada YAML. Los runs fallidos también
se conservan; una repetición cambia el identificador/versionado de experimento.

| ID | Única dimensión frente al baseline | Hipótesis |
|---|---|---|
| exp-001 | `world.obstacles` | Más columnas aumentan colisiones y tiempo de ruta. |
| exp-002 | `battery.capacity_wh` | Menos capacidad reduce éxito y reserva de batería. |
| exp-003 | `wind.acceleration_bounds` | Viento horizontal reduce éxito y aumenta correcciones de trayectoria. |
| exp-004 | `physics.mass_kg` | Más masa reduce aceleración y alarga las misiones exitosas. |

Las hipótesis completas y valores ejecutables están en [`configs/experiments/`](../configs/experiments/).
Cada experimento permanece en la familia `drone-rescue-v2` porque la interfaz de
estado y acciones no cambia; los manifiestos de modelo deben guardar la config
resuelta para distinguir políticas entrenadas bajo dinámicas diferentes.
