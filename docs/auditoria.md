# Auditoría frente a la guía del proyecto

Revisión del 17 de noviembre de 2026 sobre el commit `715894d` y los cambios de esta documentación. Contrasta el repositorio con la sección 17 de la guía («Entregables y criterios de aceptación»), con las comprobaciones obligatorias de la sección 16 y con la lista de lo que debe poder revisarse en el repositorio. Reemplaza la auditoría anterior, que se escribió sin el PDF de requisitos y quedó desactualizada.

Estados: **Cumple**, **Parcial** (existe, pero con una limitación que se explica) y **Abierto** (falta o depende de una decisión del equipo).

## Lo que pide la lista de revisión del repositorio

| Requisito | Estado | Evidencia | Qué falta o limita |
|---|---|---|---|
| README con misión, arquitectura, instalación, ejecución, reproducción y variables | Cumple | [README.md](../README.md) | Las instrucciones de Windows y Linux no se volvieron a probar con las dependencias actuales (todo se entrenó en macOS) |
| Ficha del ambiente | Cumple | [ficha_ambiente.md](ficha_ambiente.md) (incluye `drone-rescue-v3`) | — |
| Tres niveles de dificultad | Cumple | [niveles.md](niveles.md), `configs/levels/` | — |
| Al menos cuatro configuraciones con hipótesis, runs y estados | Cumple | 19 perfiles y configuraciones con `hypothesis`; exp-001 a exp-004 cambian una dimensión cada uno; runs en MLflow | Todos los runs figuran `FINISHED`: los rechazos quedan como decisión (`selection_decision`, [decision.md](decision.md), [pipeline.md](pipeline.md)), no como estado del run |
| Evaluaciones por caso | Cumple | `evaluacion.json` de cada paquete: casos × episodios con causa, batería, latencia y retorno. La del modelo oficial se reproduce con `scripts/reproduce_evaluation.py` | — |
| Modelos recuperables con manifiestos, normalización, hashes y dependencias | Cumple | [versionamiento.md](versionamiento.md); 41 manifiestos (`git_dirty` es falso en los 23 que lo registran); release `modelos-20261117`; `requirements.txt` fijado | `requirements_sha256` y `git_dirty` solo están en los 23 manifiestos recientes; 5 manifiestos históricos no son reproducibles y están marcados como tales |
| Pipeline documentado | Cumple | [pipeline.md](pipeline.md): 8 etapas, criterios, 5 corridas reales | — |
| Prueba de rollback documentada | Cumple | [pipeline.md](pipeline.md): v0003 restaura v0001 y la API sirve su SHA-256 | — |
| Incidente de degradación e intervención | Parcial | [informe.md](informe.md), sección 5: antes 99.2 %, degradación a 0.0 %, intervención, después 55.2 a 61.8 %, decisión | Ocurre en la cuadrícula, no en la misión principal; en N3 solo se observa la degradación (0 %) sin una intervención que la resuelva |
| Informe breve de decisiones, resultados, límites y contribuciones | Parcial | [informe.md](informe.md) | Las contribuciones se derivan del historial de Git: el equipo debe revisarlas y completarlas antes de entregar |
| Servicios disponibles y cómo acceder sin pasar contraseñas | Parcial | README (acceso para la revisión) y [despliegue.md](despliegue.md) | La API corre en un equipo del grupo con ngrok y solo existe mientras esté encendido. La demostración no exige clave (decisión abierta) |

## Entregables de la sección 17

| Entregable | Estado | Evidencia |
|---|---|---|
| URL de Vercel y repositorio con release o tag identificable | Parcial | Vercel responde. Los tags `modelos-*` son pre-releases («no es una entrega»); falta un tag final sobre el código definitivo |
| README con instalación, ejecución, variables, arquitectura y reproducción | Cumple | [README.md](../README.md) |
| Ficha del ambiente y tres niveles | Cumple | Ver arriba |
| Cuatro configuraciones con hipótesis y sus runs y estados en MLflow | Cumple | Ver arriba |
| Modelos finales recuperables, normalización, manifiestos y referencias de artefactos | Cumple | Release de GitHub, DVC en DagsHub y Hugging Face (solo el oficial) |
| Evaluaciones por caso, comparación de finalistas y presupuesto utilizado | Cumple | [decision.md](decision.md) (finalistas de Q-learning), [informe.md](informe.md) (resultados y presupuesto por run) |
| Pipeline, evidencia de degradación e intervención y prueba de rollback | Cumple | [pipeline.md](pipeline.md), [informe.md](informe.md), `artifacts/results/pipeline/` |
| Informe breve y presentación final | Parcial | El informe está; la presentación no es parte del repositorio |

## Comprobaciones obligatorias de la sección 16

| Comprobación | Estado | Cómo se comprobó |
|---|---|---|
| La URL abre | Cumple | El 17 de noviembre de 2026, con Chrome sin pantalla: la raíz abre el visor 3D y `/app` la consola con «API conectada» |
| La API usa HTTPS y admite el origen del frontend (CORS) | Cumple | La API pública respondió con `access-control-allow-origin` para el origen de Vercel |
| Las rutas de producción no apuntan a localhost | Cumple | `web/src/api.ts` rechaza `localhost` y las URL sin HTTPS en producción |
| Los modelos y escenarios se recuperan sin depender de carpetas personales | Parcial | Release y DVC los recuperan, pero la API y la web locales necesitan además la base de MLflow (351 MB), que la release no incluye |
| Se puede evaluar un modelo antiguo y el recomendado | Parcial | Con la API pública se comprobó el catálogo (35 runs), la carga del modelo oficial (SHA-256 coincidente) y el avance de un episodio. No se ejecutó una evaluación completa (escribiría en MLflow) ni se probó con clics desde la web |
| Un error de servicio se muestra como error; un replay se identifica como replay | Sin verificar | Comportamiento de la interfaz; no se revisó en esta auditoría |
| La release y el deployment identifican el código entregado | Parcial | Producción se reconstruyó desde `main` (deployment `dpl_H2zeMp…`, commit `ca046da`, alias de rama `git-main`); el visor de la raíz es idéntico byte a byte a `drone_view_city_hard.html` y el bundle de la consola, al construido de `web/`. `GET /version` da el commit de la API y coincide con el tag `modelos-20261117`; desde ahora se fija al arrancar. Falta un tag de entrega final |
| El docente no debe instalar un servidor | Cumple | La web y la API se abren por URL |
| Backend y registro disponibles durante la revisión | Abierto | Depende de un equipo encendido ([despliegue.md](despliegue.md)) |

## Qué se comprobó y cómo

| Comprobación | Resultado |
|---|---|
| `python -m pytest` | 124 pruebas aprobadas |
| `python scripts/verify_packages.py` en una copia limpia con la release descargada de GitHub | 97 archivos, 0 problemas |
| `python scripts/reproduce_evaluation.py` (modelo oficial) | N1 100 %, N2 0 %, N3 0 %; 30 de 30 episodios idénticos a los registrados |
| Copia limpia con solo la release: `GET /health`, `GET /catalog`, `POST /load` | `/health` responde; `/catalog` vacío y `/load` falla por falta de la base de MLflow |
| `GET /health`, `GET /version` y `GET /catalog` de la API pública | Responden sin clave; `/version` mostraba el commit `715894d` |
| Construir `web/` con la `VITE_API_BASE_URL` de producción y comparar con el bundle que sirve Vercel | Mismo archivo con hash (`assets/index-CjrhFWLv.js`): lo publicado es el código de `web/` del repositorio |
| `vercel inspect https://drone-rescue-snowy.vercel.app` tras `vercel redeploy ... --target production` | `target production`, `Ready`, alias `drone-rescue-snowy` y `git-main`; construido desde `main` |
| SHA-256 de `/`, `/visor` y `/visor/` en producción frente a `drone_view_city_hard.html` | `57b35d73c5cb9bb3…` en los tres: idénticos |
| Capturas con Chrome sin pantalla de `/` y de `/app` | Visor de la ciudad con su animación; consola con panel de control e indicador «API conectada» |
| API pública: `GET /catalog`, `POST /load` del modelo oficial, `GET /episode/step` | 35 runs; SHA-256 servido `bdf35f2e…` igual al de la publicación v0004; el episodio avanza |

## Pendiente o decisión del equipo

1. **API sin clave.** Hoy cualquiera con la URL puede usarla, incluido lanzar entrenamientos de hasta 100 000 pasos. Decidir si se activa `DRONE_RESCUE_API_TOKEN` y se entrega por el canal del curso ([despliegue.md](despliegue.md)).
2. **Tag de entrega.** Crear un tag final sobre el commit definitivo, reiniciar la API para que `/version` muestre ese commit y, si `web/` cambió, actualizar producción (`vercel redeploy ... --target production`). Un `git push` a `main` solo crea Previews en Vercel; decidir si se cambia la rama de producción a `main` ([despliegue.md](despliegue.md)).
3. **La release no trae MLflow.** Para que la revisión local funcione sin cuenta de DagsHub, incluir una base de MLflow reducida en la release o hacer que la API cargue los modelos desde `manifests/` y `artifacts/models/` cuando no encuentre el run.
4. **Test final reservado.** No existe; está declarado como límite ([informe.md](informe.md), sección 7).
5. **Contribuciones.** Revisar la sección 8 del informe: se derivó del historial de Git y no refleja el trabajo no versionado.
6. **Licencia del repositorio.** No hay `LICENSE`; el README declara que es material académico.
