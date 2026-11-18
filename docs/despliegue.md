# Despliegue compartido

## Despliegue en uso (demostración y revisión)

| Pieza | Dónde | Cómo se comprueba |
|---|---|---|
| Web | Vercel ([drone-rescue-snowy.vercel.app](https://drone-rescue-snowy.vercel.app)), con `web/` como Root Directory. `/` abre el visor 3D de la ciudad (`web/public/visor/index.html`, sin API) y `/app` abre la consola, que usa `VITE_API_BASE_URL` | El visor abre solo; la consola muestra el catálogo y, si la API no responde, «API no disponible» |
| API | FastAPI en un equipo del grupo, publicada por HTTPS con ngrok mediante [`scripts/serve_public.sh`](../scripts/serve_public.sh) | `GET /health` (estado) y `GET /version` (commit del código y modelo servido) |
| Experimentos y datos | MLflow local (`artifacts/mlflow/mlflow.db`) en el equipo de la API; copia compartida en DagsHub (MLflow y DVC) | [versionamiento.md](versionamiento.md) |
| Modelos sin cuenta | Release de GitHub (todos los modelos) y Hugging Face (solo el oficial) | `SHA256SUMS` y `scripts/verify_packages.py` |

La API solo existe mientras ese equipo esté encendido, sin suspensión y con `serve_public.sh` en marcha. No hay un servidor permanente.

**Producción en Vercel**

- Proyecto `visualizacion1/drone-rescue`: Root Directory `web`, Vite, `npm ci` y `npm run build`, salida `dist`. La variable `VITE_API_BASE_URL` existe solo en el entorno Production.
- [`web/vercel.json`](../web/vercel.json) define las rutas: `/` y `/visor` sirven el visor (`web/public/visor/index.html`, copia exacta de `drone_view_city_hard.html`), `/app` sirve la consola y todo lo demás sale de los archivos construidos. La consola no cambia: usa enlaces con hash y recursos con rutas absolutas.
- Producción vigente: deployment `dpl_H2zeMpFCQmZ4AtxCZPR2gSNR2ps8`, reconstruido desde `main` (commit `ca046da`) el 17 de noviembre de 2026 a las 21:27 (hora de Colombia) con `vercel redeploy ... --target production`. Sirve `drone-rescue-snowy.vercel.app`. Anteriores, que sirven de respaldo (`vercel rollback` o `vercel promote`): `dpl_2ZRB4YzTXNCTSyLzjbUiCQi1Ys48` (commit `9cfaddd`, con la consola en `/` y sin visor) y `dpl_rgXZnnqoftvpSkrpg1a2kKLVoDam` (11 de noviembre, rama `feature/integra-main`).
- Un `git push` a `main` crea un deployment de **Preview**, no de producción (se ve con `vercel ls`), y ese Preview no tiene `VITE_API_BASE_URL`, así que no llega a la API. Para actualizar producción desde `main`: `vercel ls` para ver la URL del Preview del commit, `vercel redeploy <esa URL> --target production` desde `web/`, y comprobar con `vercel inspect https://drone-rescue-snowy.vercel.app` (target `production`, alias `git-main`) y abriendo la web («API conectada»). La alternativa permanente es cambiar la rama de producción a `main` en los ajustes del proyecto de Vercel; no se hizo.
- Un Preview no se puede abrir sin sesión de Vercel; se comprueba con `vercel curl <ruta> --deployment <URL del Preview>`. Sus páginas HTML de `/` llevan 163 bytes más (un script de comentarios que Vercel inyecta solo en Preview); en producción el visor sale idéntico byte a byte.
- Solo `drone-rescue-snowy.vercel.app` es pública. `drone-rescue-visualizacion1.vercel.app` redirige al inicio de sesión de Vercel (protección del proyecto), así que no sirve para la revisión.

**Antes de una revisión**

1. Equipo enchufado y con la suspensión desactivada. Desde la raíz del proyecto: `./scripts/serve_public.sh` (requiere haber ejecutado una vez `ngrok config add-authtoken <token>`).
2. Comprobar `GET /health` (`{"status":"ok",...}`) y `GET /version` (el `git_commit` debe ser el del tag `v1.0.0`) en la URL pública de la API, abrir la web y cargar un modelo.
3. En Vercel, `vercel inspect https://drone-rescue-snowy.vercel.app` debe mostrar `target production` y el alias `git-main`; si `web/` cambió desde el último deployment de producción, actualizarlo como se explica arriba. `VITE_API_BASE_URL` debe ser la URL HTTPS vigente de la API. Sin la CLI: construir `web/` con esa misma variable (`npm run build -- --outDir /tmp/web --emptyOutDir`) y comparar el nombre `assets/index-*.js` con el de la web publicada; si coinciden, el código es el mismo.
4. Si cambió el código, reiniciar la API: `/version` lee el commit al arrancar.

**Acceso sin compartir contraseñas**

- La demostración corre en modo `development` y **no exige clave**: cualquiera con la URL de la API puede usarla, incluido lanzar entrenamientos, limitados por `DRONE_RESCUE_TRAINING_MAX_TIMESTEPS` (100 000 pasos en `serve_public.sh`). Está pensada solo para demostraciones.
- Para exigir clave: `DRONE_RESCUE_API_TOKEN=<cadena larga> ./scripts/serve_public.sh`. Con la variable definida, la API pide `X-API-Key` en todo salvo `/health` y `/version`, y la clave se escribe en el campo de la web (se guarda solo en `sessionStorage`). La clave se entrega por el canal del curso, nunca por correo ni en el repositorio.
- DagsHub (MLflow y DVC) exige una cuenta gratuita incluso para leer. Cada persona usa su propia cuenta y su propio token, que solo queda en `.dvc/config.local`. Sin cuenta, los modelos se recuperan de la release de GitHub o de Hugging Face.
- **Si la API no está disponible**, sin cuenta se pueden verificar y reproducir los modelos desde la release (`scripts/verify_packages.py` y `scripts/reproduce_evaluation.py`, ver el [README](../README.md)). Levantar la web y la API en el equipo de quien revisa necesita además los runs de MLflow (`dvc pull artifacts/mlflow`, con cuenta de DagsHub): la release no incluye la base de MLflow y, sin ella, el catálogo queda vacío y `POST /load` falla (probado con una copia limpia y solo la release).

## Arquitectura preparada (Render y DagsHub)

En uso: DagsHub como remoto DVC y servidor MLflow (`scripts/setup_dagshub.sh`). No desplegado: Render; `render.yaml` queda preparado.

- La web Vite se publica como sitio estático en Vercel, con `web/` como Root
  Directory y `VITE_API_BASE_URL` como variable pública con la URL HTTPS de la
  API.
- `render.yaml` prepara una API FastAPI en Render Free. Las claves y los
  orígenes permitidos se ingresan en el panel como variables secretas; no se
  versionan.
- MLflow remoto se configura con `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD`. DagsHub ofrece un
  servidor MLflow y almacenamiento DVC para un repositorio compartido; crea
  primero un repositorio en DagsHub y usa un token de acceso como contraseña.
- La clave `DRONE_RESCUE_API_TOKEN` protege catálogo, sesiones, carga de
  modelos, entrenamiento y evaluación. Cada persona la ingresa en el campo de
  la web; solo se conserva en `sessionStorage` durante esa pestaña. No uses
  `VITE_*` para esta clave.

## Preparación paso a paso

Atajo para DagsHub (pasos 1 a 3): conecta el repositorio de GitHub en DagsHub, crea un token y ejecuta `./scripts/setup_dagshub.sh` en tu terminal. Pide usuario y token sin mostrarlos, configura el remoto DVC, hace `dvc push` y copia los runs de MLflow.

1. Crea un repositorio de equipo en DagsHub y agrega a las personas que deban
   registrar/consultar runs. Configura su URI MLflow y token únicamente en
   variables de entorno/secretos.
2. Migra los runs y artefactos históricos al tracking remoto. Desde la raíz
   del repositorio, configura credenciales en variables temporales de entorno
   (no en archivos versionados) y ejecuta:

   ```bash
   read -rp "Usuario DagsHub: " MLFLOW_TRACKING_USERNAME
   read -rsp "Token DagsHub: " MLFLOW_TRACKING_PASSWORD
   printf '\n'
   export MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD
   .venv/bin/python scripts/migrate_mlflow_to_remote.py \
     --source-uri "sqlite:////ruta/absoluta/al/proyecto/mlflow.db" \
     --destination-uri "https://dagshub.com/<usuario-dagshub>/drone_rescue.mlflow" \
     --dry-run
   ```

   Si el número de runs es correcto, repite sin `--dry-run`. La migración
   conserva los IDs originales en el catálogo, registra el nuevo ID remoto en
   cada manifiesto y crea runs remotos nuevos; no modifica ni borra el tracking
   local. Revisa los logs y verifica la carga de un modelo antiguo antes de
   desplegar la API. Una migración parcial se marca como fallida y no se
   sobrescribe automáticamente. Si un proceso se interrumpió y dejó un run con
   estado `RUNNING`, verifica primero que no haya otra migración activa y repite
   con `--retry-incomplete`; el intento previo se conserva y marca como fallido.
   Al terminar, ejecuta
   `unset MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD`.
3. Configura el remoto DVC de DagsHub usando los comandos que muestra el panel
   de ese repositorio. El patrón habitual es:

   ```bash
   read -rp "Usuario DagsHub: " DAGSHUB_USER
   read -rsp "Token DagsHub: " DAGSHUB_TOKEN
   printf '\n'
   dvc remote add -d dagshub s3://dvc
   dvc remote modify dagshub endpointurl "https://dagshub.com/${DAGSHUB_USER}/drone_rescue.s3"
   dvc remote modify dagshub --local access_key_id "$DAGSHUB_TOKEN"
   dvc remote modify dagshub --local secret_access_key "$DAGSHUB_TOKEN"
   dvc push
   unset DAGSHUB_TOKEN
   ```

   Las credenciales se guardan en la configuración local de DVC (`--local`),
   nunca en Git. Confirma que `dvc status` esté limpio y que otra copia pueda
   descargar los artefactos.
4. En Render, crea un Blueprint desde este repositorio y suministra:
   `DRONE_RESCUE_API_TOKEN` (una cadena aleatoria larga), el origen HTTPS
   exacto de Vercel en `DRONE_RESCUE_CORS_ORIGINS`, el URI del servidor
   MLflow, usuario/token y `MLFLOW_UI_URL`.
5. Comprueba `/health` y `/version` por HTTPS. Configura Vercel con Root
   Directory `web`, Build Command `npm run build`, Output Directory `dist` y
   `VITE_API_BASE_URL` igual a la URL pública de Render. Añade el origen
   definitivo de Vercel a CORS y vuelve a desplegar la API.
6. Ingresa la clave API en la web, prueba catálogo, carga, entrenamiento,
   replay y evaluación, y confirma que runs, manifiestos y pesos nuevos estén
   en el MLflow/almacenamiento compartido.

## Limitaciones del plan gratuito

**Modelos PPO.** Servir PPO necesita `torch` y `stable-baselines3`, incluidos en `requirements-api.txt`. Render Free tiene 512 MB de memoria y no se ha comprobado que la API con `torch` quepa ni cuánto tarda en iniciar. Si no cabe, hay que usar un plan con más memoria o exportar la política a un formato más ligero, validando que dé las mismas acciones. Entrenar PPO desde la API lanza varios procesos de simulación, así que en Render Free conviene limitar `DRONE_RESCUE_TRAINING_MAX_TIMESTEPS` o entrenar en el equipo y migrar el run.

Render Free no es una opción de disponibilidad permanente: suspende el
servicio tras 15 minutos sin tráfico, puede tardar aproximadamente un minuto
en iniciar, limita las horas mensuales y su sistema de archivos local es efímero. No tiene
disco persistente. El tracking MLflow local (`mlflow.db`) y los artefactos
locales se perderían al reiniciar; por eso el despliegue debe apuntar al
tracking remoto y no depender del disco del servicio. Los trabajos de
entrenamiento en memoria tampoco sobreviven a suspensión/reinicio. Para uptime
continuo o trabajos largos se necesita un plan/worker permanente y una
instancia de almacenamiento apropiada.

Render no se ha desplegado: su Blueprint requiere crear la cuenta y cargar los
secretos en el panel. Si se usa, no publiques una URL de API sin cambiar la
clave de ejemplo y verificar CORS y autenticación.
