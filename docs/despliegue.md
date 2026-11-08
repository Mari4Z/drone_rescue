# Despliegue compartido

## Arquitectura preparada

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

Render Free no es una opción de disponibilidad permanente: suspende el
servicio tras 15 minutos sin tráfico, puede tardar aproximadamente un minuto
en iniciar, limita las horas mensuales y su sistema de archivos local es efímero. No tiene
disco persistente. El tracking MLflow local (`mlflow.db`) y los artefactos
locales se perderían al reiniciar; por eso el despliegue debe apuntar al
tracking remoto y no depender del disco del servicio. Los trabajos de
entrenamiento en memoria tampoco sobreviven a suspensión/reinicio. Para uptime
continuo o trabajos largos se necesita un plan/worker permanente y una
instancia de almacenamiento apropiada.

El despliegue no queda activo hasta crear las cuentas/repositorios y cargar los
secretos en sus respectivos paneles. No publiques una URL de API sin cambiar
la clave de ejemplo y verificar CORS y autenticación.
