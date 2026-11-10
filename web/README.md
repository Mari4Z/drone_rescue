# Aplicación web Drone Rescue

Interfaz Vite + TypeScript + Three.js. La escena solo representa la posición,
obstáculos y telemetría recibidos de FastAPI; toda transición y regla física se
ejecuta en Python.

## Desarrollo

```bash
npm ci
npm run dev
```

Inicia la API Python en otra terminal. En desarrollo, si
`VITE_API_BASE_URL` no está definida, el cliente usa `http://127.0.0.1:8000`.
Para una API remota crea `.env.local` a partir de `.env.example`. Si la API
requiere clave compartida, ingrésala en el campo de la cabecera; el navegador
la guarda solo en `sessionStorage`. Cada pestaña obtiene una sesión propia de
simulación.

El modo manual puede iniciar el ambiente sin modelo. Replay reproduce el
modelo/ambiente cargados de forma determinista con la semilla elegida.
Entrenamiento inicia un trabajo registrado como run de MLflow y muestra su
progreso; el proceso puede perderse si el servicio se reinicia.

## Vercel

Configura `web/` como Root Directory, deja el Build Command `npm run build` y
Output Directory `dist`. Define `VITE_API_BASE_URL` con una URL HTTPS accesible
desde el navegador y agrega el origen de Vercel a
`DRONE_RESCUE_CORS_ORIGINS` en la API.

Las variables `VITE_*` se incrustan en los recursos públicos del navegador:
no pongas secretos, tokens ni credenciales en ellas. La aplicación no tiene
rutas de servidor; el contexto de run/episodio usa enlaces hash para funcionar
con enlaces profundos y recargas en hosting estático. La clave API nunca debe
configurarse en `VITE_*`; cada usuario la ingresa al abrir la aplicación.
