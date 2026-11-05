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
Para una API remota crea `.env.local` a partir de `.env.example`.

## Vercel

Configura `web/` como Root Directory, deja el Build Command `npm run build` y
Output Directory `dist`. Define `VITE_API_BASE_URL` con una URL HTTPS accesible
desde el navegador y agrega el origen de Vercel a
`DRONE_RESCUE_CORS_ORIGINS` en la API.

Las variables `VITE_*` se incrustan en los recursos públicos del navegador:
no pongas secretos, tokens ni credenciales en ellas. La aplicación no tiene
rutas de servidor; el contexto de run/episodio usa enlaces hash para funcionar
con enlaces profundos y recargas en hosting estático.
