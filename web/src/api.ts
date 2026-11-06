import type {
  ApiFailure,
  CatalogResponse,
  EpisodeMode,
  EpisodeState,
  EvaluationResponse,
  LoadResponse,
  TrainingStatus,
} from "./types";

const configuredApiUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = (configuredApiUrl || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "")).replace(
  /\/+$/,
  "",
);
const invalidProductionApiUrl = (() => {
  if (!import.meta.env.PROD || !apiBaseUrl) return null;
  try {
    const url = new URL(apiBaseUrl);
    const hostname = url.hostname.toLowerCase();
    if (
      url.protocol !== "https:"
      || hostname === "localhost"
      || hostname.endsWith(".localhost")
      || hostname.startsWith("127.")
      || hostname === "::1"
      || hostname === "[::1]"
    ) return "En producción, VITE_API_BASE_URL debe usar HTTPS y no puede apuntar a localhost.";
    return null;
  } catch {
    return "VITE_API_BASE_URL no contiene una URL válida.";
  }
})();

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (invalidProductionApiUrl) throw new ApiError(invalidProductionApiUrl, 0);
  if (!apiBaseUrl) {
    throw new ApiError("Configura VITE_API_BASE_URL para conectar con la API desplegada.", 0);
  }

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? `No se pudo contactar la API: ${error.message}` : "No se pudo contactar la API.",
      0,
    );
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as ApiFailure | null)?.detail;
    throw new ApiError(
      typeof detail === "string" ? detail : `La API respondió HTTP ${response.status}.`,
      response.status,
    );
  }
  return body as T;
}

export function getCatalog(): Promise<CatalogResponse> {
  return request("/catalog");
}

export function loadRun(runId: string, seed: number): Promise<LoadResponse> {
  return request("/load", {
    method: "POST",
    body: JSON.stringify({ run_id: runId, seed }),
  });
}

export function resetEpisode(mode: EpisodeMode, seed: number): Promise<EpisodeState> {
  return request("/episode/reset", {
    method: "POST",
    body: JSON.stringify({ mode, seed }),
  });
}

export function stepEpisode(action?: number): Promise<EpisodeState> {
  const query = action === undefined ? "" : `?action=${encodeURIComponent(action)}`;
  return request(`/episode/step${query}`);
}

export function getTraining(): Promise<TrainingStatus> {
  return request("/training");
}

export function startTraining(profile: string, seed: number, timesteps: number): Promise<TrainingStatus> {
  return request("/training", {
    method: "POST",
    body: JSON.stringify({ profile, seed, timesteps }),
  });
}

export function evaluateModel(): Promise<EvaluationResponse> {
  return request("/evaluate", { method: "POST" });
}
