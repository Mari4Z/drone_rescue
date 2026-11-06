import "./style.css";
import {
  ApiError,
  evaluateModel,
  getCatalog,
  loadRun,
  resetEpisode,
  stepEpisode,
} from "./api";
import type {
  CatalogRun,
  ConfigChange,
  EpisodeMode,
  EpisodeState,
  EvaluationResponse,
  LoadResponse,
} from "./types";

// Carga visual inicial de la barra de puntos: puntos = POINTS_START + retorno del episodio.
// No forma parte de la recompensa, que sigue calculándose solo en el ambiente Python.
const POINTS_START = 100;

const modes: EpisodeMode[] = [
  "manual",
  "aleatorio",
  "referencia",
  "evaluacion",
  "entrenamiento",
  "replay",
];
const modeLabels: Record<EpisodeMode, string> = {
  manual: "Manual",
  aleatorio: "Aleatorio",
  referencia: "Referencia",
  evaluacion: "Evaluación",
  entrenamiento: "Entrenamiento",
  replay: "Replay",
};
const actionLabels = [
  { action: 0, label: "Este", glyph: "→" },
  { action: 1, label: "Oeste", glyph: "←" },
  { action: 2, label: "Norte", glyph: "↑" },
  { action: 3, label: "Sur", glyph: "↓" },
  { action: 4, label: "Ascender", glyph: "↟" },
  { action: 5, label: "Descender", glyph: "↡" },
  { action: 6, label: "Entregar", glyph: "◇" },
];

interface EpisodeLink {
  runId: string;
  seed: number;
  mode: EpisodeMode;
  steps: number;
  actions: number[];
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("No se encontró el elemento #app");

app.innerHTML = `
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="#/" aria-label="Drone Rescue, inicio">
        <span class="brand-mark" aria-hidden="true">DR</span>
        <span><strong>DRONE RESCUE</strong><small>CONTROL MLOPS · UNIVERSIDAD EXTERNADO</small></span>
      </a>
      <div class="topbar-right">
        <span class="authority"><i></i> SIMULACIÓN PYTHON · AUTORIDAD</span>
        <span class="service-indicator" id="service-status">Conectando API…</span>
      </div>
    </header>

    <main>
      <section class="intro">
        <div>
          <p class="eyebrow">LABORATORIO DE POLÍTICAS</p>
          <h1>Misión de rescate</h1>
          <p class="lede">Cada transición viene del ambiente Gymnasium. La escena 3D refleja su estado; no simula la física.</p>
        </div>
        <div class="intro-meta">
          <span class="meta-label">AMBIENTE SERVIDO</span>
          <strong id="environment-version">Sin cargar</strong>
          <span id="model-family">Drone Rescue · 3D</span>
        </div>
      </section>

      <div class="error-banner" id="error-banner" role="alert" hidden>
        <span class="error-icon" aria-hidden="true">!</span>
        <div><strong id="error-title">Error de servicio</strong><p id="error-message"></p></div>
        <button class="icon-button" id="dismiss-error" type="button" aria-label="Cerrar error">×</button>
      </div>

      <section class="workspace">
        <div class="scene-column">
          <div class="scene-heading">
            <div><p class="eyebrow">VISTA DE MISIÓN</p><h2>Entorno activo</h2></div>
            <div class="scene-badges">
              <span class="live-badge"><i></i> ESTADO REAL</span>
              <span class="stage-pill" id="stage-pill">Sin episodio</span>
            </div>
          </div>
          <div class="scene-frame">
            <div class="scene-canvas" id="scene-canvas"></div>
            <div class="scene-legend">
              <span><i class="legend-base"></i> BASE</span>
              <span><i class="legend-drop"></i> ENTREGA</span>
              <span><i class="legend-obstacle"></i> OBSTÁCULOS</span>
            </div>
            <div class="scene-coordinates" id="scene-coordinates">X 0.0 · Y 0.0 · Z 0.0 m</div>
          </div>
          <div class="telemetry-grid">
            <article class="telemetry-card"><span>EPISODIO · SEMILLA</span><strong id="episode-number">—</strong><small id="episode-steps">pasos del ambiente</small></article>
            <article class="telemetry-card"><span>ETAPA</span><strong id="episode-stage">—</strong><small id="supply-state">Sin carga</small></article>
            <article class="telemetry-card"><span>PUNTOS</span><strong id="episode-points">—</strong><div class="meter" id="points-meter" role="meter" aria-label="Puntos del episodio" aria-valuemin="0" aria-valuemax="100"><i></i></div><small id="episode-reward">inicia en ${POINTS_START}; es solo visual</small></article>
            <article class="telemetry-card"><span>BATERÍA</span><strong id="episode-battery">—</strong><div class="meter" id="battery-meter" role="meter" aria-label="Batería restante" aria-valuemin="0" aria-valuemax="100"><i></i></div><small id="episode-contacts">0 contactos</small></article>
          </div>
        </div>

        <aside class="control-column">
          <section class="panel active-panel">
            <div class="panel-heading"><div><p class="eyebrow">POLÍTICA Y AMBIENTE</p><h2>Sesión activa</h2></div><span class="mode-badge" id="active-mode">MANUAL</span></div>
            <div class="active-model">
              <span class="model-orb">✳</span>
              <div><strong id="active-model-name">Sin modelo cargado</strong><small id="active-model-run">Selecciona un run del catálogo</small></div>
            </div>
            <div class="active-facts">
              <div><span>AMBIENTE</span><strong id="active-environment">—</strong></div>
              <div><span>FALLO / TERMINACIÓN</span><strong id="failure-cause">—</strong></div>
            </div>
            <label class="mode-select-label" for="mode-select">Modo de operación</label>
            <select id="mode-select" aria-label="Modo de operación">
              ${modes.map((mode) => `<option value="${mode}">${modeLabels[mode]}</option>`).join("")}
            </select>
            <div class="control-actions">
              <button class="button button-secondary" id="reset-episode" type="button" disabled><span>↻</span> Reiniciar episodio</button>
              <button class="button button-primary" id="evaluate-model" type="button" disabled><span>◉</span> Ejecutar evaluación</button>
            </div>
            <div class="manual-controls" id="manual-controls">
              <div class="control-title"><span>CONTROL MANUAL</span><small>cada acción avanza la física 1 paso</small></div>
              <div class="action-grid">
                ${actionLabels.map(({ action, label, glyph }) => `
                  <button class="action-button${action === 6 ? " action-release" : ""}" type="button" data-action="${action}" disabled aria-label="${label}">
                    <span>${glyph}</span><small>${label}</small>
                  </button>
                `).join("")}
              </div>
            </div>
            <div class="training-notice" id="mode-notice">Elige un run compatible en el catálogo para comenzar.</div>
          </section>

          <section class="panel result-panel" id="evaluation-panel" hidden>
            <div class="panel-heading"><div><p class="eyebrow">SUITE PÚBLICA</p><h2>Resultado</h2></div><span class="result-icon">✓</span></div>
            <div id="evaluation-summary"></div>
          </section>
        </aside>
      </section>

      <section class="catalog-section">
        <div class="section-heading">
          <div><p class="eyebrow">TRACKING EXPERIMENTAL</p><h2>Catálogo de experimentos</h2><p>Runs y manifiestos de MLflow · Selecciona dos o más para comparar.</p></div>
          <div class="catalog-actions">
            <span class="selection-count" id="selection-count">0 seleccionados</span>
            <button class="button button-secondary" id="compare-runs" type="button" disabled>⇄ Comparar seleccionados</button>
            <button class="button button-secondary" id="refresh-catalog" type="button">↻ Actualizar</button>
          </div>
        </div>
        <div class="compare-panel" id="compare-panel" hidden></div>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th class="select-column"><span class="sr-only">Comparar</span></th>
              <th>EXPERIMENTO / HIPÓTESIS</th><th>NIVEL · AMBIENTE</th><th>CAMBIOS VS BASELINE</th>
              <th>RUN · SEMILLA</th><th>PASOS · TIEMPO</th><th>MÉTRICAS</th><th>ESTADO · MODELO</th><th>COMMIT</th><th>MLFLOW</th><th></th>
            </tr></thead>
            <tbody id="catalog-body"><tr><td colspan="11" class="empty-state">Consultando la API…</td></tr></tbody>
          </table>
        </div>
        <div class="catalog-footer"><span id="catalog-count">—</span><span>Los cambios de entorno se calculan contra <code>configs/drone_rescue.yaml</code>.</span></div>
      </section>
    </main>
    <footer><span>DRONE RESCUE · MLOPS</span><span>Python decide · Three.js representa</span></footer>
  </div>
`;

const element = <T extends HTMLElement>(selector: string): T => {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`No se encontró ${selector}`);
  return found;
};
const { DroneScene } = await import("./scene");
const scene = new DroneScene(element("#scene-canvas"));
const catalogBody = element<HTMLTableSectionElement>("#catalog-body");
const errorBanner = element<HTMLDivElement>("#error-banner");
const errorMessage = element<HTMLParagraphElement>("#error-message");
const modeSelect = element<HTMLSelectElement>("#mode-select");
const selectedRuns = new Set<string>();
let catalogRuns: CatalogRun[] = [];
let activeRun: CatalogRun | null = null;
let activeLoad: LoadResponse | null = null;
let episode: EpisodeState | null = null;
let episodeSeed = 0;
let episodeActions: number[] = [];
let episodeSteps = 0;
let currentMode: EpisodeMode = "manual";
let operationBusy = false;
let restoringLink = false;

function showError(error: unknown): void {
  const status = error instanceof ApiError && error.status ? `HTTP ${error.status} · ` : "";
  element("#error-title").textContent = error instanceof ApiError && error.status === 0
    ? "API no disponible"
    : "Error de servicio";
  errorMessage.textContent = `${status}${error instanceof Error ? error.message : "Ocurrió un error inesperado."}`;
  errorBanner.hidden = false;
}

function clearError(): void {
  errorBanner.hidden = true;
  errorMessage.textContent = "";
}

function setBusy(busy: boolean): void {
  operationBusy = busy;
  document.querySelectorAll<HTMLButtonElement>(".action-button").forEach((button) => {
    button.disabled = busy || !activeLoad || currentMode !== "manual" || Boolean(episode?.terminated || episode?.truncated);
  });
  element<HTMLButtonElement>("#reset-episode").disabled = busy || !activeLoad;
  element<HTMLButtonElement>("#evaluate-model").disabled = busy || !activeLoad;
  modeSelect.disabled = busy || !activeLoad;
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function escapeHtml(value: unknown): string {
  return String(value ?? "—").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character] ?? character);
}

function displayMetrics(run: CatalogRun): { primary: string; secondary: string } {
  const metrics = run.metrics;
  const preferred = [
    ["success", metrics["evaluation/success"]],
    ["éxito", metrics["evaluation/success_rate"]],
  ] as const;
  const metric = preferred.find(([, value]) => typeof value === "number");
  const primary = metric
    ? `${metric[0]} ${formatNumber((metric[1] as number) * 100, 1)}%`
    : (() => {
      const successful = Object.entries(metrics).find(([key]) => key.endsWith("/success_rate"));
      if (successful) return `éxito ${formatNumber(successful[1] * 100, 1)}%`;
      const success = metrics["training/success"];
      return typeof success === "number" ? `éxito ${formatNumber(success * 100, 1)}%` : "Métricas no disponibles";
    })();
  const successful = Object.entries(metrics).find(([key]) => key.endsWith("/success_rate"));
  const evaluationReturn = metrics["evaluation/return"];
  const secondary = typeof evaluationReturn === "number"
    ? `retorno ${formatNumber(evaluationReturn)}`
    : run.mean_steps !== null
      ? `media ${formatNumber(run.mean_steps, 1)} pasos`
      : successful
        ? `suite ${formatNumber(successful[1] * 100, 1)}%`
        : "métricas adicionales —";
  return { primary, secondary };
}

function formatConfigValue(value: unknown): string {
  if (Array.isArray(value) && value.some((entry) => typeof entry === "object" && entry !== null)) {
    return `${value.length} elementos`;
  }
  return JSON.stringify(value);
}

function formatChanges(changes: ConfigChange[] | null): string {
  if (changes === null) return "Sin comparación";
  if (!changes.length) return "Sin cambios";
  return changes.slice(0, 3).map((change) => `
    <span class="change-line"><code>${escapeHtml(change.path)}</code>
      <small>${escapeHtml(formatConfigValue(change.baseline))} → ${escapeHtml(formatConfigValue(change.run))}</small>
    </span>
  `).join("") + (changes.length > 3 ? `<small class="muted">+${changes.length - 3} cambios</small>` : "");
}

function renderCatalog(): void {
  catalogBody.innerHTML = catalogRuns.map((run) => {
    const statusClass = run.status === "FINISHED" ? "status-finished" : run.status === "FAILED" ? "status-failed" : "status-other";
    const mlflowLink = run.mlflow_link
      ? `<a class="mlflow-link" href="${escapeHtml(run.mlflow_link)}" target="_blank" rel="noreferrer">Abrir ↗</a>`
      : `<span class="muted" title="${escapeHtml(run.mlflow_tracking_uri)}">Local</span>`;
    const metrics = displayMetrics(run);
    const hypothesis = run.hypothesis || "Hipótesis no registrada";
    const details = [
      run.config_error ? `<small class="config-error" title="${escapeHtml(run.config_error)}">Config no disponible</small>` : "",
      run.evaluation_error ? `<small class="config-error" title="${escapeHtml(run.evaluation_error)}">Pasos no disponibles</small>` : "",
    ].join("");
    return `<tr class="${activeRun?.run_id === run.run_id ? "row-active" : ""}">
      <td class="select-column"><input type="checkbox" class="run-select" data-run-id="${escapeHtml(run.run_id)}" ${selectedRuns.has(run.run_id) ? "checked" : ""} aria-label="Comparar run ${escapeHtml(run.run_id)}"></td>
      <td class="hypothesis-cell"><span class="experiment-name">${escapeHtml(run.experiment_id || run.experiment || run.level || "Experimento")}</span><strong>${escapeHtml(hypothesis)}</strong></td>
      <td><span class="level-pill">${escapeHtml(run.level)}</span><small>env ${escapeHtml(run.environment_version)}</small></td>
      <td class="changes-cell">${formatChanges(run.changes_vs_baseline)}${details}</td>
      <td><code class="run-id">${escapeHtml(run.run_id)}</code><small>semilla ${escapeHtml(run.seed)}</small></td>
      <td><strong>${formatNumber(run.mean_steps, 1)} pasos</strong><small>${formatNumber(run.duration_seconds, 1)} s · ${escapeHtml(run.training_episodes ?? "—")} ep.</small></td>
      <td><span class="metric-chip">${escapeHtml(metrics.primary)}</span><small>${escapeHtml(metrics.secondary)}</small><small>${escapeHtml(run.checkpoint_status || "sin estado de modelo")}</small></td>
      <td><span class="status ${statusClass}">${escapeHtml(run.status)}</span><small>${escapeHtml(run.model_version ? `v${run.model_version}` : "sin versión")}</small></td>
      <td><code class="commit">${escapeHtml(run.commit ? run.commit.slice(0, 8) : "—")}</code></td>
      <td>${mlflowLink}</td>
      <td><button class="button button-load" type="button" data-load-run="${escapeHtml(run.run_id)}" ${run.model_available ? "" : "disabled"}>Cargar</button></td>
    </tr>`;
  }).join("") || `<tr><td colspan="11" class="empty-state">No hay runs disponibles.</td></tr>`;
  element("#catalog-count").textContent = `${catalogRuns.length} runs · ${catalogRuns.filter((run) => run.manifest_available).length} con manifiesto`;
  updateSelectionControls();
}

function updateSelectionControls(): void {
  const count = selectedRuns.size;
  element("#selection-count").textContent = `${count} seleccionado${count === 1 ? "" : "s"}`;
  element<HTMLButtonElement>("#compare-runs").disabled = count < 2;
}

// Mismo medidor para puntos y batería. Los bonos pueden pasar de 100 pts: la barra se
// queda llena y el número muestra el valor real.
function setMeter(selector: string, fraction: number): void {
  const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  const meter = element<HTMLElement>(selector);
  meter.setAttribute("aria-valuenow", String(percent));
  meter.dataset.level = percent > 50 ? "high" : percent > 20 ? "mid" : "low";
  element<HTMLElement>(`${selector} i`).style.width = `${percent}%`;
}

function updateEpisodeView(state: EpisodeState): void {
  episode = state;
  currentMode = state.mode;
  element("#active-mode").textContent = modeLabels[state.mode].toUpperCase();
  element<HTMLSelectElement>("#mode-select").value = state.mode;
  element("#episode-number").textContent = `#${episodeSeed}`;
  element("#episode-steps").textContent = `${state.telemetry.steps} pasos`;
  element("#episode-stage").textContent = state.telemetry.stage;
  element("#stage-pill").textContent = state.terminated || state.truncated
    ? state.cause || "terminado"
    : state.telemetry.stage;
  element("#supply-state").textContent = state.telemetry.has_supply ? "Suministro a bordo" : "Sin carga";
  const points = POINTS_START + state.episode_return;
  element("#episode-points").textContent = `${formatNumber(points, 1)} pts`;
  setMeter("#points-meter", points / POINTS_START);
  element("#episode-reward").textContent = `retorno ${formatNumber(state.episode_return)} · último paso ${formatNumber(state.reward)}`;
  setMeter("#battery-meter", state.telemetry.battery_fraction);
  element("#episode-battery").textContent = `${formatNumber(state.telemetry.battery_wh, 1)} Wh`;
  element("#episode-contacts").textContent = `${state.telemetry.contacts} contacto${state.telemetry.contacts === 1 ? "" : "s"}`;
  element("#failure-cause").textContent = state.cause || (state.terminated || state.truncated ? "Terminado" : "Sin fallo");
  const [x, y, z] = state.position_m;
  element("#scene-coordinates").textContent = `X ${x.toFixed(1)} · Y ${y.toFixed(1)} · Z ${z.toFixed(1)} m`;
  scene.update(state);
  element("#service-status").textContent = "API conectada";
  element("#service-status").className = "service-indicator service-online";
  setBusy(operationBusy);
}

function updateDeepLink(): void {
  if (!activeRun || restoringLink) return;
  const params = new URLSearchParams({
    run: activeRun.run_id,
    seed: String(episodeSeed),
    mode: currentMode,
    steps: String(episodeSteps),
  });
  if (currentMode === "manual" && episodeActions.length) {
    params.set("actions", episodeActions.join(","));
  }
  history.replaceState(null, "", `#/run/${encodeURIComponent(activeRun.run_id)}?${params}`);
}

function readDeepLink(): EpisodeLink | null {
  const hash = window.location.hash;
  if (!hash.startsWith("#/run/")) return null;
  const question = hash.indexOf("?");
  const path = question < 0 ? hash.slice(6) : hash.slice(6, question);
  const query = new URLSearchParams(question < 0 ? "" : hash.slice(question + 1));
  const runId = decodeURIComponent(path.split("/")[0] || query.get("run") || "");
  const seed = Number(query.get("seed") ?? 0);
  const mode = query.get("mode") as EpisodeMode | null;
  const steps = Number(query.get("steps") ?? 0);
  const actionsValue = query.get("actions");
  const actions = actionsValue ? actionsValue.split(",").map(Number) : [];
  if (
    !runId
    || !Number.isInteger(seed)
    || !Number.isInteger(steps)
    || steps < 0
    || !mode
    || !modes.includes(mode)
    || actions.some((action) => !Number.isInteger(action) || action < 0 || action > 6)
  ) return null;
  return { runId, seed, mode, steps, actions };
}

async function loadSelectedRun(run: CatalogRun, seed = run.seed ?? 0, link?: EpisodeLink): Promise<void> {
  clearError();
  setBusy(true);
  try {
    const loaded = await loadRun(run.run_id, seed);
    activeRun = run;
    activeLoad = loaded;
    episodeSeed = seed;
    episodeActions = [];
    episodeSteps = 0;
    scene.setEnvironment(loaded.config);
    element("#environment-version").textContent = `v${loaded.environment.version}`;
    element("#model-family").textContent = loaded.environment.model_family;
    element("#active-model-name").textContent = run.experiment || run.level || "Modelo Q-table";
    element("#active-model-run").textContent = `${run.run_id} · ${run.model_version ? `v${run.model_version}` : "sin versión"} · ${run.model_sha256?.slice(0, 12) ?? "SHA no disponible"}`;
    element("#active-environment").textContent = `${loaded.environment.level || "—"} · v${loaded.environment.version}`;
    element<HTMLButtonElement>("#reset-episode").disabled = false;
    element<HTMLButtonElement>("#evaluate-model").disabled = false;
    const mode = link?.mode ?? "manual";
    const reset = await resetEpisode(mode, seed);
    updateEpisodeView(reset);
    if (link) {
      if (link.mode === "manual") {
        for (const action of link.actions) {
          const restored = await stepEpisode(action);
          episodeActions.push(action);
          episodeSteps += 1;
          updateEpisodeView(restored);
        }
      } else {
        for (let index = 0; index < link.steps; index += 1) {
          const restored = await stepEpisode();
          episodeSteps += 1;
          updateEpisodeView(restored);
        }
      }
    }
    selectedRuns.add(run.run_id);
    renderCatalog();
    updateDeepLink();
  } catch (error) {
    if (!activeLoad) {
      element("#service-status").textContent = "API con error";
      element("#service-status").className = "service-indicator service-offline";
    }
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function changeMode(mode: EpisodeMode): Promise<void> {
  if (!activeLoad) return;
  clearError();
  setBusy(true);
  try {
    const state = await resetEpisode(mode, episodeSeed);
    currentMode = mode;
    episodeActions = [];
    episodeSteps = 0;
    updateEpisodeView(state);
    updateDeepLink();
    element("#mode-notice").textContent = mode === "manual"
      ? "Cada acción manual avanza exactamente un paso del ambiente."
      : `La política ${modeLabels[mode]} selecciona acciones en Python.`;
  } catch (error) {
    modeSelect.value = currentMode;
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function advanceEpisode(action?: number): Promise<void> {
  if (!activeLoad || operationBusy) return;
  clearError();
  setBusy(true);
  try {
    const state = await stepEpisode(action);
    if (currentMode === "manual" && action !== undefined) episodeActions.push(action);
    episodeSteps += 1;
    updateEpisodeView(state);
    updateDeepLink();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function restoreDeepLink(link: EpisodeLink): Promise<void> {
  const run = catalogRuns.find((candidate) => candidate.run_id === link.runId);
  if (!run) {
    showError(new ApiError(`El run ${link.runId} no aparece en el catálogo.`, 404));
    return;
  }
  restoringLink = true;
  try {
    await loadSelectedRun(run, link.seed, link);
  } finally {
    restoringLink = false;
  }
}

function renderComparison(): void {
  const runs = catalogRuns.filter((run) => selectedRuns.has(run.run_id));
  if (runs.length < 2) return;
  const metricEntries = [
    ["Éxito", "evaluation/success", "percent"],
    ["Retorno evaluación", "evaluation/return", "number"],
    ["Retorno entrenamiento", "training/return", "number"],
    ["Éxito entrenamiento", "training/success", "percent"],
  ] as const;
  const comparison = element<HTMLDivElement>("#compare-panel");
  comparison.hidden = false;
  comparison.innerHTML = `
    <div class="compare-heading"><div><p class="eyebrow">COMPARACIÓN DE RUNS</p><h3>${runs.length} políticas · mismo catálogo MLflow</h3></div><button class="icon-button" id="close-compare" type="button" aria-label="Cerrar comparación">×</button></div>
    <div class="compare-grid">${runs.map((run) => `
      <article class="compare-card"><span>${escapeHtml(run.level)} · semilla ${escapeHtml(run.seed)}</span><strong>${escapeHtml(run.experiment_id || run.experiment || run.run_id)}</strong><small><code>${escapeHtml(run.run_id)}</code></small>
        <div class="compare-metrics">${metricEntries.map(([label, key, format]) => {
          const value = run.metrics[key];
          const display = typeof value !== "number" ? "—" : format === "percent" ? `${formatNumber(value * 100, 1)}%` : formatNumber(value, 2);
          return `<div><span>${label}</span><b>${display}</b></div>`;
        }).join("")}</div>
      </article>
    `).join("")}</div>`;
  element<HTMLButtonElement>("#close-compare").addEventListener("click", () => {
    comparison.hidden = true;
  });
}

async function runEvaluation(): Promise<void> {
  if (!activeLoad || operationBusy) return;
  clearError();
  setBusy(true);
  const button = element<HTMLButtonElement>("#evaluate-model");
  button.innerHTML = `<span class="spinner"></span> Evaluando suite…`;
  try {
    const response: EvaluationResponse = await evaluateModel();
    const summary = response.result.cases.map((testCase) => {
      const success = testCase.metrics.success_rate;
      const steps = testCase.metrics.mean_steps;
      return `<div class="evaluation-row"><span>${escapeHtml(testCase.case_id)}</span><strong>${typeof success === "number" ? `${(success * 100).toFixed(1)}% éxito` : "—"}</strong><small>${typeof steps === "number" ? `${steps.toFixed(1)} pasos` : "—"}</small></div>`;
    }).join("");
    element("#evaluation-summary").innerHTML = `${summary}<p class="result-path">Resultado · <code>${escapeHtml(response.result_path)}</code></p>`;
    element<HTMLDivElement>("#evaluation-panel").hidden = false;
  } catch (error) {
    showError(error);
  } finally {
    button.innerHTML = `<span>◉</span> Ejecutar evaluación`;
    setBusy(false);
  }
}

async function refreshCatalog(): Promise<void> {
  const button = element<HTMLButtonElement>("#refresh-catalog");
  button.disabled = true;
  clearError();
  try {
    const response = await getCatalog();
    catalogRuns = response.runs;
    renderCatalog();
    element("#service-status").textContent = "API conectada";
    element("#service-status").className = "service-indicator service-online";
    const link = readDeepLink();
    if (link && !restoringLink && activeRun?.run_id !== link.runId) await restoreDeepLink(link);
  } catch (error) {
    catalogBody.innerHTML = `<tr><td colspan="11" class="empty-state empty-error">No se pudo cargar el catálogo: ${escapeHtml(error instanceof Error ? error.message : error)}</td></tr>`;
    element("#service-status").textContent = "API con error";
    element("#service-status").className = "service-indicator service-offline";
    showError(error);
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const loadButton = target.closest<HTMLButtonElement>("[data-load-run]");
  if (loadButton) {
    const run = catalogRuns.find((candidate) => candidate.run_id === loadButton.dataset.loadRun);
    if (run) void loadSelectedRun(run);
  }
  const actionButton = target.closest<HTMLButtonElement>("[data-action]");
  if (actionButton && actionButton.dataset.action) void advanceEpisode(Number(actionButton.dataset.action));
});

catalogBody.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.matches(".run-select")) return;
  const runId = target.dataset.runId;
  if (!runId) return;
  if (target.checked) selectedRuns.add(runId);
  else selectedRuns.delete(runId);
  updateSelectionControls();
});

modeSelect.addEventListener("change", () => {
  const mode = modeSelect.value as EpisodeMode;
  if (modes.includes(mode)) void changeMode(mode);
});
element<HTMLButtonElement>("#reset-episode").addEventListener("click", () => {
  void changeMode(currentMode);
});
element<HTMLButtonElement>("#evaluate-model").addEventListener("click", () => {
  void runEvaluation();
});
element<HTMLButtonElement>("#compare-runs").addEventListener("click", renderComparison);
element<HTMLButtonElement>("#refresh-catalog").addEventListener("click", () => void refreshCatalog());
element<HTMLButtonElement>("#dismiss-error").addEventListener("click", clearError);

window.addEventListener("hashchange", () => {
  const link = readDeepLink();
  if (link && link.runId !== activeRun?.run_id) void restoreDeepLink(link);
});
window.addEventListener("beforeunload", () => scene.dispose(), { once: true });

void refreshCatalog();
