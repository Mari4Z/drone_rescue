export type EpisodeMode =
  | "manual"
  | "aleatorio"
  | "referencia"
  | "evaluacion"
  | "entrenamiento"
  | "replay";

export interface ConfigChange {
  path: string;
  baseline: unknown;
  run: unknown;
}

export interface CatalogRun {
  run_id: string;
  status: string;
  checkpoint_status: string | null;
  hypothesis: string | null;
  commit: string | null;
  level: string | null;
  experiment_id: string | null;
  mlflow_experiment: string | null;
  experiment: string | null;
  environment_version: string | null;
  seed: number | null;
  training_episodes: number | null;
  training_timesteps: number | null;
  algorithm: string;
  model_family: string;
  mean_steps: number | null;
  duration_seconds: number | null;
  model_version: string | null;
  model_available: boolean;
  registry_uri: string | null;
  model_sha256: string | null;
  changed_dimension: string | null;
  changes_vs_baseline: ConfigChange[] | null;
  config_error: string | null;
  evaluation_error: string | null;
  metrics: Record<string, number>;
  manifest_available: boolean;
  mlflow_link: string | null;
  mlflow_tracking_uri: string;
  mlflow_run_uri: string;
}

export interface CatalogResponse {
  runs: CatalogRun[];
}

export interface LoadResponse {
  session_id: string;
  model: {
    run_id: string;
    sha256: string;
    level: string | null;
    commit: string | null;
  };
  environment: {
    version: string;
    reward_version: string;
    level: string | null;
    model_family: string;
    algorithm: string;
  };
  config: DroneConfig;
  config_sha256: string;
  mode: EpisodeMode;
}

export interface ManualSessionResponse {
  session_id: string;
  environment: LoadResponse["environment"];
  config: DroneConfig;
  mode: "manual";
  state: EpisodeState;
}

export interface TrainingJob {
  job_id: string;
  status: "RUNNING" | "FINISHED" | "FAILED";
  profile_id: string;
  algorithm?: "ppo" | "q_learning";
  seed: number;
  episode: number;
  total_episodes: number | null;
  timesteps?: number;
  total_timesteps?: number | null;
  evaluation_success?: number;
  evaluation_delivered?: number;
  return?: number;
  success?: boolean;
  epsilon?: number;
  cause?: string;
  steps?: number;
  run_id: string | null;
  manifest_path?: string;
  checkpoint_path?: string;
  error: string | null;
}

export interface DroneConfig {
  world: {
    cell_size_m: number;
    grid_cells: [number, number, number];
    base_center_m: [number, number, number];
    base_radius_m: number;
    delivery_center_m: [number, number, number];
    delivery_radius_m: number;
    obstacles: Array<{ min_m: [number, number, number]; max_m: [number, number, number] }>;
  };
  physics: {
    drone_radius_m: number;
  };
}

export interface EpisodeState {
  run_id: string;
  mode: EpisodeMode;
  position_m: [number, number, number];
  telemetry: {
    velocity_mps: [number, number, number];
    battery_wh: number;
    battery_fraction: number;
    has_supply: boolean;
    stage: string;
    contacts: number;
    steps: number;
    wind_mps2: [number, number, number];
    environment_version: string;
  };
  reward: number;
  episode_return: number;
  cause: string | null;
  terminated: boolean;
  truncated: boolean;
}

export interface EvaluationResponse {
  evaluation_id: string;
  run_id: string;
  evaluation_run_id: string;
  result_path: string;
  result: {
    cases: Array<{
      case_id: string;
      metrics: Record<string, number | null>;
    }>;
  };
}

export interface ApiFailure {
  detail?: string;
}
