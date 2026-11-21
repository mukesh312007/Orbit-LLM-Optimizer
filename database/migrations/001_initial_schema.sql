-- Execute with: mysql -u USER -p DATABASE < database/migrations/001_initial_schema.sql
-- Clean-install schema for the optional local MySQL telemetry mirror.
-- JSONL remains the authoritative evidence store.

CREATE TABLE IF NOT EXISTS models (
  model_id VARCHAR(128) PRIMARY KEY,
  family VARCHAR(128) NOT NULL,
  quantization VARCHAR(64) NULL,
  metadata_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS hardware_profiles (
  hardware_profile_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  created_at DATETIME(6) NOT NULL,
  profile_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_id VARCHAR(128) PRIMARY KEY,
  source TEXT NULL,
  manifest_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_configurations (
  configuration_id VARCHAR(128) PRIMARY KEY,
  model_id VARCHAR(128) NOT NULL,
  backend VARCHAR(32) NOT NULL,
  settings_json JSON NOT NULL,
  CONSTRAINT fk_runtime_model FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE IF NOT EXISTS benchmark_experiments (
  experiment_id VARCHAR(128) PRIMARY KEY,
  started_at DATETIME(6) NOT NULL,
  model_id VARCHAR(128) NULL,
  configuration_id VARCHAR(128) NULL,
  dataset_id VARCHAR(128) NULL,
  kind VARCHAR(64) NOT NULL,
  metadata_json JSON NOT NULL,
  CONSTRAINT fk_experiment_model FOREIGN KEY (model_id) REFERENCES models(model_id),
  CONSTRAINT fk_experiment_configuration FOREIGN KEY (configuration_id) REFERENCES runtime_configurations(configuration_id),
  CONSTRAINT fk_experiment_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS benchmark_results (
  result_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  experiment_id VARCHAR(128) NOT NULL,
  prompt_id VARCHAR(128) NULL,
  trial_number INTEGER NULL,
  response_text TEXT NULL,
  CONSTRAINT uq_benchmark_trial UNIQUE (experiment_id, prompt_id, trial_number),
  CONSTRAINT fk_result_experiment FOREIGN KEY (experiment_id) REFERENCES benchmark_experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS quality_results (
  quality_result_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  result_id INTEGER NOT NULL,
  metric VARCHAR(64) NOT NULL,
  score DOUBLE NULL,
  source VARCHAR(32) NOT NULL,
  details_json JSON NOT NULL,
  CONSTRAINT fk_quality_result FOREIGN KEY (result_id) REFERENCES benchmark_results(result_id)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
  metric_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  result_id INTEGER NOT NULL,
  ttft_seconds DOUBLE NULL,
  tpot_ms DOUBLE NULL,
  end_to_end_seconds DOUBLE NULL,
  generation_tokens_per_second DOUBLE NULL,
  generated_tokens INTEGER NULL,
  CONSTRAINT fk_performance_result FOREIGN KEY (result_id) REFERENCES benchmark_results(result_id)
);

CREATE TABLE IF NOT EXISTS resource_metrics (
  metric_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  result_id INTEGER NOT NULL,
  process_memory_mb DOUBLE NULL,
  system_memory_percent DOUBLE NULL,
  cpu_percent DOUBLE NULL,
  gpu_utilization_percent DOUBLE NULL,
  CONSTRAINT fk_resource_result FOREIGN KEY (result_id) REFERENCES benchmark_results(result_id)
);

CREATE TABLE IF NOT EXISTS energy_metrics (
  metric_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  result_id INTEGER NOT NULL,
  energy_joules DOUBLE NULL,
  energy_proxy_joules DOUBLE NULL,
  source VARCHAR(128) NULL,
  measured BOOLEAN NOT NULL DEFAULT FALSE,
  measurement_kind VARCHAR(32) NOT NULL DEFAULT 'unavailable',
  proxy_method TEXT NULL,
  CONSTRAINT fk_energy_result FOREIGN KEY (result_id) REFERENCES benchmark_results(result_id)
);

CREATE TABLE IF NOT EXISTS optimization_recommendations (
  recommendation_id INTEGER AUTO_INCREMENT PRIMARY KEY,
  created_at DATETIME(6) NOT NULL,
  request_json JSON NOT NULL,
  selected_configuration_id VARCHAR(128) NULL,
  explanation TEXT NOT NULL,
  predicted BOOLEAN NOT NULL DEFAULT TRUE
);
