# Database schema

The SQLAlchemy schema contains `models`, `hardware_profiles`, `datasets`, `runtime_configurations`, `benchmark_experiments`, `benchmark_results`, `quality_results`, `performance_metrics`, `resource_metrics`, `energy_metrics`, and `optimization_recommendations`. Credentials are only read from `DATABASE_URL`.

For an existing local MySQL80 installation, run `scripts/setup-orbit-mysql.ps1` in PowerShell. It prompts locally for the MySQL administrator password, creates a dedicated `orbit_telemetry` account, writes the connection string to git-ignored `.env`, and initializes the schema. It does not remove or replace JSONL benchmark logs.
