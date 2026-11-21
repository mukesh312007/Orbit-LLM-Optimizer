# API documentation

Run only on loopback by default: `uvicorn api.app:app --host 127.0.0.1 --port 8001`.

`Start-Agent.ps1` starts this API on port 8001 by default. It refuses to use a
port occupied by a non-Orbit application instead of silently talking to it.

- `GET /health` reports backend reachability.
- `GET /api/v1/models` returns registry entries and local availability.
- `GET /api/v1/readiness` reports available models, safe matrix capacity,
  calibrated-profile readiness, and energy-measurement limitations without
  starting a benchmark.
- `POST /api/v1/optimize` accepts `task_type`, optional constraints, and `preference` (`quality`, `latency`, `memory`, `energy`, `balanced`).
- `POST /api/v1/generate` adds `prompt`, optional `model_id`, and `max_output_tokens`; live Spider threshold selection requires `reference_sql` and compares the normalized SQL text with that reference. It returns the selected configuration and clearly labelled inference telemetry.
- `GET /api/v1/experiments` exposes imported historical profile rows.
- `GET /api/v1/recommendations` returns a default optimization result.

The API returns HTTP 503 when inference is unavailable; it never produces a synthetic answer as a fallback.
