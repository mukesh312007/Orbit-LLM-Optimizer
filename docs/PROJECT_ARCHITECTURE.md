# Architecture

`datasets` prepares official records into a small normalized JSONL format. `benchmarks` validates records and selects task-specific metrics. `runtime` talks only to loopback llama.cpp or Ollama. `hardware` samples resources and never turns utilisation into energy. `database` persists MySQL experiment history; `optimizer` consumes explicitly measured profiles and returns feasibility, alternatives, and a Pareto set. `api` wires selection to inference. `dashboard.py` renders real recorded data.
