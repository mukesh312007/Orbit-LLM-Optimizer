# Responsible AI and measurement governance

Orbit is local-first: llama.cpp, Streamlit, and the Orbit API bind to loopback
addresses. No prompt, model response, benchmark result, or meter reading is
sent to a cloud service by this project.

## Measurement taxonomy

| Value | Classification | Selection use |
|---|---|---|
| Spider SQL execution score | Task-valid measured quality | Quality admission gate for Spider only |
| TTFT, end-to-end time, generation tokens/s, model RSS | Direct local runtime measurements | Cost comparison |
| CPU-utilisation energy proxy | Estimated, not physical energy | Exploratory comparison only |
| Cumulative joule meter delta from an approved physical source | Actual energy measurement | Energy constraints when source is present, physical, and monotonic |
| General chat quality without a labelled reference | Unavailable | Never treated as passing a quality threshold |

## Human oversight

The user can inspect every live candidate, configuration, rejected quality
candidate, RAM safety skip, metric provenance, and deferred run. A full matrix
is never silently run by default. The user may override the recommendation by
selecting a different local model or running a different benchmark plan.

## Data handling

Live response text is redacted from JSONL and MySQL telemetry by default.
Set `ORBIT_RETAIN_LIVE_RESPONSE_TEXT=true` only after approving local retention
for the prompts being handled. Benchmark records are retained because they are
reproducibility evidence; use only licensed/public benchmark data and do not
place private prompts in benchmark datasets.

## Risk register

| Risk | Safe behaviour | Remaining limitation |
|---|---|---|
| Insufficient RAM | Preserve a 1.5 GiB reserve and skip unsafe candidates | Estimate may be conservative or imperfect for unusual workloads |
| Missing quality evidence | Reject a thresholded live selection | General task quality suites must still be populated locally |
| Missing, simulated, or proxy energy meter | Return unavailable actual energy | Proxy must not be called measured energy or used for automatic ranking |
| Runtime/model failure | Record failure and do not select it | Recovery remains a local operator action |
| Retrieval mismatch | Show source path/chunk metadata | Lexical retrieval is not semantic grounding proof |
| Uncertain experiments | Record sample count and lower confidence bound | Repeated trials and larger held-out samples are still required |
