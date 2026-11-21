# Orbit configuration selection

Orbit first applies the requested quality threshold to calibrated benchmark
profiles. It rejects a configuration with unavailable quality or quality below
the threshold. Among eligible configurations, it uses normalized latency,
TTFT, RAM, throughput, and clearly labelled energy-proxy efficiency to select
the lowest-cost option.

Actual joules are kept unavailable until a physical power meter or supported
counter supplies cumulative energy readings. A configuration with a higher
quality score does not automatically win after it has passed the quality gate.
