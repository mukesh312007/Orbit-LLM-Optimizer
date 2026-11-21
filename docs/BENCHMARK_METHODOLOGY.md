# Benchmark methodology

Use repeated trials and retain a unique experiment ID for each run. Record cold/warm state explicitly, interleave conditions where feasible, and report mean/median/stddev from raw trials. TTFT is client-observed; llama.cpp server timing is separately reported. Energy is null unless an external meter or a supported hardware counter supplies joules. Historical CSV data is evidence, not a controlled quantization comparison.
