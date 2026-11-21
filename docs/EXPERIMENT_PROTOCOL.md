# Reproducible experiment protocol

The primary research question is: **which local configuration has the lowest
measured cost while meeting a task-valid quality threshold?**

## Required controlled matrix

Independent variables: local model/GGUF quantization, CPU threads (2/4/6/8),
context (1024/2048/4096), prompt-cache policy, compression mode, and retrieval
on/off. Controlled variables: prompt template, llama.cpp build, CPU-only mode,
temperature zero, output limit, local machine, and dataset split.

Run at least 10 diverse Spider development databases with three trials per
configuration. Separate one warm-up from measured trials. Preserve every
trial, failure, and RAM safety skip. Do not compare a cache-warm condition with
an unrelated cold condition.

## Quality and success criteria

For Spider, score normalized SQL exact matching against the matching reference
SQL in the dataset file. A configuration is eligible only if its conservative quality
lower confidence bound satisfies the requested threshold. The selected
configuration must have lower measured cost than the declared baseline while
meeting the same threshold.

For general chat, coding, reasoning, and summarization, add a locally supplied
licensed held-out dataset and task-specific evaluator before enabling a quality
threshold. Do not use Spider performance as proof of general-chat quality.

## Measurements

Record TTFT, end-to-end latency, prompt processing time, generation throughput,
peak model-process RSS, source/model hash, runtime flags, hardware snapshot,
warm-up state, and trial number. Actual joules require a monotonic cumulative
physical-meter export. CPU-based values are labelled estimates.

## Execution sequence

1. Run `/api/v1/readiness` or a full-matrix dry run.
2. Run a reduced safe matrix for one model and validate raw JSONL.
3. Expand to all safe models/configurations.
4. Build profiles with `--minimum-samples 3`.
5. Select at least two quality thresholds and verify the winner/rejections.
6. Repeat on a quiet machine state and report mean, median, standard deviation,
   sample count, and limitations.
