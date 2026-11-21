# Limitations

- MySQL is optional. The local MySQL service may be available, but every deployment must independently verify its configured credentials and persistence path; JSONL remains the source-of-evidence fallback.
- Local llama.cpp inference can be verified only for the models and configurations actually benchmarked on the target hardware. A running server does not validate the entire configuration matrix.
- External datasets are intentionally not downloaded automatically; source licences and versions must be confirmed before preparation.
- MBPP code execution is disabled by default (`ALLOW_UNSAFE_CODE_EVAL` is not set) because a temporary Python process on Windows is not a hardened security boundary. Enable it only inside a disposable container/VM with explicit resource/network restrictions.
- Integrated graphics reporting and CPU/GPU utilisation are not energy measurements. Energy values remain unavailable without supported telemetry.
- A quality threshold is currently task-valid for Spider text-to-SQL only when a gold SQL reference is supplied. General chat, reasoning, coding, and summarization thresholds require their own held-out evaluation datasets before they can be claimed as measured.
- Historic CSV schemas lack some required metadata. The optimizer preserves this uncertainty rather than claiming quality or energy measurements.
