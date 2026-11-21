# Orbit local model inventory

Orbit currently supports local Qwen GGUF files through llama.cpp using CPU
inference by default. The available model sizes are 0.5B, 1.5B, and 3B. The
1.5B model is available in Q4_K_M, Q5_K_M, and Q6_K quantizations. Higher
quantization can improve fidelity but normally uses more RAM and may reduce
generation throughput.
