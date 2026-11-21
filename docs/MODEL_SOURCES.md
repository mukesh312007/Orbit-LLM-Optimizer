# Local model acquisition record

The benchmark artifacts in this project remain local. This document records the
source and integrity data for manually approved model acquisition; it contains
no model binary and no credentials.

## Qwen2.5 1.5B Instruct quantization study

Official compatible source: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` on Hugging Face.

| Quantization | Filename | Bytes | SHA-256 |
|---|---|---:|---|
| Q5_K_M | `qwen2.5-1.5b-instruct-q5_k_m.gguf` | 1,285,494,304 | `b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c` |
| Q6_K | `qwen2.5-1.5b-instruct-q6_k.gguf` | 1,464,178,720 | `e16d94f3b1eb243f6f6be9eee51090ef5dfd741324394fd5b6e0e425c33df5c7` |

Store validated files under `models/quantizations/` using the lower-case
filenames shown above. The controlled comparison harness refuses missing or
invalid GGUF files and never downloads or converts artifacts itself.
