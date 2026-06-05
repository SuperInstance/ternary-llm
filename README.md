# ternary-llm

Ternary LLM building blocks for the SuperInstance {-1, 0, +1} ecosystem.

## Features

- **BitNet 1.58-bit quantization** — `bitnet_quantize` maps float weights to trits with a per-tensor scale; `bitnet_dequantize` recovers approximations
- **TokenEmbedding** — vocab-to-ternary lookup table; `embed_sequence` for batches
- **TernaryLinear** — ternary weight matrix + float scale + bias; `forward_float` for full-precision input
- **TernaryTransformerBlock** — pre-norm → ternary attention head → residual → pre-norm → ternary FFN → residual
- **KvCache** — key/value cache with ternary compression; evicts oldest when full
- **TernaryLM** — end-to-end model with `forward` and greedy `generate`
- `compression_ratio` — theoretical bits-per-weight calculation (~20× vs float32)

## Usage

```rust
use ternary_llm::{TernaryLM, bitnet_quantize};

let model = TernaryLM::new(256, 64, 16, 128);
let generated = model.generate(&[0, 1, 2], 10);
```

## Tests

21 tests covering quantization, embeddings, linear layers, FFN, transformer blocks, KV-cache, and generation.
