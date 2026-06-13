# ternary-llm
[![Migration: Binary → Ternary](https://img.shields.io/badge/Migration-Binary%E2%86%92Ternary-blueviolet)](https://github.com/SuperInstance/ternary-types)


**Ternary language model building blocks: a complete miniature LLM with {-1, 0, +1} weights.**

This crate implements the full transformer stack — token embedding, multi-head attention, feed-forward network, KV-cache, and autoregressive decoding — using only ternary weights. It's a working proof that you can build an entire LLM where every weight is in {-1, 0, +1}.

---

## Why This Matters

Microsoft's BitNet b1.58 (2024) demonstrated that 1.58-bit LLMs (ternary weights) match the quality of float16 models at 70B parameter scale. The key insight: neural networks are massively over-parameterized, and {-1, 0, +1} provides enough expressivity when you have enough weights.

**Per-weight cost comparison:**

| Format | Bits/weight | Memory (7B params) | Inference energy |
|--------|-------------|---------------------|------------------|
| float32 | 32 | 28 GB | 1.0x (baseline) |
| float16 | 16 | 14 GB | ~0.5x |
| int8 | 8 | 7 GB | ~0.25x |
| **ternary** | **1.58** | **~1.4 GB** | **~0.05x** |

---

## Architecture

```
Input tokens
    │
    ▼
TokenEmbedding ─── maps token IDs to ternary vectors
    │
    ▼
TernaryTransformerBlock ── repeated N times
    ├── rms_norm
    ├── TernaryAttentionHead (Q,K,V ∈ {-1,0,+1})
    ├── residual connection
    ├── rms_norm
    ├── TernaryFFN (up-project → ReLU → down-project)
    └── residual connection
    │
    ▼
argmax decoding ─── next token prediction
```

### Key Types

- **`TokenEmbedding`** — Embeds tokens into ternary weight space with per-tensor scaling
- **`TernaryLinear`** — Matrix multiply with {-1,0,+1} weights and INT scaling
- **`TernaryAttentionHead`** — Single-head attention with ternary Q,K,V projections
- **`TernaryFFN`** — SwiGLU-style feed-forward with ternary up/down projections
- **`TernaryTransformerBlock`** — Full transformer block: norm → attention → norm → FFN
- **`KvCache`** — Caches ternary key/value pairs across generation steps
- **`TernaryLM`** — Complete language model: embed → N blocks → decode

### BitNet Quantization

```rust
use ternary_llm::{bitnet_quantize, bitnet_dequantize, Trit};

let float_weights = vec![0.23, -0.87, 0.01, 1.42, -0.55];
let (trits, scale) = bitnet_quantize(&float_weights);
// trits: [+1, -1, 0, +1, -1], scale: 0.87
// Each weight ≈ scale × trit

let reconstructed = bitnet_dequantize(&trits, scale);
```

---

## Quick Start

```toml
[dependencies]
ternary-llm = "0.1.0"
```

```rust
use ternary_llm::{TernaryLM, TernaryTransformerBlock, TokenEmbedding, KvCache};

let vocab_size = 32000;
let d_model = 512;
let n_heads = 8;

let mut model = TernaryLM::new(vocab_size, d_model, n_heads, 6);

// Forward pass
let tokens = vec![1, 42, 1337, 2024];
let logits = model.forward(&tokens);

// Autoregressive generation
let generated = model.generate(&[1, 42], 50, 0.8);
```

---

## Performance

The `TernaryLinear` layer replaces float multiplications with additions:
- **float32 matmul**: M×N×K multiply-accumulate operations
- **ternary matmul**: M×N×K add/subtract operations (no multiplication!)

Each weight ∈ {-1, 0, +1} means the multiply becomes: skip (0), negate (-1), or pass through (+1).

---

## Ecosystem

- **ternary-tnn** — Lower-level ternary neural network layers (conv1d/2d, LUT matmul)
- **ternary-attention** — Standalone attention mechanisms for ternary inputs
- **ternary-grad** — Training utilities: STE, ternary Adam/SGD optimizers
- **ternary-cookbook** — Working demos and tutorials

## License

MIT
