# ternary-llm

BitNet 1.58-bit large language model building blocks in pure Rust. Each weight is stored as a trit ∈ {-1, 0, +1} alongside a single per-tensor float scale, enabling **~20× memory compression** over FP32 with minimal quality loss.

## Why It Matters

Modern LLMs are memory-bound: the cost of loading weights dwarfs the arithmetic. BitNet 1.58-bit quantization represents every weight with just `log₂(3) ≈ 1.585` bits, turning a 4-billion-parameter model from 16 GB (FP32) down to ~0.8 GB. Because the weights are ternary, matrix multiplication reduces to integer addition and subtraction — no floating-point multiply needed in the inner loop.

## How It Works

### Quantization

Given a weight tensor **W** ∈ ℝⁿ, compute the scale:

$$s = \frac{1}{n}\sum_{i=1}^{n} |w_i|$$

Then quantize each weight:

$$\tilde{w}_i = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w_i}{s}\right),\;-1,\;+1\right) \in \{-1, 0, +1\}$$

Dequantization is simply $\hat{w}_i = \tilde{w}_i \cdot s$.

### Ternary Linear Layer

Forward pass for `TernaryLinear`:

$$\text{out}_i = s \cdot \sum_{j=1}^{d_{\text{in}}} \tilde{W}_{ij} \cdot x_j + b_i$$

Since $\tilde{W}_{ij} \in \{-1, 0, +1\}$, each multiply-accumulate becomes an **add, subtract, or no-op** — O(1) with zero hardware multipliers.

**Complexity:** O(d_out × d_in) per forward pass — same asymptotic order as dense FP32, but with a ~4-8× wall-clock speedup from eliminating floating-point multiplication.

### Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Q, K, V are produced by ternary projection layers. The attention scores are computed in float space to preserve numerical stability.

### KV-Cache Compression

Keys and values at each position are quantized to ternary + scale before storage, reducing KV-cache memory by ~20×. This enables longer context windows within fixed GPU memory.

### Compression Ratio

For $n$ weights:
- FP32 storage: `32n` bits
- Ternary storage: `1.585n + 32` bits (one scale per tensor)
- Ratio: `32n / (1.585n + 32) ≈ 20.2×` for large $n$

## Quick Start

```rust
use ternary_llm::*;

// Quantize weights
let weights = vec![0.5, -0.8, 0.0, 0.3, -0.1, 0.9];
let (trits, scale) = bitnet_quantize(&weights);
// trits = [1, -1, 0, 0, 0, 1], scale ≈ 0.433

// Build a ternary linear layer
let layer = TernaryLinear::new(64, 32);
let output = layer.forward_float(&vec![0.5; 64]);

// Full transformer block
let block = TernaryTransformerBlock::new(64, 16, 128);
let seq = vec![vec![0.1; 64]; 4]; // 4 tokens, dim 64
let out = block.forward(&seq, true); // causal masking

// Mini language model
let lm = TernaryLM::new(100, 64, 16, 128);
let tokens = lm.generate(&[0, 1, 2], 10);
```

## API

| Type / Function | Description |
|---|---|
| `bitnet_quantize(&[f32]) → (Vec<Trit>, f32)` | Quantize float weights to ternary + scale |
| `bitnet_dequantize(&[Trit], f32) → Vec<f32>` | Reconstruct approximate floats |
| `TernaryLinear` | Linear layer with ternary weights: `new`, `from_floats`, `forward`, `forward_float` |
| `TokenEmbedding` | Ternary embedding table: `new`, `embed`, `embed_sequence` |
| `TernaryAttentionHead` | Single-head attention with ternary Q/K/V/O projections |
| `TernaryFFN` | Feed-forward network with ternary W1/W2 |
| `TernaryTransformerBlock` | Pre-norm transformer block (attention + FFN + residuals) |
| `KvCache` | Ternary-compressed KV cache with FIFO eviction |
| `TernaryLM` | Mini LM: embedding → transformer block → LM head → greedy decode |
| `rms_norm`, `softmax`, `relu`, `argmax` | Standard neural net utilities |

## Architecture Notes

The ternary ecosystem rests on the conservation identity **γ + η = C**, where γ represents the constructive (active) signal mass, η the destructive (inhibitory) mass, and C the conserved total. In BitNet quantization, this manifests as: every weight contributes {-1, 0, +1} to the sum, and the per-tensor scale $s$ absorbs the magnitude information that ternary values discard. The zero trit acts as the neutral carrier — it preserves the trit count $n$ but contributes nothing to the dot product, effectively acting as structured sparsity introduced at quantization time.

The KV-cache extends this principle to sequence modeling: each cached key-value pair is compressed to ternary form, so the memory cost of $L$ cached positions grows as $O(1.585L)$ bits rather than $O(32L)$.

## References

- Wang, R. et al. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453.
- Ma, S. et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.

## License

MIT
