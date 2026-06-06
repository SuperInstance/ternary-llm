# PLUG_AND_PLAY — Llm

> Ternary LLM building blocks: BitNet 1.58-bit quantization

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-llm = { git = "https://github.com/SuperInstance/ternary-llm" }
```

Use in your code:

```rust
use ternary_llm::{bitnet_quantize, TernaryLM};

let (trits, scale) = bitnet_quantize(&[0.5, -0.3, 0.0, 1.2]);
let mut lm = TernaryLM::new(1000, 128);
let output = lm.generate("Hello");
```

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
