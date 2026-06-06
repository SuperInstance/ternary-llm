# Getting Started — ternary-llm

> *Estimated time to complete: 5 minutes*

## Prerequisites

- **Rust 1.75+** (MSRV)
- Cargo (included with Rust)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary_llm = "0.1.0"
```

Or build from source:

```bash
git clone https://github.com/SuperInstance/ternary-llm.git
cd ternary-llm
cargo build --release
cargo test
```

## Core Concepts

This crate is part of the SuperInstance ternary ecosystem. It provides:

- Ternary {-1, 0, +1} semantics for llm
- Integration with the ternary type system via ternary-core traits
- Zero-Safe design: 0 is not nothing, it is a meaningful "neutral" state

## Quick Start

```rust
use {ternary_llm};
// TernaryLinear is the primary type
let instance = {TernaryLinear::new();
// ... use it
```

## Ternary Example

```rust
use ternary_llm::*;

// The (-1, 0, 1) values work with ternary-core arithmetic
let a = 1i8;  // Positive
let b = -1i8; // Negative
// Z₃ addition: 1 + (-1) = 0 (cancellation)

## Running Tests

```bash
cargo test
```

## Next Steps

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Internal design and data flow
- [PLUG_AND_PLAY.md](./PLUG_AND_PLAY.md) — Integration and configuration
- [CONTRIBUTING.md](./CONTRIBUTING.md) — How to contribute

## Ecosystem

This crate is part of the **[SuperInstance Fleet](https://github.com/SuperInstance)**.
- [ternary-core](https://github.com/SuperInstance/ternary-core) — shared traits and Z₃ arithmetic
- [ternary-types](https://github.com/SuperInstance/ternary-types) — type-level ternary encodings
- [ternary-compiler](https://github.com/SuperInstance/ternary-compiler) — expression compiler
