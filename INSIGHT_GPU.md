# GPU Integration Insights

> Analysis of how this ternary crate maps to GPU execution in the Flux→PTX runtime.
> From the Ternary Math Scout (Kimi Code) and Hermes 405B architectural analysis.

---

# Ternary Mathematics & Physics: GPU Mapping Analysis

> Scout report on the SuperInstance ternary {-1, 0, +1} ecosystem.
> 16 repositories studied. ~7,300 lines of Rust. 300+ tests.

---

## Table of Contents

1. [Foundations: `ternary-types` & `ternary-core`](#1-foundations-ternary-types--ternary-core)
2. [`ternary-tensor`: Multi-Dimensional Ternary Arrays](#2-ternary-tensor-multi-dimensional-ternary-arrays)
3. [`ternary-room`: Recursive Room-Tensor Architecture](#3-ternary-room-recursive-room-tensor-architecture)
4. [`ternary-warp`: Clamp, Quantize, Fold, Warp](#4-ternary-warp-clamp-quantize-fold-warp)
5. [`ternary-algebra` & `ternary-logic`: Z₃ and Kleene](#5-ternary-algebra--ternary-logic-z₃-and-kleene)
6. [`ternary-tnn`: Ternary Neural Network Layers](#6-ternary-tnn-ternary-neural-network-layers)
7. [`ternary-llm`: BitNet b1.58 Ternary LLM](#7-ternary-llm-bitnet-b158-ternary-llm)
8. [`ternary-attention`: Ternary Attention Mechanism](#8-ternary-attention-ternary-attention-mechanism)
9. [`ternary-grad`: STE and Ternary Training](#9-ternary-grad-ste-and-ternary-training)
10. [`ternary-electromagnetism` / `hamiltonian` / `noether`: Physics on Ternary](#10-ternary-electromagnetism--hamiltonian--noether-physics-on-ternary)
11. [`ternary-spiral`: RPS Spiral Wave Dynamics](#11-ternary-spiral-rps-spiral-wave-dynamics)
12. [`ternary-diehard`: Fault Tolerance & Hardening](#12-ternary-diehard-fault-tolerance--hardening)
13. [`ternary-genetic`: Ternary Chromosomes](#13-ternary-genetic-ternary-chromosomes)
14. [GPU Mapping: SIMD, Warp Shuffle, Tensor Cores](#14-gpu-mapping-simd-warp-shuffle-tensor-cores)
15. [Conservation Laws & Noether for GPU Computation](#15-conservation-laws--noether-for-gpu-computation)
16. [Ternary GPU Kernels: Native {-1,0,+1} Compute](#16-ternary-gpu-kernels-native--10-1-compute)

---

## 1. Foundations: `ternary-types` & `ternary-core`

### The `Ternary` Enum

`ternary-types` defines the fundamental type:

```rust
pub enum Ternary {
    Negative,  // -1
    Neutral,   // 0
    Positive,  // +1
}
```

Conversions: `TryFrom<i8>` (fails on values outside {-1,0,1}), `From<Ternary> for i8`,
`Neg` (flips sign, Neutral stays Neutral), and an `ExactSizeIterator` over all three variants.

### Z₃ Arithmetic in `ternary-core`

`ternary-core` implements the cyclic group Z₃ (integers modulo 3, mapped to {-1,0,1}):

| Op | Signature | Z₃ semantics |
|----|-----------|-------------|
| `tadd(a,b)` | `i8 × i8 → i8` | `(a+b) mod 3`, remapped to {-1,0,1} |
| `tsub(a,b)` | `i8 × i8 → i8` | `tadd(a, -b)` |
| `tmul(a,b)` | `i8 × i8 → i8` | `(a*b) mod 3`, remapped |
| `tneg(a)` | `i8 → i8` | additive inverse |
| `tinv(a)` | `i8 → Option<i8>` | multiplicative inverse (1→1, -1→-1, 0→None) |
| `tclamp(v)` | `i8 → i8` | hard clamp to [-1, 1] |
| `tdist(a,b)` | `i8 × i8 → i8` | shortest path on Z₃ circle |
| `tdot(a,b)` | `&[i8] × &[i8] → i8` | inner product modulo 3 |

Key identity: `1 + 1 = -1` in Z₃ (since 2 ≡ -1 mod 3). This is the rock-paper-scissors cyclic structure.

### TernaryGrid & TernaryGraph

`TernaryGrid` is a 2D dense array of `i8` with:
- `get`/`set` with bounds checking and clamping
- `sum`, `average`, `histogram` (counts of -1/0/+1)
- `map`, `zip_with` for element-wise operations
- `laplacian_at` using von Neumann neighbors with Z₃ addition
- `neighbor_count` for Moore or von Neumann neighborhoods

`TernaryGraph` is an adjacency-matrix graph with ternary edge weights:
- Positive edges are traversable (`bfs`, `is_connected`, `components`)
- Negative edges are excluded from connectivity
- `total_weight` sums all edge weights

### Core Traits

```rust
pub trait TernaryValue: Copy + PartialEq + Eq {
    fn to_ternary(self) -> i8;
    fn from_ternary(v: i8) -> Self;
}

pub trait TernaryDynamics {
    type State;
    fn step(&mut self);
    fn run(&mut self, steps: usize);
}

pub trait TernaryMeasure<T> {
    fn measure(&self, input: &T) -> i8;
}
```

These traits form the behavioral contract for the entire ecosystem.

---

## 2. `ternary-tensor`: Multi-Dimensional Ternary Arrays

### The `Trit` Type

```rust
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}
```

`Trit::multiply` implements ternary multiplication truth table:
- `Zero × anything = Zero` (absorbing element)
- `Pos × x = x` (identity)
- `Neg × Neg = Pos`, `Neg × Pos = Neg`

`Trit::add` clamps integer sum to {-1,0,1}.

### Dense Tensor: `TernaryTensor`

- Row-major N-dimensional storage
- `zeros`, `filled`, `from_vec`, `matrix` constructors
- `map`, `elementwise`, `add`, `multiply`, `negate`
- **Broadcasting**: numpy-like rules, dimensions of size 1 stretched
- `sum` → `i32` (unclamped accumulation)
- `count_nonzero` → sparsity metric

### Matrix Multiplication

```rust
pub fn matmul(a: &TernaryTensor, b: &TernaryTensor) -> TernaryTensor
```

Standard O(m×k×n) triple loop, but the accumulate step is:
```
sum += a_ik * b_kj   // i32 accumulation
result = clamp(sum, -1, 1)   // project back to ternary
```

This is the **key ternary tensor op**: multiplication of two ternary values produces {-1,0,1}, but summing K of them can range [-K, K], so a clamp projects back.

### Sparse Tensor: `SparseTernaryTensor`

- HashMap storage: `HashMap<TensorIndex, Trit>`
- Only non-Zero values stored
- `from_dense` / `to_dense` conversion
- `nnz`, `density` metrics
- Element-wise `add` with zero-cleanup

### CP Decomposition

A simplified sign-based CP decomposition:
- For each rank-r component, factor vectors derived from sign patterns along each mode
- Weights are the dominant sign (Pos/Neg/Zero) of the tensor

### Contraction

```rust
pub fn contract(tensor: &TernaryTensor, axes: &[usize]) -> TernaryTensor
```

Sum-reduce along specified axes, clamp results to {-1,0,1}.

---

## 3. `ternary-room`: Recursive Room-Tensor Architecture

### Concept

"Every program is a room. Every room is a cell in the tensor. Rooms contain rooms. Tiles are projections. Connections are alive."

The recursion IS the architecture:
```
Dance floor → DJ board → instrument panel → signal path → code → metal → bits
Same shape at every scale.
```

### Core Types

| Type | Role |
|------|------|
| `Connection` | Living link between two rooms with time_weight, distance, familiarity, attraction, rhythm_sync, strength, trend |
| `Tile` | A room's projection into another room's perspective (brightness, warmth, pulse_phase, color) |
| `Room` | A perspective containing tiles, connections, children, and a ternary `state: i8` |
| `TensorLayer` | All rooms at one depth, arranged in a grid |
| `RecursiveTensor` | Full stack of layers from Floor to Metal |
| `CrossLayerLink` | Connections between rooms at different depths |

### Room Depth Hierarchy

```rust
pub enum RoomDepth {
    Floor,   // Dancers
    Board,   // DJ control board
    Panel,   // Instrument panel
    Path,    // Signal path
    Code,    // Code
    Metal,   // Transistors/registers
}
```

Each room has `state: i8 ∈ {-1, 0, +1}` — the ternary value propagates through the recursive structure.

### Connection Dynamics

`Connection::compute_strength()`:
```
time_factor = 1 - exp(-time_weight * 0.1)   // saturating growth
dist_factor = 1 - distance
base = time_factor * dist_factor * familiarity * attraction
sync_bonus = rhythm_sync * 0.3
strength = clamp(base + sync_bonus, 0, 1)
```

Connections have flavors: `Electric`, `Deep`, `Resonant`, `Fading`, `Steady`.

### GPU Mapping Insight

The recursive tensor maps naturally to GPU hierarchy:
- **Floor** → thread
- **Board** → warp
- **Panel** → block
- **Path** → SM (streaming multiprocessor)
- **Code** → kernel
- **Metal** → hardware register/file

Cross-layer links are like warp shuffle, shared memory, and global memory transfers.

---

## 4. `ternary-warp`: Clamp, Quantize, Fold, Warp

### Operations

```rust
pub fn clamp(values: &[i8], min: i8, max: i8) -> Vec<i8>
pub fn quantize(values: &[f64], thresholds: (f64, f64)) -> Vec<i8>
pub fn fold(values: &[i8], f: fn(i8, i8) -> i8) -> Vec<i8>
pub fn warp(values: &[i8], map: fn(i8) -> i8) -> Vec<i8>
pub fn smooth(values: &[i8], radius: usize) -> Vec<i8>
pub fn differentiate(values: &[i8]) -> Vec<i8>
```

### Quantization

```rust
if v < threshold_low { -1 }
else if v > threshold_high { 1 }
else { 0 }
```

This is the **entry point from float to ternary**: every ternary neural network starts here.

### Smooth (Majority Filter)

For each position, count {-1, 0, +1} in a window and output the majority.
This is a **ternary median filter** — highly parallel, perfect for SIMD.

### Differentiate

`diff[i] = clamp(values[i] - values[i-1], -1, 1)`
Discrete derivative on the ternary ring.

### GPU Mapping

All six operations are **embarrassingly parallel** and **branch-free friendly**:
- `clamp` → one `vmin/vmax` SIMD instruction per element
- `quantize` → two compares + blend (AVX-512 `vcmplt` + `vpblendvb`)
- `smooth` → population count in window (can use GPU ballot/shuffle)
- `warp` → lookup table (3-entry LUT fits in a single 32-bit constant)
- `fold` → reduction pattern, map to warp shuffle reduce
- `differentiate` → vector subtract + clamp

---

## 5. `ternary-algebra` & `ternary-logic`: Z₃ and Kleene

> Note: `ternary-algebra` repo did not exist in the fleet. Algebraic structures are embedded in `ternary-core`. Logic is in `ternary-logic`.

### Z₃ Algebraic Structure (from `ternary-core`)

The set {-1, 0, 1} with `tadd` and `tmul` forms a **commutative ring with identity**:
- `(Z₃, tadd)` is an abelian group with identity 0
- `(Z₃ \ {0}, tmul)` is a cyclic group of order 2: {1, -1}
- `tmul` distributes over `tadd`
- 1 is the multiplicative identity

This is isomorphic to the standard Z₃ = {0, 1, 2} via mapping:
```
ternary: -1  →  standard: 2
ternary:  0  →  standard: 0
ternary:  1  →  standard: 1
```

### Ternary Logic Systems (`ternary-logic`)

Four three-valued logics implemented:

| System | Unknown handling | Key property |
|--------|-----------------|--------------|
| **Kleene (K3)** | Unknown propagates through AND/OR | Strong — most widely used |
| **Łukasiewicz (L3)** | Unknown → True in implication | Weak — material implication |
| **Bochvar (B3)** | Unknown is "poison" — any op with Unknown → Unknown | Internal / null-logic |
| **Gödel-Dummett (G3)** | Conjunction = min, ordered chain | Fuzzy-like |

### Kleene Truth Tables

**AND (Conjunction):**
```
      F  U  T
    ┌─────────┐
  F │ F  F  F │
  U │ F  U  U │
  T │ F  U  T │
    └─────────┘
```

**OR (Disjunction):**
```
      F  U  T
    ┌─────────┐
  F │ F  U  T │
  U │ U  U  T │
  T │ T  T  T │
    └─────────┘
```

### Modal Operators

```rust
pub fn necessity(val: Ternary, system: LogicSystem) -> Ternary
pub fn possibility(val: Ternary, system: LogicSystem) -> Ternary
```

Kleene: □T = T, □F = F, □U = F (necessity requires certainty)
Kleene: ◇T = T, ◇F = F, ◇U = T (possibility allows uncertainty)

### GPU Mapping

Ternary logic ops are **perfect LUT candidates**:
- Each binary op is a 3×3 = 9-entry table
- Each unary op is a 3-entry table
- A single 32-bit integer can encode the entire truth table
- GPU: load LUT constant, index by `(a+1)*3 + (b+1)`, read result
- No branching, one integer multiply-add + load

---

## 6. `ternary-tnn`: Ternary Neural Network Layers

### Weight Quantization (BitNet 1.58-bit)

```rust
scale = mean(abs(weights))          // per-row L1 mean
wn = weight / scale               // normalize
if wn > 0.5 { 1 }                 // threshold at 0.5
else if wn < -0.5 { -1 }
else { 0 }
```

Storage: 2 bits per weight + one f32 scale per output neuron.
For a 7B parameter model: ~1.4 GB vs 28 GB at float32.

### LUT MatMul

The core operation — zero floating-point multiplications:

```rust
match w {
    1 => accumulate +x,    // pass through
   -1 => accumulate -x,    // negate
    0 => skip,             // no-op
}
```

Operation count for M×N×K matmul:
- float32: 2·M·N·K ops (1 mul + 1 add per element)
- ternary: M·N·K ops (1 add-or-sub per element, ~50% skipped due to zeros)

### Layer Types

| Layer | Forward pass |
|-------|-------------|
| `TernaryLinear` | LUT matmul + per-row scale multiply |
| `TernaryConv1d` | Ternary kernel, add/sub/skip per tap |
| `TernaryConv2d` | 2D ternary kernel, same principle |

### Straight-Through Estimator (STE)

The sign function has derivative 0 almost everywhere. Bengio's STE:

```rust
pub fn ste_gradient(grad_output: f32, weight: f32) -> f32 {
    if weight.abs() <= 1.0 { grad_output } else { 0.0 }
}
```

During forward: use quantized weights.
During backward: pretend quantization was identity, pass gradient through if |weight| ≤ 1.
This creates a "funnel" pushing weights toward {-1, 0, +1}.

---

## 7. `ternary-llm`: BitNet b1.58 Ternary LLM

### Full Transformer Stack

```
Input tokens
    │
    ▼
TokenEmbedding ─── ternary lookup table
    │
    ▼
TernaryTransformerBlock × N
    ├── rms_norm
    ├── TernaryAttentionHead (Q,K,V ∈ {-1,0,+1})
    ├── residual
    ├── rms_norm
    ├── TernaryFFN (up-project → ReLU → down-project)
    └── residual
    │
    ▼
argmax decoding
```

### BitNet Quantization

```rust
pub fn bitnet_quantize(weights: &[f32]) -> (Vec<Trit>, f32) {
    scale = mean(abs(weights));
    trits = weights.map(|w| clamp(round(w/scale), -1, 1));
    (trits, scale)
}
```

Reconstruction: `weight ≈ scale * trit`

### TernaryLinear Forward

```rust
acc: i32 = Σ (w as i32) * (v.round() as i32)   // integer accumulate
output = acc as f32 * scale + bias
```

The inner loop is **integer-only**. The single f32 multiply happens once per output neuron.

### KV-Cache with Ternary Compression

```rust
pub struct KvCacheEntry {
    key_trits: Vec<Trit>,      // 2 bits per element
    key_scale: f32,            // 1 f32 per vector
    value_trits: Vec<Trit>,
    value_scale: f32,
}
```

At inference time, KV-cache bandwidth dominates. Ternary compression reduces cache bandwidth by ~16×.

### Complete Model: `TernaryLM`

```rust
pub struct TernaryLM {
    embedding: TokenEmbedding,      // vocab × dim ternary table
    block: TernaryTransformerBlock, // attention + FFN
    lm_head: TernaryLinear,         // dim × vocab projection
}
```

Generates autoregressively via greedy argmax.

---

## 8. `ternary-attention`: Ternary Attention Mechanism

### Ternary-to-Dense Embedding

```rust
pub fn ternary_to_dense(sequence: &[Ternary], dim: usize) -> Matrix {
    sequence.map(|t| {
        let base = t.to_f64();  // -1, 0, or 1
        (0..dim).map(|i| base * (i+1) / dim).collect()
    })
}
```

Ternary values are expanded into dense vectors for attention computation. The ternary state determines the sign and magnitude pattern.

### Scaled Dot-Product Attention

Standard attention, but Q/K/V projections come from ternary `TernaryLinear` layers:

```
scores[i][j] = (Q[i] · K[j]) / sqrt(dim)
weights[i] = softmax(scores[i])
output[i] = Σ_j weights[i][j] * V[j]
```

The Q/K/V matrices are ternary; the attention scores and weights are float (required for softmax).

### Multi-Head Attention

```rust
pub struct MultiHeadAttention {
    n_heads: usize,
    dim: usize,
    head_dim: usize,  // dim / n_heads
}
```

Splits dense representation into head_dim chunks, runs attention independently per head, concatenates.

### Ternary Compatibility Score

A direct ternary-ternary similarity (no dense expansion):

```rust
pub fn ternary_compatibility(query: &[Ternary], key: &[Ternary]) -> f64 {
    score = Σ (q.to_f64() * k.to_f64()) / len    // +1 match, -1 opposite, 0 neutral
}
```

This could replace dot-product in some attention variants — purely ternary, no float intermediates.

### GPU Mapping

- **Ternary Q/K/V**: LUT matmul (add/sub/skip) → very fast
- **Q·Kᵀ**: after dequantization, standard GEMM on tensor cores
- **Softmax**: still requires float; could use ternary approximations
- **AttentionPattern heatmap**: can be rendered with 4-level ASCII (██ ▓▓ ▒▓ ░░ ··)

---

## 9. `ternary-grad`: STE and Ternary Training

### Straight-Through Estimator

```rust
pub fn straight_through(x: f64, threshold: f64) -> i8 {
    if x > threshold { 1 }
    else if x < -threshold { -1 }
    else { 0 }
}
```

The threshold controls sparsity: higher threshold → more zeros.

### STE Gradient

```rust
pub fn ste_gradient(x: f64) -> f64 {
    if x.abs() <= 1.0 { 1.0 } else { 0.0 }
}
```

This is the **clipped identity STE**. Alternatives:
- Full identity: `g_ste = 1` everywhere (risky divergence)
- Tanh STE: `g_ste = 1 - tanh²(x)` (smooth)

### Ternary Adam

Standard Adam with STE-modified gradients:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          // first moment
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         // second moment
m̂_t = m_t / (1-β₁^t)                   // bias correction
v̂_t = v_t / (1-β₂^t)
w_t = w_{t-1} - α · m̂_t / (√v̂_t + ε)
```

The latent weights `w` are float; only the forward pass uses ternary quantization.

### Training Diagnostics

- `quantization_error(weights, threshold)`: L₂ distance to nearest ternary point
- `ternary_accuracy(weights, threshold)`: fraction of weights within threshold of {-1,0,+1}
- `cosine_lr`, `step_lr`: learning rate schedules
- `clip_ternary_gradient`: gradient clipping
- `ternary_weight_decay`: L₂ regularization

---

## 10. `ternary-electromagnetism` / `hamiltonian` / `noether`: Physics on Ternary

### `ternary-electromagnetism`

EM fields discretized onto a ternary lattice where charges/currents ∈ {-1, 0, +1}.

**ElectricField**: Coulomb's law with ternary charges
```
F = k * q1 * q2 / r²    // q1, q2 ∈ {-1, 0, +1}
```

**MagneticField**: Biot-Savart with ternary currents
```
B = μ * I / (2πr)       // I ∈ {-1, 0, +1}
```

**YeeLattice**: Discrete Maxwell's equations
- Staggered E and B fields
- Leapfrog (Störmer-Verlet) integration
- Exact discrete charge conservation
- CFL stability: dt ≤ dx / (c·√2)

**WavePropagation**: Pulse injection + energy tracking
```
E_total = ½ Σ (Ex² + Ey² + Bz²)
```

**Polarization**: Ternary polarization states
- Horizontal (-1) → Jones vector (1, 0)
- None (0) → Jones vector (0, 0)
- Vertical (+1) → Jones vector (0, 1)

**Interference**: Double-slit + ternary phase interference
```
I(x) = cos²(π·d·x/λ)
ternary_phase_interference(a, b) = clamp(a + b, -1, 1)
```

### `ternary-hamiltonian`

Hamiltonian mechanics on ternary phase space (q, p) ∈ {-1,0,+1}²ⁿ.

**Hamiltonian**: H = T + V (kinetic + potential)

**Symplectic Integrators**:
1. **Symplectic Euler**: preserves structure but drifts in energy
2. **Störmer-Verlet (leapfrog)**: time-reversible, second-order accurate

After each continuous update, values are **rounded and clamped** back to {-1,0,+1}.

**Poisson Bracket**: discrete approximation via central finite differences
```
{f, g} ≈ Σ_i (∂f/∂q_i · ∂g/∂p_i − ∂f/∂p_i · ∂g/∂q_i)
∂f/∂q_i ≈ (f(q=+1) − f(q=−1)) / 2
```

**Liouville's Theorem**: verified by counting distinct occupied phase-space cells.

### `ternary-noether`

Discrete analogue of Noether's theorem for ternary symmetries.

| Symmetry | Generator | Conserved Quantity |
|----------|-----------|-------------------|
| Time translation | `t → t + δ` | Energy E = Σ(p²/2 + x²/2) |
| Space translation | `x → clamp(x + δ)` | Momentum P = Σ p_i |
| Rotation | 90° × k | Angular momentum L = Σ(x·p_y − y·p_x) |
| Reflection(X) | `(x,y) → (-x,y)` | Parity / Momentum |
| Reflection(Y) | `(x,y) → (x,-y)` | Parity / Momentum |

All transformations clamp back to {-1,0,+1}, ensuring the group action closes on the discrete state space.

### GPU Mapping for Physics

The Yee lattice update is a **stencil operation** — perfect for GPU:
```
for each cell (i,j) in parallel:
    Ex[i][j] += dt * (Bz[i][j] - Bz[i-1][j])
    Ey[i][j] -= dt * (Bz[i][j] - Bz[i][j-1])
```

- Each cell updates independently → massive parallelism
- Only nearest-neighbor reads → fits in shared memory
- Ternary charges mean no float storage for sources → read from 2-bit packed arrays

---

## 11. `ternary-spiral`: RPS Spiral Wave Dynamics

### Rock-Paper-Scissors Cellular Automaton

Each cell is one of three states mapped to trits:
- `-1` → Rock (beats Scissors)
- `0` → Paper (beats Rock)
- `+1` → Scissors (beats Paper)

### Update Rule (Invasion + Majority)

For each cell, inspect 4 von Neumann neighbors (toroidal wrapping):
1. Find neighbors that **beat** the focal cell
2. If any exist, focal cell converts to the **majority type** among beaters
3. Ties broken by order: Rock > Paper > Scissors

This is the microscopic mechanism that nucleates spiral wave pairs at topological defects.

### Biodiversity Metrics

```
Shannon entropy: H = -Σ p_i ln(p_i)
Simpson index: λ = 1 - Σ p_i²
Evenness: J = H / ln(3)
```

A perfectly mixed 1:1:1 grid maximizes Shannon entropy at ln(3).

### GPU Mapping

This CA is an **ideal GPU kernel**:
- One thread per cell
- Each thread reads 4 neighbors (shared memory tile)
- Branchless implementation via lookup table:
  ```
  // Encode cell state + 4 neighbors → next state
  // 3^5 = 243 possible inputs → fits in a 256-byte LUT
  ```
- No floating point anywhere
- Can update in-place with double-buffering

---

## 12. `ternary-diehard`: Fault Tolerance & Hardening

### Three-State Cellular Automata

Extends Conway's Life with a quiescent "Idle" state:
- `-1` → Dead (apoptotic)
- `0` → Idle (refractory / G₀-arrested)
- `+1` → Alive (mitotically active)

### Rule Families

| Variant | Birth | Survival | Idle transition |
|---------|-------|----------|-----------------|
| ThreeStateLife | 3 | 2 or 3 | Alive→Idle on exactly 2 neighbors |
| HighLifeTernary | 3 or 6 | 2 or 3 | Alive→Idle otherwise; Idle→Dead |
| DayAndNightTernary | {3,6,7,8} | {3,4,6,7,8} | complex active-neighbor logic |

### Fault Tolerance Properties

The Idle state acts as:
1. **One-generation memory** — recent activity is remembered
2. **Graceful degradation** — cells don't die immediately
3. **Noise buffer** — transient perturbations are absorbed by Idle transition
4. **Self-healing** — 2×2 blocks are stable (still life), forming resilient cores

### Analysis Utilities

- `detect_oscillation(history)` → finds periods 2-10 in population time-series
- `PopulationStats` → min, max, mean, variance; `is_stable()` flags near-constant populations
- `find_still_life(grid)` → tests fixed-point stability

### GPU Mapping

Same stencil pattern as ternary-spiral:
- Moore neighborhood (8 neighbors)
- One thread per cell
- 3^9 = 19,683 possible neighborhood configurations → fits in a 20KB LUT
- Or: count Alive neighbors + count Active neighbors → branch on two integers
- Double-buffered grid update

---

## 13. `ternary-genetic`: Ternary Chromosomes

### Ternary Genome

```rust
pub type Trit = i8;  // -1, 0, or +1

pub struct TernaryChromosome {
    genes: Vec<Trit>,
    length: usize,
}
```

Search space: {-1, 0, +1}ᴸ with 3ᴸ vertices.

### Fitness Landscapes

| Landscape | Formula | Character |
|-----------|---------|-----------|
| `MaxSumFitness` | Σ trits | Smooth, unimodal |
| `TargetFitness` | 1 / (1 + Hamming distance) | Needle-in-haystack |
| `OneMaxFitness` | count of +1 trits | Classic OneMax generalization |

### Genetic Operators

- **Tournament selection**: pick k random, return fittest
- **Roulette selection**: fitness-proportional with negative-fitness shifting
- **Single-point crossover**: suffix exchange at random locus
- **Uniform crossover**: each trit independently from either parent (50/50)
- **Trit-flip mutation**: each trit → random {-1,0,+1} with probability `rate`
- **Elitism**: top N individuals cloned to next generation

### Convergence

The mutation operator makes the Markov chain **ergodic**: any state reachable from any other. Elitism guarantees best-so-far is never lost → finite expected absorption time.

### GPU Mapping

- **Population**: array of chromosomes, one block per individual or one thread per gene
- **Fitness evaluation**: embarrassingly parallel across population
- **Selection**: warp-level tournament (shuffle-based)
- **Crossover**: coalesced memory access, one thread per gene position
- **Mutation**: LCG PRNG per thread (state in shared memory)
- **Elitism**: parallel reduction to find top-N

---

## 14. GPU Mapping: SIMD, Warp Shuffle, Tensor Cores

### Ternary Data Packing

Two bits per trit (enough for {-1, 0, +1}):
```
Encoding: -1 = 0b10, 0 = 0b00, +1 = 0b01
(0b11 is unused — can be reserved for future)
```

Packing density:
- 32 trits per 64-bit word
- 128 trits per 256-bit AVX register
- 512 trits per 1024-bit GPU vector

### SIMD Operations

**Ternary addition (clamped)**:
```
// SSE/AVX: no native ternary add, but can simulate:
// a + b in i8, then vmin(vmax(result, -1), 1)
_mm256_min_epi8(_mm256_max_epi8(sum, neg1), pos1)
```

**Ternary multiplication (LUT)**:
```
// Truth table for multiply:
//   × | -1  0 +1
//  ---|-----------
//  -1| +1  0 -1
//   0|  0  0  0
//  +1| -1  0 +1
//
// Can encode as: result = a * b with 0-absorbing property
// Actually: result = (a == 0 || b == 0) ? 0 : (a == b ? 1 : -1)
```

**Branchless multiply using bit tricks**:
```
// For packed 2-bit trits:
// Use lookup in a 256-entry table (4 bits → 2 bits)
// Or: multiply as signed integers, then mask
```

### Warp Shuffle for Ternary

NVIDIA warp shuffle (`shfl.sync`) can exchange ternary values between threads in a warp:
```cuda
__device__ int8_t ternary_shuffle(int8_t val, int src_lane) {
    // Pack 16 trits into a 32-bit int
    // Shuffle the 32-bit word
    // Unpack at destination
}
```

Use cases:
- **Ternary reduction**: sum trits across warp using shuffle + clamp
- **CA neighborhood**: exchange edge cells with adjacent threads
- **Tournament selection**: compare fitness via shuffle

### Tensor Cores

Standard tensor cores do `D = A×B + C` in float16/bfloat16/TF32. For ternary:

**Option A: Emulation via int8 tensor cores**
- Represent {-1, 0, +1} as signed int8
- Use int8 tensor core MMA (matrix multiply accumulate)
- Accumulate in int32, then clamp to {-1,0,+1}
- NVIDIA Ampere+ supports int8 MMA at 2× throughput vs fp16

**Option B: Bit-serial multiplication**
- Decompose ternary matmul into two binary matmuls:
  - Positive mask: `A_pos = (A == 1)`
  - Negative mask: `A_neg = (A == -1)`
  - `A × B = A_pos × B - A_neg × B`
- Each binary matmul uses 1-bit tensor core operations
- NVIDIA's CUTLASS has bit-serial GEMM primitives

**Option C: Sparse tensor cores**
- ~50% of ternary weights are zero
- Skip zero computations
- NVIDIA Hopper has structured sparsity (2:4) support
- Ternary sparsity is unstructured but very high

### Memory Bandwidth

The primary win for ternary on GPU:
```
float32 matmul: 32 bits/weight bandwidth bound
ternary matmul: ~2 bits/weight → 16× less memory traffic
```

For large matrices, memory bandwidth is the bottleneck. Ternary weights turn compute-bound problems into more compute-bound problems by reducing bandwidth pressure.

---

## 15. Conservation Laws & Noether for GPU Computation

### Analogy: GPU Computation as Physical System

| Physics | GPU Computation |
|---------|----------------|
| Hamiltonian H | Total energy / FLOP budget |
| Phase space (q,p) | (program counter, register state) |
| Symplectic integrator | Time-stepped simulation kernel |
| Energy conservation | No numerical drift in conserved quantities |
| Liouville's theorem | Reversible computation / no information loss |

### Discrete Noether for GPU Kernels

A GPU kernel has **discrete symmetries** that imply conservation laws:

**1. Time-translation symmetry → Energy conservation**
- If a kernel's behavior is independent of absolute time step index
- Then the "computational energy" (sum of squared ternary values) is conserved
- In ternary Yee lattice: `Σ(Ex² + Ey² + Bz²)` is constant (vacuum case)

**2. Space-translation symmetry → Momentum conservation**
- If a stencil kernel is invariant under grid translation
- Then total "momentum" (sum of cell states) is conserved
- In ternary CA: `Σ cells` is conserved under certain rules

**3. Rotation symmetry → Angular momentum conservation**
- If a 2D kernel is invariant under 90° rotation
- Then discrete angular momentum is conserved
- Verified in `ternary-noether` by applying `DiscreteSymmetry::Rotation` and checking `Verification::verify_angular_momentum()`

### Why This Matters for GPUs

**Conservation laws are invariants that detect bugs:**
- If energy should be conserved but drifts → numerical instability
- If momentum should be conserved but changes → boundary condition error
- If phase-space volume collapses → information loss / irreversibility

**Ternary makes conservation exact:**
- In float, conservation is approximate (roundoff error accumulates)
- In ternary with discrete updates, conservation can be **exactly verified**
- `LiouvilleTheorem::check_conservation()` compares `HashSet` sizes — exact, no epsilon

**Symplectic integrators on GPU:**
- Störmer-Verlet (leapfrog) is time-reversible
- Ternary clamping after each half-step preserves discrete phase-space topology
- Energy drift is bounded (second-order accuracy)

---

## 16. Ternary GPU Kernels: Native {-1,0,+1} Compute

### Kernel 1: Ternary MatMul (LUT-based)

```cuda
// Each thread computes one output element
// Weights are packed: 16 trits per 32-bit word
__global__ void ternary_matmul_lut(
    const uint32_t* weights_packed,  // [M, K/16] packed ternary weights
    const float*    input,           // [K] float input
    float*          output,          // [M] float output
    const float*    scales,          // [M] per-row scales
    int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += 16) {
        uint32_t wpack = weights_packed[row * (K/16) + k/16];
        #pragma unroll
        for (int b = 0; b < 16; b++) {
            int w = (wpack >> (b * 2)) & 0x3;  // extract 2-bit trit
            float x = input[k + b];
            // Decode: 0b01 = +1, 0b10 = -1, 0b00 = 0
            acc += (w == 1) ? x : (w == 2) ? -x : 0.0f;
        }
    }
    output[row] = acc * scales[row];
}
```

**Key properties:**
- No floating-point multiplies in inner loop
- Weight bandwidth: 2 bits per element
- Input bandwidth: 32 bits per element (unchanged)
- One f32 multiply per output row (scale)

### Kernel 2: Ternary 2D Convolution

```cuda
__global__ void ternary_conv2d(
    const uint32_t* kernel_packed,   // [OC, IC, KH*KW/16]
    const float*    input,           // [IC, H, W]
    float*          output,          // [OC, OH, OW]
    int IC, int KH, int KW, int H, int W, int OH, int OW
) {
    int oc = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh + kh - KH/2;
                int iw = ow + kw - KW/2;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float x = input[ic*H*W + ih*W + iw];
                    int w = unpack_trit(kernel_packed, oc, ic, kh, kw);
                    sum += (w == 1) ? x : (w == -1) ? -x : 0.0f;
                }
            }
        }
    }
    output[oc*OH*OW + oh*OW + ow] = sum;
}
```

### Kernel 3: Ternary CA Update (RPS Spiral)

```cuda
__global__ void rps_step(
    const uint8_t* grid_in,   // 2 bits per cell, packed
    uint8_t*       grid_out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Read 4 neighbors with toroidal wrapping
    int left  = read_trit(grid_in, (x-1+width)%width, y, width);
    int right = read_trit(grid_in, (x+1)%width, y, width);
    int up    = read_trit(grid_in, x, (y-1+height)%height, width);
    int down  = read_trit(grid_in, x, (y+1)%height, width);
    int self  = read_trit(grid_in, x, y, width);
    
    // Encode neighborhood as 10-bit index (5 trits × 2 bits)
    int idx = (self << 8) | (left << 6) | (right << 4) | (up << 2) | down;
    
    // 3^5 = 243 entries → LUT in shared memory or constant cache
    int next = rps_lut[idx];
    
    write_trit(grid_out, x, y, width, next);
}
```

**Properties:**
- One thread per cell
- Shared memory tile for halo exchange
- 243-entry LUT (can fit in 256 bytes)
- Zero floating point
- Memory bandwidth: 2 bits read + 2 bits write per cell

### Kernel 4: Ternary STE Backward Pass

```cuda
__global__ void ste_backward(
    const float* grad_output,   // dL/dq
    const float* latent_weights, // w (float, before quantization)
    float*       grad_input,     // dL/dw
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float w = latent_weights[i];
        // Clipped identity STE: pass gradient if |w| <= 1
        grad_input[i] = (fabsf(w) <= 1.0f) ? grad_output[i] : 0.0f;
    }
}
```

### Kernel 5: Ternary Quantization (BitNet)

```cuda
__global__ void bitnet_quantize_row(
    const float* weights,    // [in_features]
    uint32_t*    trits_packed, // [in_features/16] output
    float*       scale,        // [1] output
    int in_features
) {
    // Block-level reduction for mean(abs(weights))
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        local_sum += fabsf(weights[i]);
    }
    // warp shuffle reduction → block reduction
    float mean_abs = block_reduce_sum(local_sum) / in_features;
    float s = (mean_abs < 1e-8f) ? 1.0f : mean_abs;
    
    if (threadIdx.x == 0) *scale = s;
    __syncthreads();
    
    // Quantize and pack
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        float wn = weights[i] / s;
        int t = (wn > 0.5f) ? 1 : (wn < -0.5f) ? 2 : 0;  // 1=+1, 2=-1, 0=0
        // Pack into shared buffer, then write coalesced
        // ...
    }
}
```

### Performance Model

For a large M×K matrix-vector product:

| Metric | float32 | ternary (LUT) | Speedup |
|--------|---------|---------------|---------|
| Weight memory | 32MK bits | 2MK bits | 16× |
| Weight bandwidth | 32MK | 2MK | 16× |
| Arithmetic ops | 2MK (mul+add) | MK (add/sub) | 2× fewer |
| FLOP intensity | 1/16 (BW bound) | 1/2 (compute bound) | Better utilization |
| Energy per op | ~1.0 pJ (fmul) | ~0.1 pJ (add) | 10× |

The ternary kernel is **bandwidth-bound at much higher throughput** because the weight matrix is 16× smaller.

---

## Summary: The Ternary-GPU Stack

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  ├── Ternary LLM (ternary-llm)                              │
│  ├── Ternary CA Physics (ternary-spiral, diehard)           │
│  └── Ternary Genetics (ternary-genetic)                     │
├─────────────────────────────────────────────────────────────┤
│  NEURAL NETWORK LAYER                                       │
│  ├── TernaryAttention (Q,K,V projections)                   │
│  ├── TernaryLinear (LUT matmul)                             │
│  ├── TernaryConv (1D/2D ternary kernels)                    │
│  └── STE Gradient (clipped identity backward)               │
├─────────────────────────────────────────────────────────────┤
│  TENSOR LAYER                                               │
│  ├── TernaryTensor (dense N-dim)                            │
│  ├── SparseTernaryTensor (HashMap non-zero)                 │
│  ├── matmul, contract, broadcast                            │
│  └── CP decomposition                                       │
├─────────────────────────────────────────────────────────────┤
│  CORE LAYER                                                 │
│  ├── Z₃ arithmetic (tadd, tmul, tdot, tdist)                │
│  ├── TernaryGrid (2D stencil ops)                           │
│  ├── TernaryGraph (ternary-weighted edges)                  │
│  └── TernaryLogic (Kleene, Łukasiewicz, Bochvar, Gödel)     │
├─────────────────────────────────────────────────────────────┤
│  TYPE LAYER                                                 │
│  └── Ternary enum {-1, 0, +1}                               │
├─────────────────────────────────────────────────────────────┤
│  GPU KERNEL LAYER                                           │
│  ├── Packed trit loads (2 bits/element)                     │
│  ├── LUT-based matmul (no multiplies)                       │
│  ├── Warp shuffle reductions                                │
│  ├── Shared memory stencil tiles                            │
│  └── Tensor core int8 emulation                             │
└─────────────────────────────────────────────────────────────┘
```

The entire ecosystem is built on a single invariant: **values live in {-1, 0, +1}**. This constraint propagates upward, enabling:
- **16× memory reduction** for neural networks
- **2× operation reduction** (adds instead of multiply-adds)
- **Branch-free logic** via lookup tables
- **Exact conservation laws** via discrete Noether theorem
- **Natural sparsity** via the zero state
- **Cyclic dynamics** via Z₃ group structure

Ternary is not just a quantization scheme — it is a **computational physics** founded on the simplest non-trivial ring.
