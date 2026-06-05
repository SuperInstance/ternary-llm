#![forbid(unsafe_code)]

//! Ternary LLM building blocks.
//!
//! BitNet 1.58-bit quantization: each weight is stored as a trit {-1, 0, +1}
//! alongside a per-tensor float scale. Forward pass multiplies by scale after
//! the integer accumulation, keeping arithmetic cheap.

/// A single ternary weight value.
pub type Trit = i8;

/// Quantize a float slice to trits {-1, 0, +1} using BitNet 1.58-bit scheme.
/// Returns (trits, scale) where scale = mean(|weights|).
pub fn bitnet_quantize(weights: &[f32]) -> (Vec<Trit>, f32) {
    if weights.is_empty() {
        return (vec![], 1.0);
    }
    let scale: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;
    let eps = 1e-8_f32;
    let s = scale.max(eps);
    let trits: Vec<Trit> = weights
        .iter()
        .map(|&w| {
            let v = (w / s).round() as i8;
            v.max(-1).min(1)
        })
        .collect();
    (trits, s)
}

/// Reconstruct approximate float weights from trits and scale.
pub fn bitnet_dequantize(trits: &[Trit], scale: f32) -> Vec<f32> {
    trits.iter().map(|&t| t as f32 * scale).collect()
}

/// Ternary linear layer: out = scale * (W_trit @ x) + bias.
/// W_trit has shape [out_dim, in_dim].
pub struct TernaryLinear {
    pub weights: Vec<Vec<Trit>>,
    pub bias: Vec<f32>,
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl TernaryLinear {
    /// Initialize with quantized random-ish weights (seeded by index for determinism).
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut weights = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            let row: Vec<Trit> = (0..in_dim)
                .map(|j| {
                    let h = (i * 31 + j * 17) % 3;
                    match h {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    }
                })
                .collect();
            weights.push(row);
        }
        TernaryLinear {
            weights,
            bias: vec![0.0; out_dim],
            scale: 1.0,
            in_dim,
            out_dim,
        }
    }

    /// From pre-existing float weights.
    pub fn from_floats(float_weights: &[Vec<f32>], bias: Vec<f32>) -> Self {
        let out_dim = float_weights.len();
        let in_dim = if out_dim > 0 { float_weights[0].len() } else { 0 };
        let flat: Vec<f32> = float_weights.iter().flatten().cloned().collect();
        let (trits_flat, scale) = bitnet_quantize(&flat);
        let weights: Vec<Vec<Trit>> = trits_flat.chunks(in_dim).map(|c| c.to_vec()).collect();
        TernaryLinear { weights, bias, scale, in_dim, out_dim }
    }

    /// Forward: output[i] = scale * sum_j(W[i][j] * x[j]) + bias[i].
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "input dim mismatch");
        (0..self.out_dim)
            .map(|i| {
                let acc: i32 = self.weights[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &v)| w as i32 * (v.round() as i32).max(-127).min(127))
                    .sum();
                acc as f32 * self.scale + self.bias[i]
            })
            .collect()
    }

    /// Forward with float accumulation (no input quantization).
    pub fn forward_float(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "input dim mismatch");
        (0..self.out_dim)
            .map(|i| {
                let acc: f32 = self.weights[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &v)| w as f32 * v)
                    .sum();
                acc * self.scale + self.bias[i]
            })
            .collect()
    }
}

/// Token embedding table mapping token IDs to ternary vectors.
pub struct TokenEmbedding {
    pub table: Vec<Vec<Trit>>,
    pub dim: usize,
    pub vocab_size: usize,
    pub scale: f32,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let table: Vec<Vec<Trit>> = (0..vocab_size)
            .map(|v| {
                (0..dim)
                    .map(|d| {
                        let h = (v * 13 + d * 7) % 3;
                        match h {
                            0 => -1,
                            1 => 0,
                            _ => 1,
                        }
                    })
                    .collect()
            })
            .collect();
        TokenEmbedding { table, dim, vocab_size, scale: 1.0 }
    }

    /// Look up embedding for a token, returning float vector.
    pub fn embed(&self, token_id: usize) -> Vec<f32> {
        assert!(token_id < self.vocab_size, "token id out of range");
        self.table[token_id].iter().map(|&t| t as f32 * self.scale).collect()
    }

    /// Embed a sequence of tokens.
    pub fn embed_sequence(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        tokens.iter().map(|&t| self.embed(t)).collect()
    }
}

/// Simple RMS normalization.
pub fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|&v| v / rms).collect()
}

/// Element-wise addition.
pub fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// ReLU activation.
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Numerically stable softmax.
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Scaled dot-product attention (float).
fn scaled_dot_product(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    scale: f32,
    causal: bool,
) -> Vec<Vec<f32>> {
    let n_q = q.len();
    let n_k = k.len();
    let d_v = v[0].len();

    let mut output = vec![vec![0.0_f32; d_v]; n_q];
    for i in 0..n_q {
        let mut scores: Vec<f32> = (0..n_k)
            .map(|j| {
                if causal && j > i {
                    f32::NEG_INFINITY
                } else {
                    q[i].iter().zip(k[j].iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale
                }
            })
            .collect();

        if causal {
            let valid: Vec<f32> = scores.iter().filter(|&&s| s > f32::NEG_INFINITY / 2.0).cloned().collect();
            let soft_valid = softmax(&valid);
            let mut vi = 0;
            for j in 0..n_k {
                if scores[j] > f32::NEG_INFINITY / 2.0 {
                    scores[j] = soft_valid[vi];
                    vi += 1;
                } else {
                    scores[j] = 0.0;
                }
            }
        } else {
            let soft = softmax(&scores);
            scores = soft;
        }

        for j in 0..n_k {
            for d in 0..d_v {
                output[i][d] += scores[j] * v[j][d];
            }
        }
    }
    output
}

/// Single attention head with ternary projections.
pub struct TernaryAttentionHead {
    pub wq: TernaryLinear,
    pub wk: TernaryLinear,
    pub wv: TernaryLinear,
    pub wo: TernaryLinear,
    pub head_dim: usize,
    pub scale: f32,
}

impl TernaryAttentionHead {
    pub fn new(model_dim: usize, head_dim: usize) -> Self {
        TernaryAttentionHead {
            wq: TernaryLinear::new(model_dim, head_dim),
            wk: TernaryLinear::new(model_dim, head_dim),
            wv: TernaryLinear::new(model_dim, head_dim),
            wo: TernaryLinear::new(head_dim, model_dim),
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: &[Vec<f32>], causal: bool) -> Vec<Vec<f32>> {
        let q: Vec<Vec<f32>> = x.iter().map(|v| self.wq.forward_float(v)).collect();
        let k: Vec<Vec<f32>> = x.iter().map(|v| self.wk.forward_float(v)).collect();
        let v: Vec<Vec<f32>> = x.iter().map(|v| self.wv.forward_float(v)).collect();
        let ctx = scaled_dot_product(&q, &k, &v, self.scale, causal);
        ctx.iter().map(|c| self.wo.forward_float(c)).collect()
    }
}

/// Feed-forward network with ternary weights: FFN(x) = W2 * ReLU(W1 * x).
pub struct TernaryFFN {
    pub w1: TernaryLinear,
    pub w2: TernaryLinear,
}

impl TernaryFFN {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        TernaryFFN {
            w1: TernaryLinear::new(dim, hidden_dim),
            w2: TernaryLinear::new(hidden_dim, dim),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h = relu(&self.w1.forward_float(x));
        self.w2.forward_float(&h)
    }
}

/// Transformer block: pre-norm -> attention -> residual -> pre-norm -> FFN -> residual.
pub struct TernaryTransformerBlock {
    pub attention: TernaryAttentionHead,
    pub ffn: TernaryFFN,
    pub dim: usize,
}

impl TernaryTransformerBlock {
    pub fn new(dim: usize, head_dim: usize, ffn_hidden: usize) -> Self {
        TernaryTransformerBlock {
            attention: TernaryAttentionHead::new(dim, head_dim),
            ffn: TernaryFFN::new(dim, ffn_hidden),
            dim,
        }
    }

    pub fn forward(&self, x: &[Vec<f32>], causal: bool) -> Vec<Vec<f32>> {
        // Pre-norm + attention + residual
        let normed: Vec<Vec<f32>> = x.iter().map(|v| rms_norm(v, 1e-6)).collect();
        let attn_out = self.attention.forward(&normed, causal);
        let after_attn: Vec<Vec<f32>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(xi, ai)| vec_add(xi, ai))
            .collect();

        // Pre-norm + FFN + residual
        after_attn
            .iter()
            .map(|v| {
                let normed = rms_norm(v, 1e-6);
                let ffn_out = self.ffn.forward(&normed);
                vec_add(v, &ffn_out)
            })
            .collect()
    }
}

/// KV-cache entry: keys and values stored as ternary-compressed vectors.
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    pub key_trits: Vec<Trit>,
    pub key_scale: f32,
    pub value_trits: Vec<Trit>,
    pub value_scale: f32,
}

/// KV-cache with ternary compression for memory-efficient inference.
pub struct KvCache {
    pub entries: Vec<KvCacheEntry>,
    pub max_len: usize,
}

impl KvCache {
    pub fn new(max_len: usize) -> Self {
        KvCache { entries: Vec::new(), max_len }
    }

    /// Compress and store a (key, value) pair.
    pub fn push(&mut self, key: Vec<f32>, value: Vec<f32>) {
        if self.entries.len() >= self.max_len {
            self.entries.remove(0);
        }
        let (key_trits, key_scale) = bitnet_quantize(&key);
        let (value_trits, value_scale) = bitnet_quantize(&value);
        self.entries.push(KvCacheEntry { key_trits, key_scale, value_trits, value_scale });
    }

    /// Retrieve a decompressed key at position i.
    pub fn get_key(&self, i: usize) -> Vec<f32> {
        let e = &self.entries[i];
        bitnet_dequantize(&e.key_trits, e.key_scale)
    }

    /// Retrieve a decompressed value at position i.
    pub fn get_value(&self, i: usize) -> Vec<f32> {
        let e = &self.entries[i];
        bitnet_dequantize(&e.value_trits, e.value_scale)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Greedy argmax decoding over a logit vector.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Simple language model for generation testing.
pub struct TernaryLM {
    pub embedding: TokenEmbedding,
    pub block: TernaryTransformerBlock,
    pub lm_head: TernaryLinear,
}

impl TernaryLM {
    pub fn new(vocab_size: usize, dim: usize, head_dim: usize, ffn_hidden: usize) -> Self {
        TernaryLM {
            embedding: TokenEmbedding::new(vocab_size, dim),
            block: TernaryTransformerBlock::new(dim, head_dim, ffn_hidden),
            lm_head: TernaryLinear::new(dim, vocab_size),
        }
    }

    /// Forward pass: returns logits for each position.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let embeds = self.embedding.embed_sequence(tokens);
        let hidden = self.block.forward(&embeds, true);
        hidden.iter().map(|h| self.lm_head.forward_float(h)).collect()
    }

    /// Greedy generation: predict next tokens up to max_new_tokens.
    pub fn generate(&self, prompt: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let logits = self.forward(&tokens);
            let next = argmax(logits.last().unwrap());
            tokens.push(next);
        }
        tokens[prompt.len()..].to_vec()
    }
}

/// Compute compression ratio of ternary vs float32 storage.
pub fn compression_ratio(n_weights: usize) -> f32 {
    let float_bits = n_weights * 32;
    // 1.58 bits per weight + one f32 scale per tensor
    let ternary_bits = (n_weights as f32 * 1.58) as usize + 32;
    float_bits as f32 / ternary_bits as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_quantize_basic() {
        let w = vec![0.5_f32, -0.8, 0.0, 0.3, -0.1];
        let (trits, scale) = bitnet_quantize(&w);
        assert_eq!(trits.len(), w.len());
        for &t in &trits {
            assert!(t == -1 || t == 0 || t == 1, "trit must be in {{-1,0,1}}");
        }
        assert!(scale > 0.0);
    }

    #[test]
    fn test_bitnet_quantize_signs() {
        let w = vec![10.0_f32, -10.0, 0.0001];
        let (trits, _scale) = bitnet_quantize(&w);
        assert_eq!(trits[0], 1);
        assert_eq!(trits[1], -1);
    }

    #[test]
    fn test_bitnet_dequantize_roundtrip() {
        let w = vec![1.0_f32, -1.0, 0.5];
        let (trits, scale) = bitnet_quantize(&w);
        let recovered = bitnet_dequantize(&trits, scale);
        assert_eq!(recovered.len(), w.len());
        for &v in &recovered {
            assert!(v.abs() <= scale + 1e-5);
        }
    }

    #[test]
    fn test_bitnet_empty() {
        let (trits, scale) = bitnet_quantize(&[]);
        assert!(trits.is_empty());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_token_embedding_shape() {
        let emb = TokenEmbedding::new(100, 16);
        let v = emb.embed(0);
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn test_token_embedding_trit_values() {
        let emb = TokenEmbedding::new(50, 8);
        for tid in 0..50_usize {
            for &t in &emb.table[tid] {
                assert!(t == -1 || t == 0 || t == 1);
            }
        }
    }

    #[test]
    fn test_token_embedding_sequence() {
        let emb = TokenEmbedding::new(10, 4);
        let seq = emb.embed_sequence(&[0, 3, 7]);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0].len(), 4);
    }

    #[test]
    fn test_ternary_linear_output_shape() {
        let layer = TernaryLinear::new(8, 4);
        let x = vec![1.0_f32; 8];
        let out = layer.forward_float(&x);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_ternary_linear_from_floats() {
        let floats = vec![
            vec![1.0_f32, -1.0, 0.5],
            vec![-0.5, 0.8, -0.3],
        ];
        let bias = vec![0.0_f32, 0.0];
        let layer = TernaryLinear::from_floats(&floats, bias);
        let x = vec![1.0_f32, 0.0, -1.0];
        let out = layer.forward_float(&x);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let n = rms_norm(&x, 1e-6);
        let rms: f32 = (n.iter().map(|v| v * v).sum::<f32>() / n.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let s = softmax(&[1.0_f32, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_ffn_shape() {
        let ffn = TernaryFFN::new(8, 16);
        let x = vec![0.5_f32; 8];
        let out = ffn.forward(&x);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_transformer_block_shape() {
        let block = TernaryTransformerBlock::new(8, 4, 16);
        let seq: Vec<Vec<f32>> = (0..3).map(|_| vec![0.1_f32; 8]).collect();
        let out = block.forward(&seq, true);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_kv_cache_push_and_retrieve() {
        let mut cache = KvCache::new(16);
        cache.push(vec![1.0_f32, -1.0, 0.5], vec![0.3_f32, -0.7, 0.0]);
        assert_eq!(cache.len(), 1);
        let k = cache.get_key(0);
        assert_eq!(k.len(), 3);
        let v = cache.get_value(0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_kv_cache_max_len_eviction() {
        let mut cache = KvCache::new(3);
        for i in 0..5_usize {
            cache.push(vec![i as f32], vec![i as f32]);
        }
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![0.1_f32, 0.9, 0.3, 0.5];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-5.0_f32, -1.0, -3.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_lm_forward_shape() {
        let lm = TernaryLM::new(32, 8, 4, 16);
        let out = lm.forward(&[0, 1, 2]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 32);
    }

    #[test]
    fn test_lm_generate_length() {
        let lm = TernaryLM::new(32, 8, 4, 16);
        let new_tokens = lm.generate(&[0, 1], 5);
        assert_eq!(new_tokens.len(), 5);
    }

    #[test]
    fn test_generate_tokens_in_vocab() {
        let vocab_size = 16;
        let lm = TernaryLM::new(vocab_size, 8, 4, 16);
        let new_tokens = lm.generate(&[0], 4);
        for t in new_tokens {
            assert!(t < vocab_size);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(1_000_000);
        assert!(ratio > 15.0, "expected > 15x compression, got {}", ratio);
    }
}
