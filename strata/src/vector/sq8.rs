//! SQ8: 8-bit scalar quantisation with a per-dimension range.
//!
//! Dimension `j` maps `[min_j, min_j + 255·scale_j]` onto codes `0..=255`:
//! `x ≈ min_j + scale_j · c_j`. For a query `q`,
//!
//! ```text
//! q · x ≈ Σ q_j·min_j + Σ (q_j·scale_j)·c_j = bias + Σ w_j·c_j
//! ```
//!
//! `bias` is the same for every document, so ranking needs only `Σ w_j·c_j`.
//! The weights `w` are quantised once per query to int8 (`w ≈ alpha · v`), so
//! scoring a document is an integer dot product of its u8 codes with the i8
//! query: one `vpdpbusd` per 64 dimensions (NEON: one `usdot` per 16).

use crate::postings::bytes::Slice;

use crate::vector::kernels;

/// Per-dimension quantisation ranges.
#[derive(Clone, Debug)]
pub struct Sq8 {
    pub min: Slice<f32>,
    pub scale: Slice<f32>,
}

/// A query prepared for SQ8 scoring.
#[derive(Clone, Debug)]
pub struct Sq8Query {
    /// Quantised weights, zero-padded to the code row stride.
    pub q8: Vec<i8>,
    /// Float32 weights `q_j·scale_j` (for the asymmetric kernel), padded.
    pub w: Vec<f32>,
    pub alpha: f32,
    pub bias: f32,
}

impl Sq8Query {
    /// Approximate similarity from an integer code dot product.
    #[inline]
    pub fn score(&self, dot: i32) -> f32 {
        self.bias + self.alpha * dot as f32
    }

    /// Approximate similarity of a code row with the quantised query.
    #[inline]
    pub fn dot(&self, codes: &[u8]) -> i32 {
        kernels::dot_u8_i8(codes, &self.q8)
    }
}

impl Sq8 {
    /// Fits the per-dimension ranges to `rows` (each of length `dim`).
    pub fn fit<'a>(dim: usize, rows: impl Iterator<Item = &'a [f32]>) -> Sq8 {
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for r in rows {
            for j in 0..dim {
                lo[j] = lo[j].min(r[j]);
                hi[j] = hi[j].max(r[j]);
            }
        }
        let mut scale = vec![0f32; dim];
        for j in 0..dim {
            if lo[j] > hi[j] {
                // No rows at all.
                lo[j] = 0.0;
            } else {
                scale[j] = (hi[j] - lo[j]) / 255.0;
            }
        }
        Sq8 { min: Slice::Owned(lo), scale: Slice::Owned(scale) }
    }

    pub fn dim(&self) -> usize {
        self.min.len()
    }

    /// Codes for `x` into `out[..dim]` (values outside the fitted range clamp).
    pub fn encode(&self, x: &[f32], out: &mut [u8]) {
        for j in 0..self.dim() {
            let s = self.scale[j];
            let c = if s > 0.0 { ((x[j] - self.min[j]) / s).round() } else { 0.0 };
            out[j] = c.clamp(0.0, 255.0) as u8;
        }
    }

    /// Reconstructs a vector from its codes.
    pub fn decode(&self, codes: &[u8], out: &mut [f32]) {
        for j in 0..self.dim() {
            out[j] = self.min[j] + self.scale[j] * codes[j] as f32;
        }
    }

    /// Prepares `q` for scoring code rows of `stride` bytes.
    pub fn prepare(&self, q: &[f32], stride: usize) -> Sq8Query {
        let dim = self.dim();
        let mut w = vec![0f32; stride];
        let mut bias = 0f64;
        for j in 0..dim {
            w[j] = q[j] * self.scale[j];
            bias += q[j] as f64 * self.min[j] as f64;
        }
        let max = w.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let alpha = max / 127.0;
        let q8 = w.iter().map(|&x| if alpha > 0.0 { (x / alpha).round().clamp(-127.0, 127.0) as i8 } else { 0 }).collect();
        Sq8Query { q8, w, alpha, bias: bias as f32 }
    }

    /// [`Sq8::prepare`] without the float32 weights, for [`Sq8Query::dot`]
    /// only: graph construction prepares one per insert and per neighbour
    /// it compares candidates with.
    pub fn prepare_int8(&self, q: &[f32], stride: usize) -> Sq8Query {
        let mut bias = 0f64;
        let mut max = 0f32;
        for j in 0..self.dim() {
            max = max.max((q[j] * self.scale[j]).abs());
            bias += q[j] as f64 * self.min[j] as f64;
        }
        let alpha = max / 127.0;
        let mut q8 = vec![0i8; stride];
        if alpha > 0.0 {
            for j in 0..self.dim() {
                q8[j] = ((q[j] * self.scale[j]) / alpha).round().clamp(-127.0, 127.0) as i8;
            }
        }
        Sq8Query { q8, w: Vec::new(), alpha, bias: bias as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn round_trip_error_is_within_half_a_step() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let dim = 37;
        let rows: Vec<Vec<f32>> = (0..500).map(|_| (0..dim).map(|j| rng.gen_range(-1.0..1.0) * (j as f32 + 1.0)).collect()).collect();
        let sq = Sq8::fit(dim, rows.iter().map(|r| r.as_slice()));
        let mut c = vec![0u8; dim];
        let mut back = vec![0f32; dim];
        for r in &rows {
            sq.encode(r, &mut c);
            sq.decode(&c, &mut back);
            for j in 0..dim {
                let err = (r[j] - back[j]).abs();
                assert!(err <= sq.scale[j] * 0.5 + 1e-5 * (j as f32 + 1.0), "dim {j}: error {err} > half step {}", sq.scale[j] / 2.0);
            }
        }
        // A constant dimension encodes exactly.
        let flat = Sq8::fit(2, [[1.5f32, 0.0].as_slice(), [1.5, 1.0].as_slice()].into_iter());
        flat.encode(&[1.5, 0.25], &mut c[..2]);
        flat.decode(&c[..2], &mut back[..2]);
        assert_eq!(back[0], 1.5);
    }

    #[test]
    fn quantised_score_approximates_dot_product() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(8);
        let dim = 256;
        let unit = |rng: &mut rand::rngs::StdRng| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let rows: Vec<Vec<f32>> = (0..200).map(|_| unit(&mut rng)).collect();
        let sq = Sq8::fit(dim, rows.iter().map(|r| r.as_slice()));
        let q = unit(&mut rng);
        let qc = sq.prepare(&q, dim);
        let mut c = vec![0u8; dim];
        for r in &rows {
            sq.encode(r, &mut c);
            let exact = kernels::dot_f32(&q, r);
            let approx = qc.score(qc.dot(&c));
            let asym = qc.bias + kernels::dot_u8_f32(&c, &qc.w);
            assert!((exact - approx).abs() < 0.02, "exact {exact} approx {approx}");
            assert!((exact - asym).abs() < 0.02, "exact {exact} asym {asym}");
        }
        let q8 = sq.prepare_int8(&q, dim);
        assert_eq!((q8.q8, q8.alpha, q8.bias), (qc.q8.clone(), qc.alpha, qc.bias));
    }
}
