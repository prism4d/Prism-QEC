//! Schedule-conditioned transfer entropy between detector spike trains (`do-TIDE`).
//!
//! A *coupler pair class* `(ci, cj, lag)` is a pair of check sites whose detectors are
//! flipped together by some fault mechanism of the model: `lag = 0` couples the two checks
//! in the same detector layer, `lag = 1` couples `ci` at layer `l` with `cj` at layer `l + 1`.
//! For every class we keep the eight joint counts of
//!
//! ```text
//! y = x_cj[target layer],   c = x_cj[target layer - 1],   x = x_ci[source layer]
//! ```
//!
//! and evaluate the transfer entropy `TE = I(y ; x | c)`: the information the source spike
//! carries about the target beyond the target's own past.  The interventional reference is
//! the nominal circuit: the same statistic under the known gate schedule (`do(schedule)`),
//! obtained by sampling the nominal model.  The excess `2 N (TE_obs - TE_nominal)` is a
//! G-statistic; a class is *active* when it exceeds a threshold.
//!
//! Counts are bytes with halving at saturation (integer forgetting) and logarithms come from
//! a Q16 table, so every decision is integer-exact on CPU and GPU.

/// One coupler pair class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PairClass {
    pub ci: u16,
    pub cj: u16,
    pub lag: u8,
}

/// `L[n] = round(65536 ln n)` for `n = 1..=255`, `L[0] = 0`.
pub fn log_table_q16() -> Vec<i32> {
    (0..256).map(|n| if n == 0 { 0 } else { ((n as f64).ln() * 65536.0).round() as i32 }).collect()
}

#[inline]
pub fn bin_index(y: bool, c: bool, x: bool) -> usize {
    ((y as usize) << 2) | ((c as usize) << 1) | x as usize
}

/// Adds one observation; halves every bin once the total would exceed `cap` (<= 255).
#[inline]
pub fn bump(counts: &mut [u8; 8], bin: usize, cap: u32) {
    let total: u32 = counts.iter().map(|&c| c as u32).sum();
    if total >= cap {
        for c in counts.iter_mut() {
            *c >>= 1;
        }
    }
    counts[bin] += 1;
}

#[inline]
pub fn total(counts: &[u8; 8]) -> u32 {
    counts.iter().map(|&c| c as u32).sum()
}

/// `2 N * I(y ; x | c)` in Q16 nats from integer counts.
pub fn g_statistic_q16(counts: &[u8; 8], logt: &[i32]) -> i64 {
    let mut g = 0i64;
    for c in 0..2usize {
        let n = |y: usize, x: usize| counts[(y << 2) | (c << 1) | x] as usize;
        let n_c = n(0, 0) + n(0, 1) + n(1, 0) + n(1, 1);
        for y in 0..2usize {
            for x in 0..2usize {
                let n_ycx = n(y, x);
                if n_ycx == 0 {
                    continue;
                }
                let n_yc = n(y, 0) + n(y, 1);
                let n_cx = n(0, x) + n(1, x);
                g += n_ycx as i64 * (logt[n_ycx] as i64 + logt[n_c] as i64 - logt[n_yc] as i64 - logt[n_cx] as i64);
            }
        }
    }
    2 * g
}

/// True iff either spike train of the pair fires in 35%..65% of the samples.  A train that is
/// (close to) a fair coin through *other* sources is statistically independent of any single
/// one of them, so an inactive pair with such an endpoint is uninformative rather than
/// evidence against coupling.
pub fn pair_masked(counts: &[u8; 8]) -> bool {
    let n = total(counts);
    let cnt = |y: usize, x: usize| (counts[(y << 2) | x] as u32) + (counts[(y << 2) | 2 | x] as u32);
    let nx = cnt(0, 1) + cnt(1, 1);
    let ny = cnt(1, 0) + cnt(1, 1);
    let fair = |k: u32| 20 * k >= 7 * n && 20 * k <= 13 * n;
    n > 0 && (fair(nx) || fair(ny))
}

/// Integer square root (floor).
pub fn isqrt_u64(x: u64) -> u64 {
    if x < 2 {
        return x;
    }
    let mut r = (x as f64).sqrt() as u64;
    while r * r > x {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= x {
        r += 1;
    }
    r
}

/// Probability (Q12, in `[0, 2048]` i.e. `[0, 0.5]`) that an odd number of mechanisms flips
/// both detectors of the pair, from the first two moments of the spike trains:
/// `p = 1/2 - 1/2 sqrt(1 - 4 cov / (1 - 2<x> - 2<y> + 4<xy>))`.
pub fn pair_rate_q12(counts: &[u8; 8]) -> i32 {
    let n = total(counts) as i64;
    if n == 0 {
        return 0;
    }
    let cnt = |y: usize, x: usize| (counts[(y << 2) | x] as i64) + (counts[(y << 2) | 2 | x] as i64);
    let n11 = cnt(1, 1);
    let nx = cnt(0, 1) + n11;
    let ny = cnt(1, 0) + n11;
    let num = 4 * (n * n11 - nx * ny);
    let den = n * (n - 2 * nx - 2 * ny + 4 * n11);
    if num <= 0 {
        return 0;
    }
    if den <= 0 || num >= den {
        return 2048;
    }
    let t_q24 = (((den - num) as u64) << 24) / den as u64;
    ((4096 - isqrt_u64(t_q24) as i64) / 2) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::fixed::mix64;

    fn uniform(seed: u64, i: u64) -> f64 {
        (mix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn collect(seed: u64, n: u64, p_shared: f64, p_own: f64) -> [u8; 8] {
        let mut counts = [0u8; 8];
        let mut prev_y = false;
        for t in 0..n {
            let shared = uniform(seed, 3 * t) < p_shared;
            let x = shared ^ (uniform(seed, 3 * t + 1) < p_own);
            let y = shared ^ (uniform(seed, 3 * t + 2) < p_own);
            bump(&mut counts, bin_index(y, prev_y, x), 255);
            prev_y = y;
        }
        counts
    }

    #[test]
    fn integer_g_statistic_matches_floating_point_transfer_entropy() {
        let logt = log_table_q16();
        for seed in 0..10u64 {
            let counts = collect(seed, 200, 0.2, 0.1);
            let n: f64 = total(&counts) as f64;
            let mut te = 0.0f64;
            for c in 0..2usize {
                let get = |y: usize, x: usize| counts[(y << 2) | (c << 1) | x] as f64;
                let n_c = get(0, 0) + get(0, 1) + get(1, 0) + get(1, 1);
                for y in 0..2 {
                    for x in 0..2 {
                        let v = get(y, x);
                        if v > 0.0 {
                            te += v / n * (v * n_c / ((get(y, 0) + get(y, 1)) * (get(0, x) + get(1, x)))).ln();
                        }
                    }
                }
            }
            let g = g_statistic_q16(&counts, &logt) as f64 / 65536.0;
            assert!((g - 2.0 * n * te).abs() < 1e-2, "G {g} vs 2 N TE {}", 2.0 * n * te);
            assert!(g >= -1e-2);
        }
    }

    #[test]
    fn coupled_trains_are_separated_from_independent_ones() {
        let logt = log_table_q16();
        let thr = 24 * 65536;
        let (mut hits, mut false_hits) = (0, 0);
        for seed in 0..200u64 {
            hits += (g_statistic_q16(&collect(seed, 120, 0.25, 0.08), &logt) >= thr) as u32;
            false_hits += (g_statistic_q16(&collect(1000 + seed, 120, 0.0, 0.2), &logt) >= thr) as u32;
        }
        assert!(hits >= 190, "coupled pairs detected {hits}/200");
        assert_eq!(false_hits, 0, "independent pairs must not trigger at G >= 24");
    }

    #[test]
    fn pair_rate_recovers_the_shared_mechanism_probability() {
        for &p in &[0.05f64, 0.2, 0.45] {
            let mut acc = 0.0;
            for seed in 0..50u64 {
                acc += pair_rate_q12(&collect(seed + 77, 250, p, 0.05)) as f64 / 4096.0 / 50.0;
            }
            assert!((acc - p).abs() < 0.04, "true {p}, estimated {acc}");
        }
        assert_eq!(pair_rate_q12(&[0; 8]), 0);
        assert_eq!(isqrt_u64(1 << 24), 4096);
        assert_eq!(isqrt_u64(99), 9);
    }

    #[test]
    fn counters_forget_by_halving() {
        let mut counts = [0u8; 8];
        for _ in 0..10_000 {
            bump(&mut counts, 5, 255);
        }
        assert!(total(&counts) <= 255 && total(&counts) >= 127);
    }
}
