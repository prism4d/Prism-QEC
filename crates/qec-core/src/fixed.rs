//! Fixed-point conventions and counter-based randomness shared by the CPU reference
//! decoders and the SM120 kernels.  Every decoder decision is taken in integer arithmetic,
//! so the CPU and GPU paths agree bit-for-bit; transcendental functions enter only through
//! the lookup tables built here (computed once on the host and uploaded verbatim).

/// Belief-propagation LLR units per nat.
pub const LLR_ONE: i32 = 64;
/// Clip of prior LLR magnitudes (24 nats).
pub const PRIOR_CLIP: i32 = 24 * LLR_ONE;
/// Clip of message magnitudes.
pub const MSG_CLIP: i32 = 2047;
/// Saturation of the damped beliefs.  With `|l| <= 2^24` every intermediate of the velocity
/// projection stays inside 64-bit (and its result inside 32-bit) range, so a numerically
/// unstable gauge mode saturates instead of overflowing.
pub const POST_CLIP: i32 = 1 << 24;
/// Clip of the accumulated Hodge potential.
pub const HODGE_CLIP: i32 = 8 * LLR_ONE;
/// Nishimori energy units per nat (SMC stage).
pub const EN_ONE: i32 = 16;
/// Number of weight bit-planes of the energy engine (`EN_CLIP < 2^EN_BITS`).
pub const EN_BITS: usize = 9;
pub const EN_CLIP: i32 = (1 << EN_BITS) - 1;
/// Clip of the twist field (10 nats).
pub const TWIST_CLIP: i32 = 10 * EN_ONE;
/// Fixed-point one of the Hodge metric `W` (Q12).
pub const W_ONE: i32 = 4096;
/// Largest `|LLR|` index of the tanh table (8 nats).
pub const TANH_MAX: i32 = 8 * LLR_ONE;

/// Prior LLR `ln((1 - p) / p)` in BP units, clipped to `[-PRIOR_CLIP, PRIOR_CLIP]`.
pub fn llr_from_prob(p: f64) -> i32 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let w = ((1.0 - p) / p).ln() * LLR_ONE as f64;
    (w.round() as i64).clamp(-(PRIOR_CLIP as i64), PRIOR_CLIP as i64) as i32
}

/// Nishimori energy weight in SMC units derived from a BP-unit prior LLR.
pub fn energy_from_llr(llr: i32) -> i32 {
    let shift = LLR_ONE / EN_ONE;
    ((llr + shift / 2).div_euclid(shift)).clamp(0, EN_CLIP)
}

/// `T[a] = round(W_ONE * tanh(a / (2 * LLR_ONE)))` for `a = 0..=TANH_MAX`.
pub fn tanh_table() -> Vec<u16> {
    (0..=TANH_MAX).map(|a| (W_ONE as f64 * (a as f64 / (2.0 * LLR_ONE as f64)).tanh()).round() as u16).collect()
}

/// Bernoulli thresholds of the twisted reference: `thr[t + TWIST_CLIP] = round(2^32 / (1 + e^{t / EN_ONE}))`
/// for `t = -TWIST_CLIP..=TWIST_CLIP`; a coordinate with twist `t` starts at 1 iff `u32 < thr`.
pub fn logistic_thresholds() -> Vec<u32> {
    (-TWIST_CLIP..=TWIST_CLIP)
        .map(|t| {
            let q = 1.0 / (1.0 + (t as f64 / EN_ONE as f64).exp());
            (q * 4294967296.0).round().clamp(1.0, 4294967295.0) as u32
        })
        .collect()
}

/// Metropolis thresholds: `acc[d] = round(2^32 * exp(-d / (EN_ONE * beta_steps)))`, truncated
/// where the probability falls below `2^-32`.  A move with tempered cost `d > 0` is accepted
/// iff `d < acc.len()` and `u32 < acc[d]`.
pub fn metropolis_thresholds(beta_steps: u32) -> Vec<u32> {
    let unit = (EN_ONE as f64) * beta_steps as f64;
    let len = (unit * 22.2).ceil() as usize + 1;
    (0..len).map(|d| ((-(d as f64) / unit).exp() * 4294967296.0).round().clamp(0.0, 4294967295.0) as u32).collect()
}

/// SplitMix64 finalizer.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Counter-based generator: a pure function of `(seed, shot, walker, counter)`.
#[inline]
pub fn rng_u64(seed: u64, shot: u32, walker: u32, counter: u32) -> u64 {
    let k = mix64(seed ^ (((shot as u64) << 32) | walker as u64).wrapping_add(0x9E37_79B9_7F4A_7C15));
    mix64(k ^ (counter as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(0x2545_F491_4F6C_DD1D))
}

/// Deterministic per-site quenched disorder in `[0, 256)` used by the Hodge metric regulator.
#[inline]
pub fn site_disorder(seed: u64, site: u32) -> u32 {
    (mix64(seed ^ (site as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 56) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_llr_is_monotone_antisymmetric_and_clipped() {
        assert_eq!(llr_from_prob(0.5), 0);
        assert_eq!(llr_from_prob(0.25), -llr_from_prob(0.75));
        assert!(llr_from_prob(0.001) > llr_from_prob(0.01));
        assert_eq!(llr_from_prob(1e-30), PRIOR_CLIP);
        assert_eq!(llr_from_prob(0.01), ((0.99f64 / 0.01).ln() * 64.0).round() as i32);
        assert_eq!(energy_from_llr(llr_from_prob(0.5)), 0);
        assert_eq!(energy_from_llr(PRIOR_CLIP), 384);
    }

    #[test]
    fn tables_have_expected_endpoints() {
        let t = tanh_table();
        assert_eq!(t.len(), (TANH_MAX + 1) as usize);
        assert_eq!(t[0], 0);
        assert!(t[TANH_MAX as usize] >= 4090 && t.windows(2).all(|w| w[0] <= w[1]));
        let l = logistic_thresholds();
        assert_eq!(l[TWIST_CLIP as usize], 1u32 << 31);
        assert!(l.windows(2).all(|w| w[0] >= w[1]));
        let m = metropolis_thresholds(32);
        assert_eq!(m[0], u32::MAX);
        assert!(*m.last().unwrap() <= 2);
    }

    #[test]
    fn rng_streams_are_distinct_and_roughly_uniform() {
        let mut ones = 0u32;
        let mut seen = std::collections::HashSet::new();
        for c in 0..4096u32 {
            let v = rng_u64(7, 3, 5, c);
            ones += v.count_ones();
            assert!(seen.insert(v));
        }
        let mean = ones as f64 / 4096.0;
        assert!((mean - 32.0).abs() < 0.5, "mean popcount {mean}");
        assert_ne!(rng_u64(7, 3, 5, 0), rng_u64(7, 3, 6, 0));
        assert_ne!(rng_u64(7, 3, 5, 0), rng_u64(8, 3, 5, 0));
    }
}
