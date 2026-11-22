//! Bayesian online changepoint detection (Adams & MacKay) on a detector's defect-spike train.
//!
//! Observation model per run: spikes are (censored) Poisson with rate `lambda * e_t`, where
//! `e_t = -ln(1 - pi_t)` is the *exposure* of the detector at this position of the
//! syndrome-extraction schedule under the nominal model (`pi_t` its nominal firing
//! probability) and `lambda` is the multiplicative rate of the current regime (`lambda = 1`
//! is nominal).  With the conjugate prior `lambda ~ Gamma(1, 1)` a run that has seen `a`
//! spikes over total exposure `b` predicts `P(no spike) = ((1 + b) / (1 + b + e))^(1 + a)`:
//! an *integer* power, evaluated by repeated squaring so that the recursion uses only IEEE
//! multiplications, divisions and additions in a fixed order (bit-identical on CPU and GPU).

pub const R_MAX: usize = 32;

/// `base^exp` by binary exponentiation with a fixed operation order.
#[inline]
pub fn powi_fixed(base: f64, mut exp: u32) -> f64 {
    let mut acc = 1.0f64;
    let mut sq = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= sq;
        }
        sq *= sq;
        exp >>= 1;
    }
    acc
}

/// Run-length filter of one detector site.
#[derive(Clone, Debug, PartialEq)]
pub struct Bocpd {
    /// Run-length posterior `P(r_t = k | x_1..t)`; the last bin holds `r >= R_MAX - 1`.
    pub r: [f64; R_MAX],
    /// Spikes and accumulated exposure of each run hypothesis.
    pub a: [u32; R_MAX],
    pub b: [f64; R_MAX],
}

impl Default for Bocpd {
    fn default() -> Self {
        Self::new()
    }
}

impl Bocpd {
    pub fn new() -> Self {
        let mut r = [0.0; R_MAX];
        r[0] = 1.0;
        Self { r, a: [0; R_MAX], b: [0.0; R_MAX] }
    }

    /// Assimilates one layer: `spike` observed at exposure `e`, constant hazard `h`.
    pub fn update(&mut self, spike: bool, e: f64, h: f64) {
        let mut w = [0.0f64; R_MAX];
        let mut total = 0.0f64;
        for k in 0..R_MAX {
            let q = (1.0 + self.b[k]) / (1.0 + self.b[k] + e);
            let p0 = powi_fixed(q, 1 + self.a[k]);
            let pred = if spike { 1.0 - p0 } else { p0 };
            w[k] = self.r[k] * pred;
            total += w[k];
        }
        let grow = 1.0 - h;
        let mut nr = [0.0f64; R_MAX];
        let mut na = [0u32; R_MAX];
        let mut nb = [0.0f64; R_MAX];
        nr[0] = h * total;
        for k in 0..R_MAX - 1 {
            nr[k + 1] = w[k] * grow;
            na[k + 1] = self.a[k] + spike as u32;
            nb[k + 1] = self.b[k] + e;
        }
        // tail merge: the last bin absorbs runs that outgrow the window and keeps the
        // statistics of whichever component carries more mass
        nr[R_MAX - 1] += w[R_MAX - 1] * grow;
        if w[R_MAX - 1] >= w[R_MAX - 2] {
            na[R_MAX - 1] = self.a[R_MAX - 1] + spike as u32;
            nb[R_MAX - 1] = self.b[R_MAX - 1] + e;
        }
        let norm = nr.iter().fold(0.0f64, |s, &x| s + x);
        for k in 0..R_MAX {
            self.r[k] = nr[k] / norm;
        }
        self.a = na;
        self.b = nb;
    }

    /// Posterior mean of the rate multiplier `lambda`, marginal over run lengths.
    pub fn rate_mean(&self) -> f64 {
        let mut m = 0.0f64;
        for k in 0..R_MAX {
            m += self.r[k] * ((1.0 + self.a[k] as f64) / (1.0 + self.b[k]));
        }
        m
    }

    /// Age of the hot regime: the most probable run length among the hypotheses whose own
    /// rate estimate is at least `thr` (the plain MAP run may still be the stale long run when
    /// the posterior mean first crosses the threshold).  Returns 1 if no hypothesis is hot.
    pub fn hot_run(&self, thr: f64) -> usize {
        let mut best = 1;
        let mut best_v = 0.0f64;
        for k in 0..R_MAX {
            if (1.0 + self.a[k] as f64) / (1.0 + self.b[k]) >= thr && self.r[k] > best_v {
                best_v = self.r[k];
                best = k;
            }
        }
        best
    }

    /// Most probable run length.
    pub fn map_run(&self) -> usize {
        let mut best = 0;
        for k in 1..R_MAX {
            if self.r[k] > self.r[best] {
                best = k;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::fixed::mix64;

    fn uniform(seed: u64, i: u64) -> f64 {
        (mix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn integer_power_matches_libm() {
        for &(b, e) in &[(0.93f64, 1u32), (0.5, 7), (0.999, 33), (1.0, 0), (0.25, 20)] {
            assert!((powi_fixed(b, e) - b.powi(e as i32)).abs() < 1e-14);
        }
    }

    #[test]
    fn posterior_stays_normalised_and_tracks_a_stationary_rate() {
        let pi = 0.08f64;
        let e = -(1.0 - pi).ln();
        let mut f = Bocpd::new();
        for t in 0..2000u64 {
            f.update(uniform(1, t) < pi, e, 1.0 / 250.0);
            let s: f64 = f.r.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
        assert!((f.rate_mean() - 1.0).abs() < 0.35, "nominal stream must read lambda ~ 1: {}", f.rate_mean());
        assert_eq!(f.map_run(), R_MAX - 1, "a long stationary run saturates the window");
    }

    #[test]
    fn rate_jump_is_detected_within_ten_layers() {
        let (pi0, pi1) = (0.08f64, 0.5f64);
        let e = -(1.0 - pi0).ln();
        let mut latencies = Vec::new();
        let mut false_alarms = 0u32;
        for stream in 0..200u64 {
            let mut f = Bocpd::new();
            let mut detected = None;
            for t in 0..600u64 {
                let hot = t >= 500;
                f.update(uniform(100 + stream, t) < if hot { pi1 } else { pi0 }, e, 1.0 / 100.0);
                let alarm = f.rate_mean() >= 2.0;
                if !hot && t >= 50 && alarm {
                    false_alarms += 1;
                }
                if hot && alarm && detected.is_none() {
                    detected = Some(t - 500 + 1);
                }
            }
            latencies.push(detected.unwrap_or(1000));
        }
        latencies.sort_unstable();
        let median = latencies[latencies.len() / 2];
        let p90 = latencies[latencies.len() * 9 / 10];
        let fa_rate = false_alarms as f64 / (200.0 * 450.0);
        eprintln!("[bocpd] detection latency median={median} p90={p90} layers, false-alarm layer rate={fa_rate:.5}");
        assert!(median <= 10 && p90 <= 20, "median {median}, p90 {p90}");
        assert!(fa_rate < 0.005, "false-alarm rate {fa_rate}");
    }
}
