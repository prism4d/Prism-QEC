//! Pareto-smoothed importance sampling diagnostic (Vehtari, Simpson, Gelman, Yao, Gabry).
//!
//! `k_hat` is the shape of a generalized Pareto distribution fitted to the upper tail of the
//! importance ratios with the Zhang-Stephens empirical-Bayes estimator.  `k_hat > 0.7` means
//! the importance-weighted estimate has no usable variance bound at this sample size.

/// Number of tail draws used for a sample of size `s`: `ceil(min(s / 5, 3 sqrt(s)))`.
pub fn tail_length(s: usize) -> usize {
    let a = s as f64 / 5.0;
    let b = 3.0 * (s as f64).sqrt();
    a.min(b).ceil() as usize
}

/// Zhang-Stephens fit of a generalized Pareto distribution to sorted positive exceedances.
/// Returns `(k, sigma)` with the weakly informative prior of the PSIS paper applied to `k`.
pub fn gpd_fit(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let nf = n as f64;
    let prior_bs = 3.0;
    let prior_k = 10.0;
    let m = 30 + (nf.sqrt() as usize);
    let quartile = x[((nf / 4.0 + 0.5).floor() as usize).max(1) - 1];
    let x_max = x[n - 1];
    let mut b = vec![0.0f64; m];
    let mut len_scale = vec![0.0f64; m];
    for j in 0..m {
        let bj = (1.0 - (m as f64 / (j as f64 + 0.5)).sqrt()) / (prior_bs * quartile) + 1.0 / x_max;
        let kj = x.iter().map(|&xi| (-bj * xi).ln_1p()).sum::<f64>() / nf;
        b[j] = bj;
        len_scale[j] = nf * ((-bj / kj).ln() - kj - 1.0);
    }
    let mut w = vec![0.0f64; m];
    for j in 0..m {
        let s: f64 = len_scale.iter().map(|&l| (l - len_scale[j]).exp()).sum();
        w[j] = 1.0 / s;
    }
    let eps = 10.0 * f64::EPSILON;
    let total: f64 = w.iter().filter(|&&x| x >= eps).sum();
    let b_post: f64 = b.iter().zip(&w).filter(|(_, &wj)| wj >= eps).map(|(&bj, &wj)| bj * wj / total).sum();
    let k_post = x.iter().map(|&xi| (-b_post * xi).ln_1p()).sum::<f64>() / nf;
    let sigma = -k_post / b_post;
    let k = (nf * k_post + prior_k * 0.5) / (nf + prior_k);
    (k, sigma)
}

/// Pareto `k_hat` of a set of log importance weights (unnormalized, any offset).
///
/// Conventions for degenerate inputs: fewer than 5 tail draws gives `+inf` (no diagnostic is
/// possible); a tail of identical ratios is a bounded tail and gives `0`.
pub fn psis_khat(log_weights: &[f64]) -> f64 {
    let s = log_weights.len();
    let m = tail_length(s);
    if m < 5 || m >= s {
        return f64::INFINITY;
    }
    let mut lw: Vec<f64> = log_weights.to_vec();
    lw.sort_by(|a, b| a.total_cmp(b));
    let top = lw[s - 1];
    let cutoff = (lw[s - m - 1] - top).exp();
    let tail: Vec<f64> = lw[s - m..].iter().map(|&l| (l - top).exp() - cutoff).collect();
    if tail[m - 1] <= 1e-300 || tail[m - 1] - tail[0] <= 1e-12 * tail[m - 1] {
        return 0.0;
    }
    // exceedances are sorted ascending; guard the smallest against exact zero
    let floor = tail[m - 1] * 1e-12;
    let x: Vec<f64> = tail.iter().map(|&t| t.max(floor)).collect();
    let (k, _) = gpd_fit(&x);
    if k.is_finite() {
        k
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::fixed::mix64;

    fn uniform(seed: u64, i: usize) -> f64 {
        ((mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Draws from a generalized Pareto distribution with shape `k` by inversion.
    fn gpd_sample(k: f64, sigma: f64, seed: u64, n: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let u = uniform(seed, i);
                if k.abs() < 1e-12 { -sigma * (1.0 - u).ln() } else { sigma * ((1.0 - u).powf(-k) - 1.0) / k }
            })
            .collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v
    }

    #[test]
    fn zhang_stephens_recovers_the_shape_parameter() {
        for &(k_true, tol) in &[(-0.3, 0.12), (0.0, 0.12), (0.4, 0.12), (0.9, 0.15)] {
            let mut acc = 0.0;
            let reps = 20;
            for seed in 0..reps {
                acc += gpd_fit(&gpd_sample(k_true, 1.5, 1000 + seed, 2000)).0;
            }
            let mean = acc / reps as f64;
            assert!((mean - k_true).abs() < tol, "k_true = {k_true}, estimated {mean}");
        }
    }

    #[test]
    fn khat_separates_light_and_heavy_tailed_weights() {
        // log-weights = sigma * standard normal: ratios are log-normal; the tail heaviness seen
        // at finite sample size grows with sigma.
        let normal = |seed: u64, i: usize| {
            let (u1, u2) = (uniform(seed, 2 * i), uniform(seed, 2 * i + 1));
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let khat_for = |sigma: f64, seed: u64| {
            let lw: Vec<f64> = (0..256).map(|i| sigma * normal(seed, i)).collect();
            psis_khat(&lw)
        };
        let light: f64 = (0..30).map(|s| khat_for(0.3, s)).sum::<f64>() / 30.0;
        let heavy: f64 = (0..30).map(|s| khat_for(4.0, s)).sum::<f64>() / 30.0;
        assert!(light < 0.5, "light-tailed weights should certify: {light}");
        assert!(heavy > 0.7, "heavy-tailed weights must be flagged: {heavy}");
        // offsets do not matter
        let lw: Vec<f64> = (0..256).map(|i| 2.0 * normal(3, i)).collect();
        let shifted: Vec<f64> = lw.iter().map(|x| x + 1234.5).collect();
        assert!((psis_khat(&lw) - psis_khat(&shifted)).abs() < 1e-9);
    }

    #[test]
    fn degenerate_inputs_follow_the_documented_conventions() {
        assert_eq!(psis_khat(&[0.0; 256]), 0.0);
        assert_eq!(psis_khat(&[0.0, 1.0, 2.0]), f64::INFINITY);
        assert_eq!(tail_length(256), 48);
        assert_eq!(tail_length(64), 13);
    }
}
