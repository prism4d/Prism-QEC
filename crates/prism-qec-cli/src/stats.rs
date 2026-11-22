//! Exact statistics for the evaluation matrix: Clopper-Pearson binomial intervals and the
//! exact two-sided McNemar test on paired discordant counts.

/// `ln Gamma(x)` for `x > 0` (Lanczos approximation, g = 7, 9 coefficients).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`, summed in log space from the mode outwards.
pub fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let (lp, lq) = (p.ln(), (-p).ln_1p());
    let term = |i: u64| (ln_choose(n, i) + i as f64 * lp + (n - i) as f64 * lq).exp();
    // sum the smaller tail for accuracy
    let mean = n as f64 * p;
    if (k as f64) < mean {
        (0..=k).rev().map(term).take_while(|&t| t > 0.0).sum::<f64>().min(1.0)
    } else {
        let upper: f64 = (k + 1..=n).map(term).take_while(|&t| t > 0.0).sum();
        (1.0 - upper).clamp(0.0, 1.0)
    }
}

/// Exact two-sided Clopper-Pearson interval for `k` successes in `n` trials at level `conf`.
pub fn clopper_pearson(k: u64, n: u64, conf: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let alpha = 1.0 - conf;
    let bisect = |f: &dyn Fn(f64) -> f64, target: f64, increasing: bool| -> f64 {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let v = f(mid);
            if (v < target) == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    // lower: P(X >= k | p) = alpha / 2, increasing in p
    let lower = if k == 0 { 0.0 } else { bisect(&|p| 1.0 - binom_cdf(k - 1, n, p), alpha / 2.0, true) };
    // upper: P(X <= k | p) = alpha / 2, decreasing in p
    let upper = if k == n { 1.0 } else { bisect(&|p| binom_cdf(k, n, p), alpha / 2.0, false) };
    (lower, upper)
}

/// Exact two-sided McNemar p-value for discordant counts `b` and `c`:
/// `min(1, 2 P(X <= min(b, c)))` with `X ~ Binomial(b + c, 1/2)`.
pub fn mcnemar_exact(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 || b == c {
        return 1.0;
    }
    (2.0 * binom_cdf(b.min(c), n, 0.5)).min(1.0)
}

/// One-sided exact p-value that the first decoder is *worse*: `P(X >= b)` with `b` its
/// failures on discordant pairs.
pub fn mcnemar_worse(b_fails: u64, c_fails: u64) -> f64 {
    let n = b_fails + c_fails;
    if n == 0 {
        return 1.0;
    }
    if b_fails == 0 {
        return 1.0;
    }
    1.0 - binom_cdf(b_fails - 1, n, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-11);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-11);
        assert!((ln_gamma(100_001.0) - 1_051_299.221_899_122).abs() < 1e-6);
    }

    #[test]
    fn binomial_cdf_matches_direct_summation() {
        // Binomial(10, 0.3): P(X <= 3) = 0.6496107184
        assert!((binom_cdf(3, 10, 0.3) - 0.649_610_718_4).abs() < 1e-9);
        assert!((binom_cdf(0, 20, 0.05) - 0.95f64.powi(20)).abs() < 1e-12);
        assert_eq!(binom_cdf(20, 20, 0.5), 1.0);
        // large n: symmetric at p = 1/2
        assert!((binom_cdf(49_999, 100_000, 0.5) + binom_cdf(49_999, 100_000, 0.5) + binom_pmf_mid() - 1.0).abs() < 1e-9);
    }

    fn binom_pmf_mid() -> f64 {
        (ln_choose(100_000, 50_000) + 100_000.0 * 0.5f64.ln()).exp()
    }

    #[test]
    fn clopper_pearson_known_intervals() {
        // classic tabulated cases
        let (lo, hi) = clopper_pearson(0, 10, 0.95);
        assert_eq!(lo, 0.0);
        assert!((hi - 0.308_497_1).abs() < 1e-6);
        let (lo, hi) = clopper_pearson(5, 10, 0.95);
        assert!((lo - 0.187_086_0).abs() < 1e-6 && (hi - 0.812_914_0).abs() < 1e-6);
        let (lo, hi) = clopper_pearson(150, 50_000, 0.95);
        assert!(lo < 0.003 && hi > 0.003 && lo > 0.0025 && hi < 0.0036);
        // rule of three: 0 failures in n trials -> upper bound ~ 3.69 / n at 95% two-sided
        let (_, hi) = clopper_pearson(0, 100_000, 0.95);
        assert!((hi * 100_000.0 - 3.6888).abs() < 1e-2);
    }

    #[test]
    fn mcnemar_known_values() {
        assert_eq!(mcnemar_exact(7, 7), 1.0);
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        // b = 1, c = 9: 2 * P(X <= 1 | n = 10) = 2 * 11 / 1024
        assert!((mcnemar_exact(1, 9) - 22.0 / 1024.0).abs() < 1e-12);
        assert!((mcnemar_exact(9, 1) - 22.0 / 1024.0).abs() < 1e-12);
        assert!(mcnemar_exact(10, 60) < 1e-8);
        // one-sided: P(X >= 9 | n = 10) = 11 / 1024
        assert!((mcnemar_worse(9, 1) - 11.0 / 1024.0).abs() < 1e-12);
        assert!(mcnemar_worse(1, 9) > 0.99);
    }
}
