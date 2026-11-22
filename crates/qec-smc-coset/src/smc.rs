//! Nishimori-temperature twisted SMC over the coset coordinates of a [`Pivot`].
//!
//! Target at `beta = 1`: `pi(z) ~ exp(-H_N(e(z)))`, `H_N = sum_i w_i e_i`, `w_i = ln((1 - p_i) / p_i)`.
//! Twisted reference at `beta = 0`: independent cycle toggles around the quenched state,
//! `mu_0(z) ~ exp(-Theta(z))`, `Theta(z) = sum_t theta_t (z_t (+) zq_t)`, where `theta_t >= 0` is
//! the energy cost of toggling cycle `t` there (the exact target if cycles did not overlap).
//! The bridge is geometric,
//! `pi_b ~ mu_0^(1 - b/B) exp(-(b/B) H_N)`, `b = 0..B`, with incremental weights
//! `log w -= (H_N - Theta) / B` and a deterministic-scan Metropolis sweep at each temperature.
//!
//! Whenever the effective sample size of the incremental weights drops below
//! `W * resample_q8 / 256` the population is resampled (systematic, integer arithmetic) and
//! the weights reset; the walkers then act as a population annealer that concentrates on the
//! low-energy cosets.  The final reweighting to `beta = 1` is never followed by a resampling:
//! the PSIS diagnostic is evaluated on those terminal, pre-resample log-weights.

use crate::pivot::Pivot;
use crate::psis::psis_khat;
use qec_core::fixed::{logistic_thresholds, metropolis_thresholds, rng_u64, EN_ONE, TWIST_CLIP};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmcConfig {
    /// Suspect columns handed to the pivot.
    pub k_cols: usize,
    /// Parallel walkers `W`.
    pub walkers: usize,
    /// Temperature steps `B` of the bridge `beta = b / B`.
    pub beta_steps: u32,
    /// Metropolis sweeps per temperature.
    pub sweeps: u32,
    pub seed: u64,
    /// Sweep budget of the zero-temperature quench that centres the twisted reference.
    pub quench_sweeps: u32,
    /// Resample when `ESS < W * resample_q8 / 256` (0 = never: plain annealed importance sampling).
    pub resample_q8: u32,
    /// HLE gap threshold `tau_cert` in nats.
    pub tau_cert: f64,
    /// HLE Pareto threshold.
    pub khat_max: f64,
}

impl Default for SmcConfig {
    fn default() -> Self {
        Self {
            k_cols: 512,
            walkers: 128,
            beta_steps: 32,
            sweeps: 1,
            seed: 0x5052_4953_4D53_4D43,
            quench_sweeps: 8,
            resample_q8: 128,
            tau_cert: 1.0,
            khat_max: 0.7,
        }
    }
}

/// Lookup tables shared verbatim with the SM120 kernel.
#[derive(Clone, Debug)]
pub struct SmcTables {
    pub logistic: Vec<u32>,
    pub metropolis: Vec<u32>,
}

impl SmcTables {
    pub fn new(beta_steps: u32) -> Self {
        Self { logistic: logistic_thresholds(), metropolis: metropolis_thresholds(beta_steps) }
    }
}

/// Terminal state of one walker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Walker {
    /// `A = sum_b (H_N - Theta)` along the path; `log w = -A / (B * EN_ONE)`.
    pub acc: i64,
    /// `L e` of the terminal state.
    pub obs: u64,
    /// Terminal Nishimori energy (SMC units).
    pub energy: i32,
}

/// Free-energy value reported when one coset side received no weight.
pub const DELTA_F_CAP: f64 = 60.0;

#[derive(Clone, Debug, Default)]
pub struct SmcResult {
    /// `Delta F_j = -ln(sum_{w: l_j = 1} w / sum_{w: l_j = 0} w)` for each logical observable.
    pub delta_f: Vec<f64>,
    /// Marginal maximum-likelihood decision: bit `j` set iff `Delta F_j < 0`.
    pub obs: u64,
    pub khat: f64,
    pub min_abs_df: f64,
    pub hle: bool,
    /// Effective sample size of the terminal weights.
    pub ess: f64,
    /// Estimate of `ln(Z_1 / Z_0)` (product of stage-mean incremental weights).
    pub log_z_ratio: f64,
    /// Minimum terminal energy over walkers and the logical word of that walker.
    pub best_energy: i32,
    pub best_obs: u64,
    /// Logical word of the coset with the largest total terminal weight (joint decision).
    pub mode_obs: u64,
    /// `-ln` of the weight fraction of that coset.
    pub mode_free_energy: f64,
    /// Minimum-weight gap heuristic: best energy outside the decided coset minus best energy
    /// inside it (`i32::MAX` if every walker ended in the decided coset).
    pub energy_gap: i32,
    /// Resampling events along the path.
    pub resamples: u32,
    pub walkers: Vec<Walker>,
}

/// Terminal population of one shot plus path statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Population {
    pub walkers: Vec<Walker>,
    pub resamples: u32,
    /// `ln` of the product of mean incremental weights over all resampling stages
    /// (excluding the terminal stage, which [`certify`] adds).
    pub log_z_path: f64,
}

/// Integer importance weight `round(2^32 exp(-(acc - acc_min) / (B * EN_ONE)))`.
#[inline]
pub fn int_weight(tables: &SmcTables, d: i64) -> u64 {
    if (d as usize) < tables.metropolis.len() {
        tables.metropolis[d as usize] as u64
    } else {
        0
    }
}

/// Runs all walkers for one shot.  Purely a function of `(pivot, cfg, shot_id)`.
pub fn run_walkers(pivot: &Pivot, cfg: &SmcConfig, tables: &SmcTables, shot_id: u32) -> Population {
    let n_free = pivot.n_free();
    let wr = pivot.wr;
    let wz = n_free.div_ceil(64).max(1);
    let nw = cfg.walkers;
    let b_total = cfg.beta_steps as i64;
    let mut rows = vec![0u64; nw * wr];
    let mut z = vec![0u64; nw * wz];
    let mut energy = vec![0i32; nw];
    let mut theta_sum = vec![0i32; nw];
    let mut obs = vec![0u64; nw];
    let mut acc = vec![0i64; nw];
    // ---- exact draws from the twisted reference centred on the quenched state
    for wk in 0..nw {
        let (r, zz) = (&mut rows[wk * wr..(wk + 1) * wr], &mut z[wk * wz..(wk + 1) * wz]);
        r.copy_from_slice(&pivot.rows_q);
        zz.copy_from_slice(&pivot.zq);
        energy[wk] = pivot.en_q;
        obs[wk] = pivot.obs_q;
        for t in 0..n_free {
            let u = (rng_u64(cfg.seed, shot_id, wk as u32, t as u32) >> 32) as u32;
            if u < tables.logistic[(pivot.theta[t] + TWIST_CLIP) as usize] {
                let zt = (zz[t >> 6] >> (t & 63)) & 1 == 1;
                energy[wk] += pivot.delta_energy(t, r, zt);
                for (x, c) in r.iter_mut().zip(pivot.col(t)) {
                    *x ^= *c;
                }
                zz[t >> 6] ^= 1 << (t & 63);
                theta_sum[wk] += pivot.theta[t];
                obs[wk] ^= pivot.cyc_obs[t];
            }
        }
    }
    let scale = 1.0 / (cfg.beta_steps as f64 * EN_ONE as f64);
    let mut pop = Population::default();
    let (mut rows_tmp, mut z_tmp) = (rows.clone(), z.clone());
    let mut anc = vec![0usize; nw];
    for b in 1..=cfg.beta_steps {
        // ---- reweight to beta_b
        for wk in 0..nw {
            acc[wk] += (energy[wk] - theta_sum[wk]) as i64;
        }
        // ---- adaptive systematic resampling (never after the terminal reweighting)
        if b < cfg.beta_steps && cfg.resample_q8 > 0 {
            let lo = *acc.iter().min().unwrap_or(&0);
            let (mut s1, mut s2, mut total) = (0u64, 0u64, 0u64);
            for &a in acc.iter() {
                let w = int_weight(tables, a - lo);
                total += w;
                s1 += w >> 16;
                s2 += (w >> 16) * (w >> 16);
            }
            if 256 * s1 * s1 < cfg.resample_q8 as u64 * nw as u64 * s2 {
                let mean: f64 = acc.iter().map(|&a| (-((a - lo) as f64) * scale).exp()).sum::<f64>() / nw as f64;
                pop.log_z_path += mean.ln() - lo as f64 * scale;
                let u0 = rng_u64(cfg.seed, shot_id, u32::MAX, b) % total;
                let (mut i, mut cum) = (0usize, int_weight(tables, acc[0] - lo));
                for (j, slot) in anc.iter_mut().enumerate() {
                    let target = u0 + j as u64 * total;
                    while cum * nw as u64 <= target {
                        i += 1;
                        cum += int_weight(tables, acc[i] - lo);
                    }
                    *slot = i;
                }
                rows_tmp.copy_from_slice(&rows);
                z_tmp.copy_from_slice(&z);
                let (e_old, t_old, o_old) = (energy.clone(), theta_sum.clone(), obs.clone());
                for (j, &a) in anc.iter().enumerate() {
                    rows[j * wr..(j + 1) * wr].copy_from_slice(&rows_tmp[a * wr..(a + 1) * wr]);
                    z[j * wz..(j + 1) * wz].copy_from_slice(&z_tmp[a * wz..(a + 1) * wz]);
                    energy[j] = e_old[a];
                    theta_sum[j] = t_old[a];
                    obs[j] = o_old[a];
                    acc[j] = 0;
                }
                pop.resamples += 1;
            }
        }
        // ---- deterministic-scan Metropolis at beta_b
        for wk in 0..nw {
            let (r, zz) = (&mut rows[wk * wr..(wk + 1) * wr], &mut z[wk * wz..(wk + 1) * wz]);
            for sweep in 0..cfg.sweeps {
                let base = (b * cfg.sweeps + sweep) << 12;
                for t in 0..n_free {
                    let zt = (zz[t >> 6] >> (t & 63)) & 1 == 1;
                    let de = pivot.delta_energy(t, r, zt);
                    let toggled = zt != ((pivot.zq[t >> 6] >> (t & 63)) & 1 == 1);
                    let dth = if toggled { -pivot.theta[t] } else { pivot.theta[t] };
                    let d = (b_total - b as i64) * dth as i64 + b as i64 * de as i64;
                    let accept = d <= 0 || {
                        (d as usize) < tables.metropolis.len()
                            && ((rng_u64(cfg.seed, shot_id, wk as u32, base | t as u32) >> 32) as u32)
                                < tables.metropolis[d as usize]
                    };
                    if accept {
                        for (x, c) in r.iter_mut().zip(pivot.col(t)) {
                            *x ^= *c;
                        }
                        zz[t >> 6] ^= 1 << (t & 63);
                        energy[wk] += de;
                        theta_sum[wk] += dth;
                        obs[wk] ^= pivot.cyc_obs[t];
                    }
                }
            }
        }
    }
    pop.walkers = (0..nw).map(|wk| Walker { acc: acc[wk], obs: obs[wk], energy: energy[wk] }).collect();
    pop
}

/// Marginal coset free-energy gaps, PSIS diagnostic and the HLE certificate from terminal walkers.
///
/// Every *decision* (joint coset, marginal bits) is taken on integer weights
/// `round(2^32 exp(-(acc - acc_min) / (B * EN_ONE)))`, so CPU and GPU agree exactly; the
/// reported free energies, ESS and `k_hat` are floating-point functions of the same data.
pub fn certify(pop: &Population, n_obs: usize, cfg: &SmcConfig, tables: &SmcTables) -> SmcResult {
    let walkers = &pop.walkers[..];
    let scale = 1.0 / (cfg.beta_steps as f64 * EN_ONE as f64);
    let lo = walkers.iter().map(|w| w.acc).min().unwrap_or(0);
    let wt: Vec<u64> = walkers.iter().map(|w| int_weight(tables, w.acc - lo)).collect();
    let total: u64 = wt.iter().sum();
    // joint coset masses: O(W^2) pairwise accumulation, first walker of a coset owns its mass
    let (mut mode_obs, mut mode_mass) = (0u64, 0u64);
    for (i, wi) in walkers.iter().enumerate() {
        if walkers[..i].iter().any(|wj| wj.obs == wi.obs) {
            continue;
        }
        let m: u64 = walkers.iter().zip(&wt).filter(|(wj, _)| wj.obs == wi.obs).map(|(_, &x)| x).sum();
        if i == 0 || m > mode_mass {
            mode_mass = m;
            mode_obs = wi.obs;
        }
    }
    let mut delta_f = Vec::with_capacity(n_obs);
    let mut obs = 0u64;
    let mut min_abs = f64::INFINITY;
    for j in 0..n_obs {
        let p1: u64 = walkers.iter().zip(&wt).filter(|(w, _)| (w.obs >> j) & 1 == 1).map(|(_, &x)| x).sum();
        let p0 = total - p1;
        let df = if p1 == 0 {
            DELTA_F_CAP
        } else if p0 == 0 {
            -DELTA_F_CAP
        } else {
            (-(p1 as f64 / p0 as f64).ln()).clamp(-DELTA_F_CAP, DELTA_F_CAP)
        };
        if p1 > p0 {
            obs |= 1 << j;
        }
        min_abs = min_abs.min(df.abs());
        delta_f.push(df);
    }
    let lw: Vec<f64> = walkers.iter().map(|w| -((w.acc - lo) as f64) * scale).collect();
    let khat = psis_khat(&lw);
    let best = walkers.iter().min_by_key(|w| w.energy).copied().unwrap_or_default();
    let sq: f64 = wt.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let mean_w: f64 = lw.iter().map(|&l| l.exp()).sum::<f64>() / walkers.len().max(1) as f64;
    SmcResult {
        delta_f,
        obs,
        khat,
        min_abs_df: min_abs,
        hle: khat > cfg.khat_max || min_abs < cfg.tau_cert,
        ess: if sq > 0.0 { (total as f64) * (total as f64) / sq } else { 0.0 },
        log_z_ratio: pop.log_z_path + mean_w.ln() - lo as f64 * scale,
        best_energy: best.energy,
        best_obs: best.obs,
        mode_obs,
        energy_gap: {
            let inside = walkers.iter().filter(|w| w.obs == mode_obs).map(|w| w.energy).min();
            let outside = walkers.iter().filter(|w| w.obs != mode_obs).map(|w| w.energy).min();
            match (inside, outside) {
                (Some(a), Some(b)) => b - a,
                _ => i32::MAX,
            }
        },
        mode_free_energy: if mode_mass > 0 { -(mode_mass as f64 / total as f64).ln() } else { DELTA_F_CAP },
        resamples: pop.resamples,
        walkers: walkers.to_vec(),
    }
}

pub fn run_smc(pivot: &Pivot, n_obs: usize, cfg: &SmcConfig, tables: &SmcTables, shot_id: u32) -> SmcResult {
    certify(&run_walkers(pivot, cfg, tables, shot_id), n_obs, cfg, tables)
}

/// Exact answers by enumeration of all `2^n_free` coset coordinates (small pivots only).
#[derive(Clone, Debug)]
pub struct ExactCosets {
    /// `ln Z_1`, `Z_1 = sum_z exp(-H_N(e(z)))`.
    pub log_z: f64,
    /// `ln Z_0` of the twisted reference.
    pub log_z_ref: f64,
    pub delta_f: Vec<f64>,
    /// Number of distinct logical cosets reachable from the pivot.
    pub cosets: usize,
    /// Exact maximum-likelihood coset (largest total weight).
    pub ml_obs: u64,
}

pub fn exact_cosets(pivot: &Pivot, n_obs: usize) -> ExactCosets {
    let n_free = pivot.n_free();
    assert!(n_free <= 24, "exact enumeration is limited to 2^24 states");
    let unit = EN_ONE as f64;
    let mut rows = pivot.e_piv.clone();
    let (mut energy, mut obs) = (pivot.en_piv, pivot.obs_piv);
    let mut gray = 0u64;
    let mut mass: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut p1 = vec![0.0f64; n_obs];
    let mut total = 0.0f64;
    let base = pivot.en_piv as f64 / unit;
    for i in 0u64..(1u64 << n_free) {
        if i > 0 {
            let t = i.trailing_zeros() as usize;
            let zt = (gray >> t) & 1 == 1;
            energy += pivot.delta_energy(t, &rows, zt);
            for (r, c) in rows.iter_mut().zip(pivot.col(t)) {
                *r ^= *c;
            }
            gray ^= 1 << t;
            obs ^= pivot.cyc_obs[t];
        }
        let w = (-(energy as f64 / unit - base)).exp();
        total += w;
        *mass.entry(obs).or_insert(0.0) += w;
        for (j, slot) in p1.iter_mut().enumerate() {
            if (obs >> j) & 1 == 1 {
                *slot += w;
            }
        }
    }
    let delta_f = p1
        .iter()
        .map(|&a| {
            let b = total - a;
            if a <= 0.0 {
                DELTA_F_CAP
            } else if b <= 0.0 {
                -DELTA_F_CAP
            } else {
                (-(a / b).ln()).clamp(-DELTA_F_CAP, DELTA_F_CAP)
            }
        })
        .collect();
    let ml_obs = mass.iter().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0))).map(|(k, _)| *k).unwrap_or(0);
    let log_z_ref: f64 = pivot.theta.iter().map(|&t| (-(t as f64) / unit).exp().ln_1p()).sum();
    ExactCosets { log_z: total.ln() - base, log_z_ref, delta_f, cosets: mass.len(), ml_obs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pivot::{build_pivot, PivotScratch};
    use qec_core::bb::css_from_checks;
    use qec_core::complex::ChainComplex;
    use qec_core::f2::SparseF2;
    use qec_core::fixed::{energy_from_llr, llr_from_prob, mix64};
    use qec_hodge_bp::TannerGraph;

    /// The `[[2 L^2, 2, L]]` toric code: a small, highly degenerate CSS code.
    pub(crate) fn toric(l: usize) -> TannerGraph {
        let idx = |x: usize, y: usize, dir: usize| (dir * l * l + (x % l) * l + (y % l)) as u32;
        let (mut hz, mut hx) = (Vec::new(), Vec::new());
        for x in 0..l {
            for y in 0..l {
                // vertex check (detects X errors): four incident edges
                hz.push(vec![idx(x, y, 0), idx(x + l - 1, y, 0), idx(x, y, 1), idx(x, y + l - 1, 1)]);
                // plaquette stabilizer: four boundary edges
                hx.push(vec![idx(x, y, 0), idx(x, y + 1, 0), idx(x, y, 1), idx(x + 1, y, 1)]);
            }
        }
        let n = 2 * l * l;
        let code = css_from_checks("toric", SparseF2::from_rows(n, &hx), SparseF2::from_rows(n, &hz)).unwrap();
        assert_eq!(code.k, 2);
        TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap())
    }

    fn uniform(seed: u64, i: usize) -> f64 {
        ((mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn instance(g: &TannerGraph, p: f64, seed: u64) -> (Vec<i32>, Vec<u64>, Vec<i32>) {
        // heterogeneous priors make the cosets genuinely unequal
        let probs: Vec<f64> = (0..g.n_var).map(|v| p * (0.5 + uniform(seed ^ 0xA5, v))).collect();
        let en: Vec<i32> = probs.iter().map(|&q| energy_from_llr(llr_from_prob(q))).collect();
        let mut e = vec![0u64; g.n_var.div_ceil(64)];
        for v in 0..g.n_var {
            if uniform(seed, v) < probs[v] {
                e[v >> 6] ^= 1 << (v & 63);
            }
        }
        let post: Vec<i32> = probs.iter().map(|&q| llr_from_prob(q)).collect();
        (en, g.syndrome_of(&e), post)
    }

    #[test]
    fn smc_matches_exact_coset_partition_functions() {
        let g = toric(3); // 18 qubits, 8 independent checks -> 2^10 coset coordinates, 2^2 cosets
        for resample_q8 in [0u32, 128] {
            exactness_run(&g, resample_q8);
        }
    }

    fn exactness_run(g: &TannerGraph, resample_q8: u32) {
        let cfg =
            SmcConfig { k_cols: g.n_var, walkers: 256, beta_steps: 64, sweeps: 2, resample_q8, ..SmcConfig::default() };
        let tables = SmcTables::new(cfg.beta_steps);
        let mut scratch = PivotScratch::default();
        let (mut worst_p, mut worst_z, mut agree, mut total) = (0.0f64, 0.0f64, 0, 0);
        for seed in 0..24u64 {
            let (en, syn, post) = instance(g, 0.12, seed);
            let pivot = build_pivot(g, &en, &post, &syn, cfg.k_cols, cfg.quench_sweeps, &mut scratch);
            assert!(pivot.solvable);
            assert_eq!(pivot.n_free(), 10);
            let exact = exact_cosets(&pivot, g.n_obs);
            assert_eq!(exact.cosets, 4, "all 2^k logical cosets are reachable");
            // Compare marginal coset probabilities P(l_j = 1) = 1 / (1 + exp(Delta F_j)); the
            // estimator is consistent, so independent replicas are averaged to shrink noise.
            let reps = 8;
            let mut p1 = vec![0.0f64; g.n_obs];
            let mut lz = 0.0;
            for rep in 0..reps {
                let res = run_smc(&pivot, g.n_obs, &cfg, &tables, (seed * 100 + rep) as u32);
                for j in 0..g.n_obs {
                    p1[j] += 1.0 / (1.0 + res.delta_f[j].exp()) / reps as f64;
                }
                lz += res.log_z_ratio.exp() / reps as f64;
                total += 1;
                // a bit may disagree with the exact marginal decision only on a near-tie
                let ok = (0..g.n_obs).all(|j| {
                    ((res.obs >> j) & 1 == 1) == (exact.delta_f[j] < 0.0) || exact.delta_f[j].abs() < 0.75
                });
                agree += ok as u32;
            }
            for j in 0..g.n_obs {
                worst_p = worst_p.max((p1[j] - 1.0 / (1.0 + exact.delta_f[j].exp())).abs());
            }
            worst_z = worst_z.max((lz.ln() - (exact.log_z - exact.log_z_ref)).abs());
        }
        eprintln!("[exact] resample_q8={resample_q8} worst marginal-probability error {worst_p:.4}, worst ln Z error {worst_z:.4}");
        assert!(worst_p < 0.05, "marginal coset probability error {worst_p}");
        assert!(worst_z < 0.08, "ln(Z_1 / Z_0) error {worst_z} nats");
        assert_eq!(agree, total, "decisions must match exact marginal ML except on near-ties");
    }

    #[test]
    fn walkers_are_a_pure_function_of_pivot_config_and_shot_id() {
        let g = toric(2);
        let cfg = SmcConfig { k_cols: g.n_var, walkers: 64, beta_steps: 16, ..SmcConfig::default() };
        let tables = SmcTables::new(cfg.beta_steps);
        let mut scratch = PivotScratch::default();
        let (en, syn, post) = instance(&g, 0.1, 3);
        let pivot = build_pivot(&g, &en, &post, &syn, cfg.k_cols, cfg.quench_sweeps, &mut scratch);
        let pop = run_walkers(&pivot, &cfg, &tables, 0);
        // determinism: a pure function of (pivot, cfg, shot id)
        assert_eq!(pop, run_walkers(&pivot, &cfg, &tables, 0));
        assert_ne!(pop, run_walkers(&pivot, &cfg, &tables, 1));
        let exact = exact_cosets(&pivot, g.n_obs);
        let res = certify(&pop, g.n_obs, &cfg, &tables);
        assert!(res.ess > 1.0 && res.khat.is_finite());
        assert!((res.log_z_ratio - (exact.log_z - exact.log_z_ref)).abs() < 0.5);
    }
}
