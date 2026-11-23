//! The immutable decoder configurations.
//!
//! Every dimensionless solver parameter lives in one frozen constant whose SHA-256 identity is
//! recorded in every receipt.  [`CanonicalDecoderConfig::FROZEN_V3`] is the active one;
//! `FROZEN_V1` and `FROZEN_V2` are deprecated and stay reachable (same behaviour, same identity
//! hash as when they were frozen).  Development may inspect only datasets of the `CALIB_2026`
//! seed domain (`SMOKE_2026` shots only exercise the pipeline); datasets of the `BLIND_HOLDOUT_2026` domain are
//! decoded with the active constant and nothing else: there is no per-code, per-distance or
//! per-noise branch.  The one size that must follow the problem, the number of suspect
//! columns handed to the pivot, is a fixed multiple of the detector count.

use crate::pipeline::PipelineConfig;
use crate::stream::NeuroConfig;
use qec_core::complex::CycleSearch;
use qec_hodge_bp::BpConfig;
use qec_smc_coset::{DecoderConfig, EscalationConfig, SmcConfig};
use sha2::{Digest, Sha256};

pub const SEED_DOMAIN_CALIB: &str = "CALIB_2026";
pub const SEED_DOMAIN_BLIND: &str = "BLIND_HOLDOUT_2026";
pub const SEED_DOMAIN_SMOKE: &str = "SMOKE_2026";

/// All fields are integers so the canonical serialisation, and therefore the hash, is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalDecoderConfig {
    /// Configuration generation: 1 = `FROZEN_V1` (deprecated), 2 = `FROZEN_V2`.
    pub version: u32,
    // ---- fast path: damped min-sum
    pub t_max: u32,
    pub t_clean: u32,
    pub alpha_q8: i32,
    pub eta_q8: i32,
    pub eps_q12: i32,
    pub disorder_q8: i32,
    pub margin: i32,
    // ---- uncertainty-gated escalation
    pub tau_margin: i32,
    pub tau_gap: i32,
    // ---- chain complex
    pub cycle_weight: u32,
    pub cell_cap: u32,
    // ---- slow path: suspects per detector (Q8), floor, walkers, schedule
    pub k_cols_per_det_q8: u32,
    pub k_cols_min: u32,
    pub walkers: u32,
    pub beta_steps: u32,
    pub sweeps: u32,
    pub quench_sweeps: u32,
    pub resample_q8: u32,
    pub tau_cert_milli: u32,
    pub khat_max_milli: u32,
    // ---- online monitor
    pub hazard_inv: u32,
    pub hot_on_milli: u32,
    pub hot_off_milli: u32,
    pub n_min: u32,
    pub g_threshold: u32,
    pub g_threshold_hot: u32,
    pub count_cap: u32,
    pub mc_blocks: u32,
    // ---- fixed randomness
    pub disorder_seed: u64,
    pub smc_seed: u64,
    pub neuro_seed: u64,
    // ---- introduced by version 2 (part of the identity hash only when `version >= 2`)
    /// Iterations after which a converged fast path counts as late (V1: `t_clean`).
    pub t_late: u32,
    /// Reliability-ordered pair exploration of the pivot relaxation (V1: off).
    pub pair_lambda: u32,
    pub pair_rounds: u32,
    /// Herald functional: 0 = PSIS-gated free-energy gap (V1), 1 = coset mass split plus
    /// soft syndrome gap.
    pub hle_mode: u32,
    /// Escalation margin over the decoded faults and their gauge alternatives (1, V1) or over
    /// the decoded faults only (0).
    pub margin_gauge: u32,
    /// Monitor: k-point coincidence estimator for mechanisms with >= 3 detectors (V1: off),
    /// its window, the coincidences required from cold checks, and the erasure snap.
    pub coincidence: u32,
    pub coin_cap: u32,
    pub coin_min: u32,
    pub g_threshold_coin: u32,
    pub snap: u32,
    // ---- introduced by version 3 (part of the identity hash only when `version >= 3`)
    /// Full-column pivot: every fault enters the elimination (V1/V2: the suspect prefix).
    pub full_columns: u32,
    /// Free coordinates kept by the full-column pivot per detector (Q8) and their floor.
    pub k_active_per_det_q8: u32,
    pub k_active_min: u32,
}

impl CanonicalDecoderConfig {
    pub const FROZEN_V1: Self = Self {
        version: 1,
        t_late: 16,
        pair_lambda: 0,
        pair_rounds: 0,
        hle_mode: 0,
        margin_gauge: 1,
        coincidence: 0,
        coin_cap: 64,
        coin_min: 2,
        g_threshold_coin: 24,
        snap: 0,
        full_columns: 0,
        k_active_per_det_q8: 32,
        k_active_min: 64,
        t_max: 64,
        t_clean: 16,
        alpha_q8: 160,
        eta_q8: 64,
        eps_q12: 1024,
        disorder_q8: 256,
        margin: 32,
        tau_margin: 64,
        tau_gap: 1,
        cycle_weight: 3,
        cell_cap: 8,
        k_cols_per_det_q8: 256,
        k_cols_min: 128,
        walkers: 128,
        beta_steps: 32,
        sweeps: 1,
        quench_sweeps: 8,
        resample_q8: 128,
        tau_cert_milli: 1000,
        khat_max_milli: 700,
        hazard_inv: 100,
        hot_on_milli: 2000,
        hot_off_milli: 1250,
        n_min: 6,
        g_threshold: 24,
        g_threshold_hot: 4,
        count_cap: 255,
        mc_blocks: 4096,
        disorder_seed: 0x5052_4953_4D48_4447,
        smc_seed: 0x5052_4953_4D53_4D43,
        neuro_seed: 0x5052_4953_4D4E_4555,
    };

    /// Identity hash `FROZEN_V1` had when it was frozen on `CALIB_2026`.
    pub const FROZEN_V1_SHA256: &'static str = "5afd73a62875827958a8ba424adfe77687b1b76ba44d8e1d6e90e1e0e60858c9";

    /// Frozen on `CALIB_2026` (bb72 and surface_d5 at p = 0.002, bb72 drift) with `SMOKE_2026`
    /// cross-checks; differs from `FROZEN_V1` in:
    ///
    /// * escalation: the late-convergence trigger is off (`t_late = t_max`) and the margin
    ///   covers the decoded faults only (same accuracy, far fewer escalations);
    /// * herald: coset mass split plus soft syndrome gap instead of the PSIS-gated gap;
    /// * monitor: k-point coincidence estimator with a 255-sample window, activation from a
    ///   single coincidence at 14 nats, the erasure snap, and a 16x larger nominal reference.
    ///
    /// Pair exploration of the pivot relaxation stays off: it did not lower the logical
    /// error rate on any development set.
    pub const FROZEN_V2: Self = Self {
        version: 2,
        t_late: 64,
        margin_gauge: 0,
        hle_mode: 1,
        coincidence: 1,
        coin_cap: 255,
        coin_min: 1,
        g_threshold_coin: 14,
        snap: 1,
        mc_blocks: 65536,
        ..Self::FROZEN_V1
    };

    /// Identity hash `FROZEN_V2` had when it was frozen.
    pub const FROZEN_V2_SHA256: &'static str = "458f87d5d0e740eec75f65fd1ef472ae96b13d9da3236603a6e1b8e1de643f7b";

    /// `FROZEN_V2` with the full-column pivot: every fault mechanism enters the
    /// reliability-ordered elimination (exact OSD-0 warm start, always solvable) and the
    /// free coordinates are the non-pivot nullspace vectors of smallest exact gain, relaxed
    /// by single-column descent and pair moves over the 32 best columns.
    ///
    /// Frozen on `CALIB_2026` only (bb72, surface_d5 and hgp_hamming at p = 0.002,
    /// bb144_gross at p = 0.003, drift profile 2 on bb72 and bb144_gross).  Measured there:
    /// a larger free-coordinate budget (per-detector 64, floor 128 or 256) and a larger pair
    /// budget (64 or 128 columns, 16 moves) change nothing beyond noise, so the smallest
    /// budgets are kept; the escalation margin of 64 is confirmed against 32 on the same sets.
    pub const FROZEN_V3: Self = Self {
        version: 3,
        full_columns: 1,
        k_active_per_det_q8: 32,
        k_active_min: 64,
        pair_lambda: 32,
        pair_rounds: 8,
        ..Self::FROZEN_V2
    };

    /// The configuration every evaluation uses unless a deprecated one is requested.
    pub const ACTIVE: Self = Self::FROZEN_V3;

    pub fn config_id(&self) -> &'static str {
        match self.version {
            0 | 1 => "FROZEN_V1",
            2 => "FROZEN_V2",
            _ => "FROZEN_V3",
        }
    }

    /// Free coordinates of the full-column pivot: `ceil(n_det * k / 256)` rounded up to a
    /// multiple of 64, floored at `k_active_min`.
    pub fn k_active(&self, n_det: usize, n_err: usize) -> usize {
        let raw = (n_det * self.k_active_per_det_q8 as usize).div_ceil(256).max(self.k_active_min as usize);
        raw.div_ceil(64).saturating_mul(64).min(n_err)
    }

    /// Canonical serialisation: `key=value` lines in declaration order.  Version 1 serialises
    /// exactly the fields it was frozen with, so its identity hash never changes.
    pub fn canonical_text(&self) -> String {
        let fields: [(&str, i128); 31] = [
            ("t_max", self.t_max as i128),
            ("t_clean", self.t_clean as i128),
            ("alpha_q8", self.alpha_q8 as i128),
            ("eta_q8", self.eta_q8 as i128),
            ("eps_q12", self.eps_q12 as i128),
            ("disorder_q8", self.disorder_q8 as i128),
            ("margin", self.margin as i128),
            ("tau_margin", self.tau_margin as i128),
            ("tau_gap", self.tau_gap as i128),
            ("cycle_weight", self.cycle_weight as i128),
            ("cell_cap", self.cell_cap as i128),
            ("k_cols_per_det_q8", self.k_cols_per_det_q8 as i128),
            ("k_cols_min", self.k_cols_min as i128),
            ("walkers", self.walkers as i128),
            ("beta_steps", self.beta_steps as i128),
            ("sweeps", self.sweeps as i128),
            ("quench_sweeps", self.quench_sweeps as i128),
            ("resample_q8", self.resample_q8 as i128),
            ("tau_cert_milli", self.tau_cert_milli as i128),
            ("khat_max_milli", self.khat_max_milli as i128),
            ("hazard_inv", self.hazard_inv as i128),
            ("hot_on_milli", self.hot_on_milli as i128),
            ("hot_off_milli", self.hot_off_milli as i128),
            ("n_min", self.n_min as i128),
            ("g_threshold", self.g_threshold as i128),
            ("g_threshold_hot", self.g_threshold_hot as i128),
            ("count_cap", self.count_cap as i128),
            ("mc_blocks", self.mc_blocks as i128),
            ("disorder_seed", self.disorder_seed as i128),
            ("smc_seed", self.smc_seed as i128),
            ("neuro_seed", self.neuro_seed as i128),
        ];
        let mut out = String::from("prism.qec.v1.canonical_decoder_config\n");
        for (k, v) in fields {
            out.push_str(&format!("{k}={v}\n"));
        }
        if self.version >= 2 {
            let added: [(&str, i128); 11] = [
                ("version", self.version as i128),
                ("t_late", self.t_late as i128),
                ("pair_lambda", self.pair_lambda as i128),
                ("pair_rounds", self.pair_rounds as i128),
                ("hle_mode", self.hle_mode as i128),
                ("margin_gauge", self.margin_gauge as i128),
                ("coincidence", self.coincidence as i128),
                ("coin_cap", self.coin_cap as i128),
                ("coin_min", self.coin_min as i128),
                ("g_threshold_coin", self.g_threshold_coin as i128),
                ("snap", self.snap as i128),
            ];
            for (k, v) in added {
                out.push_str(&format!("{k}={v}\n"));
            }
        }
        if self.version >= 3 {
            let added: [(&str, i128); 3] = [
                ("full_columns", self.full_columns as i128),
                ("k_active_per_det_q8", self.k_active_per_det_q8 as i128),
                ("k_active_min", self.k_active_min as i128),
            ];
            for (k, v) in added {
                out.push_str(&format!("{k}={v}\n"));
            }
        }
        out
    }

    pub fn sha256(&self) -> String {
        let digest = Sha256::digest(self.canonical_text().as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Suspect columns for a model with `n_det` detectors and `n_err` fault mechanisms:
    /// `ceil(n_det * k / 256)` rounded up to a multiple of 64, floored at `k_cols_min`.
    pub fn k_cols(&self, n_det: usize, n_err: usize) -> usize {
        let raw = (n_det * self.k_cols_per_det_q8 as usize).div_ceil(256).max(self.k_cols_min as usize);
        raw.div_ceil(64).saturating_mul(64).min(n_err)
    }

    pub fn cycle_search(&self) -> CycleSearch {
        CycleSearch { max_weight: self.cycle_weight as usize, per_cell_cap: self.cell_cap as usize }
    }

    pub fn neuro(&self) -> NeuroConfig {
        NeuroConfig {
            hazard: 1.0 / self.hazard_inv as f64,
            hot_on: self.hot_on_milli as f64 / 1000.0,
            hot_off: self.hot_off_milli as f64 / 1000.0,
            n_min: self.n_min,
            g_threshold: self.g_threshold,
            g_threshold_hot: self.g_threshold_hot,
            count_cap: self.count_cap,
            mc_blocks: self.mc_blocks,
            seed: self.neuro_seed,
            bocpd_only: false,
            coincidence: self.coincidence != 0,
            coin_cap: self.coin_cap,
            coin_min: self.coin_min,
            g_threshold_coin: self.g_threshold_coin,
            snap: self.snap != 0,
        }
    }

    pub fn decoder(&self, n_det: usize, n_err: usize) -> DecoderConfig {
        DecoderConfig {
            bp: BpConfig {
                t_max: self.t_max,
                t_clean: self.t_clean,
                alpha_q8: self.alpha_q8,
                eta_q8: self.eta_q8,
                eps_q12: self.eps_q12,
                disorder_q8: self.disorder_q8,
                disorder_seed: self.disorder_seed,
                margin: self.margin,
                wavefront: true,
            },
            smc: SmcConfig {
                k_cols: if self.full_columns != 0 { self.k_active(n_det, n_err) } else { self.k_cols(n_det, n_err) },
                full_columns: self.full_columns != 0,
                walkers: self.walkers as usize,
                beta_steps: self.beta_steps,
                sweeps: self.sweeps,
                seed: self.smc_seed,
                quench_sweeps: self.quench_sweeps,
                pair_lambda: self.pair_lambda,
                pair_rounds: self.pair_rounds,
                resample_q8: self.resample_q8,
                tau_cert: self.tau_cert_milli as f64 / 1000.0,
                khat_max: self.khat_max_milli as f64 / 1000.0,
            },
            esc: EscalationConfig {
                enabled: true,
                tau_margin: self.tau_margin,
                tau_gap: self.tau_gap,
                t_late: self.t_late,
                margin_gauge: self.margin_gauge != 0,
            },
        }
    }

    /// Full pipeline configuration; `neuro_on = false` is the static-DEM ablation.
    pub fn pipeline(&self, n_det: usize, n_err: usize, neuro_on: bool) -> PipelineConfig {
        PipelineConfig { decoder: self.decoder(n_det, n_err), neuro: self.neuro(), neuro_on }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hash_is_stable_and_sensitive_to_every_field() {
        let base = CanonicalDecoderConfig::FROZEN_V1;
        assert_eq!(base.sha256(), CanonicalDecoderConfig::FROZEN_V1_SHA256);
        assert_eq!((base.config_id(), CanonicalDecoderConfig::ACTIVE.config_id()), ("FROZEN_V1", "FROZEN_V3"));
        let v2 = CanonicalDecoderConfig::FROZEN_V2;
        assert_eq!((v2.config_id(), v2.sha256().as_str()), ("FROZEN_V2", CanonicalDecoderConfig::FROZEN_V2_SHA256));
        let v3 = CanonicalDecoderConfig::FROZEN_V3;
        assert!(v3.sha256() != v2.sha256() && v3.sha256() != base.sha256());
        for f in 0..3 {
            let mut w = v3;
            match f {
                0 => w.full_columns ^= 1,
                1 => w.k_active_per_det_q8 += 1,
                _ => w.k_active_min += 1,
            }
            assert_ne!(w.sha256(), v3.sha256());
        }
        assert_eq!((v3.k_active(936, 8784), v3.k_active(252, 2232), v3.k_active(24, 40)), (128, 64, 40));
        assert!(v3.decoder(936, 8784).smc.full_columns && !v2.decoder(936, 8784).smc.full_columns);
        for f in 0..10 {
            let mut w = v2;
            match f {
                9 => w.g_threshold_coin += 1,
                8 => w.margin_gauge += 1,
                0 => w.t_late += 1,
                1 => w.pair_lambda += 1,
                2 => w.pair_rounds += 1,
                3 => w.hle_mode += 1,
                4 => w.coincidence += 1,
                5 => w.coin_cap += 1,
                6 => w.coin_min += 1,
                _ => w.snap += 1,
            }
            assert_ne!(w.sha256(), v2.sha256());
        }
        let mut variants = vec![base; 6];
        variants[0].eta_q8 += 1;
        variants[1].tau_gap += 1;
        variants[2].walkers += 1;
        variants[3].k_cols_per_det_q8 += 1;
        variants[4].g_threshold_hot += 1;
        variants[5].smc_seed ^= 1;
        for v in &variants {
            assert_ne!(v.sha256(), base.sha256());
        }
        assert_eq!(base.canonical_text().lines().count(), 32);
    }

    #[test]
    fn suspect_budget_follows_the_detector_count_only() {
        let c = CanonicalDecoderConfig::FROZEN_V1;
        assert_eq!(c.k_cols(252, 2232), 256);
        assert_eq!(c.k_cols(936, 8784), 960);
        assert_eq!(c.k_cols(24, 5000), 128);
        assert_eq!(c.k_cols(936, 500), 500);
        let d = c.decoder(252, 2232);
        assert_eq!((d.bp.t_max, d.bp.t_clean, d.smc.walkers), (c.t_max, c.t_clean, c.walkers as usize));
        assert!(c.pipeline(252, 2232, true).neuro_on && !c.pipeline(252, 2232, false).neuro_on);
    }
}
