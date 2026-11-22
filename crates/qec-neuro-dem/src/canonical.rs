//! The single immutable decoder configuration.
//!
//! Every dimensionless solver parameter lives in [`CanonicalDecoderConfig::FROZEN_V1`]; its
//! SHA-256 identity is recorded in every receipt.  Calibration may inspect only datasets of
//! the `CALIB_2026` seed domain; datasets of the `BLIND_HOLDOUT_2026` domain are decoded with
//! this constant and nothing else: there is no per-code, per-distance or per-noise branch.
//! The one size that must follow the problem, the number of suspect columns handed to the
//! pivot, is a fixed multiple of the detector count.

use crate::pipeline::PipelineConfig;
use crate::stream::NeuroConfig;
use qec_core::complex::CycleSearch;
use qec_hodge_bp::BpConfig;
use qec_smc_coset::{DecoderConfig, EscalationConfig, SmcConfig};
use sha2::{Digest, Sha256};

pub const SEED_DOMAIN_CALIB: &str = "CALIB_2026";
pub const SEED_DOMAIN_BLIND: &str = "BLIND_HOLDOUT_2026";

/// All fields are integers so the canonical serialisation, and therefore the hash, is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalDecoderConfig {
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
}

impl CanonicalDecoderConfig {
    pub const CONFIG_ID: &'static str = "FROZEN_V1";

    pub const FROZEN_V1: Self = Self {
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

    /// Canonical serialisation: `key=value` lines in declaration order.
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
                k_cols: self.k_cols(n_det, n_err),
                walkers: self.walkers as usize,
                beta_steps: self.beta_steps,
                sweeps: self.sweeps,
                seed: self.smc_seed,
                quench_sweeps: self.quench_sweeps,
                resample_q8: self.resample_q8,
                tau_cert: self.tau_cert_milli as f64 / 1000.0,
                khat_max: self.khat_max_milli as f64 / 1000.0,
            },
            esc: EscalationConfig { enabled: true, tau_margin: self.tau_margin, tau_gap: self.tau_gap },
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
        assert_eq!(base.sha256().len(), 64);
        assert_eq!(base.sha256(), CanonicalDecoderConfig::FROZEN_V1.sha256());
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
