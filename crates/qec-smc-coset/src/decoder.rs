//! The two-stage decoder: damped min-sum fast path, and the coset SMC slow path.
//!
//! `d1 e_hat == s` alone is **not** accepted as success.  A converged fast path is escalated
//! into the slow path, with its own hard decision as the pivot, whenever
//!
//! * (a) convergence was late (`t > T_late`; `FROZEN_V1` used `T_late = T_clean`, `FROZEN_V2`
//!   switches the trigger off: the damped phase resolving a gauge trap is not a risk),
//! * (b) the belief margin of the decoded error is small (`min |l_i| < tau_margin` over the
//!   decoded faults; `FROZEN_V1` also included the faults sharing a 2-cell with them, i.e.
//!   their gauge alternatives, whose indecision is harmless), or
//! * (c) a single 2-cell move `e_hat (+) d2 g` lowers the Nishimori energy or lands within
//!   `tau_gap` of it (a competing valley).
//!
//! Escalated shots hand the slow path the final beliefs as reliability order, so the decoded
//! faults lead the suspect list and the forest solution reproduces `e_hat` (whenever its
//! columns are independent).  Non-converged shots use the beliefs of the unperturbed prefix.

use crate::pivot::{build_pivot, build_pivot_full, PivotScratch};
use crate::smc::{run_smc, SmcConfig, SmcTables};
use qec_hodge_bp::{BpConfig, BpWork, HodgeBp, TannerGraph};

/// Escalation reasons (bit flags in [`ShotResult::esc_reason`]).
pub const ESC_LATE: u8 = 1;
pub const ESC_MARGIN: u8 = 2;
pub const ESC_VALLEY: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EscalationConfig {
    /// `false` accepts every converged fast path (ablation only).
    pub enabled: bool,
    /// Margin threshold `tau_margin` in BP units.
    pub tau_margin: i32,
    /// Competing-valley threshold `tau_gap` in SMC energy units.
    pub tau_gap: i32,
    /// Iteration count above which a converged fast path counts as late.
    pub t_late: u32,
    /// Whether the margin also covers the gauge alternatives of the decoded faults (the
    /// `FROZEN_V1` definition) or only the decoded faults themselves.
    pub margin_gauge: bool,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self { enabled: true, tau_margin: 96, tau_gap: 16, t_late: 16, margin_gauge: true }
    }
}

/// Value of [`ShotResult::esc_margin`] / [`ShotResult::esc_gap`] when the hard decision is
/// empty (nothing to be unsure about).
pub const ESC_NONE: i32 = i32::MAX;

/// Soft margins of a converged hard decision (`post < 0` are the decoded faults).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoftMargin {
    /// Smallest belief magnitude over the decoded faults and their gauge alternatives.
    pub margin: i32,
    /// Smallest Nishimori energy change of a single 2-cell move `e_hat (+) d2 g`.
    pub gap: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DecoderConfig {
    pub bp: BpConfig,
    pub smc: SmcConfig,
    pub esc: EscalationConfig,
}

/// Everything the decoder reports for one shot.  The SM120 pipeline fills the same record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShotResult {
    /// Predicted logical observable flips: the fast-path hard decision when it was accepted,
    /// otherwise the maximum-weight coset of the slow path.
    pub obs: u64,
    pub bp_converged: bool,
    pub bp_iters: u32,
    pub bp_unsat: u32,
    pub damped_steps: u32,
    pub damped_mask: u64,
    pub check_updates: u64,
    /// Observables of the BP hard decision (the fallback when no pivot exists).
    pub bp_obs: u64,
    /// Nonzero iff a converged fast path was escalated (`ESC_*` flags).
    pub esc_reason: u8,
    /// Soft margins of a converged fast path ([`ESC_NONE`] when not converged or empty).
    pub esc_margin: i32,
    pub esc_gap: i32,
    /// True iff the slow path ran (non-converged or escalated).
    pub smc_ran: bool,
    /// False iff the syndrome is outside the span of the suspect columns.
    pub solvable: bool,
    pub rank: u32,
    pub n_free: u32,
    /// Observables and energy of the forest (pivot) solution.
    pub piv_obs: u64,
    pub piv_energy: i32,
    pub khat: f64,
    pub min_abs_df: f64,
    pub ess: f64,
    /// Minimum-energy walker: its energy and logical word.
    pub best_energy: i32,
    pub best_obs: u64,
    /// Observables of the quenched forest solution and of the heaviest coset.
    pub q_obs: u64,
    pub mode_obs: u64,
    pub marginal_obs: u64,
    /// Heuristic heralds kept for head-to-head comparison with the HLE certificate.
    pub energy_gap: i32,
    pub mode_free_energy: f64,
    pub bp_min_abs: i32,
    pub resamples: u32,
    /// Heralded Logical Erasure certificate.
    pub hle: bool,
}

/// Separation between certified and Pareto-rejected shots in [`ShotResult::confidence`];
/// any value above the free-energy cap works, it only encodes the lexicographic order.
pub const PARETO_REJECT_OFFSET: f64 = 1.0e3;

impl ShotResult {
    /// True iff the fast path's own answer was reported (converged and not escalated).
    pub fn fast_accepted(&self) -> bool {
        self.bp_converged && !self.smc_ran
    }

    /// Confidence used to trace the post-selection frontier (larger = more trustworthy).
    /// The order is lexicographic: an accepted fast path is unheralded (`+inf`); certified
    /// slow-path shots rank by their smallest marginal coset free-energy gap; Pareto-rejected
    /// shots (`k_hat > khat_max`) rank below every certified shot, again by gap; shots without
    /// a pivot rank last.
    pub fn confidence(&self, khat_max: f64) -> f64 {
        if !self.smc_ran {
            f64::INFINITY
        } else if !self.solvable {
            -2.0 * PARETO_REJECT_OFFSET
        } else if self.khat > khat_max {
            self.min_abs_df - PARETO_REJECT_OFFSET
        } else {
            self.min_abs_df
        }
    }

    /// Logical coset posterior mass outside the decided class, `1 - W(L_MAP) / sum_L W(L)`
    /// over the terminal walker population (0 when the slow path did not run).
    pub fn mass_split(&self) -> f64 {
        if self.smc_ran && self.solvable {
            (1.0 - (-self.mode_free_energy).exp()).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// `FROZEN_V2` herald: risk = coset mass split + the fraction of the `n_chk` checks the
    /// fast path left unsatisfied (the soft syndrome the walkers had to close); ties are
    /// broken by the walkers' energy gap between the decided coset and the best other one.
    /// The Pareto `k_hat` is not used: measured, it carries no information about failure.
    /// Returned as a confidence (larger = more trustworthy), `+inf` for an accepted fast path.
    pub fn confidence_v2(&self, n_chk: usize) -> f64 {
        if !self.smc_ran {
            f64::INFINITY
        } else if !self.solvable {
            -2.0 * PARETO_REJECT_OFFSET
        } else {
            let risk = self.mass_split() + self.bp_unsat as f64 / n_chk.max(1) as f64;
            -risk + 1e-9 * self.energy_gap.clamp(0, 1 << 20) as f64
        }
    }

    /// `FROZEN_V4` herald: the `FROZEN_V2` risk plus the Nishimori weight of the decoded
    /// solution in nats per check.  All three terms are dimensionless and enter with unit
    /// weight: coset mass outside the decided class, fraction of checks the fast path left
    /// unsatisfied, solution weight per check.
    pub fn confidence_v4(&self, n_chk: usize) -> f64 {
        if !self.smc_ran {
            f64::INFINITY
        } else if !self.solvable {
            -2.0 * PARETO_REJECT_OFFSET
        } else {
            let per_chk = n_chk.max(1) as f64;
            let weight = self.best_energy as f64 / qec_core::fixed::EN_ONE as f64 / per_chk;
            let risk = self.mass_split() + self.bp_unsat as f64 / per_chk + weight;
            -risk + 1e-9 * self.energy_gap.clamp(0, 1 << 20) as f64
        }
    }

    /// Herald confidence under functional `mode` (0: `FROZEN_V1`, 1: `FROZEN_V2` / `V3`,
    /// otherwise `FROZEN_V4`).
    pub fn herald_confidence(&self, mode: u32, khat_max: f64, n_chk: usize) -> f64 {
        match mode {
            0 => self.confidence(khat_max),
            1 => self.confidence_v2(n_chk),
            _ => self.confidence_v4(n_chk),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DecodeWork {
    pub bp: BpWork,
    pub pivot: PivotScratch,
}

pub struct CosetDecoder<'g> {
    pub graph: &'g TannerGraph,
    pub cfg: DecoderConfig,
    pub bp: HodgeBp<'g>,
    pub tables: SmcTables,
}

/// Soft margins behind escalation triggers (b) and (c) for a converged hard decision.
/// `gauge` extends the margin to the faults sharing a 2-cell with a decoded fault.
pub fn soft_margin(g: &TannerGraph, post: &[i32], en: &[i32], gauge: bool) -> SoftMargin {
    let mut m = SoftMargin { margin: ESC_NONE, gap: ESC_NONE };
    for v in 0..g.n_var {
        if post[v] >= 0 {
            continue;
        }
        m.margin = m.margin.min(post[v].abs());
        for &f in g.var_cells(v) {
            let mut delta = 0i32;
            for &u in g.cell_vars(f as usize) {
                let u = u as usize;
                if gauge {
                    m.margin = m.margin.min(post[u].abs());
                }
                delta += if post[u] < 0 { -en[u] } else { en[u] };
            }
            m.gap = m.gap.min(delta);
        }
    }
    m
}

/// Escalation triggers (b) and (c) from the soft margins.
pub fn escalation_flags(m: SoftMargin, esc: &EscalationConfig) -> u8 {
    (if m.margin < esc.tau_margin { ESC_MARGIN } else { 0 }) | (if m.gap < esc.tau_gap { ESC_VALLEY } else { 0 })
}

impl<'g> CosetDecoder<'g> {
    pub fn new(graph: &'g TannerGraph, cfg: DecoderConfig) -> Self {
        Self { graph, cfg, bp: HodgeBp::new(graph, cfg.bp), tables: SmcTables::new(cfg.smc.beta_steps) }
    }

    /// Decodes one shot.  `llr` are prior LLRs in BP units, `en` the Nishimori weights in SMC
    /// units (both derived from the same fault probabilities), `syndrome` is bit-packed.
    pub fn decode(&self, work: &mut DecodeWork, llr: &[i32], en: &[i32], syndrome: &[u64], shot_id: u32) -> ShotResult {
        let g = self.graph;
        let out = self.bp.decode(&mut work.bp, llr, syndrome);
        let bp_obs = work.bp.post.iter().enumerate().filter(|(_, &p)| p < 0).fold(0u64, |a, (v, _)| a ^ g.obs[v]);
        let mut res = ShotResult {
            obs: bp_obs,
            bp_converged: out.converged,
            bp_iters: out.iterations,
            bp_unsat: out.unsat,
            damped_steps: out.damped_steps,
            damped_mask: out.damped_mask,
            check_updates: out.check_updates,
            bp_obs,
            solvable: true,
            min_abs_df: f64::INFINITY,
            esc_margin: ESC_NONE,
            esc_gap: ESC_NONE,
            ..ShotResult::default()
        };
        if out.converged {
            let m = soft_margin(g, &work.bp.post, en, self.cfg.esc.margin_gauge);
            (res.esc_margin, res.esc_gap) = (m.margin, m.gap);
            if self.cfg.esc.enabled {
                if out.iterations > self.cfg.esc.t_late {
                    res.esc_reason |= ESC_LATE;
                }
                res.esc_reason |= escalation_flags(m, &self.cfg.esc);
            }
            if res.esc_reason == 0 {
                return res;
            }
        }
        res.smc_ran = true;
        // reliability order: the final beliefs for an escalated fast path (its decoded faults
        // lead the suspect list), the unperturbed-prefix beliefs otherwise
        let order: &[i32] = if out.converged { &work.bp.post } else { &work.bp.snap };
        res.bp_min_abs = order.iter().map(|x| x.abs()).min().unwrap_or(0);
        let smc_cfg = &self.cfg.smc;
        let pivot = if smc_cfg.full_columns {
            build_pivot_full(g, en, order, syndrome, smc_cfg.k_cols, smc_cfg.relax(), &mut work.pivot)
        } else {
            build_pivot(g, en, order, syndrome, smc_cfg.k_cols, smc_cfg.relax(), &mut work.pivot)
        };
        res.rank = pivot.rank as u32;
        res.n_free = pivot.n_free() as u32;
        res.solvable = pivot.solvable;
        if !pivot.solvable {
            res.hle = true;
            res.min_abs_df = 0.0;
            res.khat = f64::INFINITY;
            return res;
        }
        res.piv_obs = pivot.obs_piv;
        res.piv_energy = pivot.en_piv;
        res.q_obs = pivot.obs_q;
        let smc = run_smc(&pivot, g.n_obs, &self.cfg.smc, &self.tables, shot_id);
        // decision: the coset of largest total weight (smallest coset free energy)
        res.obs = smc.mode_obs;
        res.khat = smc.khat;
        res.min_abs_df = smc.min_abs_df;
        res.ess = smc.ess;
        res.best_energy = smc.best_energy;
        res.best_obs = smc.best_obs;
        res.mode_obs = smc.mode_obs;
        res.marginal_obs = smc.obs;
        res.energy_gap = smc.energy_gap;
        res.mode_free_energy = smc.mode_free_energy;
        res.resamples = smc.resamples;
        res.hle = smc.hle;
        res
    }
}

/// Area under the ROC curve of `score` as a detector of `positive` (higher score = more
/// likely positive); ties count one half.
pub fn auroc(score: &[f64], positive: &[bool]) -> f64 {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
    let n_pos = positive.iter().filter(|&&x| x).count() as f64;
    let n_neg = positive.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j < order.len() && score[order[j]] == score[order[i]] {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for &idx in &order[i..j] {
            if positive[idx] {
                rank_sum += avg_rank;
            }
        }
        i = j;
    }
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::bb::bb72;
    use qec_core::complex::ChainComplex;
    use qec_core::dem::Dem;
    use qec_core::fixed::{energy_from_llr, llr_from_prob, mix64};

    fn uniform(seed: u64, i: usize) -> f64 {
        ((mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    #[test]
    fn auroc_known_values() {
        assert_eq!(auroc(&[0.1, 0.2, 0.8, 0.9], &[false, false, true, true]), 1.0);
        assert_eq!(auroc(&[0.9, 0.8, 0.2, 0.1], &[false, false, true, true]), 0.0);
        assert_eq!(auroc(&[0.5, 0.5, 0.5, 0.5], &[false, true, false, true]), 0.5);
    }

    /// On a `d = 6` code a weight-3 fault and its weight-3 logical complement satisfy the same
    /// syndrome, so "syndrome matched" cannot certify the logical class.  Escalation must
    /// remove (almost all of) the fast-path false convergences without making the overall
    /// decoder worse.
    #[test]
    fn escalation_removes_fast_path_false_convergence() {
        let code = bb72().build().unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap());
        let p = 0.03;
        let llr = vec![llr_from_prob(p); g.n_var];
        let en: Vec<i32> = llr.iter().map(|&l| energy_from_llr(l)).collect();
        let smc = SmcConfig { k_cols: g.n_var, walkers: 64, beta_steps: 16, ..SmcConfig::default() };
        let gated = DecoderConfig { bp: BpConfig::default(), smc, esc: EscalationConfig::default() };
        let bare = DecoderConfig { esc: EscalationConfig { enabled: false, ..EscalationConfig::default() }, ..gated };
        let (dec_gated, dec_bare) = (CosetDecoder::new(&g, gated), CosetDecoder::new(&g, bare));
        let mut work = DecodeWork::default();
        let (mut bare_false, mut gated_false, mut bare_fail, mut gated_fail, mut escalated) = (0, 0, 0, 0, 0);
        let shots = 3000u32;
        for shot in 0..shots {
            let mut e = vec![0u64; g.n_var.div_ceil(64)];
            for v in 0..g.n_var {
                if uniform(shot as u64 + 77, v) < p {
                    e[v >> 6] ^= 1 << (v & 63);
                }
            }
            let syn = g.syndrome_of(&e);
            let truth = g.observables_of(&e);
            let a = dec_bare.decode(&mut work, &llr, &en, &syn, shot);
            let b = dec_gated.decode(&mut work, &llr, &en, &syn, shot);
            bare_false += (a.fast_accepted() && a.obs != truth) as u32;
            gated_false += (b.fast_accepted() && b.obs != truth) as u32;
            bare_fail += (a.obs != truth) as u32;
            gated_fail += (b.obs != truth) as u32;
            escalated += (b.esc_reason != 0) as u32;
        }
        eprintln!(
            "[escalation] shots={shots} escalated={escalated} false-converged bare={bare_false} gated={gated_false}; failures bare={bare_fail} gated={gated_fail}"
        );
        assert!(bare_false >= 10, "the ungated fast path must exhibit false convergence ({bare_false})");
        assert!(gated_false * 5 <= bare_false, "escalation left {gated_false} of {bare_false} false convergences");
        // these shots are intrinsically ambiguous (a weight-3 fault and its weight-3 logical
        // complement are equally likely), so escalation relocates rather than removes them
        assert!(gated_fail <= bare_fail + bare_fail / 10 + 2, "escalation cost accuracy: {gated_fail} vs {bare_fail}");
    }

    /// Head-to-head of failure heralds on a BB72 space-time (phenomenological) detector error
    /// model, restricted to shots on which the fast path stays frustrated.
    ///
    /// Heralds compared as detectors of logical failure (AUROC):
    /// * certificate gap: the smallest marginal coset free-energy gap `min_j |Delta F_j|`,
    /// * BP gap norm: smallest belief magnitude `min_v |l_v|` (soft-output heuristic),
    /// * minimum-weight gap: best energy outside the decided coset minus best energy inside,
    /// * PSIS `k_hat` alone.
    ///
    /// Measured facts asserted here: the free-energy gap is an informative herald and
    /// dominates the BP gap norm; it is statistically indistinguishable from the
    /// minimum-weight gap; HLE-flagged shots are enriched in failures.  The Pareto `k_hat` on
    /// its own is printed, not asserted: on every dataset measured so far it carries almost no
    /// information about logical failure (AUROC ~ 0.5).
    #[test]
    fn hle_certificate_versus_heuristic_gap_norms() {
        let code = bb72().build().unwrap();
        let (dem, cells) = Dem::phenomenological(&code, 3, 0.02, 0.02).unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_dem_cells(&dem, &cells).unwrap());
        let llr: Vec<i32> = dem.probs.iter().map(|&p| llr_from_prob(p)).collect();
        let en: Vec<i32> = llr.iter().map(|&l| energy_from_llr(l)).collect();
        let cfg = DecoderConfig {
            bp: BpConfig::plain(32, 160),
            smc: SmcConfig { k_cols: 256, walkers: 128, beta_steps: 32, ..SmcConfig::default() },
            esc: EscalationConfig { enabled: false, ..EscalationConfig::default() },
        };
        let dec = CosetDecoder::new(&g, cfg);
        let mut work = DecodeWork::default();
        let (mut s_gap, mut s_bp, mut s_mw, mut s_k, mut failed, mut flagged) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut forest_fail, mut unsolvable) = (0u32, 0u32);
        for shot in 0..6000u32 {
            let mut e = vec![0u64; g.n_var.div_ceil(64)];
            for v in 0..g.n_var {
                if uniform(shot as u64 + 1, v) < dem.probs[v] {
                    e[v >> 6] ^= 1 << (v & 63);
                }
            }
            let syn = g.syndrome_of(&e);
            let truth = g.observables_of(&e);
            let res = dec.decode(&mut work, &llr, &en, &syn, shot);
            if !res.smc_ran {
                continue;
            }
            if !res.solvable {
                unsolvable += 1;
                continue;
            }
            failed.push(res.mode_obs != truth);
            flagged.push(res.hle);
            forest_fail += (res.piv_obs != truth) as u32;
            // "higher score = more likely failed"
            s_gap.push(-res.min_abs_df);
            s_bp.push(-(res.bp_min_abs as f64));
            s_mw.push(-(res.energy_gap as f64));
            s_k.push(res.khat.min(1e6));
        }
        let n = failed.len();
        let n_fail = failed.iter().filter(|&&f| f).count();
        let (a_gap, a_bp, a_mw, a_k) =
            (auroc(&s_gap, &failed), auroc(&s_bp, &failed), auroc(&s_mw, &failed), auroc(&s_k, &failed));
        let n_flag = flagged.iter().filter(|&&f| f).count();
        let fail_flag = failed.iter().zip(&flagged).filter(|(f, h)| **f && **h).count();
        let rate_flag = fail_flag as f64 / n_flag.max(1) as f64;
        let rate_clear = (n_fail - fail_flag) as f64 / (n - n_flag).max(1) as f64;
        eprintln!(
            "[hle] slow_path={n} unsolvable={unsolvable} failures={n_fail} forest_failures={forest_fail} flagged={n_flag} \
             fail_rate(flagged)={rate_flag:.4} fail_rate(clear)={rate_clear:.4} AUROC gap={a_gap:.4} bp_norm={a_bp:.4} \
             min_weight_gap={a_mw:.4} khat={a_k:.4}"
        );
        assert!(n > 300 && n_fail > 30, "need a meaningful sample: {n} shots, {n_fail} failures");
        assert!((n_fail as u32) < forest_fail, "coset decision must beat the forest (OSD-0) solution");
        assert!(a_gap > 0.7, "the free-energy gap must be an informative herald: {a_gap}");
        assert!(a_gap > a_bp + 0.1, "free-energy gap {a_gap} vs BP gap norm {a_bp}");
        assert!((a_gap - a_mw).abs() < 0.06, "free-energy gap {a_gap} vs minimum-weight gap {a_mw}");
        assert!(rate_flag > 1.3 * rate_clear, "HLE-flagged shots must be enriched in failures");
    }
}
