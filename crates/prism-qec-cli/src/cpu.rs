//! CPU reference pipeline: Hodge-BP fast path + coset SMC slow path over a shot file.

use crate::dataset::{Dataset, DecodeModel, OracleWeights};
use qec_neuro_dem::{run_reference_oracle, run_reference_with_weights, PipelineConfig, ShotRecord, StreamPlan};
use qec_smc_coset::ShotResult;
use serde::Serialize;
use std::time::Instant;

/// Which slow-path estimate becomes the reported logical word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
pub enum Decision {
    /// Bitwise sign of the marginal coset free-energy gaps.
    Marginal,
    /// Coset with the largest total walker weight.
    Mode,
    /// Coset of the minimum-energy walker.
    Best,
}

pub fn decided_obs(r: &ShotResult, decision: Decision) -> u64 {
    if !r.smc_ran || !r.solvable {
        return r.obs;
    }
    match decision {
        Decision::Marginal => r.marginal_obs,
        Decision::Mode => r.mode_obs,
        Decision::Best => r.best_obs,
    }
}

pub struct CpuRun {
    pub records: Vec<ShotRecord>,
    pub results: Vec<ShotResult>,
    pub latency_ns: Vec<u64>,
    pub wall_seconds: f64,
    /// Live prior LLRs of every stream after its last block (empty for oracle-weight runs).
    pub live_llr: Vec<i32>,
}

/// Decodes the planned shots in stream order on the CPU reference pipeline.
pub fn decode_streams(ds: &Dataset, model: &DecodeModel, plan: &StreamPlan, cfg: &PipelineConfig) -> CpuRun {
    let t0 = Instant::now();
    let (records, live_llr) = run_reference_with_weights(&model.graph, &model.layout, &ds.dets, plan, cfg);
    let wall_seconds = t0.elapsed().as_secs_f64();
    let results = records.iter().map(|r| r.result).collect();
    let latency_ns = records.iter().map(|r| r.latency_ns).collect();
    CpuRun { records, results, latency_ns, wall_seconds, live_llr }
}

/// Decodes the planned shots with the oracle's true drifted weights instead of the monitor.
pub fn decode_streams_oracle(
    ds: &Dataset,
    model: &DecodeModel,
    plan: &StreamPlan,
    cfg: &PipelineConfig,
    oracle: &OracleWeights,
) -> CpuRun {
    let t0 = Instant::now();
    let records = run_reference_oracle(&model.graph, &model.layout, &ds.dets, plan, cfg, |trace, block, llr| {
        oracle.apply(trace, block, &model.layout.p_nom, llr)
    });
    let wall_seconds = t0.elapsed().as_secs_f64();
    let results = records.iter().map(|r| r.result).collect();
    let latency_ns = records.iter().map(|r| r.latency_ns).collect();
    CpuRun { records, results, latency_ns, wall_seconds, live_llr: Vec::new() }
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct Percentiles {
    pub mean: f64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
    pub max: u64,
}

pub fn percentiles(values: &[u64]) -> Percentiles {
    if values.is_empty() {
        return Percentiles::default();
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let q = |f: f64| v[((f * v.len() as f64) as usize).min(v.len() - 1)];
    Percentiles {
        mean: v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64,
        p50: q(0.5),
        p90: q(0.9),
        p99: q(0.99),
        p999: q(0.999),
        max: v[v.len() - 1],
    }
}

/// Aggregate accuracy and work statistics of a decoded shot range.
#[derive(Clone, Debug, Serialize, Default)]
pub struct DecodeStats {
    pub shots: usize,
    pub logical_failures: usize,
    pub logical_error_rate: f64,
    pub bp_converged: usize,
    pub bp_converged_fraction: f64,
    /// Converged fast paths that were escalated into the slow path, and the accepted ones.
    pub fast_path_escalated: usize,
    pub fast_path_accepted: usize,
    /// Accepted fast paths whose logical class is wrong.
    pub fast_path_false_converged: usize,
    pub slow_path_shots: usize,
    pub unsolvable: usize,
    pub hle_flagged: usize,
    pub khat_rejected: usize,
    pub mean_bp_iterations: f64,
    pub mean_check_updates: f64,
    pub mean_rank: f64,
    pub mean_free_coordinates: f64,
    pub mean_ess: f64,
    pub mean_resamples: f64,
    /// Failures of the alternative slow-path decision rules on the same shots.
    pub failures_bp_hard: usize,
    pub failures_forest: usize,
    pub failures_quench: usize,
    pub failures_best_walker: usize,
    pub failures_coset_mode: usize,
    pub failures_marginal: usize,
    /// Failure-detection AUROC over slow-path shots for the HLE confidence and the heuristics.
    pub slow_path_failures: usize,
    pub auroc_hle: f64,
    pub auroc_gap_only: f64,
    pub auroc_khat_only: f64,
    pub auroc_coset_free_energy: f64,
    pub auroc_energy_gap: f64,
    pub auroc_bp_norm: f64,
}

pub fn stats(ds: &Dataset, results: &[ShotResult], decision: Decision, khat_max: f64, hle_mode: u32, n_chk: usize) -> DecodeStats {
    let mut st = DecodeStats { shots: results.len(), ..DecodeStats::default() };
    let (mut it, mut cu, mut rk, mut nf, mut ess, mut rs) = (0u64, 0u64, 0u64, 0u64, 0.0f64, 0u64);
    for (s, r) in results.iter().enumerate() {
        let truth = ds.truth(s);
        let alt = |o: u64| if r.smc_ran && r.solvable { o } else { r.obs };
        st.logical_failures += (decided_obs(r, decision) != truth) as usize;
        st.failures_bp_hard += (r.bp_obs != truth) as usize;
        st.failures_forest += (alt(r.piv_obs) != truth) as usize;
        st.failures_quench += (alt(r.q_obs) != truth) as usize;
        st.failures_best_walker += (alt(r.best_obs) != truth) as usize;
        st.failures_coset_mode += (alt(r.mode_obs) != truth) as usize;
        st.failures_marginal += (alt(r.marginal_obs) != truth) as usize;
        st.bp_converged += r.bp_converged as usize;
        st.fast_path_escalated += (r.esc_reason != 0) as usize;
        st.fast_path_accepted += r.fast_accepted() as usize;
        st.fast_path_false_converged += (r.fast_accepted() && r.obs != truth) as usize;
        it += r.bp_iters as u64;
        cu += r.check_updates;
        if r.smc_ran {
            st.slow_path_shots += 1;
            st.unsolvable += !r.solvable as usize;
            st.hle_flagged += r.hle as usize;
            st.khat_rejected += (r.solvable && r.khat > khat_max) as usize;
            rk += r.rank as u64;
            nf += r.n_free as u64;
            ess += r.ess;
            rs += r.resamples as u64;
        }
    }
    let slow_idx: Vec<usize> = (0..results.len()).filter(|&s| results[s].smc_ran && results[s].solvable).collect();
    let failed: Vec<bool> = slow_idx.iter().map(|&s| decided_obs(&results[s], decision) != ds.truth(s)).collect();
    let score = |f: &dyn Fn(&ShotResult) -> f64| -> f64 {
        qec_smc_coset::auroc(&slow_idx.iter().map(|&s| f(&results[s])).collect::<Vec<f64>>(), &failed)
    };
    st.slow_path_failures = failed.iter().filter(|&&f| f).count();
    st.auroc_hle = score(&|r| -r.herald_confidence(hle_mode, khat_max, n_chk));
    st.auroc_gap_only = score(&|r| -r.min_abs_df);
    st.auroc_khat_only = score(&|r| r.khat.min(1e6));
    st.auroc_coset_free_energy = score(&|r| r.mode_free_energy);
    st.auroc_energy_gap = score(&|r| -(r.energy_gap as f64));
    st.auroc_bp_norm = score(&|r| -(r.bp_min_abs as f64));
    let n = results.len().max(1) as f64;
    let slow = st.slow_path_shots.max(1) as f64;
    st.logical_error_rate = st.logical_failures as f64 / n;
    st.bp_converged_fraction = st.bp_converged as f64 / n;
    st.mean_bp_iterations = it as f64 / n;
    st.mean_check_updates = cu as f64 / n;
    st.mean_rank = rk as f64 / slow;
    st.mean_free_coordinates = nf as f64 / slow;
    st.mean_ess = ess / slow;
    st.mean_resamples = rs as f64 / slow;
    st
}

/// Failure counts before and after the burst onset of a trace-structured dataset, plus the
/// monitor's reaction.
#[derive(Clone, Debug, Serialize, Default)]
pub struct BurstStats {
    pub traces: usize,
    pub blocks_per_trace: usize,
    pub onset_block: usize,
    pub pre_onset_blocks: usize,
    pub pre_onset_failures: usize,
    pub post_onset_blocks: usize,
    pub post_onset_failures: usize,
    pub post_onset_logical_error_rate: f64,
    /// Blocks after the onset block until the first modified mechanism, median over traces
    /// (`None` if the monitor never reacted in more than half of them).
    pub median_blocks_to_first_surgery: Option<usize>,
    pub traces_with_surgery: usize,
    pub pre_onset_blocks_with_surgery: usize,
}

pub fn burst_stats(ds: &Dataset, plan: &StreamPlan, obs: &[u64], reports: Option<&[ShotRecord]>) -> BurstStats {
    let onset = ds.meta_usize("onset_block").unwrap_or(0);
    let mut st = BurstStats { traces: plan.streams, blocks_per_trace: plan.steps, onset_block: onset, ..Default::default() };
    let mut reaction = Vec::new();
    for tr in 0..plan.streams {
        let mut first = None;
        for blk in 0..plan.steps {
            let Some(s) = plan.shot(tr, blk) else { continue };
            let fail = (obs[s] != ds.truth(s)) as usize;
            if blk < onset {
                st.pre_onset_blocks += 1;
                st.pre_onset_failures += fail;
            } else if blk > onset {
                st.post_onset_blocks += 1;
                st.post_onset_failures += fail;
            }
            if let Some(recs) = reports {
                let touched = recs[s].report.modified_mechanisms > 0;
                if blk < onset {
                    st.pre_onset_blocks_with_surgery += touched as usize;
                } else if touched && first.is_none() {
                    first = Some(blk - onset);
                }
            }
        }
        if let Some(f) = first {
            st.traces_with_surgery += 1;
            reaction.push(f);
        }
    }
    st.post_onset_logical_error_rate = st.post_onset_failures as f64 / st.post_onset_blocks.max(1) as f64;
    reaction.sort_unstable();
    if reaction.len() * 2 > plan.streams {
        st.median_blocks_to_first_surgery = Some(reaction[reaction.len() / 2]);
    }
    st
}

/// BLAKE3 digest of every integer field of every record: equal digests mean two runs took
/// identical decisions and trajectories on every shot.
pub fn records_digest(results: &[ShotResult]) -> String {
    let mut h = blake3::Hasher::new();
    for r in results {
        for v in [
            r.obs,
            r.bp_obs,
            r.piv_obs,
            r.q_obs,
            r.best_obs,
            r.mode_obs,
            r.marginal_obs,
            r.damped_mask,
            r.bp_iters as u64,
            r.bp_unsat as u64,
            r.rank as u64,
            r.n_free as u64,
            r.piv_energy as u64,
            r.best_energy as u64,
            r.bp_min_abs as u64,
            r.resamples as u64,
            r.esc_reason as u64,
            r.esc_margin as u64,
            r.esc_gap as u64,
            (r.bp_converged as u64) | ((r.smc_ran as u64) << 1) | ((r.solvable as u64) << 2),
        ] {
            h.update(&v.to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

/// Writes one CSV row per shot (decisions, energies, certificate values, monitor reaction) for
/// offline failure analysis against the baseline outputs of the same shots.
pub fn dump_csv(path: &std::path::Path, ds: &Dataset, records: &[ShotRecord]) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "shot,truth,obs,bp_obs,piv_obs,q_obs,best_obs,mode_obs,marginal_obs,bp_converged,bp_iters,bp_unsat,esc_reason,\
         smc_ran,solvable,rank,n_free,piv_energy,best_energy,energy_gap,mode_free_energy,min_abs_df,khat,ess,bp_min_abs,\
         resamples,damped_steps,modified,hot_checks,esc_margin,esc_gap"
    )?;
    for (s, rec) in records.iter().enumerate() {
        let r = &rec.result;
        writeln!(
            w,
            "{s},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            ds.truth(s), r.obs, r.bp_obs, r.piv_obs, r.q_obs, r.best_obs, r.mode_obs, r.marginal_obs, r.bp_converged as u8,
            r.bp_iters, r.bp_unsat, r.esc_reason, r.smc_ran as u8, r.solvable as u8, r.rank, r.n_free, r.piv_energy,
            r.best_energy, r.energy_gap, r.mode_free_energy, r.min_abs_df, r.khat, r.ess, r.bp_min_abs, r.resamples,
            r.damped_steps, rec.report.modified_mechanisms, rec.report.hot_checks, r.esc_margin, r.esc_gap
        )?;
    }
    w.flush()
}
