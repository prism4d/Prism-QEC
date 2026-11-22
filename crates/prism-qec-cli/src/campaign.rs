//! The blind-holdout campaign: every dataset below a root is decoded on the RTX 5080 under
//! the frozen configuration on the exact shots the BP-OSD / BP-LSD baselines decoded, compared pairwise,
//! and summarised in `qec_frontier_receipt.json` (then sealed).  Nothing in the receipt is
//! entered by hand: every number is computed here from the shot files and decoder outputs.

use crate::dataset::{Dataset, DecodeModel, OracleWeights};
use crate::stats::{clopper_pearson, mcnemar_exact, mcnemar_worse};
use anyhow::{bail, ensure, Context, Result};
use qec_core::bb::spec_by_name;
use qec_core::dem::ShotBits;
use qec_core::fixed::PRIOR_CLIP;
use qec_core::merkle::write_seal;
use qec_cuda::{record_mismatch, GpuDecoder, GpuTelemetry, LaunchMode};
use qec_neuro_dem::{
    run_reference, run_reference_oracle, run_reference_with_weights, CanonicalDecoderConfig, PipelineConfig, ShotRecord,
    StreamPlan, SEED_DOMAIN_BLIND,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const RECEIPT_FILE: &str = "qec_frontier_receipt.json";
const ABORT_BUDGETS: [f64; 4] = [0.001, 0.005, 0.01, 0.02];

pub struct CampaignArgs {
    pub root: PathBuf,
    /// Shots per dataset re-decoded on the CPU reference and compared record for record.
    pub parity_shots: usize,
    /// Shots decoded one launch at a time for the latency distribution of the headline cell.
    pub latency_shots: usize,
    /// Latency shots for every other stationary cell at the headline noise point.
    pub latency_shots_other: usize,
    pub streams: usize,
    pub sealed_utc: String,
    /// Explicit operator release of the blind-holdout domain.
    pub release_blind: bool,
    /// The frozen configuration under evaluation.
    pub frozen: CanonicalDecoderConfig,
}

/// Baseline outputs of one decoder on one dataset.
struct Baseline {
    shots: usize,
    pred: Vec<u64>,
    lat_ns: Vec<u64>,
    margin: Vec<f32>,
    weight: Vec<f32>,
    bp_conv: Vec<u8>,
    summary: Value,
}

fn read_exact<T: Copy + Default>(path: &Path, n: usize) -> Result<Vec<T>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let size = std::mem::size_of::<T>();
    ensure!(bytes.len() == n * size, "{}: {} bytes, expected {}", path.display(), bytes.len(), n * size);
    let mut out = vec![T::default(); n];
    // SAFETY: `out` owns `n * size` initialised bytes of plain-old-data `T`; lengths checked above.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len()) };
    Ok(out)
}

fn load_baseline(ds_dir: &Path, decoder: &str, n_obs: usize) -> Result<Baseline> {
    let base = ds_dir.join("baselines");
    let summary: Value = serde_json::from_str(
        &std::fs::read_to_string(base.join(format!("{decoder}.summary.json")))
            .with_context(|| format!("{decoder} baseline of {} is not complete", ds_dir.display()))?,
    )?;
    let shots = summary["shots"].as_u64().context("baseline summary without shots")? as usize;
    let pred = ShotBits::load_b8(&base.join(format!("{decoder}.pred.b8")), n_obs)?;
    ensure!(pred.shots == shots, "{decoder} prediction file has {} shots, summary says {shots}", pred.shots);
    Ok(Baseline {
        shots,
        pred: (0..shots).map(|s| pred.shot(s)[0]).collect(),
        lat_ns: read_exact(&base.join(format!("{decoder}.lat_ns.u64")), shots)?,
        margin: read_exact(&base.join(format!("{decoder}.margin.f32")), shots)?,
        weight: read_exact(&base.join(format!("{decoder}.weight.f32")), shots)?,
        bp_conv: read_exact(&base.join(format!("{decoder}.bpconv.u8")), shots)?,
        summary,
    })
}

fn ci_json(k: usize, n: usize) -> Value {
    let (lo, hi) = clopper_pearson(k as u64, n as u64, 0.95);
    json!([lo, hi])
}

fn percentile_us(sorted_ns: &[u64], q: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    sorted_ns[((q * sorted_ns.len() as f64) as usize).min(sorted_ns.len() - 1)] as f64 / 1e3
}

fn latency_json(ns: &[u64]) -> Value {
    let mut v = ns.to_vec();
    v.sort_unstable();
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64 / 1e3;
    json!({
        "shots": v.len(), "mean_us": mean, "p50_us": percentile_us(&v, 0.5), "p90_us": percentile_us(&v, 0.9),
        "p99_us": percentile_us(&v, 0.99), "p999_us": percentile_us(&v, 0.999),
        "max_us": v.last().copied().unwrap_or(0) as f64 / 1e3,
    })
}

/// Retained logical error rate after aborting the `budget` fraction of shots with the lowest
/// confidence (ties broken by shot index).
fn retained_ler(confidence: &[f64], failed: &[bool], budget: f64) -> (usize, usize, f64) {
    let n = confidence.len();
    let abort = (budget * n as f64).ceil() as usize;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| confidence[a].total_cmp(&confidence[b]).then(a.cmp(&b)));
    let kept = &order[abort.min(n)..];
    let errors = kept.iter().filter(|&&s| failed[s]).count();
    (abort.min(n), errors, errors as f64 / kept.len().max(1) as f64)
}

struct Decoded {
    records: Vec<ShotRecord>,
    telemetry: GpuTelemetry,
    parity_shots: usize,
    parity_mismatches: usize,
    first_mismatch: Option<String>,
}

fn decode_with_parity(
    gpu: &mut GpuDecoder,
    ds: &Dataset,
    model: &DecodeModel,
    cfg: &PipelineConfig,
    plan: &StreamPlan,
    parity_shots: usize,
) -> Result<Decoded> {
    let (records, telemetry) = gpu.decode(&model.graph, &model.layout, cfg, &ds.dets, plan, LaunchMode::Batch)?;
    // CPU reference on a prefix that shares the stream histories of the full run
    let prefix = if plan.stream_major {
        StreamPlan::traces((parity_shots / plan.steps).max(1).min(plan.streams) * plan.steps, plan.steps)
    } else {
        StreamPlan::interleaved(((parity_shots / plan.streams).max(1) * plan.streams).min(plan.n_shots), plan.streams)
    };
    let cpu = run_reference(&model.graph, &model.layout, &ds.dets, &prefix, cfg);
    let mut mismatches = 0;
    let mut first = None;
    for (shot, c) in cpu.iter().enumerate() {
        if let Some(diff) = record_mismatch(c, &records[shot], 1e-6) {
            mismatches += 1;
            first.get_or_insert(format!("shot {shot}: {diff}"));
        }
    }
    Ok(Decoded { records, telemetry, parity_shots: cpu.len(), parity_mismatches: mismatches, first_mismatch: first })
}

/// Herald functional of the configuration under evaluation, with the model size it needs.
#[derive(Clone, Copy)]
struct Herald {
    mode: u32,
    n_chk: usize,
}

impl Herald {
    fn confidence(&self, r: &ShotRecord) -> f64 {
        r.result.herald_confidence(self.mode, 0.7, self.n_chk)
    }
}

fn write_predictions(ds_dir: &Path, tag: &str, n_obs: usize, records: &[ShotRecord], herald: Herald) -> Result<()> {
    let dir = ds_dir.join("prism");
    std::fs::create_dir_all(&dir)?;
    let mut bits = ShotBits::zeros(records.len(), n_obs);
    for (s, r) in records.iter().enumerate() {
        bits.shot_mut(s)[0] = r.result.obs;
    }
    std::fs::write(dir.join(format!("{tag}.pred.b8")), bits.to_b8_bytes())?;
    let conf: Vec<u8> = records.iter().flat_map(|r| (herald.confidence(r) as f32).to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{tag}.confidence.f32")), conf)?;
    let flags: Vec<u8> = records
        .iter()
        .map(|r| {
            (r.result.bp_converged as u8) | ((r.result.smc_ran as u8) << 1) | ((r.result.hle as u8) << 2)
                | ((r.result.solvable as u8) << 3) | (r.result.esc_reason << 4)
        })
        .collect();
    std::fs::write(dir.join(format!("{tag}.flags.u8")), flags)?;
    Ok(())
}

fn stationary_cell(ds: &Dataset, dec: &Decoded, bposd: &Baseline, bplsd: &Baseline) -> Value {
    let n = dec.records.len();
    let truth: Vec<u64> = (0..n).map(|s| ds.truth(s)).collect();
    let fail = |obs: u64, s: usize| obs != truth[s];
    let (mut pe, mut oe, mut le, mut wins, mut losses, mut wins_l, mut losses_l) = (0, 0, 0, 0u64, 0u64, 0u64, 0u64);
    let (mut fp_false, mut fast, mut esc, mut slow, mut unsolv, mut hle, mut khat_rej) = (0, 0, 0, 0, 0, 0, 0);
    let (mut f_forest, mut f_best, mut f_marg) = (0, 0, 0);
    for (s, r) in dec.records.iter().enumerate() {
        let r = &r.result;
        let (p, o, l) = (fail(r.obs, s), fail(bposd.pred[s], s), fail(bplsd.pred[s], s));
        pe += p as usize;
        oe += o as usize;
        le += l as usize;
        wins += (!p && o) as u64;
        losses += (p && !o) as u64;
        wins_l += (!p && l) as u64;
        losses_l += (p && !l) as u64;
        fast += r.fast_accepted() as usize;
        fp_false += (r.fast_accepted() && p) as usize;
        esc += (r.esc_reason != 0) as usize;
        slow += r.smc_ran as usize;
        unsolv += (r.smc_ran && !r.solvable) as usize;
        hle += r.hle as usize;
        khat_rej += (r.smc_ran && r.solvable && r.khat > 0.7) as usize;
        let alt = |o: u64| if r.smc_ran && r.solvable { o } else { r.obs };
        f_forest += fail(alt(r.piv_obs), s) as usize;
        f_best += fail(alt(r.best_obs), s) as usize;
        f_marg += fail(alt(r.marginal_obs), s) as usize;
    }
    json!({
        "code_id": ds.code_id(), "p": ds.noise_p(), "dataset": ds.name, "shots": n,
        "prism_errors": pe, "bposd_errors": oe, "bplsd_errors": le,
        "prism_ler": pe as f64 / n as f64, "bposd_ler": oe as f64 / n as f64, "bplsd_ler": le as f64 / n as f64,
        "clopper_pearson_95": {"prism": ci_json(pe, n), "bposd": ci_json(oe, n), "bplsd": ci_json(le, n)},
        "prism_wins_bposd_fails": wins, "prism_fails_bposd_wins": losses,
        "mcnemar_p_value": mcnemar_exact(losses, wins),
        "mcnemar_p_prism_worse_than_bposd": mcnemar_worse(losses, wins),
        "prism_wins_bplsd_fails": wins_l, "prism_fails_bplsd_wins": losses_l,
        "mcnemar_p_value_vs_bplsd": mcnemar_exact(losses_l, wins_l),
        "fast_path_accepted": fast, "fast_path_false_converged": fp_false, "fast_path_escalated": esc,
        "slow_path_shots": slow, "slow_path_unsolvable": unsolv, "hle_flagged": hle, "khat_rejected": khat_rej,
        "alternative_decisions": {"forest_errors": f_forest, "min_energy_walker_errors": f_best, "bitwise_marginal_errors": f_marg},
        "bposd_bp_converged_fraction": bposd.summary["bp_converged_fraction"],
        "gpu": {"shots_per_sec": dec.telemetry.shots_per_sec, "wall_seconds": dec.telemetry.wall_seconds,
                "host_syncs_in_decode_loop": dec.telemetry.host_syncs_in_decode_loop,
                "fast_loop_iterations": dec.telemetry.fast_loop_iterations, "slow_loop_iterations": dec.telemetry.slow_loop_iterations,
                "f2_integrity_errors": dec.telemetry.f2_integrity_errors, "graph_mode": dec.telemetry.graph_mode},
        "gpu_vs_cpu_parity": {"shots_compared": dec.parity_shots, "mismatches": dec.parity_mismatches, "first_mismatch": dec.first_mismatch},
    })
}

/// Failures of a prediction vector split at the burst onset of a trace-structured dataset.
fn drift_counts(ds: &Dataset, plan: &StreamPlan, obs: &[u64]) -> (usize, usize, usize, usize) {
    let onset = ds.meta_usize("onset_block").unwrap_or(0);
    let (mut all, mut post, mut post_n, mut n) = (0, 0, 0, 0);
    for tr in 0..plan.streams {
        for blk in 0..plan.steps {
            let Some(s) = plan.shot(tr, blk) else { continue };
            let f = (obs[s] != ds.truth(s)) as usize;
            all += f;
            n += 1;
            if blk > onset {
                post += f;
                post_n += 1;
            }
        }
    }
    (all, n, post, post_n)
}

/// Pair-level precision / recall of the activated crosstalk hyperedges of an `xtburst`
/// dataset against the oracle's ground truth, from every trace's final live weights.
pub fn crosstalk_edge_scores(ds: &Dataset, ds_dir: &Path, model: &DecodeModel, live_llr: &[i32], traces: usize) -> Result<Value> {
    let truth: Value = serde_json::from_str(&std::fs::read_to_string(ds_dir.join("drift_truth.json"))?)?;
    let code = spec_by_name(ds.code_id())?.build()?;
    let hz_t = code.hz.transpose();
    let half = code.n / 2;
    // check set flipped by XX on the candidate pair (i, half + i): symmetric difference
    let mut by_checks: HashMap<Vec<u16>, usize> = HashMap::new();
    for i in 0..half {
        let mut set: Vec<u16> = hz_t.row(i).iter().chain(hz_t.row(half + i).iter()).map(|&c| c as u16).collect();
        set.sort_unstable();
        let sym: Vec<u16> = set.iter().copied().filter(|c| set.iter().filter(|x| *x == c).count() == 1).collect();
        by_checks.insert(sym, i);
    }
    let layout = &model.layout;
    let n_mech = layout.n_mech();
    // dormant hyperedges: nominal probability at the clip, identified with their pair
    let dormant: Vec<(usize, usize)> = (0..n_mech)
        .filter(|&e| layout.llr_nom[e] >= PRIOR_CLIP - 64 * 8)
        .filter_map(|e| by_checks.get(layout.mech_checks(e)).map(|&pair| (e, pair)))
        .collect();
    let (mut tp, mut fp, mut fnn) = (0usize, 0usize, 0usize);
    for tr in 0..traces {
        let llr = &live_llr[tr * n_mech..(tr + 1) * n_mech];
        let mut predicted = vec![false; half];
        for &(e, pair) in &dormant {
            predicted[pair] |= llr[e] < layout.llr_nom[e];
        }
        let mut actual = vec![false; half];
        for pair in truth[tr]["xt_pairs"].as_array().into_iter().flatten() {
            actual[pair[0].as_u64().unwrap_or(0) as usize] = true;
        }
        for i in 0..half {
            tp += (predicted[i] && actual[i]) as usize;
            fp += (predicted[i] && !actual[i]) as usize;
            fnn += (!predicted[i] && actual[i]) as usize;
        }
    }
    Ok(json!({
        "dormant_hyperedge_mechanisms": dormant.len(), "traces": traces,
        "true_positive_pairs": tp, "false_positive_pairs": fp, "false_negative_pairs": fnn,
        "precision": tp as f64 / (tp + fp).max(1) as f64, "recall": tp as f64 / (tp + fnn).max(1) as f64,
        "scored_at": "end of each 500-round trace",
    }))
}

struct DriftPart {
    report: Value,
    shots: usize,
    on_errors: usize,
    off_errors: usize,
    bposd_errors: usize,
    bocpd_only_errors: Option<usize>,
    wins: u64,
    losses: u64,
}

#[allow(clippy::too_many_arguments)]
fn drift_dataset(
    gpu: &mut GpuDecoder,
    ds: &Dataset,
    ds_dir: &Path,
    model: &DecodeModel,
    frozen: &CanonicalDecoderConfig,
    args: &CampaignArgs,
    with_bocpd_only: bool,
) -> Result<DriftPart> {
    let n_obs = ds.dem.n_obs;
    let bposd = load_baseline(ds_dir, "bposd", n_obs)?;
    let bplsd = load_baseline(ds_dir, "bplsd", n_obs)?;
    let plan = ds.plan(args.streams, Some(bposd.shots));
    ensure!(plan.n_shots == bposd.shots, "{}: baseline shot count does not tile whole traces", ds.name);
    let cfg_on = frozen.pipeline(ds.dem.n_det, ds.dem.n_err(), true);
    let cfg_off = frozen.pipeline(ds.dem.n_det, ds.dem.n_err(), false);
    let on = decode_with_parity(gpu, ds, model, &cfg_on, &plan, args.parity_shots)?;
    let live = gpu.last_live_llr.clone();
    let off = decode_with_parity(gpu, ds, model, &cfg_off, &plan, args.parity_shots)?;
    let herald = Herald { mode: frozen.hle_mode, n_chk: model.graph.n_chk };
    write_predictions(ds_dir, "neuro_on", n_obs, &on.records, herald)?;
    write_predictions(ds_dir, "neuro_off", n_obs, &off.records, herald)?;
    let obs_on: Vec<u64> = on.records.iter().map(|r| r.result.obs).collect();
    let obs_off: Vec<u64> = off.records.iter().map(|r| r.result.obs).collect();
    let (on_all, n, on_post, post_n) = drift_counts(ds, &plan, &obs_on);
    let (off_all, _, off_post, _) = drift_counts(ds, &plan, &obs_off);
    let (osd_all, _, osd_post, _) = drift_counts(ds, &plan, &bposd.pred);
    let (lsd_all, _, lsd_post, _) = drift_counts(ds, &plan, &bplsd.pred);
    let (mut wins, mut losses) = (0u64, 0u64);
    for s in 0..n {
        let (p, o) = (obs_on[s] != ds.truth(s), bposd.pred[s] != ds.truth(s));
        wins += (!p && o) as u64;
        losses += (p && !o) as u64;
    }
    // monitor reaction: blocks from the onset block to the first surgery of each trace
    let onset = ds.meta_usize("onset_block").unwrap_or(0);
    let mut reaction = Vec::new();
    let mut pre_surgery = 0usize;
    for tr in 0..plan.streams {
        let mut first = None;
        for blk in 0..plan.steps {
            let touched = on.records[plan.shot(tr, blk).unwrap_or(0)].report.modified_mechanisms > 0;
            if blk < onset {
                pre_surgery += touched as usize;
            } else if touched && first.is_none() {
                first = Some(blk - onset);
            }
        }
        if let Some(f) = first {
            reaction.push(f);
        }
    }
    reaction.sort_unstable();
    let rounds = ds.meta_usize("rounds").unwrap_or(1);
    let median_blocks = reaction.get(reaction.len() / 2).copied();
    let mut report = json!({
        "dataset": ds.name, "code_id": ds.code_id(), "shots": n, "traces": plan.streams, "blocks_per_trace": plan.steps,
        "onset_block": onset, "post_onset_shots": post_n,
        "errors": {"prism_neuro_on": on_all, "prism_neuro_off": off_all, "bposd_static": osd_all, "bplsd_static": lsd_all},
        "post_onset_errors": {"prism_neuro_on": on_post, "prism_neuro_off": off_post, "bposd_static": osd_post, "bplsd_static": lsd_post},
        "clopper_pearson_95": {"prism_neuro_on": ci_json(on_all, n), "prism_neuro_off": ci_json(off_all, n), "bposd_static": ci_json(osd_all, n)},
        "prism_on_wins_bposd_fails": wins, "prism_on_fails_bposd_wins": losses, "mcnemar_p_value": mcnemar_exact(losses, wins),
        "monitor": {
            "traces_with_surgery": reaction.len(), "pre_onset_blocks_with_surgery": pre_surgery,
            "median_blocks_from_onset_block_to_first_surgery": median_blocks,
            "median_rounds_from_onset_to_first_surgery_upper_bound": median_blocks.map(|b| (b + 1) * rounds),
        },
        "gpu_vs_cpu_parity": {"neuro_on": {"shots_compared": on.parity_shots, "mismatches": on.parity_mismatches, "first_mismatch": on.first_mismatch},
                              "neuro_off": {"shots_compared": off.parity_shots, "mismatches": off.parity_mismatches, "first_mismatch": off.first_mismatch}},
        "host_syncs_in_decode_loop": on.telemetry.host_syncs_in_decode_loop + off.telemetry.host_syncs_in_decode_loop,
    });
    if ds_dir.join("oracle_weights.json").is_file() {
        // reference on the CPU pipeline: the decoder handed the true drifted weights of every
        // block, i.e. the bound on what any online monitor can gain
        let oracle = OracleWeights::load(ds_dir, &ds.dem)?;
        let recs = run_reference_oracle(&model.graph, &model.layout, &ds.dets, &plan, &cfg_off, |trace, block, llr| {
            oracle.apply(trace, block, &model.layout.p_nom, llr)
        });
        let obs_or: Vec<u64> = recs.iter().map(|r| r.result.obs).collect();
        let (or_all, _, or_post, _) = drift_counts(ds, &plan, &obs_or);
        report["errors"]["prism_oracle_weights"] = json!(or_all);
        report["post_onset_errors"]["prism_oracle_weights"] = json!(or_post);
    }
    let mut bocpd_only_errors = None;
    if with_bocpd_only {
        // ablation on the CPU reference: changepoint filters without transfer entropy
        let mut cfg_bo = cfg_on;
        cfg_bo.neuro.bocpd_only = true;
        let (recs, _) = run_reference_with_weights(&model.graph, &model.layout, &ds.dets, &plan, &cfg_bo);
        let obs_bo: Vec<u64> = recs.iter().map(|r| r.result.obs).collect();
        let (bo_all, _, bo_post, _) = drift_counts(ds, &plan, &obs_bo);
        bocpd_only_errors = Some(bo_all);
        report["errors"]["prism_bocpd_only"] = json!(bo_all);
        report["post_onset_errors"]["prism_bocpd_only"] = json!(bo_post);
        report["do_tide_edge_detection"] = crosstalk_edge_scores(ds, ds_dir, model, &live, plan.streams)?;
    }
    Ok(DriftPart { report, shots: n, on_errors: on_all, off_errors: off_all, bposd_errors: osd_all, bocpd_only_errors, wins, losses })
}

fn drift_experiment(parts: &[DriftPart], description: &str) -> Value {
    let shots: usize = parts.iter().map(|p| p.shots).sum();
    let on: usize = parts.iter().map(|p| p.on_errors).sum();
    let off: usize = parts.iter().map(|p| p.off_errors).sum();
    let osd: usize = parts.iter().map(|p| p.bposd_errors).sum();
    let (wins, losses) = (parts.iter().map(|p| p.wins).sum::<u64>(), parts.iter().map(|p| p.losses).sum::<u64>());
    let ler = |k: usize| k as f64 / shots.max(1) as f64;
    let mut out = json!({
        "description": description, "shots": shots,
        "prism_neuro_on_errors": on, "prism_neuro_off_errors": off, "bposd_static_errors": osd,
        "prism_neuro_on_ler": ler(on), "prism_neuro_off_ler": ler(off), "bposd_static_ler": ler(osd),
        "ler_ratio_bposd_static_over_prism_neuro_on": if on > 0 { json!(osd as f64 / on as f64) } else { Value::Null },
        "ler_ratio_prism_neuro_off_over_on": if on > 0 { json!(off as f64 / on as f64) } else { Value::Null },
        "meets_3x_over_static_bposd": on * 3 <= osd && osd > 0,
        "prism_on_wins_bposd_fails": wins, "prism_on_fails_bposd_wins": losses, "mcnemar_p_value": mcnemar_exact(losses, wins),
        "clopper_pearson_95": {"prism_neuro_on": ci_json(on, shots), "prism_neuro_off": ci_json(off, shots), "bposd_static": ci_json(osd, shots)},
        "per_code": parts.iter().map(|p| p.report.clone()).collect::<Vec<_>>(),
    });
    if parts.iter().all(|p| p.bocpd_only_errors.is_some()) {
        let bo: usize = parts.iter().filter_map(|p| p.bocpd_only_errors).sum();
        out["prism_bocpd_only_errors"] = json!(bo);
        out["prism_bocpd_only_ler"] = json!(ler(bo));
        out["do_tide_beats_bocpd_only"] = json!(on < bo);
    }
    out
}

fn hle_frontier(ds: &Dataset, dec: &Decoded, bposd: &Baseline, herald: Herald) -> Value {
    let n = dec.records.len();
    let prism_fail: Vec<bool> = (0..n).map(|s| dec.records[s].result.obs != ds.truth(s)).collect();
    let osd_fail: Vec<bool> = (0..n).map(|s| bposd.pred[s] != ds.truth(s)).collect();
    let hle: Vec<f64> = dec.records.iter().map(|r| herald.confidence(r)).collect();
    let gap: Vec<f64> = dec.records.iter().map(|r| if r.result.smc_ran { r.result.min_abs_df } else { f64::INFINITY }).collect();
    // BP-OSD soft gap: smallest posterior |LLR| on the decoded faults; and the classic
    // "BP did not converge, then heaviest solution first" ordering
    let soft: Vec<f64> = bposd.margin.iter().map(|&m| m as f64).collect();
    let heavy: Vec<f64> =
        (0..n).map(|s| if bposd.bp_conv[s] != 0 { 1e9 - bposd.weight[s] as f64 } else { -(bposd.weight[s] as f64) }).collect();
    let points: Vec<Value> = ABORT_BUDGETS
        .iter()
        .map(|&b| {
            let (aborted, e_hle, l_hle) = retained_ler(&hle, &prism_fail, b);
            let (_, e_gap, l_gap) = retained_ler(&gap, &prism_fail, b);
            let (_, e_soft, l_soft) = retained_ler(&soft, &osd_fail, b);
            let (_, e_heavy, l_heavy) = retained_ler(&heavy, &osd_fail, b);
            json!({
                "abort_budget": b, "aborted_shots": aborted, "retained_shots": n - aborted,
                "prism_hle_retained_errors": e_hle, "prism_hle_retained_ler": l_hle,
                "prism_gap_only_retained_errors": e_gap, "prism_gap_only_retained_ler": l_gap,
                "bposd_soft_gap_retained_errors": e_soft, "bposd_soft_gap_retained_ler": l_soft,
                "bposd_nonconverged_weight_retained_errors": e_heavy, "bposd_nonconverged_weight_retained_ler": l_heavy,
                "prism_hle_beats_bposd_soft_gap": l_hle < l_soft,
            })
        })
        .collect();
    json!({
        "dataset": ds.name, "code_id": ds.code_id(), "p": ds.noise_p(), "shots": n,
        "unselected": {"prism_errors": prism_fail.iter().filter(|&&f| f).count(), "bposd_errors": osd_fail.iter().filter(|&&f| f).count()},
        "prism_hle_order": if herald.mode == 0 {
            "fast-path accepted shots last; Pareto-rejected (k_hat > 0.7) slow-path shots first, then by min_j |Delta F_j|"
        } else {
            "fast-path accepted shots last; slow-path shots by coset mass split 1 - W(L_MAP)/sum_L W(L) plus unsatisfied-check fraction, ties by walker energy gap"
        },
        "points": points,
    })
}

/// Per-shot streaming latency: one stream, one launch per shot, on the first `shots` shots.
fn latency_run(gpu: &mut GpuDecoder, ds: &Dataset, model: &DecodeModel, cfg: &PipelineConfig, shots: usize) -> Result<(Vec<u64>, GpuTelemetry)> {
    let plan = StreamPlan::interleaved(shots.min(ds.shots()), 1);
    let blocks = gpu.smc_blocks;
    gpu.smc_blocks = 1;
    let out = gpu.decode(&model.graph, &model.layout, cfg, &ds.dets, &plan, LaunchMode::PerStep);
    gpu.smc_blocks = blocks;
    let (_, tel) = out?;
    Ok((tel.launch_ns.clone(), tel))
}

pub fn run(args: &CampaignArgs) -> Result<()> {
    let frozen = args.frozen;
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(args.root.join("oracle_manifest.json"))?)?;
    let domain = manifest["seed_domain"].as_str().context("manifest without seed_domain")?.to_string();
    ensure!(domain != qec_neuro_dem::SEED_DOMAIN_CALIB, "the calibration domain must not be evaluated as a campaign");
    if domain == SEED_DOMAIN_BLIND {
        ensure!(args.release_blind, "{SEED_DOMAIN_BLIND} is quarantined: pass --release-blind once the frozen configuration is final");
    } else {
        println!("[campaign] NOTE: evaluating seed domain {domain}, not {SEED_DOMAIN_BLIND}");
    }
    let names: Vec<String> = manifest["datasets"].as_array().context("manifest without datasets")?.iter().filter_map(|v| v.as_str().map(String::from)).collect();
    let mut gpu = GpuDecoder::new(6usize << 30, 64 << 20)?;
    gpu.smc_blocks = 448;
    let f2 = gpu.f2_microbench(944, 128, 1024, 200)?;
    let info = gpu.info();
    println!(
        "[campaign] {} SMs {} = fast {} + slow {}; F2 engine {} (popc {:.0} ns, b1 {:.0} ns verified {})",
        info.device_name(), info.sm_total, info.sm_fast, info.sm_slow, f2.selected_engine, f2.popc_ns, f2.b1_ns, f2.b1_verified
    );
    let mut cells = Vec::new();
    let mut hle = Vec::new();
    let mut latency = Vec::new();
    let (mut tls, mut xtb) = (Vec::new(), Vec::new());
    let mut host_syncs = 0u64;
    let (mut parity_shots, mut parity_mismatches, mut integrity) = (0usize, 0usize, 0u64);
    let mut headline: Option<Value> = None;
    for name in &names {
        let ds_dir = args.root.join("datasets").join(name);
        let ds = Dataset::load(&ds_dir)?;
        ensure!(ds.seed_domain() == domain, "{name} is not a {domain} dataset");
        let model = DecodeModel::build(&ds.dem, frozen.cycle_search(), &frozen.neuro())?;
        let t0 = std::time::Instant::now();
        if ds.is_burst() {
            let is_xt = name.contains("xtburst");
            let part = drift_dataset(&mut gpu, &ds, &ds_dir, &model, &frozen, args, is_xt)?;
            host_syncs += part.report["host_syncs_in_decode_loop"].as_u64().unwrap_or(0);
            for key in ["neuro_on", "neuro_off"] {
                parity_shots += part.report["gpu_vs_cpu_parity"][key]["shots_compared"].as_u64().unwrap_or(0) as usize;
                parity_mismatches += part.report["gpu_vs_cpu_parity"][key]["mismatches"].as_u64().unwrap_or(0) as usize;
            }
            println!("[campaign] {name}: {} shots, errors {} ({:.1}s)", part.shots, part.report["errors"], t0.elapsed().as_secs_f64());
            if is_xt { xtb.push(part) } else { tls.push(part) }
            continue;
        }
        let n_obs = ds.dem.n_obs;
        let bposd = load_baseline(&ds_dir, "bposd", n_obs)?;
        let bplsd = load_baseline(&ds_dir, "bplsd", n_obs)?;
        ensure!(bposd.shots == bplsd.shots, "{name}: baselines decoded different shot counts");
        let cfg = frozen.pipeline(ds.dem.n_det, ds.dem.n_err(), true);
        let plan = ds.plan(args.streams, Some(bposd.shots));
        let dec = decode_with_parity(&mut gpu, &ds, &model, &cfg, &plan, args.parity_shots)?;
        let herald = Herald { mode: frozen.hle_mode, n_chk: model.graph.n_chk };
        write_predictions(&ds_dir, &frozen.config_id().to_lowercase(), n_obs, &dec.records, herald)?;
        host_syncs += dec.telemetry.host_syncs_in_decode_loop;
        parity_shots += dec.parity_shots;
        parity_mismatches += dec.parity_mismatches;
        integrity += dec.telemetry.f2_integrity_errors as u64;
        let cell = stationary_cell(&ds, &dec, &bposd, &bplsd);
        println!(
            "[campaign] {name}: shots={} prism={} bposd={} bplsd={} fp_false={} mcnemar_p={:.3e} gpu {:.0} shots/s parity {}/{} ({:.1}s)",
            cell["shots"], cell["prism_errors"], cell["bposd_errors"], cell["bplsd_errors"], cell["fast_path_false_converged"],
            cell["mcnemar_p_value"].as_f64().unwrap_or(1.0), dec.telemetry.shots_per_sec, dec.parity_mismatches, dec.parity_shots,
            t0.elapsed().as_secs_f64()
        );
        if ds.code_id() == "bb144_gross" && ds.noise_p() >= 0.0025 {
            hle.push(hle_frontier(&ds, &dec, &bposd, herald));
        }
        // streaming latency at the headline noise point
        if (ds.noise_p() - 0.003).abs() < 1e-9 {
            let is_headline = ds.code_id() == "bb144_gross";
            let shots = if is_headline { args.latency_shots } else { args.latency_shots_other };
            let (ns, tel) = latency_run(&mut gpu, &ds, &model, &cfg, shots)?;
            host_syncs += tel.host_syncs_in_decode_loop;
            let osd_ns = &bposd.lat_ns[..ns.len()];
            let lsd_ns = &bplsd.lat_ns[..ns.len()];
            let mean = |v: &[u64]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
            let entry = json!({
                "dataset": name, "code_id": ds.code_id(), "p": ds.noise_p(),
                "prism_gpu_streaming": latency_json(&ns), "bposd_cpu_single_thread": latency_json(osd_ns),
                "bplsd_cpu_single_thread": latency_json(lsd_ns),
                "mean_latency_speedup_vs_bposd": mean(osd_ns) / mean(&ns), "mean_latency_speedup_vs_bplsd": mean(lsd_ns) / mean(&ns),
                "launches": tel.launches, "host_syncs_in_decode_loop": tel.host_syncs_in_decode_loop,
                "batch_shots_per_sec": dec.telemetry.shots_per_sec,
                "batch_throughput_speedup_vs_bposd_single_thread": dec.telemetry.shots_per_sec * mean(&bposd.lat_ns) / 1e9,
            });
            if is_headline {
                headline = Some(entry.clone());
            }
            latency.push(entry);
        }
        cells.push(cell);
    }
    let headline = headline.context("the headline cell bb144_gross at p = 0.003 is missing")?;
    // runtime facts as they stand after every decode of the campaign
    let info = gpu.info();
    let lat = &headline["prism_gpu_streaming"];
    let bb72_gate: Vec<Value> = cells
        .iter()
        .filter(|c| c["code_id"] == "bb72")
        .map(|c| {
            let (pe, oe) = (c["prism_errors"].as_u64().unwrap_or(0), c["bposd_errors"].as_u64().unwrap_or(0));
            let fp = c["fast_path_false_converged"].as_u64().unwrap_or(0);
            json!({
                "p": c["p"], "prism_errors": pe, "bposd_errors": oe, "fast_path_false_converged": fp,
                "false_convergence_within_bound": pe == 0 || fp <= 3.max((0.12 * pe as f64) as u64),
                "no_significant_regression_vs_bposd": !(pe > oe && c["mcnemar_p_value"].as_f64().unwrap_or(1.0) < 0.05),
            })
        })
        .collect();
    let n1 = drift_experiment(&tls, "N1 TLS hot-qubit dropout: two data-qubit idle channels jump to X-flip probability 0.25 at round 100 of 500-round streams (p_base = 0.0015); no decoder is told");
    let n2 = drift_experiment(&xtb, "N2 schedule-conditioned coupler crosstalk (three XX pairs at 0.06 after CNOT step 3) plus a 20x transient cosmic-ray burst on a compact n/9-qubit cluster (60-round relaxation) from round 100 (p_base = 0.0015)");
    let receipt = json!({
        "schema": "prism.qec.sota.v1.frontier_receipt",
        "campaign": "bb144_sota_campaign",
        "sealed_utc": args.sealed_utc,
        "green_ctx_sms": [info.sm_fast, info.sm_slow],
        "host_syncs_in_decode_loop": host_syncs,
        "config_lock": {
            "config_id": frozen.config_id(), "config_sha256": frozen.sha256(),
            "canonical_text": frozen.canonical_text(),
            "calibration_seed_domain": qec_neuro_dem::SEED_DOMAIN_CALIB,
            "calibration_datasets": if frozen.version <= 1 {
                json!(["bb72 p=0.002", "surface_d5 p=0.002"])
            } else {
                json!(["bb72 p=0.002", "surface_d5 p=0.002", "bb72 p=0.0015 drift tls (400 traces)", "bb72 p=0.0015 drift xtburst (400 traces)"])
            },
            "development_cross_check_seed_domain": if frozen.version <= 1 { Value::Null } else { json!(qec_neuro_dem::SEED_DOMAIN_SMOKE) },
            "eval_seed_domain": domain, "per_code_tuning": false,
            "suspect_budget_rule": "k_cols = ceil(n_det * 256 / 256) rounded up to a multiple of 64, floor 128",
        },
        "gpu_sm120_telemetry": {
            "device": info.device_name(), "compute_capability": format!("{}.{}", info.cc_major, info.cc_minor),
            "sm_total": info.sm_total, "sm_fast": info.sm_fast, "sm_slow": info.sm_slow, "sm_split_request": info.split_mode,
            "graph_mode": info.graph_mode,
            "graph_topology": "one graph: WHILE(28-SM ctx){monitor -> damped min-sum -> advance} -> residual gate -> WHILE(56-SM ctx){coset SMC -> advance}",
            "p50_us": lat["p50_us"], "p99_us": lat["p99_us"], "p999_us": lat["p999_us"], "mean_us": lat["mean_us"],
            "shots_per_sec": headline["batch_shots_per_sec"],
            "speedup_vs_bposd": headline["mean_latency_speedup_vs_bposd"],
            "batch_throughput_speedup_vs_bposd_single_thread": headline["batch_throughput_speedup_vs_bposd_single_thread"],
            "headline_cell": headline["dataset"],
            "latency_definition": "wall time from graph launch to its single terminal wait, one stream, one shot per launch",
            "static_arena_bytes": info.arena_bytes, "devicemap_ring_bytes": info.ring_bytes,
            "persisting_l2": {"device_max_bytes": info.l2_max_persist, "context_limit_bytes": info.l2_limit, "limit_status": info.l2_limit_status,
                              "window_bytes": info.l2_window_bytes, "node_attribute_status": info.l2_attr_status,
                              "node_attribute_readback_persisting": info.l2_attr_readback == 1,
                              "note": "residency is requested through the access-policy window on the fast-path kernel node and verified by attribute read-back; the driver exposes no cache-occupancy counter"},
            "f2_engine_microbench": f2,
            "f2_note": "on sm_120 ptxas lowers mma .b1 and.popc to 8-bit integer tensor-core MMA (no native 1-bit BMMA)",
            "f2_integrity_errors": integrity,
            "gpu_vs_cpu_parity": {"shots_compared": parity_shots, "mismatches": parity_mismatches},
            "per_code_streaming_latency": latency,
        },
        "stationary_matrix": cells,
        "bb72_gate": bb72_gate,
        "neuro_dem_evaluation": {"tls_hot_qubit_dropout": n1, "correlated_crosstalk_and_burst": n2},
        "hle_pareto_frontier": hle.first().map(|h| h["points"].clone()).unwrap_or(json!([])),
        "hle_pareto_frontier_cells": hle,
        "baselines": manifest["baseline_config"],
        "disclosures": [
            "Decoder design and early parameter exploration used development shots of bb72 and bb144_gross at p = 0.003 (dev seed, disjoint from both seed domains) before the protocol was adopted; FROZEN_V1 values were then set on CALIB_2026 only.",
            "FROZEN_V2 was developed after FROZEN_V1 had been measured on CALIB_2026 and SMOKE_2026 shots (bb72, bb144_gross, surface_d5, hgp_hamming); its values were chosen on CALIB_2026 and cross-checked on SMOKE_2026. No PRISM decode of a BLIND_HOLDOUT_2026 shot took place before the configuration was frozen; the CLI refuses blind datasets outside this campaign.",
            "The reported decision is the maximum-weight coset; errors of the forest, minimum-energy-walker and bitwise-marginal rules on the same shots are listed per cell.",
            "PRISM latency is GPU wall time; baseline latency is single-thread CPU time measured while other baseline workers were running.",
            "Escalated fast paths reuse the reliability-ordered pivot (their decoded faults lead the suspect list); the pivot is not free.",
            "Drift experiments report errors over all blocks of every trace; post-onset errors are listed separately.",
        ],
    });
    std::fs::write(args.root.join(RECEIPT_FILE), serde_json::to_string_pretty(&receipt)? + "\n")?;
    let seal = write_seal(&args.root, &args.sealed_utc)?;
    println!("[campaign] receipt written; sealed {} files, sha256 root {}, blake3 root {}", seal.leaf_count, seal.merkle_root_sha256, seal.merkle_root_blake3);
    if parity_mismatches > 0 {
        bail!("GPU and CPU reference disagree on {parity_mismatches} of {parity_shots} compared shots");
    }
    Ok(())
}
