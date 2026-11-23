//! The SM120 decode pipeline: uploads the model into the static arena, writes the syndromes
//! into the device-mapped ring, launches the conditional decode graph and reads back one
//! record per shot.  Results are bit-identical to `qec_neuro_dem::run_reference`.

use crate::params::*;
use crate::runtime::{RtInfo, Runtime};
use anyhow::{ensure, Result};
use qec_core::dem::ShotBits;
use qec_core::f2::parity_and;
use qec_core::fixed::{logistic_thresholds, metropolis_thresholds, mix64, tanh_table, EN_ONE, TANH_MAX, TWIST_CLIP};
use qec_hodge_bp::bp::site_regulators;
use qec_hodge_bp::TannerGraph;
use qec_neuro_dem::{BlockReport, NeuroLayout, PipelineConfig, ShotRecord, StreamPlan};
use qec_smc_coset::ShotResult;
use serde::Serialize;
use std::time::Instant;

/// How the time steps of a plan are grouped into graph launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchMode {
    /// One launch decodes every time step (throughput).
    Batch,
    /// One launch per time step; with a single stream this is per-shot latency.
    PerStep,
}

/// What the device and the shim measured during one decode call.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuTelemetry {
    pub shots: usize,
    pub streams: usize,
    pub steps: usize,
    pub launches: u64,
    pub wall_seconds: f64,
    pub shots_per_sec: f64,
    /// Blocking host calls issued while a decode launch was in flight.
    pub host_syncs_in_decode_loop: u64,
    /// Device-side WHILE iterations of the two loops.
    pub fast_loop_iterations: u32,
    pub slow_loop_iterations: u32,
    pub slow_path_records: u32,
    pub slow_path_wakes: u32,
    /// Walker bits disagreeing with the F2 bit-matrix recomputation (must be 0).
    pub f2_integrity_errors: u32,
    pub f2_engine: i32,
    pub graph_mode: i32,
    pub arena_used_bytes: u64,
    pub l2_window_bytes: u64,
    /// Wall time of every launch (ns): per-shot latency when one stream decodes one step.
    pub launch_ns: Vec<u64>,
}

/// Result of the F2 engine microbenchmark.
#[derive(Clone, Debug, Default, Serialize)]
pub struct F2Bench {
    pub rows: usize,
    pub cols: usize,
    pub inner_bits: usize,
    pub popc_ns: f64,
    pub popc_verified: bool,
    pub b1_ns: f64,
    pub b1_verified: bool,
    /// 0: `__popcll(a & b) & 1`, 1: `.b1` `and.popc` tensor-core path (fastest verified).
    pub selected_engine: i32,
}

pub struct GpuDecoder {
    rt: Runtime,
    /// Engine used for the in-kernel F2 integrity check (set by [`GpuDecoder::f2_microbench`]).
    pub f2_engine: i32,
    /// Blocks per slow-path loop iteration.
    pub smc_blocks: u32,
    /// 0: try the single multi-context graph, fall back to two chained graphs.
    pub force_graph_mode: i32,
    /// Live prior LLRs of every stream after the last decode (`streams x n_mech`).
    pub last_live_llr: Vec<i32>,
}

fn out_to_record(o: &QecShotOut) -> ShotRecord {
    ShotRecord {
        result: ShotResult {
            obs: o.obs,
            bp_converged: o.bp_converged != 0,
            bp_iters: o.bp_iters as u32,
            bp_unsat: o.bp_unsat as u32,
            damped_steps: o.damped_steps as u32,
            damped_mask: o.damp_mask,
            check_updates: 0,
            bp_obs: o.bp_obs,
            esc_reason: o.esc_reason,
            esc_margin: o.esc_margin,
            esc_gap: o.esc_gap,
            smc_ran: o.smc_ran != 0,
            solvable: o.solvable != 0,
            rank: o.rank as u32,
            n_free: o.n_free as u32,
            piv_obs: o.piv_obs,
            piv_energy: o.piv_energy,
            khat: o.khat,
            min_abs_df: o.min_abs_df,
            ess: o.ess,
            best_energy: o.best_energy,
            best_obs: o.best_obs,
            q_obs: o.q_obs,
            mode_obs: o.mode_obs,
            marginal_obs: o.marginal_obs,
            energy_gap: o.energy_gap,
            mode_free_energy: o.mode_free_energy,
            bp_min_abs: o.bp_min_abs,
            resamples: o.resamples as u32,
            hle: o.hle != 0,
            piv_span: o.piv_span as u32,
            amb_gap: o.amb_gap,
            smc_sampled: o.smc_sampled != 0,
        },
        report: BlockReport {
            hot_checks: o.hot_checks as u32,
            modified_mechanisms: o.modified_mechanisms as u32,
            weight_writes: o.weight_writes as u32,
            armed: o.armed as u32,
            ..BlockReport::default()
        },
        latency_ns: 0,
    }
}

impl GpuDecoder {
    pub fn new(arena_bytes: usize, ring_bytes: usize) -> Result<Self> {
        Ok(Self {
            rt: Runtime::new(arena_bytes, ring_bytes)?,
            f2_engine: 0,
            smc_blocks: 192,
            force_graph_mode: 0,
            last_live_llr: Vec::new(),
        })
    }

    pub fn info(&self) -> RtInfo {
        self.rt.info()
    }

    /// Decodes the planned shots.  `cfg` is the same configuration the CPU reference takes.
    pub fn decode(
        &mut self,
        g: &TannerGraph,
        layout: &NeuroLayout,
        cfg: &PipelineConfig,
        dets: &ShotBits,
        plan: &StreamPlan,
        mode: LaunchMode,
    ) -> Result<(Vec<ShotRecord>, GpuTelemetry)> {
        let dc = &cfg.decoder;
        ensure!(g.n_var == layout.n_mech(), "graph and monitor layout disagree on the mechanism count");
        ensure!(!cfg.neuro.bocpd_only, "the BOCPD-only ablation runs on the CPU reference");
        ensure!(layout.n_checks < 4096, "coupler pair packing supports at most 4095 check sites");
        ensure!(
            (32..=256).contains(&dc.smc.walkers) && dc.smc.walkers.is_multiple_of(32),
            "walkers must be a multiple of 32 in [32, 256]"
        );
        ensure!(plan.n_shots <= dets.shots, "plan covers more shots than the syndrome file holds");
        let rt = &mut self.rt;
        rt.arena_reset();
        let (n_chk, n_var, n_shots) = (g.n_chk, g.n_var, plan.n_shots);
        let (streams, steps) = (plan.streams, plan.steps);
        let k_cols = dc.smc.k_cols.min(n_var);
        let full = dc.smc.full_columns;
        let k_list = if full { n_var } else { k_cols };
        let walkers = dc.smc.walkers;
        let smc_blocks = (self.smc_blocks as usize).min(n_shots.max(1)) as u32;
        let ws = dets.wps;
        let wc = layout.n_checks.div_ceil(64);
        let wr = n_chk.div_ceil(64);
        let wm = (k_list + 1).div_ceil(64);
        let wz = k_cols.div_ceil(64).div_ceil(4) * 4;
        let metro = metropolis_thresholds(dc.smc.beta_steps);
        let mut p = QecParams { magic: QEC_PARAMS_MAGIC, ..QecParams::default() };
        // ---- static model
        p.chk_ptr = rt.upload(&g.chk_ptr)?;
        p.chk_var = rt.upload(&g.chk_var)?;
        p.var_ptr = rt.upload(&g.var_ptr)?;
        p.var_edge = rt.upload(&g.var_edge)?;
        p.var_chk = rt.upload(&g.var_chk)?;
        p.cell_ptr = rt.upload(&g.cell_ptr)?;
        p.cell_var = rt.upload(&g.cell_var)?;
        p.vcell_ptr = rt.upload(&g.vcell_ptr)?;
        p.vcell = rt.upload(&g.vcell)?;
        p.obs = rt.upload(&g.obs)?;
        p.eps_site = rt.upload(&site_regulators(n_var, &dc.bp))?;
        p.tanh_tab = rt.upload(&tanh_table())?;
        p.logistic = rt.upload(&logistic_thresholds())?;
        p.metropolis = rt.upload(&metro)?;
        // ---- static monitor tables
        p.det_of = rt.upload(&layout.det_of)?;
        p.exposure = rt.upload(&layout.exposure)?;
        let pairs: Vec<u32> =
            layout.pairs.iter().map(|pc| pc.ci as u32 | ((pc.cj as u32) << 12) | ((pc.lag as u32) << 24)).collect();
        p.pairs = rt.upload(&pairs)?;
        p.pair_nom_te = rt.upload(&layout.pair_nom_te_q16)?;
        p.mech_pair_ptr = rt.upload(&layout.mech_pair_ptr)?;
        p.mech_pair = rt.upload(&layout.mech_pair)?;
        p.mech_size = rt.upload(&layout.mech_size)?;
        p.llr_nom = rt.upload(&layout.llr_nom)?;
        p.p_nom_q12 = rt.upload(&layout.p_nom_q12)?;
        p.log_q16 = rt.upload(&layout.log_q16)?;
        p.llr_of_p = rt.upload(&layout.llr_of_p)?;
        // mechanism classes (every table is padded to one element when there is no class)
        fn padded<T: Copy + Default>(v: &[T]) -> Vec<T> {
            if v.is_empty() {
                vec![T::default()]
            } else {
                v.to_vec()
            }
        }
        let n_cls = layout.n_cls();
        ensure!(cfg.neuro.coincidence || n_cls == 0, "class tables without the coincidence estimator");
        p.cls_ptr = rt.upload(&layout.cls_ptr)?;
        p.cls_det = rt.upload(&padded(&layout.cls_det))?;
        p.cls_span = rt.upload(&padded(&layout.cls_span))?;
        p.cls_pi_q32 = rt.upload(&padded(&layout.cls_pi_q32))?;
        p.cls_lnpi_q16 = rt.upload(&padded(&layout.cls_lnpi_q16))?;
        p.cls_sup_ptr = rt.upload(&layout.cls_sup_ptr)?;
        p.cls_sup = rt.upload(&padded(&layout.cls_sup))?;
        p.mech_cls = rt.upload(&layout.mech_cls)?;
        // ---- per-stream monitor state
        let n_pairs = pairs.len();
        let bo = streams * layout.n_checks * QEC_R_MAX;
        let mut bo_r = vec![0.0f64; bo];
        for slot in bo_r.chunks_exact_mut(QEC_R_MAX) {
            slot[0] = 1.0;
        }
        p.bo_r = rt.upload(&bo_r)?;
        p.bo_b = rt.alloc_zeroed(bo * 8)?;
        p.bo_a = rt.alloc_zeroed(bo * 4)?;
        p.hot = rt.alloc_zeroed(streams * layout.n_checks)?;
        p.trans = rt.alloc_zeroed(streams * layout.n_checks)?;
        p.counts = rt.alloc_zeroed(streams * n_pairs * 8)?;
        p.hist_bits = rt.alloc_zeroed(streams * QEC_HISTORY * wc * 8)?;
        p.hist_layer = rt.alloc_zeroed(streams * QEC_HISTORY)?;
        p.hist_pos = rt.alloc_zeroed(streams * 8)?;
        p.cur_bits = rt.alloc_zeroed(streams * wc * 8)?;
        p.pair_rate = rt.alloc_zeroed(streams * n_pairs * 4)?;
        p.explained = rt.alloc_zeroed(streams * n_pairs * 4)?;
        p.cls_cnt = rt.alloc_zeroed((streams * n_cls * 2).max(8))?;
        p.cls_rate = rt.alloc_zeroed((streams * n_cls * 4).max(8))?;
        p.arm_state = rt.alloc_zeroed(streams * 16)?;
        // live DEM weights of every stream, contiguous: the persisting-L2 window
        let en_nom: Vec<i32> = layout.llr_nom.iter().map(|&l| qec_core::fixed::energy_from_llr(l)).collect();
        let mut live = Vec::with_capacity(streams * n_var);
        for _ in 0..streams {
            live.extend_from_slice(&layout.llr_nom);
        }
        p.llr = rt.upload(&live)?;
        live.clear();
        for _ in 0..streams {
            live.extend_from_slice(&en_nom);
        }
        p.en = rt.upload(&live)?;
        // ---- per-stream fast-path state
        p.c2v = rt.alloc(streams * g.n_edge * 2)?;
        p.bp = rt.alloc(streams * n_var * 4)?;
        p.post = rt.alloc(streams * n_var * 4)?;
        p.snap = rt.alloc(streams * n_var * 4)?;
        p.winv = rt.alloc(streams * n_var * 4)?;
        p.ycell = rt.alloc(streams * g.n_cell.max(1) * 8)?;
        p.unsat_flag = rt.alloc(streams * n_chk)?;
        // ---- syndromes: written by the host straight into the device-mapped ring
        ensure!(n_shots * ws <= rt.ring().len(), "syndrome ring too small: {} shots x {} words", n_shots, ws);
        rt.ring()[..n_shots * ws].copy_from_slice(&dets.data[..n_shots * ws]);
        p.syn = rt.ring_dev();
        p.out = rt.alloc_zeroed(n_shots * std::mem::size_of::<QecShotOut>())?;
        p.rec_shot = rt.alloc(n_shots * 4)?;
        p.rec_list = rt.alloc(n_shots * k_list * 4)?;
        p.rec_en = rt.alloc(if full { 8 } else { n_shots * k_cols * 4 })?;
        p.ctrl = rt.alloc_zeroed(CTRL_WORDS * 4)?;
        // ---- per-block slow-path scratch
        let sb = smc_blocks as usize;
        p.s_pos = rt.alloc(sb * n_var.max(k_cols) * 4)?;
        p.s_mat = rt.alloc(sb * n_chk * wm * 8)?;
        p.s_pivcol = rt.alloc(sb * n_chk * 4)?;
        p.s_enrow = rt.alloc(sb * n_chk * 4)?;
        p.s_planes = rt.alloc(sb * QEC_EN_BITS * wr * 8)?;
        p.s_epiv = rt.alloc(sb * wr * 8)?;
        p.s_free = rt.alloc(sb * k_cols * 4)?;
        p.s_col = rt.alloc(sb * k_cols * wr * 8)?;
        p.s_rowt = rt.alloc(sb * n_chk * wz * 8)?;
        p.s_enfree = rt.alloc(sb * k_cols * 4)?;
        p.s_scol = rt.alloc(sb * k_cols * 4)?;
        p.s_theta = rt.alloc(sb * k_cols * 4)?;
        p.s_cycobs = rt.alloc(sb * k_cols * 8)?;
        p.s_zq = rt.alloc(sb * wz * 8)?;
        let full_scratch = if full { sb * n_var * 4 } else { 8 };
        p.s_list = rt.alloc(full_scratch)?;
        p.s_len = rt.alloc(full_scratch)?;
        p.s_gain = rt.alloc(full_scratch)?;
        p.w_rows = rt.alloc(sb * 2 * walkers * wr * 8)?;
        p.w_z = rt.alloc(sb * 2 * walkers * wz * 8)?;
        p.w_i32 = rt.alloc(sb * 2 * walkers * 2 * 4)?;
        p.w_obs = rt.alloc(sb * 2 * walkers * 8)?;
        p.w_acc = rt.alloc(sb * walkers * 8)?;
        p.w_anc = rt.alloc(sb * walkers * 4)?;
        p.w_wt = rt.alloc(sb * walkers * 8)?;
        p.w_lw = rt.alloc(sb * (2 * walkers + 128) * 8)?;
        // ---- scalars
        p.seed_smc = dc.smc.seed;
        p.blk_mu_q8 = layout.blk_mu_q8;
        p.blk_sigma_q8 = layout.blk_sigma_q8;
        p.hazard = cfg.neuro.hazard;
        p.hot_on = cfg.neuro.hot_on;
        p.hot_off = cfg.neuro.hot_off;
        p.tau_cert = dc.smc.tau_cert;
        p.khat_max = dc.smc.khat_max;
        p.n_chk = n_chk as u32;
        p.n_var = n_var as u32;
        p.n_edge = g.n_edge as u32;
        p.n_cell = g.n_cell as u32;
        p.n_obs = g.n_obs as u32;
        p.n_checks = layout.n_checks as u32;
        p.layers = layout.layers as u32;
        p.n_pairs = n_pairs as u32;
        p.n_cls = n_cls as u32;
        p.streams = streams as u32;
        p.steps = steps as u32;
        p.n_shots = n_shots as u32;
        p.stream_major = plan.stream_major as u32;
        p.ws = ws as u32;
        p.wc = wc as u32;
        p.wr = wr as u32;
        p.wm = wm as u32;
        p.wz = wz as u32;
        p.k_cols = k_cols as u32;
        p.k_list = k_list as u32;
        p.h_rank = g.h_rank as u32;
        p.walkers = walkers as u32;
        p.smc_blocks = smc_blocks;
        p.t_max = dc.bp.t_max as i32;
        p.t_clean = dc.bp.t_clean as i32;
        p.alpha_q8 = dc.bp.alpha_q8;
        p.eta_q8 = dc.bp.eta_q8;
        p.margin = dc.bp.margin;
        p.tau_margin = dc.esc.tau_margin;
        p.tau_gap = dc.esc.tau_gap;
        p.escalate = dc.esc.enabled as i32;
        p.t_late = dc.esc.t_late as i32;
        p.margin_gauge = dc.esc.margin_gauge as i32;
        p.beta_steps = dc.smc.beta_steps as i32;
        p.sweeps = dc.smc.sweeps as i32;
        p.quench_sweeps = dc.smc.quench_sweeps as i32;
        p.pair_lambda = dc.smc.pair_lambda as i32;
        p.pair_rounds = dc.smc.pair_rounds as i32;
        p.full_columns = full as i32;
        p.tau_ambig = dc.smc.tau_ambig;
        p.resample_q8 = dc.smc.resample_q8 as i32;
        p.tanh_max = TANH_MAX;
        p.twist_clip = TWIST_CLIP;
        p.metro_len = metro.len() as i32;
        p.en_one = EN_ONE;
        p.neuro_on = cfg.neuro_on as i32;
        p.n_min = cfg.neuro.n_min as i32;
        p.g_threshold = cfg.neuro.g_threshold as i32;
        p.g_threshold_hot = cfg.neuro.g_threshold_hot as i32;
        p.count_cap = cfg.neuro.count_cap as i32;
        p.coincidence = cfg.neuro.coincidence as i32;
        p.coin_cap = cfg.neuro.coin_cap as i32;
        p.coin_min = cfg.neuro.coin_min as i32;
        p.g_threshold_coin = cfg.neuro.g_threshold_coin as i32;
        p.coin_snap = cfg.neuro.snap as i32;
        p.arm_h_q8 = cfg.neuro.arm_h_q8 as i32;
        p.arm_kappa_q8 = cfg.neuro.arm_kappa_q8 as i32;
        p.arm_calm = cfg.neuro.arm_calm as i32;
        p.llr_deadband = cfg.neuro.llr_deadband;
        p.f2_engine = self.f2_engine;
        p.steps_per_launch = match mode {
            LaunchMode::Batch => 0,
            LaunchMode::PerStep => 1,
        };
        let params_dptr = rt.upload(std::slice::from_ref(&p))?;
        rt.build_graph(params_dptr, streams as u32, smc_blocks, walkers as u32, p.llr, streams * n_var * 8, self.force_graph_mode)?;
        // ---- decode: launches are enqueued without blocking; one terminal wait each
        let before = rt.info();
        let launches = match mode {
            LaunchMode::Batch => 1,
            LaunchMode::PerStep => steps,
        };
        let mut launch_ns = Vec::with_capacity(launches);
        let t0 = Instant::now();
        for _ in 0..launches {
            let t = Instant::now();
            rt.launch()?;
            rt.wait()?;
            launch_ns.push(t.elapsed().as_nanos() as u64);
        }
        let wall = t0.elapsed().as_secs_f64();
        let after = rt.info();
        let outs: Vec<QecShotOut> = rt.read(p.out, n_shots)?;
        let ctrl: Vec<u32> = rt.read(p.ctrl, CTRL_WORDS)?;
        self.last_live_llr = rt.read(p.llr, streams * n_var)?;
        ensure!(ctrl[CTRL_STEP] as usize == steps, "device executed {} of {} time steps", ctrl[CTRL_STEP], steps);
        ensure!(ctrl[CTRL_CURSOR] == ctrl[CTRL_NREC], "slow path left {} records undecoded", ctrl[CTRL_NREC] - ctrl[CTRL_CURSOR]);
        let records = outs.iter().map(out_to_record).collect();
        let telemetry = GpuTelemetry {
            shots: n_shots,
            streams,
            steps,
            launches: after.launches - before.launches,
            wall_seconds: wall,
            shots_per_sec: n_shots as f64 / wall.max(1e-12),
            host_syncs_in_decode_loop: after.host_syncs_in_loop - before.host_syncs_in_loop,
            fast_loop_iterations: ctrl[CTRL_FAST_ITERS],
            slow_loop_iterations: ctrl[CTRL_SLOW_ITERS],
            slow_path_records: ctrl[CTRL_NREC],
            slow_path_wakes: ctrl[7],
            f2_integrity_errors: ctrl[CTRL_INTEGRITY],
            f2_engine: self.f2_engine,
            graph_mode: after.graph_mode,
            arena_used_bytes: after.arena_used,
            l2_window_bytes: after.l2_window_bytes,
            launch_ns,
        };
        Ok((records, telemetry))
    }

    /// Microbenchmarks both exact-F2 engines on random `rows x cols` products over
    /// `inner_bits` (multiples of 16, 8 and 256), verifies each against the CPU popcount
    /// product, and selects the fastest verified one for the in-kernel integrity check.
    pub fn f2_microbench(&mut self, rows: usize, cols: usize, inner_bits: usize, reps: i32) -> Result<F2Bench> {
        ensure!(
            rows.is_multiple_of(16) && cols.is_multiple_of(8) && inner_bits.is_multiple_of(256),
            "f2 microbench shape must be multiples of (16, 8, 256)"
        );
        let rt = &mut self.rt;
        rt.arena_reset();
        let kw = inner_bits / 64;
        let wn = cols.div_ceil(64);
        let fill = |seed: u64, n: usize| -> Vec<u64> {
            (0..n).map(|i| mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))).collect()
        };
        let a = fill(0xA11CE, rows * kw);
        let b = fill(0xB0B, cols * kw);
        let mut expect = vec![0u64; rows * wn];
        for i in 0..rows {
            for j in 0..cols {
                if parity_and(&a[i * kw..(i + 1) * kw], &b[j * kw..(j + 1) * kw]) == 1 {
                    expect[i * wn + (j >> 6)] |= 1 << (j & 63);
                }
            }
        }
        let (da, db) = (rt.upload(&a)?, rt.upload(&b)?);
        let dc = rt.alloc_zeroed(rows * wn * 8)?;
        let mut bench = F2Bench { rows, cols, inner_bits, ..F2Bench::default() };
        bench.popc_ns = rt.f2_bench(0, da, db, dc, rows as i32, cols as i32, kw as i32, reps)?;
        bench.popc_verified = rt.read::<u64>(dc, rows * wn)? == expect;
        // the tensor-core path is optional: a device that rejects it simply stays on popcount
        if let Ok(ns) = rt.f2_bench(1, da, db, dc, rows as i32, cols as i32, kw as i32, reps) {
            bench.b1_ns = ns;
            bench.b1_verified = rt.read::<u64>(dc, rows * wn)? == expect;
        }
        ensure!(bench.popc_verified, "the popcount F2 engine failed verification against the CPU product");
        bench.selected_engine = (bench.b1_verified && bench.b1_ns < bench.popc_ns) as i32;
        self.f2_engine = bench.selected_engine;
        Ok(bench)
    }
}

/// Field-by-field comparison of a GPU record with the CPU reference record of the same shot.
/// Integer fields must match exactly; floating-point certificate values within `tol`.
pub fn record_mismatch(cpu: &ShotRecord, gpu: &ShotRecord, tol: f64) -> Option<String> {
    let (a, b) = (&cpu.result, &gpu.result);
    macro_rules! exact {
        ($($f:ident),*) => { $( if a.$f != b.$f { return Some(format!("{}: cpu {:?} gpu {:?}", stringify!($f), a.$f, b.$f)); } )* };
    }
    exact!(
        obs, bp_converged, bp_iters, bp_unsat, damped_steps, damped_mask, bp_obs, esc_reason, esc_margin, esc_gap, smc_ran, solvable, rank, n_free, piv_obs,
        piv_energy, best_energy, best_obs, q_obs, mode_obs, marginal_obs, energy_gap, bp_min_abs, resamples, piv_span, amb_gap,
        smc_sampled
    );
    let close = |x: f64, y: f64| (x == y) || (x - y).abs() <= tol * (1.0 + x.abs().min(y.abs()));
    macro_rules! approx {
        ($($f:ident),*) => { $( if !close(a.$f, b.$f) { return Some(format!("{}: cpu {} gpu {}", stringify!($f), a.$f, b.$f)); } )* };
    }
    approx!(khat, min_abs_df, ess, mode_free_energy);
    if cpu.report.modified_mechanisms != gpu.report.modified_mechanisms
        || cpu.report.hot_checks != gpu.report.hot_checks
        || cpu.report.weight_writes != gpu.report.weight_writes
        || cpu.report.armed != gpu.report.armed
    {
        return Some(format!(
            "monitor: cpu modified={} hot={} writes={} armed={} gpu modified={} hot={} writes={} armed={}",
            cpu.report.modified_mechanisms, cpu.report.hot_checks, cpu.report.weight_writes, cpu.report.armed,
            gpu.report.modified_mechanisms, gpu.report.hot_checks, gpu.report.weight_writes, gpu.report.armed
        ));
    }
    None
}
