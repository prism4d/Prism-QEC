//! Live RTX 5080 tests (Gate 4).  They need the GPU lease and are therefore `#[ignore]`d:
//! run them through `./harness/qec-ctl.sh gpu-exec ... cargo test -- --include-ignored`.

#![allow(clippy::needless_range_loop)]

use qec_core::bb::bb72;
use qec_core::complex::ChainComplex;
use qec_core::dem::{Dem, ShotBits};
use qec_cuda::{record_mismatch, GpuDecoder, LaunchMode};
use qec_hodge_bp::TannerGraph;
use qec_neuro_dem::{run_reference, CanonicalDecoderConfig, NeuroLayout, PipelineConfig, StreamPlan};

struct Bench {
    dem: Dem,
    graph: TannerGraph,
    layout: NeuroLayout,
    cfg: PipelineConfig,
    n: usize,
}

fn bench(p: f64) -> Bench {
    let code = bb72().build().unwrap();
    let (dem, cells) = Dem::phenomenological(&code, 5, p, p).unwrap();
    let canon = CanonicalDecoderConfig::ACTIVE;
    let cfg = canon.pipeline(dem.n_det, dem.n_err(), true);
    let layout = NeuroLayout::build(&dem, &cfg.neuro).unwrap();
    let graph = TannerGraph::from_complex(&ChainComplex::from_dem_cells(&dem, &cells).unwrap());
    Bench { dem, graph, layout, cfg, n: code.n }
}

/// `streams x steps` blocks; from `onset` on, the six data qubits of one check of every
/// stream are maximally mixed, so the monitor, the surgery and the slow path are all exercised.
fn drifting_shots(b: &Bench, streams: usize, steps: usize, onset: usize, seed: u64) -> (ShotBits, StreamPlan) {
    let plan = StreamPlan::traces(streams * steps, steps);
    let layers = b.layout.layers;
    let mut shots = ShotBits::zeros(streams * steps, b.dem.n_det);
    for s in 0..streams {
        let mut hot = b.dem.probs.clone();
        for t in 0..layers {
            for q in 0..6 {
                hot[t * b.n + (7 * s + 11 * q) % b.n] = 0.5;
            }
        }
        for step in 0..steps {
            let probs = if step >= onset { &hot } else { &b.dem.probs };
            let (dets, _) = b.dem.sample(probs, seed + s as u64, step as u64);
            shots.shot_mut(plan.shot(s, step).unwrap()).copy_from_slice(&dets);
        }
    }
    (shots, plan)
}

fn decoder() -> GpuDecoder {
    GpuDecoder::new(768 << 20, 16 << 20).expect("RTX 5080 runtime (run under qec-ctl.sh gpu-exec)")
}

#[test]
#[ignore]
fn green_contexts_partition_the_device_and_loops_run_on_it() {
    let b = bench(0.006);
    let (shots, plan) = drifting_shots(&b, 8, 12, 6, 100);
    let mut gpu = decoder();
    let info = gpu.info();
    eprintln!(
        "[sm120] {} cc {}.{} SMs total={} fast={} slow={} (align {}, split mode {}), L2 persist max={} limit={}",
        info.device_name(), info.cc_major, info.cc_minor, info.sm_total, info.sm_fast, info.sm_slow, info.sm_align,
        info.split_mode, info.l2_max_persist, info.l2_limit
    );
    assert!(info.sm_fast > 0 && info.sm_slow > 0 && info.sm_fast + info.sm_slow <= info.sm_total);
    let (records, tel) = gpu.decode(&b.graph, &b.layout, &b.cfg, &shots, &plan, LaunchMode::Batch).unwrap();
    eprintln!("[sm120] telemetry {:?}", serde_like(&tel));
    assert_eq!(records.len(), plan.n_shots);
    assert_eq!(tel.launches, 1, "one launch decodes every time step");
    assert_eq!(tel.host_syncs_in_decode_loop, 0, "no blocking host call while the loop is on the device");
    assert_eq!(tel.fast_loop_iterations as usize, plan.steps, "the WHILE loop iterated on the device");
    assert!(tel.slow_path_records > 0 && tel.slow_loop_iterations > 0, "the residual gate woke the slow path");
    assert_eq!(tel.f2_integrity_errors, 0);
    assert!(gpu.info().l2_window_bytes > 0 || gpu.info().l2_attr_status != 0);
}

fn serde_like(t: &qec_cuda::GpuTelemetry) -> String {
    format!(
        "shots={} launches={} wall={:.4}s host_syncs_in_loop={} fast_iters={} slow_iters={} records={} wakes={} integrity={} graph_mode={} arena={}",
        t.shots, t.launches, t.wall_seconds, t.host_syncs_in_decode_loop, t.fast_loop_iterations, t.slow_loop_iterations,
        t.slow_path_records, t.slow_path_wakes, t.f2_integrity_errors, t.graph_mode, t.arena_used_bytes
    )
}

#[test]
#[ignore]
fn gpu_matches_the_cpu_reference_record_for_record() {
    let b = bench(0.006);
    let (shots, plan) = drifting_shots(&b, 16, 30, 12, 7);
    let cpu = run_reference(&b.graph, &b.layout, &shots, &plan, &b.cfg);
    let mut gpu = decoder();
    for engine in [0, 1] {
        gpu.f2_engine = engine;
        let (records, tel) = gpu.decode(&b.graph, &b.layout, &b.cfg, &shots, &plan, LaunchMode::Batch).unwrap();
        let mut slow = 0;
        for (shot, (c, g)) in cpu.iter().zip(records.iter()).enumerate() {
            if let Some(diff) = record_mismatch(c, g, 1e-6) {
                panic!("engine {engine}, shot {shot}: {diff}");
            }
            slow += c.result.smc_ran as usize;
        }
        assert!(slow > 20, "the comparison must cover the slow path ({slow} shots)");
        assert!(cpu.iter().any(|r| r.report.modified_mechanisms > 0), "the comparison must cover weight surgery");
        assert_eq!((tel.host_syncs_in_decode_loop, tel.f2_integrity_errors), (0, 0));
    }
    // per-step launches (the latency mode) give the same records
    let (records, tel) = gpu.decode(&b.graph, &b.layout, &b.cfg, &shots, &plan, LaunchMode::PerStep).unwrap();
    assert_eq!(tel.launches as usize, plan.steps);
    assert_eq!(tel.host_syncs_in_decode_loop, 0);
    for (shot, (c, g)) in cpu.iter().zip(records.iter()).enumerate() {
        assert!(record_mismatch(c, g, 1e-6).is_none(), "per-step launch diverged at shot {shot}");
    }
    // and the static-DEM ablation as well
    let frozen = PipelineConfig { neuro_on: false, ..b.cfg };
    let cpu_static = run_reference(&b.graph, &b.layout, &shots, &plan, &frozen);
    let (gpu_static, _) = gpu.decode(&b.graph, &b.layout, &frozen, &shots, &plan, LaunchMode::Batch).unwrap();
    for (shot, (c, g)) in cpu_static.iter().zip(gpu_static.iter()).enumerate() {
        assert!(record_mismatch(c, g, 1e-6).is_none(), "static-DEM run diverged at shot {shot}");
    }
}

/// The paths the active configuration leaves off must keep GPU/CPU parity too: pair
/// exploration of the pivot relaxation, and the deprecated `FROZEN_V1` (pairwise monitor,
/// gauge margin, late-convergence trigger).
#[test]
#[ignore]
fn optional_and_deprecated_paths_keep_parity() {
    let b = bench(0.006);
    let (shots, plan) = drifting_shots(&b, 8, 24, 10, 21);
    let mut gpu = decoder();
    let mut paired = b.cfg;
    paired.decoder.smc.pair_lambda = 48;
    paired.decoder.smc.pair_rounds = 4;
    let cpu = run_reference(&b.graph, &b.layout, &shots, &plan, &paired);
    let (records, _) = gpu.decode(&b.graph, &b.layout, &paired, &shots, &plan, LaunchMode::Batch).unwrap();
    let plain = run_reference(&b.graph, &b.layout, &shots, &plan, &b.cfg);
    let mut moved = 0;
    for (shot, (c, g)) in cpu.iter().zip(records.iter()).enumerate() {
        if let Some(diff) = record_mismatch(c, g, 1e-6) {
            panic!("pair exploration, shot {shot}: {diff}");
        }
        moved += (c.result.q_obs != plain[shot].result.q_obs || c.result.best_energy != plain[shot].result.best_energy) as usize;
    }
    eprintln!("[sm120] pair exploration changed {moved} of {} records", cpu.len());
    for canon in [CanonicalDecoderConfig::FROZEN_V1, CanonicalDecoderConfig::FROZEN_V2, CanonicalDecoderConfig::FROZEN_V3] {
        let id = canon.config_id();
        let old = canon.pipeline(b.dem.n_det, b.dem.n_err(), true);
        let layout = NeuroLayout::build(&b.dem, &old.neuro).unwrap();
        let cpu = run_reference(&b.graph, &layout, &shots, &plan, &old);
        let (records, tel) = gpu.decode(&b.graph, &layout, &old, &shots, &plan, LaunchMode::Batch).unwrap();
        for (shot, (c, g)) in cpu.iter().zip(records.iter()).enumerate() {
            if let Some(diff) = record_mismatch(c, g, 1e-6) {
                panic!("{id}, shot {shot}: {diff}");
            }
        }
        assert!(cpu.iter().any(|r| r.report.modified_mechanisms > 0), "the {id} comparison must cover weight surgery");
        assert_eq!(tel.host_syncs_in_decode_loop, 0);
    }
}

#[test]
#[ignore]
fn f2_engines_are_verified_against_the_cpu_product() {
    let mut gpu = decoder();
    let bench = gpu.f2_microbench(944, 128, 1024, 20).unwrap();
    eprintln!(
        "[f2] {}x{} over {} bits: popc {:.0} ns (verified {}), b1 and.popc {:.0} ns (verified {}), selected engine {}",
        bench.rows, bench.cols, bench.inner_bits, bench.popc_ns, bench.popc_verified, bench.b1_ns, bench.b1_verified,
        bench.selected_engine
    );
    assert!(bench.popc_verified, "exact popcount parity must match the CPU");
    assert!(bench.b1_verified, "the .b1 and.popc path must reproduce F2 products exactly");
    assert_eq!(gpu.f2_engine, bench.selected_engine);
}
