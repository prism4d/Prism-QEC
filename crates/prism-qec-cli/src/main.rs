//! `prism-qec`: end-to-end CLI of the PRISM-QEC decoder stack.

// Index-driven loops mirror the CUDA kernels statement for statement.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod campaign;
mod cpu;
mod dataset;
mod stats;

use anyhow::{ensure, Result};
use clap::{Args, Parser, Subcommand};
use cpu::Decision;
use dataset::{Dataset, DecodeModel};
use qec_neuro_dem::{CanonicalDecoderConfig, PipelineConfig, SEED_DOMAIN_BLIND, SEED_DOMAIN_CALIB};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "prism-qec", version, about = "PRISM-QEC: damped min-sum + Nishimori coset SMC decoder")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the frozen configuration and its SHA-256 identity.
    Config,
    /// Print the chain-complex and monitor statistics of a dataset's detector error model.
    Inspect {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Calibration run on a `CALIB_2026` dataset: `FROZEN_V1` with explicit overrides.
    /// Refuses datasets of any other seed domain.
    Calibrate {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        shots: Option<usize>,
        #[command(flatten)]
        over: Overrides,
    },
    /// Decode a dataset with the CPU reference pipeline under `FROZEN_V1` (no overrides).
    CpuDecode {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        shots: Option<usize>,
        /// Freeze the detector error model at its nominal weights (static-DEM ablation).
        #[arg(long)]
        static_dem: bool,
    },
    /// Run the whole evaluation campaign of a dataset root on the RTX 5080 under `FROZEN_V1`
    /// (requires the GPU lease), write `qec_frontier_receipt.json` and seal the directory.
    Campaign {
        #[arg(long)]
        root: PathBuf,
        /// Shots per dataset re-decoded on the CPU reference and compared record for record.
        #[arg(long, default_value_t = 2048)]
        parity_shots: usize,
        /// Per-shot latency launches on the headline cell (bb144_gross, p = 0.003).
        #[arg(long, default_value_t = 10000)]
        latency_shots: usize,
        /// Per-shot latency launches on every other cell at p = 0.003.
        #[arg(long, default_value_t = 2000)]
        latency_shots_other: usize,
    },
    /// Decode a shot prefix on the RTX 5080 and on the CPU reference under `FROZEN_V1` and
    /// compare the records field by field (requires the GPU lease).
    GpuParity {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = 2048)]
        shots: usize,
        #[arg(long)]
        static_dem: bool,
        /// Use flooding instead of the lazy wavefront schedule on the CPU side.
        #[arg(long)]
        flooding: bool,
    },
    /// Seal a directory with the dual SHA-256 + BLAKE3 Merkle tree.
    Seal {
        #[arg(long)]
        root: PathBuf,
    },
    /// Re-hash a sealed directory and verify both Merkle roots.
    VerifySeal {
        #[arg(long)]
        root: PathBuf,
    },
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ` (civil-from-days, no external clock crate).
fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Calibration-only overrides of individual `FROZEN_V1` fields.
#[derive(Args, Clone, Debug)]
pub struct Overrides {
    #[arg(long)]
    pub t_max: Option<u32>,
    #[arg(long)]
    pub t_clean: Option<u32>,
    #[arg(long)]
    pub alpha_q8: Option<i32>,
    #[arg(long)]
    pub eta_q8: Option<i32>,
    #[arg(long)]
    pub eps_q12: Option<i32>,
    #[arg(long)]
    pub disorder_q8: Option<i32>,
    #[arg(long)]
    pub margin: Option<i32>,
    #[arg(long)]
    pub tau_margin: Option<i32>,
    #[arg(long)]
    pub tau_gap: Option<i32>,
    #[arg(long)]
    pub cycle_weight: Option<u32>,
    #[arg(long)]
    pub cell_cap: Option<u32>,
    #[arg(long)]
    pub k_cols_per_det_q8: Option<u32>,
    #[arg(long)]
    pub walkers: Option<u32>,
    #[arg(long)]
    pub beta_steps: Option<u32>,
    #[arg(long)]
    pub sweeps: Option<u32>,
    #[arg(long)]
    pub quench_sweeps: Option<u32>,
    #[arg(long)]
    pub resample_q8: Option<u32>,
    /// Accept every converged fast path (escalation ablation).
    #[arg(long)]
    pub no_escalation: bool,
    #[arg(long)]
    pub static_dem: bool,
    /// Recompute every check every iteration instead of the lazy wavefront schedule
    /// (must give the identical record digest).
    #[arg(long)]
    pub flooding: bool,
    #[arg(long, value_enum, default_value_t = Decision::Mode)]
    pub decision: Decision,
}

impl Overrides {
    fn apply(&self, mut c: CanonicalDecoderConfig) -> CanonicalDecoderConfig {
        macro_rules! set {
            ($($f:ident),*) => { $( if let Some(v) = self.$f { c.$f = v; } )* };
        }
        set!(
            t_max, t_clean, alpha_q8, eta_q8, eps_q12, disorder_q8, margin, tau_margin, tau_gap, cycle_weight, cell_cap,
            k_cols_per_det_q8, walkers, beta_steps, sweeps, quench_sweeps, resample_q8
        );
        c
    }
}

fn run_and_report(
    ds: &Dataset,
    canon: &CanonicalDecoderConfig,
    cfg: &PipelineConfig,
    shots: Option<usize>,
    decision: Decision,
) -> Result<()> {
    let model = DecodeModel::build(&ds.dem, canon.cycle_search(), &cfg.neuro)?;
    let plan = ds.plan(256, shots);
    let run = cpu::decode_streams(ds, &model, &plan, cfg);
    let st = cpu::stats(ds, &run.results, decision, cfg.decoder.smc.khat_max);
    println!("{}", serde_json::to_string_pretty(&st)?);
    if ds.is_burst() {
        let obs: Vec<u64> = run.results.iter().map(|r| cpu::decided_obs(r, decision)).collect();
        println!("{}", serde_json::to_string_pretty(&cpu::burst_stats(ds, &plan, &obs, Some(&run.records)))?);
    }
    println!("records_digest={}", cpu::records_digest(&run.results));
    println!(
        "latency_ns={} wall={:.2}s cells={} k_cols={}",
        serde_json::to_string(&cpu::percentiles(&run.latency_ns))?,
        run.wall_seconds,
        model.graph.n_cell,
        cfg.decoder.smc.k_cols
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let frozen = CanonicalDecoderConfig::FROZEN_V1;
    match cli.cmd {
        Cmd::Config => {
            print!("{}", frozen.canonical_text());
            println!("config_id={} config_sha256={}", CanonicalDecoderConfig::CONFIG_ID, frozen.sha256());
        }
        Cmd::Inspect { dataset } => {
            let ds = Dataset::load(&dataset)?;
            let model = DecodeModel::build(&ds.dem, frozen.cycle_search(), &frozen.neuro())?;
            let g = &model.graph;
            let covered = (0..g.n_var).filter(|&v| !g.var_cells(v).is_empty()).count();
            println!(
                "{} [{} / {} / p={}]: shots={} detectors={} mechanisms={} edges={} observables={}",
                ds.name,
                ds.code_id(),
                ds.seed_domain(),
                ds.noise_p(),
                ds.shots(),
                g.n_chk,
                g.n_var,
                g.n_edge,
                g.n_obs
            );
            println!(
                "2-cells={} covering {} mechanisms ({:.1}%); d1*d2 == 0 and L*d2 == 0 verified; k_cols={}",
                g.n_cell,
                covered,
                100.0 * covered as f64 / g.n_var as f64,
                frozen.k_cols(g.n_chk, g.n_var)
            );
            println!(
                "monitor: checks={} layers={} coupler pair classes={} re-weightable mechanisms={}",
                model.layout.n_checks,
                model.layout.layers,
                model.layout.pairs.len(),
                (0..g.n_var).filter(|&e| !model.layout.mech_pairs(e).is_empty()).count()
            );
        }
        Cmd::Calibrate { dataset, shots, over } => {
            let ds = Dataset::load(&dataset)?;
            ensure!(
                ds.seed_domain() == SEED_DOMAIN_CALIB,
                "calibration may only inspect {SEED_DOMAIN_CALIB} datasets; '{}' is {}",
                ds.name,
                ds.seed_domain()
            );
            let canon = over.apply(frozen);
            let mut cfg = canon.pipeline(ds.dem.n_det, ds.dem.n_err(), !over.static_dem);
            cfg.decoder.esc.enabled = !over.no_escalation;
            cfg.decoder.bp.wavefront = !over.flooding;
            println!("calibration config sha256={} (FROZEN_V1 is {})", canon.sha256(), frozen.sha256());
            run_and_report(&ds, &canon, &cfg, shots, over.decision)?;
        }
        Cmd::CpuDecode { dataset, shots, static_dem } => {
            let ds = Dataset::load(&dataset)?;
            let cfg = frozen.pipeline(ds.dem.n_det, ds.dem.n_err(), !static_dem);
            if ds.seed_domain() == SEED_DOMAIN_BLIND {
                println!("blind evaluation under {} sha256={}", CanonicalDecoderConfig::CONFIG_ID, frozen.sha256());
            }
            run_and_report(&ds, &frozen, &cfg, shots, Decision::Mode)?;
        }
        Cmd::Campaign { root, parity_shots, latency_shots, latency_shots_other } => {
            campaign::run(&campaign::CampaignArgs {
                root,
                parity_shots,
                latency_shots,
                latency_shots_other,
                streams: 256,
                sealed_utc: utc_now(),
            })?;
        }
        Cmd::GpuParity { dataset, shots, static_dem, flooding } => {
            let ds = Dataset::load(&dataset)?;
            let mut cfg = frozen.pipeline(ds.dem.n_det, ds.dem.n_err(), !static_dem);
            let model = DecodeModel::build(&ds.dem, frozen.cycle_search(), &cfg.neuro)?;
            let plan = ds.plan(256, Some(shots));
            let mut gpu = qec_cuda::GpuDecoder::new(2usize << 30, 64 << 20)?;
            let (records, tel) =
                gpu.decode(&model.graph, &model.layout, &cfg, &ds.dets, &plan, qec_cuda::LaunchMode::Batch)?;
            cfg.decoder.bp.wavefront = !flooding;
            let cpu = qec_neuro_dem::run_reference(&model.graph, &model.layout, &ds.dets, &plan, &cfg);
            let mut bad = 0;
            for (shot, (c, g)) in cpu.iter().zip(records.iter()).enumerate() {
                if let Some(diff) = qec_cuda::record_mismatch(c, g, 1e-6) {
                    bad += 1;
                    if bad <= 8 {
                        let mut c2 = *c;
                        c2.result.bp_unsat = g.result.bp_unsat;
                        println!(
                            "shot {shot}: {diff} | ignoring bp_unsat: {:?} | bp_obs {:x}/{:x} piv_energy {}/{} rank {}/{} mode_obs {:x}/{:x}",
                            qec_cuda::record_mismatch(&c2, g, 1e-6), c.result.bp_obs, g.result.bp_obs, c.result.piv_energy,
                            g.result.piv_energy, c.result.rank, g.result.rank, c.result.mode_obs, g.result.mode_obs
                        );
                    }
                }
            }
            println!(
                "gpu_parity shots={} mismatches={bad} host_syncs_in_decode_loop={} integrity={} graph_mode={}",
                cpu.len(), tel.host_syncs_in_decode_loop, tel.f2_integrity_errors, tel.graph_mode
            );
        }
        Cmd::Seal { root } => {
            let seal = qec_core::merkle::write_seal(&root, &utc_now())?;
            println!("sealed {} files: sha256 {} blake3 {}", seal.leaf_count, seal.merkle_root_sha256, seal.merkle_root_blake3);
        }
        Cmd::VerifySeal { root } => {
            let seal = qec_core::merkle::verify_seal(&root)?;
            println!("seal verified: {} files, sha256 {} blake3 {}", seal.leaf_count, seal.merkle_root_sha256, seal.merkle_root_blake3);
        }
    }
    Ok(())
}
