//! Oracle dataset loading (`model.dem`, `dets.b8`, `obs.b8`, `meta.json`) and the decode model
//! (chain complex, Tanner graph, fixed-point priors) derived from it.

use anyhow::{ensure, Context, Result};
use qec_core::complex::{ChainComplex, CycleSearch};
use qec_core::dem::{Dem, ShotBits};
use qec_hodge_bp::TannerGraph;
use qec_neuro_dem::{NeuroConfig, NeuroLayout, StreamPlan};
use serde_json::Value;
use std::path::Path;

pub struct Dataset {
    pub name: String,
    pub meta: Value,
    pub dem: Dem,
    pub dets: ShotBits,
    pub obs: ShotBits,
}

impl Dataset {
    pub fn load(dir: &Path) -> Result<Self> {
        let meta: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json")).with_context(|| format!("reading {}/meta.json", dir.display()))?,
        )?;
        let dem = Dem::load(&dir.join("model.dem"))?;
        let n_det = meta["num_detectors"].as_u64().unwrap_or(dem.n_det as u64) as usize;
        let n_obs = meta["num_observables"].as_u64().unwrap_or(dem.n_obs as u64) as usize;
        ensure!(dem.n_det <= n_det, "DEM references detector {} but meta declares {}", dem.n_det, n_det);
        let mut dem = dem;
        dem.n_det = n_det;
        dem.n_obs = n_obs;
        dem.det_coords.resize(n_det, [f64::NAN; 3]);
        let dets = ShotBits::load_b8(&dir.join("dets.b8"), n_det)?;
        let obs = ShotBits::load_b8(&dir.join("obs.b8"), n_obs)?;
        ensure!(dets.shots == obs.shots, "detector and observable shot counts differ");
        if let Some(expected) = meta["num_error_mechanisms"].as_u64() {
            ensure!(expected as usize == dem.n_err(), "DEM has {} mechanisms, oracle recorded {}", dem.n_err(), expected);
        }
        let name = meta["name"].as_str().map(str::to_string).unwrap_or_else(|| {
            dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
        });
        Ok(Self { name, meta, dem, dets, obs })
    }

    pub fn shots(&self) -> usize {
        self.dets.shots
    }

    /// Ground-truth observable flips of a shot.
    pub fn truth(&self, shot: usize) -> u64 {
        self.obs.shot(shot)[0]
    }

    /// Seed domain recorded by the oracle (`CALIB_2026`, `BLIND_HOLDOUT_2026`, or `DEV`).
    pub fn seed_domain(&self) -> &str {
        self.meta["seed_domain"].as_str().unwrap_or("DEV")
    }

    pub fn code_id(&self) -> &str {
        self.meta["code"].as_str().unwrap_or("unknown")
    }

    pub fn noise_p(&self) -> f64 {
        self.meta["p"].as_f64().unwrap_or(0.0)
    }

    pub fn is_burst(&self) -> bool {
        self.meta["kind"].as_str().is_some_and(|k| k != "stationary")
    }

    pub fn meta_usize(&self, key: &str) -> Option<usize> {
        self.meta[key].as_u64().map(|x| x as usize)
    }
}

/// Everything a decoder needs that is derived once per DEM.
pub struct DecodeModel {
    pub graph: TannerGraph,
    pub layout: NeuroLayout,
}

impl DecodeModel {
    pub fn build(dem: &Dem, cycle_search: CycleSearch, neuro: &NeuroConfig) -> Result<Self> {
        let complex = ChainComplex::from_dem(dem, &cycle_search)?;
        let graph = TannerGraph::from_complex(&complex);
        let layout = NeuroLayout::build(dem, neuro)?;
        Ok(Self { graph, layout })
    }
}

impl Dataset {
    /// Stream layout of the shot file: burst datasets are consecutive traces, stationary
    /// datasets are dealt round-robin over `streams` independent streams.
    pub fn plan(&self, streams: usize, limit: Option<usize>) -> StreamPlan {
        let n = limit.unwrap_or(usize::MAX).min(self.shots());
        match (self.is_burst(), self.meta_usize("blocks_per_trace")) {
            (true, Some(blocks)) => StreamPlan::traces(n - n % blocks, blocks),
            _ => StreamPlan::interleaved(n, streams),
        }
    }
}
