//! Stream-ordered CPU reference of the complete decoder: for every stream and time step,
//! the neuromorphic monitor ingests the block, performs the weight surgery, and the coset
//! decoder runs on the live weights.  The SM120 pipeline reproduces these records exactly.

use crate::stream::{BlockReport, NeuroConfig, NeuroLayout, StreamState};
use qec_core::dem::ShotBits;
use qec_hodge_bp::TannerGraph;
use qec_smc_coset::{CosetDecoder, DecodeWork, DecoderConfig, ShotResult};
use rayon::prelude::*;
use std::time::Instant;

/// How a flat shot file is laid out as `streams x steps` blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPlan {
    pub streams: usize,
    pub steps: usize,
    pub n_shots: usize,
    /// `true`: shot = stream * steps + step (one trace after another, burst datasets);
    /// `false`: shot = step * streams + stream (independent shots dealt round-robin).
    pub stream_major: bool,
}

impl StreamPlan {
    /// Consecutive traces of `steps` blocks each.
    pub fn traces(n_shots: usize, steps: usize) -> Self {
        Self { streams: n_shots.div_ceil(steps.max(1)), steps: steps.max(1), n_shots, stream_major: true }
    }

    /// Independent shots dealt over `streams` parallel streams.
    pub fn interleaved(n_shots: usize, streams: usize) -> Self {
        let streams = streams.clamp(1, n_shots.max(1));
        Self { streams, steps: n_shots.div_ceil(streams), n_shots, stream_major: false }
    }

    #[inline]
    pub fn shot(&self, stream: usize, step: usize) -> Option<usize> {
        let s = if self.stream_major { stream * self.steps + step } else { step * self.streams + stream };
        (s < self.n_shots).then_some(s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipelineConfig {
    pub decoder: DecoderConfig,
    pub neuro: NeuroConfig,
    /// `false` freezes every stream at the nominal model (static-DEM ablation).
    pub neuro_on: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShotRecord {
    pub result: ShotResult,
    pub report: BlockReport,
    pub latency_ns: u64,
}

/// Decodes every shot in stream order.  Streams are independent and run in parallel; the
/// output is indexed by shot.
pub fn run_reference(
    graph: &TannerGraph,
    layout: &NeuroLayout,
    dets: &ShotBits,
    plan: &StreamPlan,
    cfg: &PipelineConfig,
) -> Vec<ShotRecord> {
    run_reference_with_weights(graph, layout, dets, plan, cfg).0
}

/// Records of one stream (tagged with their shot index) and its final live prior LLRs.
type StreamOutput = (Vec<(usize, ShotRecord)>, Vec<i32>);

/// As [`run_reference`], also returning every stream's live prior LLRs after its last block
/// (`streams x n_mech`, stream-major).
pub fn run_reference_with_weights(
    graph: &TannerGraph,
    layout: &NeuroLayout,
    dets: &ShotBits,
    plan: &StreamPlan,
    cfg: &PipelineConfig,
) -> (Vec<ShotRecord>, Vec<i32>) {
    let dec = CosetDecoder::new(graph, cfg.decoder);
    let per_stream: Vec<StreamOutput> = (0..plan.streams)
        .into_par_iter()
        .map(|stream| {
            let mut state = StreamState::new(layout);
            let mut work = DecodeWork::default();
            let mut out = Vec::with_capacity(plan.steps);
            for step in 0..plan.steps {
                let Some(shot) = plan.shot(stream, step) else { continue };
                let syn = dets.shot(shot);
                let t0 = Instant::now();
                let report =
                    if cfg.neuro_on { state.ingest_block(layout, &cfg.neuro, syn) } else { BlockReport::default() };
                let result = dec.decode(&mut work, &state.llr, &state.en, syn, shot as u32);
                out.push((shot, ShotRecord { result, report, latency_ns: t0.elapsed().as_nanos() as u64 }));
            }
            (out, state.llr)
        })
        .collect();
    let mut records = vec![ShotRecord::default(); plan.n_shots];
    let mut weights = Vec::with_capacity(plan.streams * layout.n_mech());
    for (recs, llr) in per_stream {
        for (shot, rec) in recs {
            records[shot] = rec;
        }
        weights.extend_from_slice(&llr);
    }
    (records, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_cover_every_shot_exactly_once() {
        for plan in [StreamPlan::traces(84 * 5, 84), StreamPlan::interleaved(1000, 64), StreamPlan::interleaved(7, 64)] {
            let mut seen = vec![0u32; plan.n_shots];
            for stream in 0..plan.streams {
                for step in 0..plan.steps {
                    if let Some(s) = plan.shot(stream, step) {
                        seen[s] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "{plan:?}");
        }
        let p = StreamPlan::traces(10, 5);
        assert_eq!((p.shot(1, 2), p.shot(2, 0)), (Some(7), None));
        let q = StreamPlan::interleaved(10, 4);
        assert_eq!((q.shot(1, 2), q.shot(2, 2)), (Some(9), None));
    }
}
