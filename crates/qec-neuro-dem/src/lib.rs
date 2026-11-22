//! Neuromorphic defect-spike monitor: per-detector Bayesian online changepoint detection,
//! schedule-conditioned transfer entropy per coupler pair, and online surgery of the
//! detector-error-model weights.

// Index-driven loops mirror the CUDA kernels statement for statement.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod bocpd;
pub mod canonical;
pub mod pipeline;
pub mod stream;
pub mod tide;

pub use bocpd::{Bocpd, R_MAX};
pub use canonical::{CanonicalDecoderConfig, SEED_DOMAIN_BLIND, SEED_DOMAIN_CALIB};
pub use pipeline::{run_reference, run_reference_with_weights, PipelineConfig, ShotRecord, StreamPlan};
pub use stream::{BlockReport, NeuroConfig, NeuroLayout, StreamState};
pub use tide::PairClass;
