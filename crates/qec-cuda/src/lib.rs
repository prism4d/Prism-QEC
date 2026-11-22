//! SM120 (RTX 5080) execution of the PRISM-QEC decoder: one static VRAM arena, a
//! zero-copy device-mapped syndrome ring, `[28, 56]`-SM green contexts and a conditional
//! `WHILE` decode graph with no host synchronisation inside the decode loop.

// Index-driven loops mirror the CUDA kernels statement for statement.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod params;
pub mod pipeline;
pub mod runtime;

pub use pipeline::{record_mismatch, F2Bench, GpuDecoder, GpuTelemetry, LaunchMode};
pub use runtime::{verify_layout, RtInfo, Runtime, CUBIN, PTXAS_LOG};
