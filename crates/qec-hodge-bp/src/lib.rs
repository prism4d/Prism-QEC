//! Wavefront normalized min-sum belief propagation coupled inline with the matrix-free
//! coexact curl projection of the chain complex `C2 -> C1 -> C0`.

// Index-driven loops mirror the CUDA kernels statement for statement.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod bp;
pub mod graph;
pub mod hodge;

pub use bp::{BpConfig, BpOutcome, BpWork, HodgeBp};
pub use graph::TannerGraph;
