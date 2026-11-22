//! Spanning-forest manifold pivot, Nishimori twisted SMC over coset coordinates, marginal
//! coset free-energy gaps and the PSIS-certified Heralded Logical Erasure.

// Index-driven loops mirror the CUDA kernels statement for statement.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod decoder;
pub mod pivot;
pub mod psis;
pub mod smc;

pub use decoder::{
    auroc, escalation_flags, soft_margin, CosetDecoder, DecodeWork, DecoderConfig, EscalationConfig, ShotResult, SoftMargin,
    ESC_LATE, ESC_MARGIN, ESC_NONE, ESC_VALLEY,
};
pub use pivot::{build_pivot, Pivot, PivotScratch, Relax};
pub use smc::{certify, exact_cosets, run_smc, run_walkers, Population, SmcConfig, SmcResult, SmcTables, Walker};
