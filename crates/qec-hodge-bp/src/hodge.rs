//! Matrix-free coexact (curl) projection and velocity-space Rayleigh gauge damping.
//!
//! The projector onto the gauge directions of the chain complex, in the metric
//! `W = diag(tanh(|l| / 2) + eps)` and with the Jacobi (diagonal) approximation of
//! `d2^T W^-1 d2`, is
//!
//! ```text
//! Pi(x) = W^-1 d2 diag(d2^T W^-1 d2)^-1 d2^T x .
//! ```
//!
//! It is applied **only to the inter-iteration velocity** of the beliefs, restricted to the
//! frustrated sub-complex `F_t` (2-cells touching a parity-violating check or a low-margin
//! fault):
//!
//! ```text
//! l^(t) = l_BP^(t) - eta * Pi_{F_t}( l_BP^(t) - l^(t-1) )
//! ```
//!
//! Static beliefs are never projected.  Consequences, all tested in this crate:
//!
//! 1. **Zero fixed-point bias.**  At an exact BP fixed point the velocity vanishes, so the
//!    correction vanishes: damping cannot move a fixed point away from the Bayes posterior.
//! 2. **Gauge structure.**  The correction lies in `W^-1 im_R(d2)`; for every real cochain
//!    `u` with `d2^T u = 0` the pairing `<u, W l>` is untouched.  For Bivariate Bicycle codes
//!    the signed lift `d1_Z = [B^T | -A^T]` satisfies `d1_Z d2 = 0` over the integers, so the
//!    signed syndrome functional `d1_Z W l` and every harmonic (`H^1`) pairing are conserved.
//! 3. **Cohomology.**  Over F2, moving a hard decision along `d2 g` changes neither `d1 e`
//!    nor `L e`, because `d1 d2 = 0` and `L d2 = 0` are asserted when the complex is built.
//! 4. **Dissipation.**  A belief oscillating in the pure gauge mode of a 2-cell (all faults
//!    of a stabilizer flipping together, the signature of a symmetric trapping set) has its
//!    cell-mean velocity removed; with `eta = 1` the common mode of an isolated cell stops.
//!
//! The decoder uses the integer form [`project_fixed`]; [`project_f64`] is the real reference
//! it is checked against.

use crate::graph::TannerGraph;
use qec_core::fixed::{LLR_ONE, TANH_MAX, W_ONE};

/// Real reference: returns `eta * Pi_F(field)` for a metric evaluated at beliefs `llr`
/// (nats).  `eps[v]` is the per-site regulator, `active[f]` selects the sub-complex.
pub fn project_f64(g: &TannerGraph, field: &[f64], llr: &[f64], eta: f64, eps: &[f64], active: &[bool]) -> Vec<f64> {
    let winv: Vec<f64> = llr.iter().zip(eps).map(|(l, e)| 1.0 / ((l.abs() / 2.0).tanh() + e)).collect();
    let mut y = vec![0.0f64; g.n_cell];
    for f in 0..g.n_cell {
        if !active[f] {
            continue;
        }
        let (mut curl, mut diag) = (0.0, 0.0);
        for &v in g.cell_vars(f) {
            curl += field[v as usize];
            diag += winv[v as usize];
        }
        y[f] = curl / diag;
    }
    (0..g.n_var).map(|v| eta * winv[v] * g.var_cells(v).iter().map(|&f| y[f as usize]).sum::<f64>()).collect()
}

/// Truncating arithmetic shift that is symmetric about zero (identical in Rust and CUDA C).
#[inline]
pub fn shr_sym(x: i64, s: u32) -> i64 {
    if x >= 0 {
        x >> s
    } else {
        -((-x) >> s)
    }
}

/// Fixed-point `W^-1` in Q12 for a belief in BP units.
#[inline]
pub fn winv_fixed(llr: i32, tanh: &[u16], eps_site: i32) -> i32 {
    let a = llr.unsigned_abs().min(TANH_MAX as u32) as usize;
    (1i32 << 24) / (tanh[a] as i32 + eps_site)
}

/// Scratch buffers of the integer projection.
#[derive(Clone, Debug, Default)]
pub struct HodgeScratch {
    pub winv: Vec<i32>,
    pub y: Vec<i64>,
}

/// Integer projection: writes `corr[v] = eta * Pi_F(field)[v]` (BP units) for the metric at
/// beliefs `llr`.  `active[f] != 0` selects the projected 2-cells.
#[allow(clippy::too_many_arguments)]
pub fn project_fixed(
    g: &TannerGraph,
    field: &[i32],
    llr: &[i32],
    tanh: &[u16],
    eps_site: &[i32],
    eta_q8: i32,
    active: &[u8],
    scratch: &mut HodgeScratch,
    corr: &mut [i32],
) {
    scratch.winv.resize(g.n_var, 0);
    scratch.y.resize(g.n_cell, 0);
    for v in 0..g.n_var {
        scratch.winv[v] = winv_fixed(llr[v], tanh, eps_site[v]);
    }
    for f in 0..g.n_cell {
        if active[f] == 0 {
            scratch.y[f] = 0;
            continue;
        }
        let (mut curl, mut diag) = (0i64, 0i64);
        for &v in g.cell_vars(f) {
            curl += field[v as usize] as i64;
            diag += scratch.winv[v as usize] as i64;
        }
        // y = curl / D scaled by 2^16 (Q12 metric: real x = y / 16); division truncates.
        scratch.y[f] = (curl << 16) / diag;
    }
    for v in 0..g.n_var {
        let acc: i64 = g.var_cells(v).iter().map(|&f| scratch.y[f as usize]).sum();
        corr[v] = shr_sym(eta_q8 as i64 * scratch.winv[v] as i64 * acc, 24) as i32;
    }
}

/// Converts a BP-unit LLR to nats.
pub fn to_nats(x: i32) -> f64 {
    x as f64 / LLR_ONE as f64
}

/// Converts the fixed-point regulator to the real metric.
pub fn eps_to_real(eps_q12: i32) -> f64 {
    eps_q12 as f64 / W_ONE as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::bb::{bb72, gross144, BbSpec, CssCode};
    use qec_core::complex::ChainComplex;
    use qec_core::dem::Dem;
    use qec_core::fixed::{mix64, tanh_table};

    fn graph_of(code: &CssCode) -> TannerGraph {
        TannerGraph::from_complex(&ChainComplex::from_css_x_errors(code).unwrap())
    }

    fn pseudo(seed: u64, i: usize) -> f64 {
        (mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The signed integer lift `d1_Z = [B^T | -A^T]` of `H_Z`: entry sign is `+` on the left
    /// block and `-` on the right block.
    fn signed_check_rows(code: &CssCode) -> Vec<Vec<(usize, f64)>> {
        let half = code.n / 2;
        (0..code.hz.rows)
            .map(|c| {
                code.hz.row(c).iter().map(|&q| (q as usize, if (q as usize) < half { 1.0 } else { -1.0 })).collect()
            })
            .collect()
    }

    /// Basis of the real null space of `d2^T` by Gaussian elimination with partial pivoting.
    fn real_kernel_of_d2t(g: &TannerGraph) -> Vec<Vec<f64>> {
        let (rows, cols) = (g.n_cell, g.n_var);
        let mut m = vec![vec![0.0f64; cols]; rows];
        for (f, row) in m.iter_mut().enumerate() {
            for &v in g.cell_vars(f) {
                row[v as usize] = 1.0;
            }
        }
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..cols {
            if r == rows {
                break;
            }
            let (best, mag) =
                (r..rows).map(|i| (i, m[i][c].abs())).fold((r, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            if mag < 1e-9 {
                continue;
            }
            m.swap(r, best);
            let piv = m[r][c];
            for x in m[r].iter_mut() {
                *x /= piv;
            }
            for i in 0..rows {
                if i != r && m[i][c].abs() > 0.0 {
                    let factor = m[i][c];
                    let src = m[r].clone();
                    for (x, s) in m[i].iter_mut().zip(src.iter()) {
                        *x -= factor * s;
                    }
                }
            }
            pivots.push(c);
            r += 1;
        }
        let free: Vec<usize> = (0..cols).filter(|c| !pivots.contains(c)).collect();
        free.iter()
            .map(|&f| {
                let mut u = vec![0.0; cols];
                u[f] = 1.0;
                for (row, &p) in pivots.iter().enumerate() {
                    u[p] = -m[row][f];
                }
                u
            })
            .collect()
    }

    #[test]
    fn signed_lift_is_a_chain_complex_over_the_integers() {
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let g = graph_of(&code);
            for row in signed_check_rows(&code) {
                let mut dense = vec![0.0f64; code.n];
                for &(q, s) in &row {
                    dense[q] = s;
                }
                for f in 0..g.n_cell {
                    let sum: f64 = g.cell_vars(f).iter().map(|&v| dense[v as usize]).sum();
                    assert_eq!(sum, 0.0, "d1_Z * d2 must vanish over Z, not only mod 2");
                }
            }
        }
    }

    fn check_invariants(spec: BbSpec, seed: u64) {
        let code = spec.build().unwrap();
        let g = graph_of(&code);
        let llr: Vec<f64> = (0..g.n_var).map(|v| 8.0 * pseudo(seed, v) - 3.0).collect();
        let vel: Vec<f64> = (0..g.n_var).map(|v| 4.0 * pseudo(seed ^ 0x55, v) - 2.0).collect();
        let eps: Vec<f64> = (0..g.n_var).map(|v| 0.05 + 0.05 * pseudo(seed ^ 77, v)).collect();
        let corr = project_f64(&g, &vel, &llr, 0.7, &eps, &vec![true; g.n_cell]);
        let w: Vec<f64> = llr.iter().zip(&eps).map(|(l, e)| (l.abs() / 2.0).tanh() + e).collect();
        assert!(corr.iter().map(|x| x.abs()).sum::<f64>() > 1.0, "the projection must act on a generic velocity");
        // (1) the signed syndrome functional d1_Z W l is untouched by the correction
        for row in signed_check_rows(&code) {
            let moved: f64 = row.iter().map(|&(q, s)| s * w[q] * corr[q]).sum();
            assert!(moved.abs() < 1e-9, "syndrome boundary moved by {moved}");
        }
        // (2) every u in ker d2^T (gradient + harmonic H^1 part) pairs to zero with W * corr
        let kernel = real_kernel_of_d2t(&g);
        assert!(kernel.len() >= g.n_var - g.n_cell && !kernel.is_empty());
        for u in &kernel {
            for f in 0..g.n_cell {
                let s: f64 = g.cell_vars(f).iter().map(|&v| u[v as usize]).sum();
                assert!(s.abs() < 1e-8, "kernel vector is not annihilated by d2^T");
            }
            let moved: f64 = (0..g.n_var).map(|v| u[v] * w[v] * corr[v]).sum();
            assert!(moved.abs() < 1e-7);
        }
    }

    #[test]
    fn projection_conserves_syndrome_boundary_and_cohomology_pairings() {
        check_invariants(bb72(), 1);
        check_invariants(bb72(), 2);
        check_invariants(gross144(), 3);
    }

    #[test]
    fn zero_velocity_gives_exactly_zero_correction() {
        // the defining property of velocity-space damping: no bias at a fixed point
        let code = gross144().build().unwrap();
        let g = graph_of(&code);
        let tanh = tanh_table();
        let llr: Vec<i32> = (0..g.n_var).map(|v| ((9.0 * pseudo(8, v) - 4.0) * 64.0) as i32).collect();
        let eps_site = vec![256i32; g.n_var];
        let mut corr = vec![7i32; g.n_var];
        let mut scratch = HodgeScratch::default();
        project_fixed(&g, &vec![0; g.n_var], &llr, &tanh, &eps_site, 256, &vec![1u8; g.n_cell], &mut scratch, &mut corr);
        assert!(corr.iter().all(|&c| c == 0));
        let real = project_f64(&g, &vec![0.0; g.n_var], &vec![1.5; g.n_var], 1.0, &vec![0.1; g.n_var], &vec![true; g.n_cell]);
        assert!(real.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn gauge_moves_preserve_syndrome_and_logical_class_mod_two() {
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let g = graph_of(&code);
            for trial in 0..32u64 {
                let mut e = vec![0u64; g.n_var.div_ceil(64)];
                for v in 0..g.n_var {
                    if pseudo(trial, v) < 0.1 {
                        e[v >> 6] ^= 1 << (v & 63);
                    }
                }
                let mut moved = e.clone();
                for f in 0..g.n_cell {
                    if pseudo(trial ^ 0xABCD, f) < 0.5 {
                        for &v in g.cell_vars(f) {
                            moved[v as usize >> 6] ^= 1 << (v & 63);
                        }
                    }
                }
                assert_ne!(e, moved);
                assert_eq!(g.syndrome_of(&e), g.syndrome_of(&moved));
                assert_eq!(g.observables_of(&e), g.observables_of(&moved));
            }
        }
        // and on a space-time (detector error model) complex
        let code = bb72().build().unwrap();
        let (dem, cells) = Dem::phenomenological(&code, 2, 0.01, 0.01).unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_dem_cells(&dem, &cells).unwrap());
        let mut e = vec![0u64; g.n_var.div_ceil(64)];
        e[0] = 0b1011;
        let mut moved = e.clone();
        for f in (0..g.n_cell).step_by(3) {
            for &v in g.cell_vars(f) {
                moved[v as usize >> 6] ^= 1 << (v & 63);
            }
        }
        assert_eq!(g.syndrome_of(&e), g.syndrome_of(&moved));
        assert_eq!(g.observables_of(&e), g.observables_of(&moved));
    }

    #[test]
    fn unit_damping_stops_the_common_mode_of_an_isolated_cell() {
        let code = bb72().build().unwrap();
        let g = graph_of(&code);
        let mut active = vec![false; g.n_cell];
        active[5] = true;
        // all six faults of the stabilizer swing together by the same amount
        let llr = vec![0.8f64; g.n_var];
        let mut vel = vec![0.0f64; g.n_var];
        for &v in g.cell_vars(5) {
            vel[v as usize] = -1.6;
        }
        let eps: Vec<f64> = (0..g.n_var).map(|v| 0.05 + 0.1 * pseudo(9, v)).collect();
        let corr = project_f64(&g, &vel, &llr, 1.0, &eps, &active);
        let residual: f64 = g.cell_vars(5).iter().map(|&v| vel[v as usize] - corr[v as usize]).sum();
        assert!(residual.abs() < 1e-12, "eta = 1 removes the whole gauge-mode velocity of a single cell");
        for v in 0..g.n_var {
            if !g.cell_vars(5).contains(&(v as u32)) {
                assert_eq!(corr[v], 0.0, "faults off the cell are untouched");
            }
        }
    }

    #[test]
    fn fixed_point_projection_tracks_the_real_reference() {
        let code = gross144().build().unwrap();
        let g = graph_of(&code);
        let tanh = tanh_table();
        let eps_site: Vec<i32> = (0..g.n_var).map(|v| 256 + (pseudo(4, v) * 128.0) as i32).collect();
        let llr_q: Vec<i32> = (0..g.n_var).map(|v| ((6.0 * pseudo(5, v) - 2.0) * 64.0) as i32).collect();
        let vel_q: Vec<i32> = (0..g.n_var).map(|v| ((5.0 * pseudo(6, v) - 2.5) * 64.0) as i32).collect();
        let llr: Vec<f64> = llr_q.iter().map(|&x| to_nats(x)).collect();
        let vel: Vec<f64> = vel_q.iter().map(|&x| to_nats(x)).collect();
        let eps: Vec<f64> = eps_site.iter().map(|&e| eps_to_real(e)).collect();
        let reference = project_f64(&g, &vel, &llr, 0.5, &eps, &vec![true; g.n_cell]);
        let mut corr = vec![0i32; g.n_var];
        let mut scratch = HodgeScratch::default();
        project_fixed(&g, &vel_q, &llr_q, &tanh, &eps_site, 128, &vec![1u8; g.n_cell], &mut scratch, &mut corr);
        let worst = (0..g.n_var).map(|v| (to_nats(corr[v]) - reference[v]).abs()).fold(0.0f64, f64::max);
        assert!(corr.iter().any(|&c| c != 0));
        assert!(worst < 0.05, "fixed-point deviates from the real projection by {worst} nats");
    }
}
