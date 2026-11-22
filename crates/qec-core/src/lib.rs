//! Core F2 cellular chain-complex, Bivariate Bicycle qLDPC, and dual-Merkle primitives.

/// Verifies the fundamental CSS / space-time chain-complex identity d1 * d2 == 0 (mod 2)
/// on dense row-major bit-packed or byte matrices `[m_rows x n_cols]` and `[n_cols x p_cols]`.
pub fn check_chain_boundary_zero(d1: &[u8], m_rows: usize, n_cols: usize, d2: &[u8], p_cols: usize) -> bool {
    if d1.len() != m_rows * n_cols || d2.len() != n_cols * p_cols {
        return false;
    }
    for i in 0..m_rows {
        for j in 0..p_cols {
            let mut acc = 0u8;
            for k in 0..n_cols {
                acc ^= (d1[i * n_cols + k] & 1) & (d2[k * p_cols + j] & 1);
            }
            if acc != 0 {
                return false;
            }
        }
    }
    true
}
