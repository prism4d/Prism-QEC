//! Bivariate Bicycle codes over the group algebra F2[x, y] / (x^l - 1, y^m - 1).
//!
//! With `x = S_l (x) I_m`, `y = I_l (x) S_m` (S the cyclic shift), a pair of trinomials
//! `A`, `B` defines the CSS code `H_X = [A | B]`, `H_Z = [B^T | A^T]`.  Because the algebra
//! is commutative, `H_X H_Z^T = AB + BA = 0 (mod 2)`.

use crate::f2::{parity_and, BitMat, SparseF2};
use anyhow::{bail, ensure, Result};

/// The monomial `x^x * y^y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Monomial {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Debug)]
pub struct BbSpec {
    pub name: &'static str,
    pub l: usize,
    pub m: usize,
    pub a: [Monomial; 3],
    pub b: [Monomial; 3],
    pub k: usize,
    pub d: usize,
}

const POLY_A: [Monomial; 3] = [Monomial { x: 3, y: 0 }, Monomial { x: 0, y: 1 }, Monomial { x: 0, y: 2 }];
const POLY_B: [Monomial; 3] = [Monomial { x: 0, y: 3 }, Monomial { x: 1, y: 0 }, Monomial { x: 2, y: 0 }];

/// The `[[72, 12, 6]]` code: `l = m = 6`, `A = x^3 + y + y^2`, `B = y^3 + x + x^2`.
pub fn bb72() -> BbSpec {
    BbSpec { name: "bb72", l: 6, m: 6, a: POLY_A, b: POLY_B, k: 12, d: 6 }
}

/// The `[[144, 12, 12]]` Gross code: `l = 12`, `m = 6`, same trinomials.
pub fn gross144() -> BbSpec {
    BbSpec { name: "bb144_gross", l: 12, m: 6, a: POLY_A, b: POLY_B, k: 12, d: 12 }
}

/// The `[[90, 8, 10]]` code: `l = 15`, `m = 3`, `A = x^9 + y + y^2`, `B = 1 + x^2 + x^7`.
pub fn bb90() -> BbSpec {
    BbSpec {
        name: "bb90",
        l: 15,
        m: 3,
        a: [Monomial { x: 9, y: 0 }, Monomial { x: 0, y: 1 }, Monomial { x: 0, y: 2 }],
        b: [Monomial { x: 0, y: 0 }, Monomial { x: 2, y: 0 }, Monomial { x: 7, y: 0 }],
        k: 8,
        d: 10,
    }
}

pub fn spec_by_name(name: &str) -> Result<BbSpec> {
    match name {
        "bb72" => Ok(bb72()),
        "bb90" => Ok(bb90()),
        "bb144" | "bb144_gross" | "gross" => Ok(gross144()),
        other => bail!("unknown BB code '{other}' (expected bb72, bb90 or bb144_gross)"),
    }
}

/// A CSS code with explicit logical bases satisfying `L_X L_Z^T = I_k`.
#[derive(Clone, Debug)]
pub struct CssCode {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub hx: SparseF2,
    pub hz: SparseF2,
    pub lx: BitMat,
    pub lz: BitMat,
}

impl BbSpec {
    pub fn n(&self) -> usize {
        2 * self.l * self.m
    }

    /// Image of group element `(i, j)` under multiplication by the monomial.
    fn shift(&self, i: usize, j: usize, t: Monomial) -> usize {
        ((i + t.x) % self.l) * self.m + (j + t.y) % self.m
    }

    /// Row supports of the `lm x lm` circulant matrix of a trinomial.
    fn poly_rows(&self, poly: &[Monomial; 3]) -> Vec<Vec<u32>> {
        let mut rows = Vec::with_capacity(self.l * self.m);
        for i in 0..self.l {
            for j in 0..self.m {
                rows.push(poly.iter().map(|&t| self.shift(i, j, t) as u32).collect());
            }
        }
        rows
    }

    pub fn build(&self) -> Result<CssCode> {
        let half = self.l * self.m;
        let n = 2 * half;
        let a = self.poly_rows(&self.a);
        let b = self.poly_rows(&self.b);
        let a_s = SparseF2::from_rows(half, &a);
        let b_s = SparseF2::from_rows(half, &b);
        let at = a_s.transpose();
        let bt = b_s.transpose();
        let mut hx_rows = Vec::with_capacity(half);
        let mut hz_rows = Vec::with_capacity(half);
        for r in 0..half {
            let mut row: Vec<u32> = a_s.row(r).to_vec();
            row.extend(b_s.row(r).iter().map(|&c| c + half as u32));
            hx_rows.push(row);
            let mut row: Vec<u32> = bt.row(r).to_vec();
            row.extend(at.row(r).iter().map(|&c| c + half as u32));
            hz_rows.push(row);
        }
        let hx = SparseF2::from_rows(n, &hx_rows);
        let hz = SparseF2::from_rows(n, &hz_rows);
        let code = css_from_checks(self.name, hx, hz)?;
        ensure!(code.k == self.k, "{}: k = {} but expected {}", self.name, code.k, self.k);
        Ok(code)
    }
}

/// Builds a CSS code from its parity checks, verifying `H_X H_Z^T = 0` and deriving a
/// symplectically paired logical basis.
pub fn css_from_checks(name: &str, hx: SparseF2, hz: SparseF2) -> Result<CssCode> {
    ensure!(hx.cols == hz.cols, "H_X and H_Z act on different qubit counts");
    let n = hx.cols;
    ensure!(hz.product_is_zero(&hx), "CSS orthogonality H_Z * H_X^T != 0 (mod 2)");
    let hxd = hx.to_dense();
    let hzd = hz.to_dense();
    let k = n - hxd.rank() - hzd.rank();
    let lz0 = quotient_basis(&hxd.nullspace(), &hzd);
    let lx0 = quotient_basis(&hzd.nullspace(), &hxd);
    ensure!(lz0.rows == k && lx0.rows == k, "logical basis has wrong dimension");
    let (lx, lz) = symplectic_pair(lx0, lz0)?;
    Ok(CssCode { name: name.to_string(), n, k, hx, hz, lx, lz })
}

/// Rows of `kernel` that extend `stabilizers` to a basis of `span(kernel)`:
/// a basis of `span(kernel) / rowspace(stabilizers)`.
fn quotient_basis(kernel: &BitMat, stabilizers: &BitMat) -> BitMat {
    let cols = kernel.cols;
    let mut pivots: Vec<(usize, Vec<u64>)> = Vec::new();
    let reduce = |v: &mut Vec<u64>, pivots: &Vec<(usize, Vec<u64>)>| {
        for (p, row) in pivots {
            if (v[p >> 6] >> (p & 63)) & 1 == 1 {
                for (a, b) in v.iter_mut().zip(row.iter()) {
                    *a ^= *b;
                }
            }
        }
    };
    let lead = |v: &[u64]| -> Option<usize> {
        v.iter().enumerate().find(|(_, w)| **w != 0).map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
    };
    for i in 0..stabilizers.rows {
        let mut v = stabilizers.row(i).to_vec();
        reduce(&mut v, &pivots);
        if let Some(p) = lead(&v) {
            pivots.push((p, v));
        }
    }
    let mut out_rows = Vec::new();
    for i in 0..kernel.rows {
        let mut v = kernel.row(i).to_vec();
        reduce(&mut v, &pivots);
        if let Some(p) = lead(&v) {
            pivots.push((p, v));
            out_rows.push(kernel.row_support(i));
        }
    }
    BitMat::from_rows(cols, &out_rows)
}

/// Gram-Schmidt over F2 so that `lx[i] . lz[j] = delta_ij`.
fn symplectic_pair(mut lx: BitMat, mut lz: BitMat) -> Result<(BitMat, BitMat)> {
    let k = lx.rows;
    for i in 0..k {
        let mut found = None;
        for j in i..k {
            if parity_and(lx.row(i), lz.row(j)) == 1 {
                found = Some(j);
                break;
            }
        }
        let Some(j) = found else { bail!("logical X row {i} commutes with every remaining logical Z") };
        lz.swap_rows(i, j);
        for r in 0..k {
            if r != i && parity_and(lx.row(r), lz.row(i)) == 1 {
                lx.xor_row(i, r);
            }
        }
        for r in 0..k {
            if r != i && parity_and(lx.row(i), lz.row(r)) == 1 {
                lz.xor_row(i, r);
            }
        }
    }
    Ok((lx, lz))
}

/// Exhaustive minimum weight of a vector in `ker(check) \ rowspace(stab)` up to `max_weight`
/// (the X- or Z-distance of a CSS code).  Returns `None` when no such vector has weight
/// `<= max_weight`.  `logical_dual` must pair non-degenerately with the nontrivial classes.
pub fn min_logical_weight(check: &SparseF2, logical_dual: &BitMat, max_weight: usize) -> Option<usize> {
    let n = check.cols;
    assert!(check.rows <= 128, "syndrome must fit in 128 bits");
    assert!(logical_dual.rows <= 64);
    let ct = check.transpose();
    let col_syn: Vec<u128> = (0..n).map(|q| ct.row(q).iter().fold(0u128, |a, &r| a | (1u128 << r))).collect();
    let col_log: Vec<u64> = (0..n)
        .map(|q| (0..logical_dual.rows).fold(0u64, |a, j| a | ((logical_dual.get(j, q) as u64) << j)))
        .collect();
    fn rec(start: usize, left: usize, syn: u128, log: u64, cs: &[u128], cl: &[u64]) -> bool {
        if left == 0 {
            return syn == 0 && log != 0;
        }
        for q in start..cs.len() + 1 - left {
            if rec(q + 1, left - 1, syn ^ cs[q], log ^ cl[q], cs, cl) {
                return true;
            }
        }
        false
    }
    (1..=max_weight).find(|&w| rec(0, w, 0, 0, &col_syn, &col_log))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_chain_boundary_zero;

    #[test]
    fn bb72_bb90_and_gross_have_published_parameters() {
        assert_eq!(spec_by_name("bb90").unwrap().build().unwrap().k, 8);
        for spec in [bb72(), bb90(), gross144()] {
            let code = spec.build().unwrap();
            assert_eq!(code.n, spec.n());
            assert_eq!(code.k, spec.k);
            assert_eq!(code.hx.rows, code.n / 2);
            for r in 0..code.hx.rows {
                assert_eq!(code.hx.row(r).len(), 6, "weight-6 X checks");
                assert_eq!(code.hz.row(r).len(), 6, "weight-6 Z checks");
            }
        }
    }

    #[test]
    fn chain_boundary_vanishes_on_bb72_and_gross() {
        // C2 (X stabilizers) --d2 = H_X^T--> C1 (qubits) --d1 = H_Z--> C0 (Z checks)
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let d1 = code.hz.to_dense().to_bytes_dense();
            let d2 = code.hx.to_dense().transpose().to_bytes_dense();
            assert!(check_chain_boundary_zero(&d1, code.hz.rows, code.n, &d2, code.hx.rows));
            assert!(code.hz.product_is_zero(&code.hx));
            assert!(code.hx.product_is_zero(&code.hz));
        }
    }

    #[test]
    fn logical_bases_are_symplectic_and_commute_with_checks() {
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let hx = code.hx.to_dense();
            let hz = code.hz.to_dense();
            assert!(hz.mul(&code.lx.transpose()).is_zero(), "L_X in ker H_Z");
            assert!(hx.mul(&code.lz.transpose()).is_zero(), "L_Z in ker H_X");
            assert_eq!(code.lx.mul(&code.lz.transpose()), BitMat::identity(code.k));
            assert_eq!(hx.vstack(&code.lx).rank(), hx.rank() + code.k);
        }
    }

    #[test]
    fn bb72_distance_is_exactly_six() {
        let code = bb72().build().unwrap();
        // X-type logicals live in ker(H_Z) and are detected by L_Z; symmetric for Z.
        // Exhaustive over all supports of weight <= 6: no logical below 6, one at 6.
        assert_eq!(min_logical_weight(&code.hz, &code.lz, 6), Some(6));
        assert_eq!(min_logical_weight(&code.hx, &code.lx, 6), Some(6));
    }

    #[test]
    fn tampered_check_matrix_is_rejected() {
        let code = bb72().build().unwrap();
        let mut rows: Vec<Vec<u32>> = (0..code.hx.rows).map(|r| code.hx.row(r).to_vec()).collect();
        rows[0][0] = (rows[0][0] + 1) % code.n as u32;
        let bad = SparseF2::from_rows(code.n, &rows);
        assert!(css_from_checks("tampered", bad, code.hz.clone()).is_err());
    }
}
