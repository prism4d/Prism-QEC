//! The 3-term cellular chain complex `C2 --d2--> C1 --d1--> C0` over F2.
//!
//! * `C1`: fault mechanisms (1-cells), `C0`: detectors (0-cells), `d1 = H`.
//! * `C2`: trivial cycles (2-cells): sets of mechanisms whose detector AND observable
//!   symptoms cancel.  `d2` is their incidence into `C1`.
//!
//! The load-time identity `d1 . d2 = 0 (mod 2)` (and `L . d2 = 0`) is asserted by
//! [`ChainComplex::new`]; no complex violating it can be constructed.

use crate::bb::CssCode;
use crate::dem::Dem;
use crate::f2::SparseF2;
use anyhow::{ensure, Result};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct ChainComplex {
    /// `d1`: `n0 x n1`, row `c` lists the 1-cells on 0-cell `c`.
    pub d1: SparseF2,
    /// `d1^T`: `n1 x n0`, row `e` lists the 0-cells of 1-cell `e`.
    pub d1_t: SparseF2,
    /// `d2^T`: `n2 x n1`, row `f` lists the 1-cells on the boundary of 2-cell `f`.
    pub d2_t: SparseF2,
    /// `d2`: `n1 x n2`, row `e` lists the 2-cells containing 1-cell `e`.
    pub d2: SparseF2,
    /// Logical observable mask of every 1-cell.
    pub obs: Vec<u64>,
    pub n_obs: usize,
}

/// Parameters of the low-weight trivial-cycle search on a DEM.
#[derive(Clone, Copy, Debug)]
pub struct CycleSearch {
    /// Largest cycle weight searched (3 or 4).
    pub max_weight: usize,
    /// A cycle is kept only while one of its mechanisms lies in fewer than this many cells.
    pub per_cell_cap: usize,
}

impl Default for CycleSearch {
    fn default() -> Self {
        Self { max_weight: 4, per_cell_cap: 8 }
    }
}

impl ChainComplex {
    /// Builds the complex and asserts `d1 . d2 = 0` and `L . d2 = 0` (mod 2).
    pub fn new(d1: SparseF2, cells: &[Vec<u32>], obs: Vec<u64>, n_obs: usize) -> Result<Self> {
        ensure!(obs.len() == d1.cols, "observable mask length != number of 1-cells");
        let d2_t = SparseF2::from_rows(d1.cols, cells);
        ensure!(d1.product_is_zero(&d2_t), "chain complex violated: d1 * d2 != 0 (mod 2)");
        for f in 0..d2_t.rows {
            let acc = d2_t.row(f).iter().fold(0u64, |a, &e| a ^ obs[e as usize]);
            ensure!(acc == 0, "2-cell {f} acts nontrivially on the logical observables (L * d2 != 0)");
        }
        let d1_t = d1.transpose();
        let d2 = d2_t.transpose();
        Ok(Self { d1, d1_t, d2_t, d2, obs, n_obs })
    }

    pub fn n0(&self) -> usize {
        self.d1.rows
    }

    pub fn n1(&self) -> usize {
        self.d1.cols
    }

    pub fn n2(&self) -> usize {
        self.d2_t.rows
    }

    /// Code-capacity complex for X errors: `d1 = H_Z`, `d2 = H_X^T`, observables `L_Z`.
    pub fn from_css_x_errors(code: &CssCode) -> Result<Self> {
        let cells: Vec<Vec<u32>> = (0..code.hx.rows).map(|g| code.hx.row(g).to_vec()).collect();
        let obs = (0..code.n)
            .map(|q| (0..code.k).fold(0u64, |a, j| a | ((code.lz.get(j, q) as u64) << j)))
            .collect();
        Self::new(code.hz.clone(), &cells, obs, code.k)
    }

    /// Complex of a DEM with explicitly supplied 2-cells.
    pub fn from_dem_cells(dem: &Dem, cells: &[Vec<u32>]) -> Result<Self> {
        Self::new(dem.check_matrix(), cells, dem.err_obs.clone(), dem.n_obs)
    }

    /// Complex of a DEM whose 2-cells are the low-weight trivial cycles found by search.
    pub fn from_dem(dem: &Dem, search: &CycleSearch) -> Result<Self> {
        let cells = find_low_weight_cycles(dem, search);
        Self::from_dem_cells(dem, &cells)
    }

    /// `dim H^1 = dim ker d1 - rank d2` by dense elimination (small complexes only).
    pub fn betti1(&self) -> usize {
        let ker = self.n1() - self.d1.to_dense().rank();
        ker - self.d2_t.to_dense().rank()
    }

    /// Rank of the logical map restricted to `ker d1`, i.e. the number of independent
    /// observable classes realised by undetectable faults.
    pub fn logical_rank(&self) -> usize {
        let ker = self.d1.to_dense().nullspace();
        let mut rows = Vec::new();
        for i in 0..ker.rows {
            let mask = ker.row_support(i).iter().fold(0u64, |a, &e| a ^ self.obs[e as usize]);
            rows.push((0..self.n_obs as u32).filter(|&j| (mask >> j) & 1 == 1).collect::<Vec<u32>>());
        }
        crate::f2::BitMat::from_rows(self.n_obs.max(1), &rows).rank()
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Finds trivial cycles of weight 3 (and 4) in a DEM.
///
/// Symptoms are hashed with an XOR-homomorphic 128-bit fingerprint (a random word per
/// detector and per observable), so `sig(i) ^ sig(j)` is looked up in O(1).  Every candidate
/// is re-verified exactly.  Because a zero-sum set covers each detector an even number of
/// times, every cycle contains a pair sharing a detector, which seeds the enumeration.
pub fn find_low_weight_cycles(dem: &Dem, search: &CycleSearch) -> Vec<Vec<u32>> {
    let n_err = dem.n_err();
    let mut seed = 0x5052_4953_4D51_4543u64;
    let det_key: Vec<u128> =
        (0..dem.n_det).map(|_| ((splitmix(&mut seed) as u128) << 64) | splitmix(&mut seed) as u128).collect();
    let obs_key: Vec<u128> =
        (0..64).map(|_| ((splitmix(&mut seed) as u128) << 64) | splitmix(&mut seed) as u128).collect();
    let fp: Vec<u128> = (0..n_err)
        .map(|e| {
            let mut h = dem.dets(e).iter().fold(0u128, |a, &d| a ^ det_key[d as usize]);
            for (o, key) in obs_key.iter().enumerate() {
                if (dem.err_obs[e] >> o) & 1 == 1 {
                    h ^= key;
                }
            }
            h
        })
        .collect();
    let by_fp: HashMap<u128, u32> = fp.iter().enumerate().map(|(e, &h)| (h, e as u32)).collect();
    let h = dem.check_matrix();
    let exact_zero = |cell: &[u32]| -> bool {
        let mut acc: HashMap<u32, u8> = HashMap::new();
        let mut obs = 0u64;
        for &e in cell {
            for &d in dem.dets(e as usize) {
                *acc.entry(d).or_insert(0) ^= 1;
            }
            obs ^= dem.err_obs[e as usize];
        }
        obs == 0 && acc.values().all(|&v| v == 0)
    };
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut tri: Vec<Vec<u32>> = Vec::new();
    let mut quad: Vec<Vec<u32>> = Vec::new();
    for d in 0..dem.n_det {
        let row = h.row(d);
        for (a, &i) in row.iter().enumerate() {
            for &j in &row[a + 1..] {
                let s = fp[i as usize] ^ fp[j as usize];
                if let Some(&k) = by_fp.get(&s) {
                    if k != i && k != j {
                        let mut cell = vec![i, j, k];
                        cell.sort_unstable();
                        if exact_zero(&cell) && seen.insert(cell.clone()) {
                            tri.push(cell);
                        }
                    }
                }
                if search.max_weight < 4 {
                    continue;
                }
                // sig(i) ^ sig(j) = sig(k) ^ sig(l): k must touch the smallest detector that
                // appears exactly once among {i, j}.
                let (di, dj) = (dem.dets(i as usize), dem.dets(j as usize));
                let lone = di.iter().find(|x| !dj.contains(x)).or_else(|| dj.iter().find(|x| !di.contains(x)));
                let Some(&d0) = lone else { continue };
                for &k in h.row(d0 as usize) {
                    if k == i || k == j {
                        continue;
                    }
                    if let Some(&l) = by_fp.get(&(s ^ fp[k as usize])) {
                        if l != i && l != j && l != k {
                            let mut cell = vec![i, j, k, l];
                            cell.sort_unstable();
                            if !seen.contains(&cell) && exact_zero(&cell) {
                                seen.insert(cell.clone());
                                quad.push(cell);
                            }
                        }
                    }
                }
            }
        }
    }
    tri.sort();
    quad.sort();
    let mut load = vec![0usize; n_err];
    let mut cells = Vec::new();
    for cell in tri.into_iter().chain(quad) {
        if cell.iter().any(|&e| load[e as usize] < search.per_cell_cap) {
            for &e in &cell {
                load[e as usize] += 1;
            }
            cells.push(cell);
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bb::{bb72, gross144};

    #[test]
    fn css_complex_satisfies_boundary_identity_and_has_k_dimensional_cohomology() {
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let cx = ChainComplex::from_css_x_errors(&code).unwrap();
            assert_eq!((cx.n0(), cx.n1(), cx.n2()), (code.n / 2, code.n, code.n / 2));
            assert_eq!(cx.betti1(), 12);
            assert_eq!(cx.logical_rank(), 12);
        }
    }

    #[test]
    fn spacetime_complex_has_exactly_logical_cohomology() {
        let code = bb72().build().unwrap();
        let (dem, cells) = Dem::phenomenological(&code, 2, 0.01, 0.01).unwrap();
        let cx = ChainComplex::from_dem_cells(&dem, &cells).unwrap();
        // every undetectable, observable-preserving fault set is generated by the 2-cells
        assert_eq!(cx.betti1(), code.k);
        assert_eq!(cx.logical_rank(), code.k);
    }

    #[test]
    fn construction_rejects_non_cycles_and_logical_cells() {
        let code = bb72().build().unwrap();
        let obs: Vec<u64> =
            (0..code.n).map(|q| (0..code.k).fold(0u64, |a, j| a | ((code.lz.get(j, q) as u64) << j))).collect();
        // a single qubit is not a cycle
        assert!(ChainComplex::new(code.hz.clone(), &[vec![0u32]], obs.clone(), code.k).is_err());
        // a logical X operator is a cycle of d1 but must not be admitted as a 2-cell
        let logical = code.lx.row_support(0);
        assert!(ChainComplex::new(code.hz.clone(), &[logical], obs, code.k).is_err());
    }

    #[test]
    fn cycle_search_finds_verified_cells_in_a_spacetime_dem() {
        let code = bb72().build().unwrap();
        // one noisy round: data(q,0) + data(q,1) + 3 measurement faults are weight-5 cells,
        // while a merged "data then measure" fault creates genuine weight-3 relations.
        let (base, _) = Dem::phenomenological(&code, 1, 0.01, 0.01).unwrap();
        let mut mechs: Vec<(f64, Vec<u32>, u64)> =
            (0..base.n_err()).map(|e| (base.probs[e], base.dets(e).to_vec(), base.err_obs[e])).collect();
        // composite fault = mechanism 0 XOR mechanism 1
        let mut dets = base.dets(0).to_vec();
        dets.extend_from_slice(base.dets(1));
        mechs.push((0.001, dets, base.err_obs[0] ^ base.err_obs[1]));
        let dem = Dem::from_mechanisms(base.n_det, base.n_obs, mechs, base.det_coords.clone()).unwrap();
        let cells = find_low_weight_cycles(&dem, &CycleSearch { max_weight: 4, per_cell_cap: 64 });
        let composite = (dem.n_err() - 1) as u32;
        assert!(cells.contains(&vec![0, 1, composite]));
        let cx = ChainComplex::from_dem_cells(&dem, &cells).unwrap();
        assert!(cx.n2() >= 1);
    }
}
