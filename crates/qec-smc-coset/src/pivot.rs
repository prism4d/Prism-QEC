//! Exact manifold pivot.
//!
//! The `K` least reliable fault mechanisms (ascending BP posterior, ties by index) form the
//! suspect set.  Greedy Gauss-Jordan elimination over them in reliability order selects a
//! maximum-reliability basis of the column matroid of `d1` (its *spanning forest*): pivot
//! columns are the forest, every remaining suspect column is a chord that closes exactly one
//! fundamental cycle with the forest.  Hence
//!
//! ```text
//! e(z) = e_piv  (+)  sum_t z_t * cycle_t ,     d1 e(z) = s  for every z in F2^(K - rank)
//! ```
//!
//! parameterizes *all* syndrome-consistent errors supported on the suspect set without any
//! hard constraint.  Cycles with `L cycle = 0` are gauge directions (`im d2`), the others
//! move between logical cosets, i.e. `e(g, l) = e_piv (+) d2 g (+) L l` after a change of
//! basis of `z`; the sampler never needs that split because it tracks `L e` per walker.
//!
//! # Full-column mode (`FROZEN_V3`)
//!
//! [`build_pivot_full`] hands *every* fault mechanism to the elimination, again in reliability
//! order, so the forest solution is exactly the OSD-0 solution and the syndrome is always in
//! span.  Each of the `n_var - rank` non-pivot columns `c` then owns the nullspace vector
//! `eta_c = u_c (+) H_piv^-1 H u_c` (`H eta_c = 0`): together they span every space-time loop
//! and every logical class of the detector error model, not only the local 2-cells.  The
//! exact energy change `gain_c` of adding `eta_c` to the forest solution is evaluated for all
//! of them (the complete weight-1 sweep of combination-sweep OSD), and the `K_active` columns
//! of smallest gain (ties in reliability order) become the free coordinates, in that order.
//! The deterministic descent therefore starts with the best single column and can only
//! improve on it; pair moves and the twisted SMC then explore higher-order combinations and
//! sum the coset free energies over the same coordinates.
//!
//! Every step is integer and order-deterministic so the SM120 kernel reproduces it exactly.

use qec_core::fixed::{EN_BITS, TWIST_CLIP};
use qec_hodge_bp::TannerGraph;

pub const BUCKETS: usize = 1024;

/// Reliability bucket of a posterior LLR (1/8-nat resolution, saturating).
#[inline]
pub fn bucket_of(post: i32) -> usize {
    ((post >> 3) + 512).clamp(0, BUCKETS as i32 - 1) as usize
}

/// Budget of the deterministic zero-temperature search started at the forest solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relax {
    /// Sweeps of single-cycle descent per quench.
    pub quench_sweeps: u32,
    /// Pair exploration over the `pair_lambda` most suspicious chords (0 disables it).
    pub pair_lambda: u32,
    /// Maximum number of accepted pair moves.
    pub pair_rounds: u32,
}

impl Relax {
    pub fn quench_only(quench_sweeps: u32) -> Self {
        Self { quench_sweeps, pair_lambda: 0, pair_rounds: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Pivot {
    /// Number of suspect columns.
    pub k: usize,
    /// Suspect position -> variable index, in reliability order.
    pub list: Vec<u32>,
    pub rank: usize,
    /// False iff the syndrome is not in the span of the suspect columns.
    pub solvable: bool,
    /// Words per row-indexed bitset (`n_chk` bits).
    pub wr: usize,
    /// Per check row: suspect position of its pivot column, or -1.
    pub piv_col: Vec<i32>,
    /// Row bitset: value of `e_piv` on the pivot column of each pivot row.
    pub e_piv: Vec<u64>,
    /// Suspect positions of the free (chord) columns: ascending, or in gain order for the
    /// full-column pivot.
    pub free_pos: Vec<u32>,
    /// `n_free x wr`: for chord `t`, the pivot rows on its fundamental cycle.
    pub col_bits: Vec<u64>,
    /// Energy weight of each pivot row's column (0 on non-pivot rows) as bit-planes.
    pub planes: Vec<u64>,
    pub en_row: Vec<i32>,
    pub en_free: Vec<i32>,
    /// `S_t`: total pivot-row weight on cycle `t`.
    pub s_col: Vec<i32>,
    /// Logical action `L cycle_t`.
    pub cyc_obs: Vec<u64>,
    /// `L e_piv` and the Nishimori energy of `e_piv`.
    pub obs_piv: u64,
    pub en_piv: i32,
    /// Quenched state: the forest solution relaxed by deterministic single-cycle descent.
    /// `zq` are its chord bits, `rows_q` its pivot-row bits.
    pub zq: Vec<u64>,
    pub rows_q: Vec<u64>,
    pub en_q: i32,
    pub obs_q: u64,
    /// Twist field: energy change of toggling cycle `t` at the quenched state, clipped.
    pub theta: Vec<i32>,
}

/// Scratch reused across shots.
#[derive(Clone, Debug, Default)]
pub struct PivotScratch {
    pos: Vec<i32>,
    mat: Vec<u64>,
    used: Vec<u8>,
}

/// Weighted popcount `sum_r en_row[r]` over `r in (a & b)` via the weight bit-planes.
#[inline]
pub fn plane_dot(planes: &[u64], wr: usize, a: &[u64], b: &[u64]) -> i32 {
    let mut acc = 0u32;
    for bit in 0..EN_BITS {
        let pl = &planes[bit * wr..(bit + 1) * wr];
        let mut c = 0u32;
        for w in 0..wr {
            c += (a[w] & b[w] & pl[w]).count_ones();
        }
        acc += c << bit;
    }
    acc as i32
}

impl Pivot {
    pub fn n_free(&self) -> usize {
        self.free_pos.len()
    }

    #[inline]
    pub fn col(&self, t: usize) -> &[u64] {
        &self.col_bits[t * self.wr..(t + 1) * self.wr]
    }

    /// Energy change of flipping coordinate `t` given the pivot-row state `e_rows` and `z_t`.
    #[inline]
    pub fn delta_energy(&self, t: usize, e_rows: &[u64], z_t: bool) -> i32 {
        let own = if z_t { -self.en_free[t] } else { self.en_free[t] };
        own + self.s_col[t] - 2 * plane_dot(&self.planes, self.wr, self.col(t), e_rows)
    }

    /// Full-length bit-packed error for pivot-row state `e_rows` and chord bits `z`.
    pub fn error_bits(&self, n_var: usize, e_rows: &[u64], z: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; n_var.div_ceil(64).max(1)];
        for (r, &pc) in self.piv_col.iter().enumerate() {
            if pc >= 0 && (e_rows[r >> 6] >> (r & 63)) & 1 == 1 {
                let v = self.list[pc as usize] as usize;
                out[v >> 6] |= 1 << (v & 63);
            }
        }
        for (t, &fp) in self.free_pos.iter().enumerate() {
            if (z[t >> 6] >> (t & 63)) & 1 == 1 {
                let v = self.list[fp as usize] as usize;
                out[v >> 6] |= 1 << (v & 63);
            }
        }
        out
    }
}

/// Gain bucket of a single-column energy change (SMC units, saturating at +-32 nats).
#[inline]
pub fn gain_bucket(gain: i32) -> usize {
    (gain + 512).clamp(0, BUCKETS as i32 - 1) as usize
}

/// Builds the pivot for one shot over the `k_cols` least reliable faults.  `en` are Nishimori
/// weights (SMC units), `post` the BP posteriors that order the suspects, `syndrome` the
/// bit-packed detector syndrome.  `relax` bounds the zero-temperature search that centres the
/// twisted reference.
pub fn build_pivot(
    g: &TannerGraph,
    en: &[i32],
    post: &[i32],
    syndrome: &[u64],
    k_cols: usize,
    relax: Relax,
    scratch: &mut PivotScratch,
) -> Pivot {
    build_pivot_mode(g, en, post, syndrome, k_cols, false, relax, scratch)
}

/// Full-column pivot (see the module documentation): every fault enters the elimination and
/// the `k_active` non-pivot columns of smallest exact gain become the free coordinates.
pub fn build_pivot_full(
    g: &TannerGraph,
    en: &[i32],
    post: &[i32],
    syndrome: &[u64],
    k_active: usize,
    relax: Relax,
    scratch: &mut PivotScratch,
) -> Pivot {
    build_pivot_mode(g, en, post, syndrome, k_active, true, relax, scratch)
}

fn build_pivot_mode(
    g: &TannerGraph,
    en: &[i32],
    post: &[i32],
    syndrome: &[u64],
    k_cols: usize,
    full: bool,
    relax: Relax,
    scratch: &mut PivotScratch,
) -> Pivot {
    let n_var = g.n_var;
    let n_chk = g.n_chk;
    let k = if full { n_var } else { k_cols.min(n_var) };
    // ---- suspects: stable counting sort by reliability bucket
    let mut start = [0u32; BUCKETS + 1];
    for &p in post.iter() {
        start[bucket_of(p) + 1] += 1;
    }
    for b in 0..BUCKETS {
        start[b + 1] += start[b];
    }
    let mut list = vec![0u32; k];
    scratch.pos.clear();
    scratch.pos.resize(n_var, -1);
    for (v, &p) in post.iter().enumerate() {
        let b = bucket_of(p);
        let slot = start[b] as usize;
        start[b] += 1;
        if slot < k {
            list[slot] = v as u32;
            scratch.pos[v] = slot as i32;
        }
    }
    // ---- augmented matrix [H_K | s], one bit-packed row per check
    let wm = (k + 1).div_ceil(64);
    scratch.mat.clear();
    scratch.mat.resize(n_chk * wm, 0);
    for c in 0..n_chk {
        let row = &mut scratch.mat[c * wm..(c + 1) * wm];
        for e in g.chk_edges(c) {
            let j = scratch.pos[g.chk_var[e] as usize];
            if j >= 0 {
                row[j as usize >> 6] |= 1 << (j as usize & 63);
            }
        }
        if (syndrome[c >> 6] >> (c & 63)) & 1 == 1 {
            row[k >> 6] |= 1 << (k & 63);
        }
    }
    // ---- Gauss-Jordan in reliability order, lowest unused row as pivot
    scratch.used.clear();
    scratch.used.resize(n_chk, 0);
    let mut piv_col = vec![-1i32; n_chk];
    let mut is_free = vec![true; k];
    let mut rank = 0usize;
    for j in 0..k {
        if rank == n_chk {
            break; // every row has its pivot: all remaining columns are non-pivot
        }
        let (w, m) = (j >> 6, 1u64 << (j & 63));
        let mut pr = usize::MAX;
        for r in 0..n_chk {
            if scratch.used[r] == 0 && scratch.mat[r * wm + w] & m != 0 {
                pr = r;
                break;
            }
        }
        if pr == usize::MAX {
            continue;
        }
        scratch.used[pr] = 1;
        piv_col[pr] = j as i32;
        is_free[j] = false;
        rank += 1;
        let (lo, hi) = scratch.mat.split_at_mut(pr * wm);
        let (prow, hi) = hi.split_at_mut(wm);
        for other in lo.chunks_exact_mut(wm).chain(hi.chunks_exact_mut(wm)) {
            if other[w] & m != 0 {
                for (d, s) in other.iter_mut().zip(prow.iter()) {
                    *d ^= *s;
                }
            }
        }
    }
    // ---- consistency, pivot solution and coset structure
    let wr = n_chk.div_ceil(64).max(1);
    let aug = |r: usize| (scratch.mat[r * wm + (k >> 6)] >> (k & 63)) & 1 == 1;
    let mut solvable = true;
    let mut e_piv = vec![0u64; wr];
    let mut en_row = vec![0i32; n_chk];
    let mut planes = vec![0u64; EN_BITS * wr];
    let (mut obs_piv, mut en_piv) = (0u64, 0i32);
    for r in 0..n_chk {
        if piv_col[r] < 0 {
            solvable &= !aug(r);
            continue;
        }
        let v = list[piv_col[r] as usize] as usize;
        en_row[r] = en[v];
        for bit in 0..EN_BITS {
            if (en[v] >> bit) & 1 == 1 {
                planes[bit * wr + (r >> 6)] |= 1 << (r & 63);
            }
        }
        if aug(r) {
            e_piv[r >> 6] |= 1 << (r & 63);
            obs_piv ^= g.obs[v];
            en_piv += en[v];
        }
    }
    let free_pos: Vec<u32> = if full && solvable {
        // gain_j = en_j + sum over the pivot rows of column j of (-/+) their weight: the exact
        // energy change of eta_j at the forest solution, accumulated row by row
        let mut gain = vec![0i32; wm * 64];
        for r in 0..n_chk {
            if piv_col[r] < 0 {
                continue;
            }
            let signed = if (e_piv[r >> 6] >> (r & 63)) & 1 == 1 { -en_row[r] } else { en_row[r] };
            for (w, &word) in scratch.mat[r * wm..(r + 1) * wm].iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    gain[(w << 6) | bits.trailing_zeros() as usize] += signed;
                    bits &= bits - 1;
                }
            }
        }
        let mut start = [0u32; BUCKETS + 1];
        for j in (0..k).filter(|&j| is_free[j]) {
            gain[j] += en[list[j] as usize];
            start[gain_bucket(gain[j]) + 1] += 1;
        }
        for b in 0..BUCKETS {
            start[b + 1] += start[b];
        }
        let n_active = k_cols.min(k - rank);
        let mut chosen = vec![0u32; n_active];
        for j in (0..k).filter(|&j| is_free[j]) {
            let slot = &mut start[gain_bucket(gain[j])];
            if (*slot as usize) < n_active {
                chosen[*slot as usize] = j as u32;
            }
            *slot += 1;
        }
        chosen
    } else if full {
        Vec::new()
    } else {
        (0..k as u32).filter(|&j| is_free[j as usize]).collect()
    };
    let n_free = free_pos.len();
    let mut col_bits = vec![0u64; n_free * wr];
    let mut en_free = Vec::with_capacity(n_free);
    let mut s_col = Vec::with_capacity(n_free);
    let mut cyc_obs = Vec::with_capacity(n_free);
    for (t, &fp) in free_pos.iter().enumerate() {
        let (w, m) = (fp as usize >> 6, 1u64 << (fp as usize & 63));
        let v = list[fp as usize] as usize;
        let mut s = 0i32;
        let mut o = g.obs[v];
        for r in 0..n_chk {
            if scratch.mat[r * wm + w] & m != 0 {
                col_bits[t * wr + (r >> 6)] |= 1 << (r & 63);
                s += en_row[r];
                o ^= g.obs[list[piv_col[r] as usize] as usize];
            }
        }
        en_free.push(en[v]);
        s_col.push(s);
        cyc_obs.push(o);
    }
    let mut pivot = Pivot {
        k,
        list,
        rank,
        solvable,
        wr,
        piv_col,
        e_piv,
        free_pos,
        col_bits,
        planes,
        en_row,
        en_free,
        s_col,
        cyc_obs,
        obs_piv,
        en_piv,
        zq: Vec::new(),
        rows_q: Vec::new(),
        en_q: 0,
        obs_q: 0,
        theta: Vec::new(),
    };
    pivot.relax(relax);
    pivot
}

/// Mutable search state: pivot-row bits, chord bits, energy and logical word.
struct SearchState {
    rows: Vec<u64>,
    z: Vec<u64>,
    energy: i32,
    obs: u64,
}

impl Pivot {
    fn toggle(&self, st: &mut SearchState, t: usize, de: i32) {
        for (r, c) in st.rows.iter_mut().zip(self.col(t)) {
            *r ^= *c;
        }
        st.z[t >> 6] ^= 1 << (t & 63);
        st.energy += de;
        st.obs ^= self.cyc_obs[t];
    }

    /// Deterministic-scan descent: a cycle is toggled whenever that strictly lowers the
    /// Nishimori energy; stops after `max_sweeps` sweeps or at the first sweep without a move.
    fn quench_from(&self, st: &mut SearchState, max_sweeps: u32) {
        for _ in 0..max_sweeps {
            let mut moved = false;
            for t in 0..self.n_free() {
                let de = self.delta_energy(t, &st.rows, (st.z[t >> 6] >> (t & 63)) & 1 == 1);
                if de < 0 {
                    self.toggle(st, t, de);
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }

    /// Zero-temperature search from the forest solution, then the twist field.
    ///
    /// 1. single-cycle quench;
    /// 2. reliability-ordered pair exploration: among the `pair_lambda` most suspicious chords
    ///    (they lead the suspect list), the pair of cycles whose joint toggle lowers the energy
    ///    most is applied (ties: smallest `(i, j)`), followed by another quench; repeated until
    ///    no pair improves or `pair_rounds` pairs were taken.
    ///
    /// Every step is a deterministic function of the pivot, so the SM120 kernel mirrors it.
    pub fn relax(&mut self, relax: Relax) {
        let n_free = self.n_free();
        let mut st = SearchState {
            rows: self.e_piv.clone(),
            z: vec![0u64; n_free.div_ceil(64).max(1)],
            energy: self.en_piv,
            obs: self.obs_piv,
        };
        self.quench_from(&mut st, relax.quench_sweeps);
        let lam = (relax.pair_lambda as usize).min(n_free);
        let mut tmp = vec![0u64; self.wr];
        for _ in 0..relax.pair_rounds {
            let (mut best, mut bi, mut bj) = (0i32, usize::MAX, usize::MAX);
            for i in 0..lam {
                let zi = (st.z[i >> 6] >> (i & 63)) & 1 == 1;
                let de_i = self.delta_energy(i, &st.rows, zi);
                for (d, (r, c)) in tmp.iter_mut().zip(st.rows.iter().zip(self.col(i))) {
                    *d = *r ^ *c;
                }
                for j in i + 1..lam {
                    let tot = de_i + self.delta_energy(j, &tmp, (st.z[j >> 6] >> (j & 63)) & 1 == 1);
                    if tot < best {
                        (best, bi, bj) = (tot, i, j);
                    }
                }
            }
            if bi == usize::MAX {
                break;
            }
            let de_i = self.delta_energy(bi, &st.rows, (st.z[bi >> 6] >> (bi & 63)) & 1 == 1);
            self.toggle(&mut st, bi, de_i);
            let de_j = self.delta_energy(bj, &st.rows, (st.z[bj >> 6] >> (bj & 63)) & 1 == 1);
            self.toggle(&mut st, bj, de_j);
            self.quench_from(&mut st, relax.quench_sweeps);
        }
        self.theta = (0..n_free)
            .map(|t| self.delta_energy(t, &st.rows, (st.z[t >> 6] >> (t & 63)) & 1 == 1).clamp(-TWIST_CLIP, TWIST_CLIP))
            .collect();
        self.zq = st.z;
        self.rows_q = st.rows;
        self.en_q = st.energy;
        self.obs_q = st.obs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::bb::{bb72, gross144};
    use qec_core::complex::ChainComplex;
    use qec_core::fixed::{energy_from_llr, llr_from_prob, mix64};

    fn random_error(n: usize, p: f64, seed: u64) -> Vec<u64> {
        let mut e = vec![0u64; n.div_ceil(64)];
        for v in 0..n {
            let u = (mix64(seed ^ (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 / (1u64 << 53) as f64;
            if u < p {
                e[v >> 6] ^= 1 << (v & 63);
            }
        }
        e
    }

    #[test]
    fn pivot_satisfies_the_syndrome_identically_and_cycles_are_closed() {
        for spec in [bb72(), gross144()] {
            let code = spec.build().unwrap();
            let g = TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap());
            let en = vec![energy_from_llr(llr_from_prob(0.03)); g.n_var];
            let mut scratch = PivotScratch::default();
            for seed in 0..25u64 {
                let truth = random_error(g.n_var, 0.05, seed);
                let syn = g.syndrome_of(&truth);
                // posterior: noisy pseudo-reliabilities, errors slightly less reliable
                let post: Vec<i32> = (0..g.n_var)
                    .map(|v| {
                        let noise = (mix64(seed * 977 + v as u64) % 400) as i32;
                        if (truth[v >> 6] >> (v & 63)) & 1 == 1 { noise - 150 } else { noise + 50 }
                    })
                    .collect();
                let p = build_pivot(&g, &en, &post, &syn, g.n_var, Relax::quench_only(0), &mut scratch);
                assert!(p.solvable, "all columns are suspects, so the syndrome is always in span");
                assert_eq!(p.rank + p.n_free(), g.n_var);
                let zero_z = vec![0u64; p.n_free().div_ceil(64).max(1)];
                let e0 = p.error_bits(g.n_var, &p.e_piv, &zero_z);
                assert_eq!(g.syndrome_of(&e0), syn, "d1 * e_piv == s");
                assert_eq!(g.observables_of(&e0), p.obs_piv);
                // every fundamental cycle is in ker d1 and its logical action is tracked
                let zero_rows = vec![0u64; p.wr];
                for t in 0..p.n_free() {
                    let mut z = zero_z.clone();
                    z[t >> 6] |= 1 << (t & 63);
                    let mut rows = zero_rows.clone();
                    for (w, c) in rows.iter_mut().zip(p.col(t)) {
                        *w ^= *c;
                    }
                    let cyc = p.error_bits(g.n_var, &rows, &z);
                    assert!(g.syndrome_of(&cyc).iter().all(|&w| w == 0));
                    assert_eq!(g.observables_of(&cyc), p.cyc_obs[t]);
                }
                // twist = exact energy change of a single cycle move from the pivot
                for t in 0..p.n_free() {
                    let mut rows = p.e_piv.clone();
                    for (w, c) in rows.iter_mut().zip(p.col(t)) {
                        *w ^= *c;
                    }
                    let mut z = zero_z.clone();
                    z[t >> 6] |= 1 << (t & 63);
                    let moved = p.error_bits(g.n_var, &rows, &z);
                    let weight = |e: &[u64]| e.iter().map(|w| w.count_ones() as i32).sum::<i32>() * en[0];
                    assert_eq!(p.delta_energy(t, &p.e_piv, false), weight(&moved) - weight(&e0));
                }
            }
        }
    }

    /// Full-column pivot: always solvable, full rank, free coordinates sorted by exact gain,
    /// and the quench is never worse than the best single column (the weight-1 sweep of
    /// combination-sweep OSD) nor than the prefix pivot's forest solution.
    #[test]
    fn full_column_pivot_dominates_the_single_column_sweep() {
        let code = gross144().build().unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap());
        let en: Vec<i32> = (0..g.n_var).map(|v| 60 + (mix64(v as u64 + 5) % 40) as i32).collect();
        let mut scratch = PivotScratch::default();
        let (mut improved, mut beat_prefix) = (0, 0);
        for seed in 0..25u64 {
            let truth = random_error(g.n_var, 0.04, seed + 300);
            let syn = g.syndrome_of(&truth);
            let post: Vec<i32> = (0..g.n_var)
                .map(|v| {
                    let noise = (mix64(seed * 613 + v as u64) % 400) as i32;
                    if (truth[v >> 6] >> (v & 63)) & 1 == 1 { noise - 120 } else { noise + 40 }
                })
                .collect();
            let k_active = 24;
            let p = build_pivot_full(&g, &en, &post, &syn, k_active, Relax::quench_only(8), &mut scratch);
            assert!(p.solvable && p.k == g.n_var);
            assert_eq!(p.n_free(), k_active.min(g.n_var - p.rank));
            // the forest solution is the OSD-0 solution: it satisfies the syndrome exactly
            let zero_z = vec![0u64; p.n_free().div_ceil(64).max(1)];
            let e0 = p.error_bits(g.n_var, &p.e_piv, &zero_z);
            assert_eq!(g.syndrome_of(&e0), syn);
            let weight = |e: &[u64]| (0..g.n_var).filter(|&v| (e[v >> 6] >> (v & 63)) & 1 == 1).map(|v| en[v]).sum::<i32>();
            assert_eq!(weight(&e0), p.en_piv);
            // free coordinates are in non-decreasing gain order and closed cycles
            let gains: Vec<i32> = (0..p.n_free()).map(|t| p.delta_energy(t, &p.e_piv, false)).collect();
            assert!(gains.windows(2).all(|w| gain_bucket(w[0]) <= gain_bucket(w[1])), "{gains:?}");
            // exhaustive single-column sweep over every non-pivot column, by brute force
            let mut is_piv = vec![false; g.n_var];
            for &pc in p.piv_col.iter().filter(|&&pc| pc >= 0) {
                is_piv[p.list[pc as usize] as usize] = true;
            }
            let reference = build_pivot(&g, &en, &post, &syn, g.n_var, Relax::quench_only(0), &mut scratch);
            assert_eq!((reference.rank, reference.en_piv, reference.obs_piv), (p.rank, p.en_piv, p.obs_piv));
            let best_single = (0..reference.n_free()).map(|t| reference.delta_energy(t, &reference.e_piv, false)).min().unwrap();
            assert_eq!(gain_bucket(gains[0]), gain_bucket(best_single), "the first free coordinate is the best single column");
            assert!(p.en_q <= p.en_piv + gains[0].min(0), "quench {} vs best single {}", p.en_q, p.en_piv + gains[0]);
            let eq = p.error_bits(g.n_var, &p.rows_q, &p.zq);
            assert_eq!(g.syndrome_of(&eq), syn);
            assert_eq!((weight(&eq), g.observables_of(&eq)), (p.en_q, p.obs_q));
            improved += (p.en_q < p.en_piv) as u32;
            // the prefix pivot over the same number of suspects as rank + k_active sees less
            let prefix = build_pivot(&g, &en, &post, &syn, p.rank + k_active, Relax::quench_only(8), &mut scratch);
            beat_prefix += (!prefix.solvable || p.en_q < prefix.en_q) as u32;
        }
        assert!(improved >= 10, "the single-column sweep improved only {improved}/25 forest solutions");
        eprintln!("[full-column] improved {improved}/25 forest solutions; strictly below the prefix pivot in {beat_prefix}/25");
    }

    #[test]
    fn pivot_basis_prefers_the_least_reliable_columns() {
        let code = bb72().build().unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap());
        let en = vec![64; g.n_var];
        let truth = random_error(g.n_var, 0.06, 5);
        assert!(truth.iter().map(|w| w.count_ones()).sum::<u32>() >= 3);
        let syn = g.syndrome_of(&truth);
        // exact knowledge: true faults strongly negative -> they must all be pivots or
        // reproduce the truth exactly up to the logical class
        let post: Vec<i32> =
            (0..g.n_var).map(|v| if (truth[v >> 6] >> (v & 63)) & 1 == 1 { -500 } else { 500 + v as i32 }).collect();
        let mut scratch = PivotScratch::default();
        let p = build_pivot(&g, &en, &post, &syn, g.n_var, Relax::quench_only(0), &mut scratch);
        let e0 = p.error_bits(g.n_var, &p.e_piv, &vec![0u64; p.n_free().div_ceil(64).max(1)]);
        assert_eq!(e0, truth, "with a perfect reliability order the forest solution is the truth");
        // too few suspects: reported as unsolvable rather than returning a wrong solution
        let tiny = build_pivot(&g, &en, &post, &syn, 2, Relax::quench_only(0), &mut scratch);
        assert!(!tiny.solvable);
    }

    #[test]
    fn quench_never_raises_energy_and_ends_in_a_single_cycle_minimum() {
        let code = gross144().build().unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_css_x_errors(&code).unwrap());
        let en: Vec<i32> = (0..g.n_var).map(|v| 40 + (mix64(v as u64) % 60) as i32).collect();
        let mut scratch = PivotScratch::default();
        let mut improved = 0;
        let mut pair_gain = 0;
        for seed in 0..20u64 {
            let truth = random_error(g.n_var, 0.04, 100 + seed);
            let syn = g.syndrome_of(&truth);
            // uninformative reliabilities: the forest solution is far from minimal
            let post: Vec<i32> = (0..g.n_var).map(|v| (mix64(seed * 31 + v as u64) % 1000) as i32).collect();
            let p = build_pivot(&g, &en, &post, &syn, g.n_var, Relax::quench_only(64), &mut scratch);
            assert!(p.en_q <= p.en_piv);
            // pair exploration can only lower the energy further and keeps every invariant
            let relax = Relax { quench_sweeps: 64, pair_lambda: 48, pair_rounds: 16 };
            let pp = build_pivot(&g, &en, &post, &syn, g.n_var, relax, &mut scratch);
            assert!(pp.en_q <= p.en_q);
            pair_gain += (pp.en_q < p.en_q) as u32;
            let ee = pp.error_bits(g.n_var, &pp.rows_q, &pp.zq);
            assert_eq!(g.syndrome_of(&ee), syn);
            assert_eq!(g.observables_of(&ee), pp.obs_q);
            let direct: i32 = (0..g.n_var).filter(|&v| (ee[v >> 6] >> (v & 63)) & 1 == 1).map(|v| en[v]).sum();
            assert_eq!(direct, pp.en_q);
            assert!(pp.theta.iter().all(|&t| t >= 0));
            improved += (p.en_q < p.en_piv) as u32;
            let e = p.error_bits(g.n_var, &p.rows_q, &p.zq);
            assert_eq!(g.syndrome_of(&e), syn, "the quenched state still satisfies the syndrome");
            assert_eq!(g.observables_of(&e), p.obs_q);
            let direct: i32 = (0..g.n_var).filter(|&v| (e[v >> 6] >> (v & 63)) & 1 == 1).map(|v| en[v]).sum();
            assert_eq!(direct, p.en_q, "incremental energy bookkeeping is exact");
            assert!(p.theta.iter().all(|&t| t >= 0), "no single cycle move lowers the energy");
        }
        assert!(improved >= 15, "quench improved only {improved}/20 forest solutions");
        assert!(pair_gain >= 1, "pair exploration never found a lower state");
    }
}
