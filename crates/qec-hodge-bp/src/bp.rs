//! Wavefront normalized min-sum BP with inline velocity-space Rayleigh gauge damping.
//!
//! Iteration `t` computes the plain min-sum proposal `l_BP^(t)` from the beliefs `l^(t-1)`.
//! During the unperturbed prefix (`t <= t_clean`) the proposal is accepted as is.  Afterwards
//! the coexact projection of the *velocity* on the frustrated sub-complex is removed:
//!
//! ```text
//! l^(t) = l_BP^(t) - eta * Pi_{F_t}( l_BP^(t) - l^(t-1) ),
//! ```
//!
//! where `F_t` are the 2-cells containing a fault adjacent to a parity-violating check of
//! `l^(t-1)` or a fault of margin `|l_BP^(t)| < margin`.  Nothing is accumulated: the
//! correction is a function of the current velocity only, so every exact BP fixed point is a
//! fixed point of the damped iteration with zero bias.
//!
//! All state is integer (see `qec_core::fixed`), so this reference and the SM120 kernel
//! produce identical messages, beliefs and hard decisions.  Wavefront scheduling of the check
//! updates is *exact* lazy evaluation of flooding: a check's outputs are a function of its
//! variables' beliefs and of its own previous outgoing messages (the extrinsic subtraction),
//! so it is recomputed iff one of those beliefs changed since its last update or that update
//! itself changed one of its messages.

use crate::graph::TannerGraph;
use crate::hodge::{project_fixed, HodgeScratch};
use qec_core::fixed::{site_disorder, tanh_table, MSG_CLIP, POST_CLIP};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BpConfig {
    /// Iteration budget `T_max`.
    pub t_max: u32,
    /// Unperturbed prefix `T_clean`: plain min-sum for `t <= t_clean`.
    pub t_clean: u32,
    /// Normalized min-sum scaling `alpha * 256`.
    pub alpha_q8: i32,
    /// Rayleigh damping `eta * 256` (0 disables the projection: plain min-sum throughout).
    pub eta_q8: i32,
    /// Metric regulator `eps` in Q12.
    pub eps_q12: i32,
    /// Quenched disorder strength `kappa * 256`: `eps_v = eps * (1 + kappa * xi_v)`, `xi_v` in [0, 1).
    pub disorder_q8: i32,
    pub disorder_seed: u64,
    /// Low-margin threshold (BP units) that admits a fault's 2-cells into `F_t`.
    pub margin: i32,
    /// Lazy (dirty-flag) check scheduling; `false` recomputes every check every iteration.
    pub wavefront: bool,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            t_max: 64,
            t_clean: 16,
            alpha_q8: 160,
            eta_q8: 64,
            eps_q12: 1024,
            disorder_q8: 256,
            disorder_seed: 0x5052_4953_4D48_4447,
            margin: 32,
            wavefront: true,
        }
    }
}

impl BpConfig {
    pub fn plain(t_max: u32, alpha_q8: i32) -> Self {
        Self { t_max, alpha_q8, eta_q8: 0, ..Self::default() }
    }

    /// True iff iteration `it` is damped.
    #[inline]
    pub fn damped(&self, it: u32) -> bool {
        self.eta_q8 != 0 && it > self.t_clean
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BpOutcome {
    pub converged: bool,
    pub iterations: u32,
    pub unsat: u32,
    /// Iterations in which a nonzero gauge correction was applied, and which ones (bit `t - 1`
    /// for iteration `t <= 64`).
    pub damped_steps: u32,
    pub damped_mask: u64,
    /// Check-node updates actually executed (the wavefront work measure).
    pub check_updates: u64,
}

/// Per-thread reusable decoder state.
#[derive(Clone, Debug, Default)]
pub struct BpWork {
    pub c2v: Vec<i16>,
    /// Plain min-sum proposal `l_BP^(t)`.
    pub bp: Vec<i32>,
    /// Beliefs `l^(t)` actually used (proposal minus gauge correction).
    pub post: Vec<i32>,
    /// Beliefs at the end of the unperturbed prefix (`l^(min(t, t_clean))`).
    pub snap: Vec<i32>,
    par: Vec<u8>,
    chk_flag: Vec<u8>,
    var_flag: Vec<u8>,
    chk_queue: Vec<u32>,
    chk_next: Vec<u32>,
    var_queue: Vec<u32>,
    hot: Vec<u8>,
    active: Vec<u8>,
    vel: Vec<i32>,
    corr: Vec<i32>,
    scratch: HodgeScratch,
}

impl BpWork {
    /// Bit-packed hard decision `post < 0`.
    pub fn hard_bits(&self) -> Vec<u64> {
        let mut out = vec![0u64; self.post.len().div_ceil(64).max(1)];
        for (v, &p) in self.post.iter().enumerate() {
            if p < 0 {
                out[v >> 6] |= 1 << (v & 63);
            }
        }
        out
    }
}

pub struct HodgeBp<'g> {
    pub graph: &'g TannerGraph,
    pub cfg: BpConfig,
    pub tanh: Vec<u16>,
    pub eps_site: Vec<i32>,
}

/// Per-site regulator `eps_v` (Q12) from the quenched disorder field.
pub fn site_regulators(n_var: usize, cfg: &BpConfig) -> Vec<i32> {
    (0..n_var)
        .map(|v| {
            let xi = site_disorder(cfg.disorder_seed, v as u32) as i64;
            let extra = (cfg.eps_q12 as i64 * cfg.disorder_q8 as i64 * xi) >> 16;
            (cfg.eps_q12 as i64 + extra).max(1) as i32
        })
        .collect()
}

impl<'g> HodgeBp<'g> {
    pub fn new(graph: &'g TannerGraph, cfg: BpConfig) -> Self {
        Self { graph, cfg, tanh: tanh_table(), eps_site: site_regulators(graph.n_var, &cfg) }
    }

    /// Decodes one syndrome (bit-packed, `n_chk` bits) under prior LLRs `weights` (BP units).
    pub fn decode(&self, work: &mut BpWork, weights: &[i32], syndrome: &[u64]) -> BpOutcome {
        let g = self.graph;
        let cfg = &self.cfg;
        let syn = |c: usize| ((syndrome[c >> 6] >> (c & 63)) & 1) as u8;
        work.c2v.clear();
        work.c2v.resize(g.n_edge, 0);
        work.bp.clear();
        work.bp.extend_from_slice(weights);
        work.post.clear();
        work.post.extend_from_slice(weights);
        work.par.clear();
        work.par.resize(g.n_chk, 0);
        work.chk_flag.clear();
        work.chk_flag.resize(g.n_chk, 1);
        work.var_flag.clear();
        work.var_flag.resize(g.n_var, 0);
        work.chk_queue.clear();
        work.chk_queue.extend(0..g.n_chk as u32);
        work.var_queue.clear();
        work.corr.clear();
        work.corr.resize(g.n_var, 0);
        let mut unsat = 0u32;
        for c in 0..g.n_chk {
            let mut p = 0u8;
            for e in g.chk_edges(c) {
                p ^= (work.post[g.chk_var[e] as usize] < 0) as u8;
            }
            work.par[c] = p;
            unsat += (p != syn(c)) as u32;
        }
        let mut out = BpOutcome { converged: unsat == 0, unsat, ..BpOutcome::default() };
        let mut snapped = false;
        if !out.converged {
            for it in 1..=cfg.t_max {
                out.iterations = it;
                // ---- check phase: recompute outgoing messages of dirty checks from l^(t-1)
                if !cfg.wavefront {
                    work.chk_queue.clear();
                    work.chk_queue.extend(0..g.n_chk as u32);
                }
                work.chk_next.clear();
                for qi in 0..work.chk_queue.len() {
                    let c = work.chk_queue[qi] as usize;
                    work.chk_flag[c] = 0;
                    out.check_updates += 1;
                    let mut moved = false;
                    let mut sign = syn(c);
                    let (mut min1, mut min2, mut arg) = (MSG_CLIP, MSG_CLIP, usize::MAX);
                    for e in g.chk_edges(c) {
                        let m = (work.post[g.chk_var[e] as usize] - work.c2v[e] as i32).clamp(-MSG_CLIP, MSG_CLIP);
                        sign ^= (m < 0) as u8;
                        let a = m.abs();
                        if a < min1 {
                            min2 = min1;
                            min1 = a;
                            arg = e;
                        } else if a < min2 {
                            min2 = a;
                        }
                    }
                    for e in g.chk_edges(c) {
                        let v = g.chk_var[e] as usize;
                        let m = (work.post[v] - work.c2v[e] as i32).clamp(-MSG_CLIP, MSG_CLIP);
                        let mag = ((if e == arg { min2 } else { min1 }) * cfg.alpha_q8) >> 8;
                        let neg = sign ^ (m < 0) as u8;
                        let new = (if neg == 1 { -mag } else { mag }) as i16;
                        if new != work.c2v[e] {
                            work.c2v[e] = new;
                            moved = true;
                            if work.var_flag[v] == 0 {
                                work.var_flag[v] = 1;
                                work.var_queue.push(v as u32);
                            }
                        }
                    }
                    if moved {
                        work.chk_next.push(c as u32);
                    }
                }
                work.chk_queue.clear();
                // a check whose own messages moved must be revisited: they enter its next
                // extrinsic subtraction even if every belief around it cancels back
                for qi in 0..work.chk_next.len() {
                    let c = work.chk_next[qi] as usize;
                    work.chk_flag[c] = 1;
                    work.chk_queue.push(c as u32);
                }
                // ---- variable phase: min-sum proposal l_BP^(t) of the dirty variables
                if !cfg.wavefront {
                    work.var_queue.clear();
                    work.var_queue.extend(0..g.n_var as u32);
                }
                for qi in 0..work.var_queue.len() {
                    let v = work.var_queue[qi] as usize;
                    work.var_flag[v] = 0;
                    let mut acc = weights[v];
                    for s in g.var_slots(v) {
                        acc += work.c2v[g.var_edge[s] as usize] as i32;
                    }
                    work.bp[v] = acc;
                }
                work.var_queue.clear();
                let damp = cfg.damped(it) && g.n_cell > 0;
                if damp && !snapped {
                    work.snap.clear();
                    work.snap.extend_from_slice(&work.post);
                    snapped = true;
                }
                if damp {
                    // ---- frustrated sub-complex F_t and velocity projection
                    work.hot.clear();
                    work.hot.resize(g.n_var, 0);
                    for c in 0..g.n_chk {
                        if work.par[c] != syn(c) {
                            for e in g.chk_edges(c) {
                                work.hot[g.chk_var[e] as usize] = 1;
                            }
                        }
                    }
                    for v in 0..g.n_var {
                        if work.bp[v].abs() < cfg.margin {
                            work.hot[v] = 1;
                        }
                    }
                    work.active.clear();
                    work.active.extend(
                        (0..g.n_cell).map(|f| g.cell_vars(f).iter().any(|&v| work.hot[v as usize] == 1) as u8),
                    );
                    work.vel.clear();
                    work.vel.extend(work.bp.iter().zip(work.post.iter()).map(|(&b, &p)| b - p));
                    project_fixed(
                        g,
                        &work.vel,
                        &work.bp,
                        &self.tanh,
                        &self.eps_site,
                        cfg.eta_q8,
                        &work.active,
                        &mut work.scratch,
                        &mut work.corr,
                    );
                    let applied = work.corr.iter().any(|&c| c != 0);
                    out.damped_steps += applied as u32;
                    out.damped_mask |= (applied as u64) << ((it - 1) & 63);
                }
                // ---- accept l^(t) = l_BP^(t) - correction
                for v in 0..g.n_var {
                    let new =
                        if damp { (work.bp[v] - work.corr[v]).clamp(-POST_CLIP, POST_CLIP) } else { work.bp[v] };
                    let old = work.post[v];
                    if new != old {
                        work.post[v] = new;
                        touch(g, work, v, old, new, &syn, &mut unsat);
                    }
                }
                if unsat == 0 {
                    out.converged = true;
                    break;
                }
            }
        }
        out.unsat = unsat;
        if !snapped {
            work.snap.clear();
            work.snap.extend_from_slice(&work.post);
        }
        out
    }
}

/// Bookkeeping after the belief of `v` moved from `old` to `new`: neighbouring checks become
/// dirty and, if the hard decision flipped, their parities and the unsatisfied count update.
#[inline]
fn touch(
    g: &TannerGraph,
    work: &mut BpWork,
    v: usize,
    old: i32,
    new: i32,
    syn: &impl Fn(usize) -> u8,
    unsat: &mut u32,
) {
    let flipped = (old < 0) != (new < 0);
    for s in g.var_slots(v) {
        let c = g.var_chk[s] as usize;
        if work.chk_flag[c] == 0 {
            work.chk_flag[c] = 1;
            work.chk_queue.push(c as u32);
        }
        if flipped {
            work.par[c] ^= 1;
            if work.par[c] != syn(c) {
                *unsat += 1;
            } else {
                *unsat -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::bb::{bb72, gross144, CssCode};
    use qec_core::complex::ChainComplex;
    use qec_core::dem::Dem;
    use qec_core::fixed::{llr_from_prob, mix64};

    fn graph_of(code: &CssCode) -> TannerGraph {
        TannerGraph::from_complex(&ChainComplex::from_css_x_errors(code).unwrap())
    }

    fn bits_from(n: usize, support: &[usize]) -> Vec<u64> {
        let mut e = vec![0u64; n.div_ceil(64)];
        for &q in support {
            e[q >> 6] ^= 1 << (q & 63);
        }
        e
    }

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
    fn zero_syndrome_converges_immediately_and_single_faults_decode() {
        let code = gross144().build().unwrap();
        let g = graph_of(&code);
        let w = vec![llr_from_prob(0.01); g.n_var];
        let bp = HodgeBp::new(&g, BpConfig::default());
        let mut work = BpWork::default();
        let zero = vec![0u64; g.n_chk.div_ceil(64)];
        let out = bp.decode(&mut work, &w, &zero);
        assert!(out.converged && out.iterations == 0);
        for q in [0usize, 17, 71, 100, 143] {
            let e = bits_from(g.n_var, &[q]);
            let out = bp.decode(&mut work, &w, &g.syndrome_of(&e));
            assert!(out.converged, "single fault on {q}");
            assert_eq!(work.hard_bits(), e);
        }
    }

    #[test]
    fn wavefront_is_bit_identical_to_flooding_and_does_less_work() {
        let code = gross144().build().unwrap();
        let g = graph_of(&code);
        let w = vec![llr_from_prob(0.02); g.n_var];
        let (mut lazy_updates, mut flood_updates) = (0u64, 0u64);
        for eta_q8 in [0, 64, 192] {
            let lazy_cfg = BpConfig { eta_q8, t_clean: 3, t_max: 96, ..BpConfig::default() };
            let flood_cfg = BpConfig { wavefront: false, ..lazy_cfg };
            let (lazy, flood) = (HodgeBp::new(&g, lazy_cfg), HodgeBp::new(&g, flood_cfg));
            let (mut wa, mut wb) = (BpWork::default(), BpWork::default());
            for seed in 0..40u64 {
                let syn = g.syndrome_of(&random_error(g.n_var, 0.03, seed));
                let a = lazy.decode(&mut wa, &w, &syn);
                let b = flood.decode(&mut wb, &w, &syn);
                assert_eq!(
                    (a.converged, a.iterations, a.unsat, a.damped_steps),
                    (b.converged, b.iterations, b.unsat, b.damped_steps)
                );
                assert_eq!(a.damped_mask, b.damped_mask);
                assert_eq!(wa.c2v, wb.c2v);
                assert_eq!(wa.post, wb.post);
                assert_eq!(wa.snap, wb.snap);
                lazy_updates += a.check_updates;
                flood_updates += b.check_updates;
            }
        }
        assert!(lazy_updates < flood_updates, "wavefront {lazy_updates} vs flooding {flood_updates}");
    }

    #[test]
    fn converged_output_always_satisfies_the_syndrome() {
        let code = bb72().build().unwrap();
        let (dem, cells) = Dem::phenomenological(&code, 3, 0.01, 0.01).unwrap();
        let g = TannerGraph::from_complex(&ChainComplex::from_dem_cells(&dem, &cells).unwrap());
        let w: Vec<i32> = dem.probs.iter().map(|&p| llr_from_prob(p)).collect();
        let bp = HodgeBp::new(&g, BpConfig::default());
        let mut work = BpWork::default();
        let mut converged = 0;
        for seed in 0..60u64 {
            let syn = g.syndrome_of(&random_error(g.n_var, 0.01, seed));
            let out = bp.decode(&mut work, &w, &syn);
            if out.converged {
                converged += 1;
                assert_eq!(g.syndrome_of(&work.hard_bits()), syn);
            } else {
                assert!(out.unsat > 0);
            }
        }
        assert!(converged >= 30, "only {converged}/60 converged");
    }

    #[test]
    fn convergence_inside_the_clean_prefix_is_identical_to_plain_min_sum() {
        // Zero fixed-point shift: the damping acts on velocity only, so whenever plain
        // min-sum settles within the unperturbed prefix the damped decoder returns the very
        // same state and never applies a correction.
        let code = gross144().build().unwrap();
        let g = graph_of(&code);
        let w = vec![llr_from_prob(0.01); g.n_var];
        let plain = HodgeBp::new(&g, BpConfig::plain(64, 160));
        let damped = HodgeBp::new(&g, BpConfig::default());
        let (mut wa, mut wb) = (BpWork::default(), BpWork::default());
        let mut checked = 0;
        for seed in 0..60u64 {
            let syn = g.syndrome_of(&random_error(g.n_var, 0.01, seed));
            let a = plain.decode(&mut wa, &w, &syn);
            if a.converged && a.iterations <= damped.cfg.t_clean {
                let b = damped.decode(&mut wb, &w, &syn);
                assert_eq!((b.converged, b.iterations, b.damped_steps), (true, a.iterations, 0));
                assert_eq!(wa.post, wb.post);
                checked += 1;
            }
        }
        assert!(checked >= 20, "only {checked} shots converged inside the clean prefix");
    }

    /// Half of a weight-6 X stabilizer is degenerate with the complementary half.  Plain
    /// min-sum locks into a period-2 oscillation of the pure gauge mode (all six beliefs
    /// equal, flipping sign every iteration) and never converges.  Velocity-space damping
    /// dissipates the oscillation and the iteration settles in the correct logical class.
    #[test]
    fn rayleigh_damping_resolves_symmetric_stabilizer_trapping_sets() {
        let code = bb72().build().unwrap();
        let g = graph_of(&code);
        let w = vec![llr_from_prob(0.02); g.n_var];
        let plain = HodgeBp::new(&g, BpConfig::plain(64, 160));
        let damped = HodgeBp::new(&g, BpConfig { t_clean: 8, ..BpConfig::default() });
        let mut work = BpWork::default();
        let (mut plain_ok, mut damped_ok, mut total) = (0, 0, 0);
        for f in 0..g.n_cell {
            let cell = g.cell_vars(f);
            for mask in 0u32..64 {
                if mask.count_ones() != 3 || mask & 1 == 0 {
                    continue; // one representative of each complementary pair
                }
                let support: Vec<usize> = (0..6).filter(|b| (mask >> b) & 1 == 1).map(|b| cell[b] as usize).collect();
                let e = bits_from(g.n_var, &support);
                let syn = g.syndrome_of(&e);
                let truth = g.observables_of(&e);
                total += 1;
                let a = plain.decode(&mut work, &w, &syn);
                plain_ok += (a.converged && g.observables_of(&work.hard_bits()) == truth) as u32;
                let b = damped.decode(&mut work, &w, &syn);
                if b.converged {
                    assert_eq!(g.syndrome_of(&work.hard_bits()), syn);
                }
                damped_ok += (b.converged && g.observables_of(&work.hard_bits()) == truth) as u32;
            }
        }
        assert_eq!(plain_ok, 0, "plain min-sum is expected to stall on every symmetric half-stabilizer");
        assert!(damped_ok * 100 >= total * 95, "damped min-sum resolved only {damped_ok}/{total}");
    }
}
