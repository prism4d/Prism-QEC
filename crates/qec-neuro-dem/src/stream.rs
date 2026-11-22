//! Per-stream online state and detector-error-model weight surgery.
//!
//! A *stream* is the time-ordered sequence of syndrome blocks of one device.  Each block is
//! `layers` detector layers of `n_checks` checks.  For every block, in causal order:
//!
//! 1. each check's BOCPD filter assimilates its spike of every layer (exposure-normalised by
//!    the nominal schedule); a check whose posterior rate multiplier crosses `hot_on` becomes
//!    HOT and the transfer-entropy counters of all its coupler pairs restart from the layers
//!    of the new regime only (the MAP run length, replayed from a short spike history), and
//!    again when it cools below `hot_off`;
//! 2. every coupler pair class accumulates its conditional joint counts;
//! 3. surgery: a fault mechanism with at least one *active* detector pair (excess transfer
//!    entropy over the nominal circuit) and no vetoing pair is re-weighted to the smallest
//!    active pair rate.  An inactive pair vetoes unless it is *masked* (one of its trains is
//!    a fair coin through other sources and therefore carries no pairwise information).
//!    A fully depolarised (TLS-dropout) qubit drives that rate to 1/2, i.e. `w = 0`: a known
//!    erasure.  A dormant correlated-crosstalk hyperedge is activated by the same rule.
//!    Two-detector mechanisms only receive the rate left unexplained by larger mechanisms.
//!
//! From `FROZEN_V2` on, mechanisms with three or more detectors are estimated by the k-point
//! *coincidence* of their own detector pattern instead of the pairwise rule: a mechanism
//! *class* is the translation-invariant pattern `(check, layer offset)` of such a mechanism,
//! and every class counts the samples in which all of its detectors fire together.  Against
//! the nominal circuit that count is a Poisson likelihood ratio; an active class is
//! re-weighted to its coincidence rate, minus the rate of the active classes whose pattern
//! contains it.  This is what resolves simultaneous (space-like) correlated faults such as a
//! crosstalk hyperedge, whose fifteen detector pairs are individually far too weak.  A
//! changepoint of the class's checks restarts the counter from the layers of the new regime
//! (replayed from the history ring) whenever that regime shows the higher rate, so a dropout
//! is re-weighted the moment the changepoint filter fires.

use crate::bocpd::Bocpd;
use crate::tide::{
    bin_index, bump, coincidence_bump, coincidence_g_q16, g_statistic_q16, log_table_q16, pair_masked, pair_rate_q12, total,
    PairClass,
};
use anyhow::{ensure, Result};
use qec_core::dem::Dem;
use qec_core::fixed::{energy_from_llr, llr_from_prob, PRIOR_CLIP};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuroConfig {
    /// Changepoint hazard per detector layer.
    pub hazard: f64,
    /// Rate-multiplier thresholds of the HOT state machine (hysteresis).
    pub hot_on: f64,
    pub hot_off: f64,
    /// Minimum samples before a pair class may become active.
    pub n_min: u32,
    /// G-statistic threshold on the excess transfer entropy (nats) for a pair of cold checks.
    pub g_threshold: u32,
    /// Threshold when BOCPD already holds both checks HOT (two-stage test: the changepoint
    /// posterior is the prior evidence, the transfer entropy only has to confirm coupling).
    pub g_threshold_hot: u32,
    /// Saturation of the pair counters (<= 255); bins halve when the total reaches it.
    pub count_cap: u32,
    /// Blocks sampled from the nominal model for the interventional reference.
    pub mc_blocks: u32,
    pub seed: u64,
    /// Ablation: ignore the transfer-entropy evidence and re-weight from the changepoint
    /// filters alone (a mechanism whose checks are all HOT has its nominal probability scaled
    /// by the smallest rate multiplier among them).  CPU reference only.
    pub bocpd_only: bool,
    /// k-point coincidence estimator for the mechanisms with at least three detectors
    /// (`false`: the pairwise rule of `FROZEN_V1`).
    pub coincidence: bool,
    /// Window of the coincidence counters (<= 255 samples): matches and exposure halve there.
    pub coin_cap: u32,
    /// Coincidences required before a class with a cold check may become active, and the
    /// likelihood-ratio threshold (nats) it must reach.
    pub coin_min: u32,
    pub g_threshold_coin: u32,
    /// Erasure snap: a class whose checks are all HOT and whose regime is younger than
    /// `n_min` samples is a known erasure (`w = 0`) as soon as it is active.
    pub snap: bool,
}

impl Default for NeuroConfig {
    fn default() -> Self {
        Self {
            hazard: 1.0 / 100.0,
            hot_on: 2.0,
            hot_off: 1.25,
            n_min: 6,
            g_threshold: 24,
            g_threshold_hot: 4,
            count_cap: 255,
            mc_blocks: 4096,
            seed: 0x5052_4953_4D4E_4555,
            bocpd_only: false,
            coincidence: false,
            coin_cap: 64,
            coin_min: 2,
            g_threshold_coin: 24,
            snap: false,
        }
    }
}

/// Static tables derived once from the nominal DEM; shared by every stream and uploaded
/// verbatim to the GPU.
#[derive(Clone, Debug)]
pub struct NeuroLayout {
    pub n_checks: usize,
    pub layers: usize,
    /// `det_of[layer * n_checks + check]` -> detector index, or [`NO_DETECTOR`] where the
    /// schedule has no detector at that site (e.g. X checks in the first layer of a Z memory).
    pub det_of: Vec<u32>,
    /// Nominal exposure `-ln(1 - pi_d)` per detector.
    pub exposure: Vec<f64>,
    pub pairs: Vec<PairClass>,
    /// Nominal transfer entropy per sample (Q16 nats) and nominal pair rate (Q12).
    pub pair_nom_te_q16: Vec<i64>,
    pub pair_nom_p_q12: Vec<i32>,
    /// check -> pair classes touching it.
    pub chk_pair_ptr: Vec<u32>,
    pub chk_pair: Vec<u32>,
    /// mechanism -> its pair classes; empty for mechanisms that are never re-weighted.
    pub mech_pair_ptr: Vec<u32>,
    pub mech_pair: Vec<u32>,
    /// Number of detectors of each mechanism.
    pub mech_size: Vec<u8>,
    /// mechanism -> distinct check sites it touches.
    pub mech_chk_ptr: Vec<u32>,
    pub mech_chk: Vec<u16>,
    pub p_nom: Vec<f64>,
    pub llr_nom: Vec<i32>,
    pub p_nom_q12: Vec<i32>,
    pub log_q16: Vec<i32>,
    /// `llr_of_p[q] = LLR(q / 4096)` for `q = 0..=2048`.
    pub llr_of_p: Vec<i32>,
    /// Mechanism classes (empty unless the coincidence estimator is on): the pattern
    /// `check | layer offset << 12` (sorted) of every mechanism with at least three detectors
    /// spanning at most two consecutive layers.
    pub cls_ptr: Vec<u32>,
    pub cls_det: Vec<u32>,
    /// Layers spanned by the pattern (1 or 2).
    pub cls_span: Vec<u8>,
    /// Nominal coincidence rate per sample (Q32) and its logarithm (Q16 nats).
    pub cls_pi_q32: Vec<u32>,
    pub cls_lnpi_q16: Vec<i32>,
    /// class -> classes whose pattern contains it (one entry per embedding).
    pub cls_sup_ptr: Vec<u32>,
    pub cls_sup: Vec<u32>,
    /// mechanism -> class, or [`NO_CLASS`].
    pub mech_cls: Vec<u32>,
}

/// Marks a mechanism that belongs to no class.
pub const NO_CLASS: u32 = u32::MAX;

fn pair_class(a: (u16, u16), b: (u16, u16)) -> Option<PairClass> {
    let ((ca, la), (cb, lb)) = (a, b);
    if la == lb {
        (ca != cb).then(|| PairClass { ci: ca.min(cb), cj: ca.max(cb), lag: 0 })
    } else if lb == la + 1 {
        Some(PairClass { ci: ca, cj: cb, lag: 1 })
    } else if la == lb + 1 {
        Some(PairClass { ci: cb, cj: ca, lag: 1 })
    } else {
        None
    }
}

impl NeuroLayout {
    pub fn build(dem: &Dem, cfg: &NeuroConfig) -> Result<Self> {
        ensure!(cfg.count_cap >= 16 && cfg.count_cap <= 255, "count_cap must be in [16, 255]");
        ensure!(cfg.coin_cap >= 8 && cfg.coin_cap <= 255, "coin_cap must be in [8, 255]");
        ensure!(cfg.coin_min >= 1, "coin_min must be at least 1");
        let mut site = Vec::with_capacity(dem.n_det);
        for (d, c) in dem.det_coords.iter().enumerate() {
            ensure!(!c[0].is_nan() && !c[1].is_nan(), "detector {d} has no (check, layer) coordinates");
            site.push((c[0] as u16, c[1] as u16));
        }
        let n_checks = site.iter().map(|s| s.0 as usize + 1).max().unwrap_or(0);
        let layers = site.iter().map(|s| s.1 as usize + 1).max().unwrap_or(0);
        ensure!(n_checks * layers >= dem.n_det, "more detectors than (check, layer) sites");
        let mut det_of = vec![u32::MAX; n_checks * layers];
        for (d, &(c, l)) in site.iter().enumerate() {
            let slot = &mut det_of[l as usize * n_checks + c as usize];
            ensure!(*slot == u32::MAX, "two detectors share site ({c}, {l})");
            *slot = d as u32;
        }
        // nominal firing probability of each detector: 1/2 (1 - prod (1 - 2 p_e))
        let mut prod = vec![1.0f64; dem.n_det];
        for e in 0..dem.n_err() {
            for &d in dem.dets(e) {
                prod[d as usize] *= 1.0 - 2.0 * dem.probs[e];
            }
        }
        let exposure: Vec<f64> = prod.iter().map(|&x| -(1.0 - 0.5 * (1.0 - x)).ln()).collect();
        // pair classes of every mechanism
        let mut class_index: BTreeMap<PairClass, u32> = BTreeMap::new();
        let mut mech_classes: Vec<Vec<PairClass>> = Vec::with_capacity(dem.n_err());
        for e in 0..dem.n_err() {
            let ds = dem.dets(e);
            let mut classes = Vec::new();
            let mut ok = ds.len() >= 2;
            for (i, &a) in ds.iter().enumerate() {
                for &b in &ds[i + 1..] {
                    match pair_class(site[a as usize], site[b as usize]) {
                        Some(pc) => classes.push(pc),
                        None => ok = false,
                    }
                }
            }
            if !ok {
                classes.clear();
            }
            classes.sort_unstable();
            classes.dedup();
            for pc in &classes {
                class_index.entry(*pc).or_insert(0);
            }
            mech_classes.push(classes);
        }
        // mechanism classes: translation-invariant patterns of the mechanisms with >= 3 detectors
        let mut mech_pattern: Vec<Option<Vec<u32>>> = vec![None; dem.n_err()];
        let mut pattern_index: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
        if cfg.coincidence {
            ensure!(n_checks < 4096, "class patterns pack at most 4095 check sites");
            for e in 0..dem.n_err() {
                let ds = dem.dets(e);
                let lmin = ds.iter().map(|&d| site[d as usize].1).min().unwrap_or(0);
                let lmax = ds.iter().map(|&d| site[d as usize].1).max().unwrap_or(0);
                if ds.len() < 3 || lmax - lmin > 1 {
                    continue;
                }
                let mut pat: Vec<u32> =
                    ds.iter().map(|&d| site[d as usize].0 as u32 | (((site[d as usize].1 - lmin) as u32) << 12)).collect();
                pat.sort_unstable();
                pattern_index.entry(pat.clone()).or_insert(0);
                mech_pattern[e] = Some(pat);
            }
        }
        let patterns: Vec<Vec<u32>> = pattern_index.keys().cloned().collect();
        for (i, pat) in patterns.iter().enumerate() {
            pattern_index.insert(pat.clone(), i as u32);
        }
        let mech_cls: Vec<u32> = mech_pattern.iter().map(|p| p.as_ref().map_or(NO_CLASS, |pat| pattern_index[pat])).collect();
        let cls_span: Vec<u8> = patterns.iter().map(|pat| 1 + pat.iter().any(|&d| d >> 12 != 0) as u8).collect();
        let mut cls_ptr = vec![0u32];
        let mut cls_det = Vec::new();
        for pat in &patterns {
            cls_det.extend_from_slice(pat);
            cls_ptr.push(cls_det.len() as u32);
        }
        // containment: T is explained by S when T shifted by 0 or 1 layers lies inside S
        let mut holders: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (j, pat) in patterns.iter().enumerate() {
            for &d in pat {
                holders.entry(d).or_default().push(j as u32);
            }
        }
        let mut cls_sup_ptr = vec![0u32];
        let mut cls_sup = Vec::new();
        for (k, pat) in patterns.iter().enumerate() {
            for shift in 0..2u32 {
                if shift as u8 + cls_span[k] > 2 {
                    continue;
                }
                let moved: Vec<u32> = pat.iter().map(|&d| d + (shift << 12)).collect();
                for &j in holders.get(&moved[0]).map(Vec::as_slice).unwrap_or(&[]) {
                    let sup = &patterns[j as usize];
                    if sup.len() > pat.len() && moved.iter().all(|d| sup.binary_search(d).is_ok()) {
                        cls_sup.push(j);
                    }
                }
            }
            cls_sup_ptr.push(cls_sup.len() as u32);
        }
        let pairs: Vec<PairClass> = class_index.keys().copied().collect();
        for (i, pc) in pairs.iter().enumerate() {
            class_index.insert(*pc, i as u32);
        }
        let mut mech_pair_ptr = vec![0u32];
        let mut mech_pair = Vec::new();
        for classes in &mech_classes {
            mech_pair.extend(classes.iter().map(|pc| class_index[pc]));
            mech_pair_ptr.push(mech_pair.len() as u32);
        }
        let mut per_check: Vec<Vec<u32>> = vec![Vec::new(); n_checks];
        for (i, pc) in pairs.iter().enumerate() {
            per_check[pc.ci as usize].push(i as u32);
            if pc.cj != pc.ci {
                per_check[pc.cj as usize].push(i as u32);
            }
        }
        let mut chk_pair_ptr = vec![0u32];
        let mut chk_pair = Vec::new();
        for list in &per_check {
            chk_pair.extend_from_slice(list);
            chk_pair_ptr.push(chk_pair.len() as u32);
        }
        // interventional reference: the same statistics under the nominal circuit
        let mut nominal = vec![[0u64; 8]; pairs.len()];
        let mut cls_nominal = vec![[0u64; 2]; patterns.len()];
        for blk in 0..cfg.mc_blocks as u64 {
            let (dets, _) = dem.sample(&dem.probs, cfg.seed, blk);
            let has = |c: u16, l: usize| det_of[l * n_checks + c as usize] != NO_DETECTOR;
            let bit = |c: u16, l: usize| {
                let d = det_of[l * n_checks + c as usize] as usize;
                (dets[d >> 6] >> (d & 63)) & 1 == 1
            };
            for l in 1..layers {
                for (i, pc) in pairs.iter().enumerate() {
                    let src = if pc.lag == 0 { l } else { l - 1 };
                    if !(has(pc.cj, l) && has(pc.cj, l - 1) && has(pc.ci, src)) {
                        continue;
                    }
                    nominal[i][bin_index(bit(pc.cj, l), bit(pc.cj, l - 1), bit(pc.ci, src))] += 1;
                }
            }
            for (k, pat) in patterns.iter().enumerate() {
                let span = cls_span[k] as usize;
                for l in span - 1..layers {
                    let at = |d: u32| ((d & 0xFFF) as u16, l + 1 - span + (d >> 12) as usize);
                    if pat.iter().all(|&d| has(at(d).0, at(d).1)) {
                        cls_nominal[k][1] += 1;
                        cls_nominal[k][0] += pat.iter().all(|&d| bit(at(d).0, at(d).1)) as u64;
                    }
                }
            }
        }
        // nominal coincidence rate; a pattern never seen in the reference gets half a count
        let cls_pi: Vec<f64> = cls_nominal.iter().map(|c| (c[0] as f64).max(0.5) / (c[1] as f64).max(1.0)).collect();
        let cls_pi_q32: Vec<u32> = cls_pi.iter().map(|&p| (p * 4294967296.0).round().clamp(1.0, 4294967295.0) as u32).collect();
        let cls_lnpi_q16: Vec<i32> = cls_pi.iter().map(|&p| (p.ln() * 65536.0).round() as i32).collect();
        let mut pair_nom_te_q16 = Vec::with_capacity(pairs.len());
        let mut pair_nom_p_q12 = Vec::with_capacity(pairs.len());
        for counts in &nominal {
            let n: f64 = (counts.iter().sum::<u64>() as f64).max(1.0);
            let mut te = 0.0f64;
            for c in 0..2usize {
                let get = |y: usize, x: usize| counts[(y << 2) | (c << 1) | x] as f64;
                let n_c = get(0, 0) + get(0, 1) + get(1, 0) + get(1, 1);
                for y in 0..2 {
                    for x in 0..2 {
                        let v = get(y, x);
                        if v > 0.0 {
                            te += v / n * (v * n_c / ((get(y, 0) + get(y, 1)) * (get(0, x) + get(1, x)))).ln();
                        }
                    }
                }
            }
            pair_nom_te_q16.push((te.max(0.0) * 65536.0).round() as i64);
            let cnt = |y: usize, x: usize| (counts[(y << 2) | x] + counts[(y << 2) | 2 | x]) as f64;
            let (pxy, px, py) = (cnt(1, 1) / n, (cnt(0, 1) + cnt(1, 1)) / n, (cnt(1, 0) + cnt(1, 1)) / n);
            let den = 1.0 - 2.0 * px - 2.0 * py + 4.0 * pxy;
            let ratio = if den > 0.0 { (4.0 * (pxy - px * py) / den).clamp(0.0, 1.0) } else { 0.0 };
            pair_nom_p_q12.push(((0.5 - 0.5 * (1.0 - ratio).sqrt()) * 4096.0).round() as i32);
        }
        let llr_nom: Vec<i32> = dem.probs.iter().map(|&p| llr_from_prob(p)).collect();
        let p_nom_q12: Vec<i32> = dem.probs.iter().map(|&p| (p * 4096.0).round().min(2048.0) as i32).collect();
        let llr_of_p: Vec<i32> =
            (0..=2048).map(|q| if q == 0 { PRIOR_CLIP } else { llr_from_prob(q as f64 / 4096.0) }).collect();
        let mut mech_chk_ptr = vec![0u32];
        let mut mech_chk = Vec::new();
        for e in 0..dem.n_err() {
            let mut cs: Vec<u16> = dem.dets(e).iter().map(|&d| site[d as usize].0).collect();
            cs.sort_unstable();
            cs.dedup();
            mech_chk.extend_from_slice(&cs);
            mech_chk_ptr.push(mech_chk.len() as u32);
        }
        Ok(Self {
            n_checks,
            layers,
            mech_chk_ptr,
            mech_chk,
            p_nom: dem.probs.clone(),
            det_of,
            exposure,
            pairs,
            pair_nom_te_q16,
            pair_nom_p_q12,
            chk_pair_ptr,
            chk_pair,
            mech_pair_ptr,
            mech_pair,
            mech_size: (0..dem.n_err()).map(|e| dem.dets(e).len().min(255) as u8).collect(),
            llr_nom,
            p_nom_q12,
            log_q16: log_table_q16(),
            llr_of_p,
            cls_ptr,
            cls_det,
            cls_span,
            cls_pi_q32,
            cls_lnpi_q16,
            cls_sup_ptr,
            cls_sup,
            mech_cls,
        })
    }

    pub fn n_cls(&self) -> usize {
        self.cls_span.len()
    }

    #[inline]
    pub fn cls_pattern(&self, k: usize) -> &[u32] {
        &self.cls_det[self.cls_ptr[k] as usize..self.cls_ptr[k + 1] as usize]
    }

    #[inline]
    pub fn cls_sups(&self, k: usize) -> &[u32] {
        &self.cls_sup[self.cls_sup_ptr[k] as usize..self.cls_sup_ptr[k + 1] as usize]
    }

    /// True iff class `k` can be sampled with its last layer at within-block layer `tl`.
    #[inline]
    pub fn cls_sampled(&self, k: usize, tl: usize) -> bool {
        let span = self.cls_span[k] as usize;
        tl + 1 >= span
            && self.cls_pattern(k).iter().all(|&d| {
                self.det_of[(tl + 1 - span + (d >> 12) as usize) * self.n_checks + (d & 0xFFF) as usize] != NO_DETECTOR
            })
    }

    /// True iff every detector of class `k` fired; `last` / `prev` are the check bits of the
    /// pattern's last layer and of the layer before it.
    #[inline]
    pub fn cls_match(&self, k: usize, last: &[u64], prev: &[u64]) -> bool {
        let two = self.cls_span[k] == 2;
        self.cls_pattern(k).iter().all(|&d| {
            let c = (d & 0xFFF) as usize;
            let row = if two && d >> 12 == 0 { prev } else { last };
            (row[c >> 6] >> (c & 63)) & 1 == 1
        })
    }

    #[inline]
    pub fn mech_pairs(&self, e: usize) -> &[u32] {
        &self.mech_pair[self.mech_pair_ptr[e] as usize..self.mech_pair_ptr[e + 1] as usize]
    }

    #[inline]
    pub fn chk_pairs(&self, c: usize) -> &[u32] {
        &self.chk_pair[self.chk_pair_ptr[c] as usize..self.chk_pair_ptr[c + 1] as usize]
    }

    #[inline]
    pub fn mech_checks(&self, e: usize) -> &[u16] {
        &self.mech_chk[self.mech_chk_ptr[e] as usize..self.mech_chk_ptr[e + 1] as usize]
    }

    pub fn n_mech(&self) -> usize {
        self.llr_nom.len()
    }
}

/// What one block changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockReport {
    pub hot_checks: u32,
    pub hot_transitions: u32,
    pub active_pairs: u32,
    /// Mechanisms whose weight currently differs from the nominal model.
    pub modified_mechanisms: u32,
    /// Mechanisms currently treated as known erasures (`w = 0`).
    pub erased_mechanisms: u32,
}

/// Online state of one stream.
#[derive(Clone, Debug)]
pub struct StreamState {
    pub bocpd: Vec<Bocpd>,
    pub hot: Vec<u8>,
    pub counts: Vec<[u8; 8]>,
    /// Current prior LLRs (BP units) and Nishimori weights (SMC units).
    pub llr: Vec<i32>,
    pub en: Vec<i32>,
    pub layers_seen: u64,
    /// Ring of the last `HISTORY` layers (check bits and within-block layer index).
    hist_bits: Vec<u64>,
    hist_layer: Vec<u8>,
    hist_head: usize,
    hist_len: usize,
    /// Scratch: transition run lengths, per pair rate, and rate explained by larger mechanisms.
    trans: Vec<u8>,
    pair_rate: Vec<i32>,
    explained: Vec<i32>,
    /// `[matches, exposure]` of every mechanism class and its active rate (Q12, 0: inactive).
    pub cls_cnt: Vec<[u8; 2]>,
    cls_rate: Vec<i32>,
}

/// Marks a (check, layer) site without a detector.
pub const NO_DETECTOR: u32 = u32::MAX;

/// Pair status codes stored next to the (non-negative) estimated rates.
pub const PAIR_VETO: i32 = -1;
pub const PAIR_MASKED: i32 = -2;

/// Layers of spike history kept per stream for regime replay.
pub const HISTORY: usize = 32;

impl StreamState {
    pub fn new(layout: &NeuroLayout) -> Self {
        Self {
            bocpd: vec![Bocpd::new(); layout.n_checks],
            hot: vec![0; layout.n_checks],
            counts: vec![[0u8; 8]; layout.pairs.len()],
            llr: layout.llr_nom.clone(),
            en: layout.llr_nom.iter().map(|&l| energy_from_llr(l)).collect(),
            layers_seen: 0,
            hist_bits: vec![0; HISTORY * layout.n_checks.div_ceil(64)],
            hist_layer: vec![0; HISTORY],
            hist_head: 0,
            hist_len: 0,
            trans: vec![0; layout.n_checks],
            pair_rate: vec![0; layout.pairs.len()],
            explained: vec![0; layout.pairs.len()],
            cls_cnt: vec![[0u8; 2]; layout.n_cls()],
            cls_rate: vec![0; layout.n_cls()],
        }
    }

    /// Ring slot of the layer `back` layers before the one being processed (`back >= 1`).
    #[inline]
    fn hist_slot(&self, back: usize) -> usize {
        (self.hist_head + HISTORY - (back - 1)) % HISTORY
    }

    #[inline]
    fn hist_row(&self, back: usize, wc: usize) -> &[u64] {
        let h = self.hist_slot(back);
        &self.hist_bits[h * wc..(h + 1) * wc]
    }

    /// Assimilates one block of detection events (bit-packed over detectors) and applies the
    /// resulting surgery to `llr` / `en`.  The updated weights are valid for decoding this
    /// same block: the decoder only runs once the block's syndrome is complete.
    pub fn ingest_block(&mut self, layout: &NeuroLayout, cfg: &NeuroConfig, dets: &[u64]) -> BlockReport {
        let nc = layout.n_checks;
        let wc = nc.div_ceil(64);
        let mut report = BlockReport::default();
        let mut cur = vec![0u64; wc];
        for l in 0..layout.layers {
            // ---- phase 1: changepoint filters; a transition records the MAP run length
            cur.iter_mut().for_each(|w| *w = 0);
            for c in 0..nc {
                self.trans[c] = 0;
                if layout.det_of[l * nc + c] == NO_DETECTOR {
                    continue;
                }
                let d = layout.det_of[l * nc + c] as usize;
                let x = (dets[d >> 6] >> (d & 63)) & 1 == 1;
                if x {
                    cur[c >> 6] |= 1 << (c & 63);
                }
                self.bocpd[c].update(x, layout.exposure[d], cfg.hazard);
                let m = self.bocpd[c].rate_mean();
                let flip = (self.hot[c] == 0 && m >= cfg.hot_on) || (self.hot[c] == 1 && m <= cfg.hot_off);
                if flip {
                    self.hot[c] ^= 1;
                    report.hot_transitions += 1;
                    // heating: replay the layers of the regime that raised the rate;
                    // cooling: plain restart
                    self.trans[c] = if self.hot[c] == 1 {
                        (self.bocpd[c].hot_run(cfg.hot_on) as u8).clamp(1, (HISTORY - 2) as u8)
                    } else {
                        1
                    };
                }
            }
            // ---- phase 2: pair counters; a pair touching a transition restarts from the
            // layers of the new regime only (replayed from the history ring)
            let get = |bits: &[u64], c: u16| (bits[c as usize >> 6] >> (c & 63)) & 1 == 1;
            // a sample needs the target detector, its predecessor and the source detector
            let sampled = |pc: &PairClass, tl: usize| {
                let has = |c: u16, layer: usize| layout.det_of[layer * nc + c as usize] != NO_DETECTOR;
                tl >= 1 && has(pc.cj, tl) && has(pc.cj, tl - 1) && has(pc.ci, if pc.lag == 0 { tl } else { tl - 1 })
            };
            for (i, pc) in layout.pairs.iter().enumerate() {
                let r = self.trans[pc.ci as usize].max(self.trans[pc.cj as usize]) as usize;
                if r > 0 {
                    self.counts[i] = [0u8; 8];
                    for k in (1..r).rev() {
                        // sample whose target layer lies k layers before the current one
                        if k + 1 > self.hist_len || !sampled(pc, self.hist_layer[self.hist_slot(k)] as usize) {
                            continue;
                        }
                        let (tgt, prev) = (self.hist_row(k, wc), self.hist_row(k + 1, wc));
                        let x = if pc.lag == 0 { get(tgt, pc.ci) } else { get(prev, pc.ci) };
                        let bin = bin_index(get(tgt, pc.cj), get(prev, pc.cj), x);
                        bump(&mut self.counts[i], bin, cfg.count_cap);
                    }
                }
                if sampled(pc, l) {
                    let prev = self.hist_row(1, wc);
                    let x = if pc.lag == 0 { get(&cur, pc.ci) } else { get(prev, pc.ci) };
                    let bin = bin_index(get(&cur, pc.cj), get(prev, pc.cj), x);
                    bump(&mut self.counts[i], bin, cfg.count_cap);
                }
            }
            // ---- phase 2b: k-point coincidences of the mechanism classes
            for k in 0..layout.n_cls() {
                let (mut r_heat, mut cooled, mut any_hot) = (0usize, false, false);
                for &d in layout.cls_pattern(k) {
                    let c = (d & 0xFFF) as usize;
                    any_hot |= self.hot[c] == 1;
                    if self.trans[c] > 0 {
                        if self.hot[c] == 1 {
                            r_heat = r_heat.max(self.trans[c] as usize);
                        } else {
                            cooled = true;
                        }
                    }
                }
                if cooled && !any_hot {
                    // the regime of this class ended
                    self.cls_cnt[k] = [0, 0];
                }
                if r_heat > 1 {
                    // coincidences of the new regime alone, replayed from the history ring;
                    // adopted when that regime shows the higher rate
                    let span = layout.cls_span[k] as usize;
                    let (mut m, mut n) = (0u32, 0u32);
                    for back in (1..r_heat).rev() {
                        if back + span - 1 > self.hist_len || !layout.cls_sampled(k, self.hist_layer[self.hist_slot(back)] as usize) {
                            continue;
                        }
                        n += 1;
                        m += layout.cls_match(k, self.hist_row(back, wc), self.hist_row(back + 1, wc)) as u32;
                    }
                    let [m0, n0] = self.cls_cnt[k];
                    if n > 0 && (n0 == 0 || m * n0 as u32 > m0 as u32 * n) {
                        self.cls_cnt[k] = [m as u8, n as u8];
                    }
                }
                if layout.cls_sampled(k, l) {
                    let hit = layout.cls_match(k, &cur, self.hist_row(1, wc));
                    coincidence_bump(&mut self.cls_cnt[k], hit, cfg.coin_cap);
                }
            }
            // ---- phase 3: push the layer into the history ring
            self.hist_head = (self.hist_head + 1) % HISTORY;
            let h = self.hist_head;
            self.hist_bits[h * wc..(h + 1) * wc].copy_from_slice(&cur);
            self.hist_layer[h] = l as u8;
            self.hist_len = (self.hist_len + 1).min(HISTORY);
        }
        self.layers_seen += layout.layers as u64;
        report.hot_checks = self.hot.iter().map(|&h| h as u32).sum();
        // ---- surgery
        if cfg.bocpd_only {
            for e in 0..layout.n_mech() {
                let checks = layout.mech_checks(e);
                let mut llr = layout.llr_nom[e];
                if !checks.is_empty() && checks.iter().all(|&c| self.hot[c as usize] == 1) {
                    let lam = checks.iter().map(|&c| self.bocpd[c as usize].rate_mean()).fold(f64::INFINITY, f64::min);
                    llr = llr.min(llr_from_prob((layout.p_nom[e] * lam).min(0.5)));
                }
                self.llr[e] = llr;
                self.en[e] = energy_from_llr(llr);
                report.modified_mechanisms += (llr != layout.llr_nom[e]) as u32;
                report.erased_mechanisms += (llr <= 0) as u32;
            }
            return report;
        }
        for i in 0..layout.pairs.len() {
            let n = total(&self.counts[i]);
            let excess = g_statistic_q16(&self.counts[i], &layout.log_q16) - 2 * n as i64 * layout.pair_nom_te_q16[i];
            let pc = layout.pairs[i];
            let both_hot = self.hot[pc.ci as usize] == 1 && self.hot[pc.cj as usize] == 1;
            let thr = (if both_hot { cfg.g_threshold_hot } else { cfg.g_threshold }) as i64 * 65536;
            let active = n >= cfg.n_min && excess >= thr;
            self.pair_rate[i] = if active {
                pair_rate_q12(&self.counts[i])
            } else if pair_masked(&self.counts[i]) {
                PAIR_MASKED
            } else {
                PAIR_VETO
            };
            self.explained[i] = 0;
            report.active_pairs += active as u32;
        }
        for k in 0..layout.n_cls() {
            let [m, n] = self.cls_cnt[k];
            let all_hot = layout.cls_pattern(k).iter().all(|&d| self.hot[(d & 0xFFF) as usize] == 1);
            let g = coincidence_g_q16(m as u32, n as u32, layout.cls_pi_q32[k], layout.cls_lnpi_q16[k], &layout.log_q16);
            let (thr, need) = if all_hot { (cfg.g_threshold_hot, 1) } else { (cfg.g_threshold_coin, cfg.coin_min) };
            self.cls_rate[k] = if (m as u32) < need || g < thr as i64 * 65536 {
                0
            } else if cfg.snap && all_hot && (n as u32) < cfg.n_min {
                2048
            } else {
                (((m as i32) << 12) / n as i32).min(2048)
            };
        }
        for phase in 0..2 {
            for e in 0..layout.n_mech() {
                let pairs = layout.mech_pairs(e);
                let big = layout.mech_size[e] >= 3;
                if pairs.is_empty() || big != (phase == 0) {
                    continue;
                }
                if layout.mech_cls[e] != NO_CLASS {
                    // k-point rule: own coincidence rate minus the active containing classes
                    let k = layout.mech_cls[e] as usize;
                    let mut p = self.cls_rate[k];
                    for &s in layout.cls_sups(k) {
                        p -= self.cls_rate[s as usize];
                    }
                    let mut llr = layout.llr_nom[e];
                    if p > layout.p_nom_q12[e] {
                        llr = llr.min(layout.llr_of_p[p.min(2048) as usize]);
                        for &pi in pairs {
                            let slot = &mut self.explained[pi as usize];
                            *slot = (*slot).max(p);
                        }
                    }
                    self.llr[e] = llr;
                    self.en[e] = energy_from_llr(llr);
                    report.modified_mechanisms += (llr != layout.llr_nom[e]) as u32;
                    report.erased_mechanisms += (llr <= 0) as u32;
                    continue;
                }
                let (mut p, mut veto) = (i32::MAX, false);
                for &pi in pairs {
                    let r = self.pair_rate[pi as usize];
                    if r == PAIR_VETO {
                        veto = true;
                    } else if r != PAIR_MASKED {
                        p = p.min(if big { r } else { r - self.explained[pi as usize] });
                    }
                }
                let mut llr = layout.llr_nom[e];
                if !veto && p != i32::MAX && p > layout.p_nom_q12[e] {
                    llr = llr.min(layout.llr_of_p[p.min(2048) as usize]);
                    if big {
                        for &pi in pairs {
                            let slot = &mut self.explained[pi as usize];
                            *slot = (*slot).max(p);
                        }
                    }
                }
                self.llr[e] = llr;
                self.en[e] = energy_from_llr(llr);
                report.modified_mechanisms += (llr != layout.llr_nom[e]) as u32;
                report.erased_mechanisms += (llr <= 0) as u32;
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qec_core::bb::bb72;
    use qec_core::complex::ChainComplex;
    use qec_hodge_bp::{BpConfig, TannerGraph};
    use qec_smc_coset::{CosetDecoder, DecodeWork, DecoderConfig, EscalationConfig, SmcConfig};

    struct Bench {
        hz_rows: Vec<Vec<usize>>,
        dem: Dem,
        layout: NeuroLayout,
        graph: TannerGraph,
        cfg: NeuroConfig,
        rounds: usize,
        n: usize,
    }

    fn bench() -> Bench {
        bench_with(NeuroConfig { mc_blocks: 2048, ..NeuroConfig::default() })
    }

    /// The `FROZEN_V2` monitor: k-point coincidence estimator with the erasure snap.
    fn coincidence_cfg() -> NeuroConfig {
        NeuroConfig {
            mc_blocks: 8192,
            coincidence: true,
            coin_cap: 255,
            coin_min: 1,
            g_threshold_coin: 14,
            snap: true,
            ..NeuroConfig::default()
        }
    }

    fn bench_with(cfg: NeuroConfig) -> Bench {
        let code = bb72().build().unwrap();
        let rounds = 5;
        let (dem, cells) = Dem::phenomenological(&code, rounds, 0.004, 0.004).unwrap();
        let layout = NeuroLayout::build(&dem, &cfg).unwrap();
        let graph = TannerGraph::from_complex(&ChainComplex::from_dem_cells(&dem, &cells).unwrap());
        let hz_rows = (0..code.hz.rows).map(|r| code.hz.row(r).iter().map(|&q| q as usize).collect()).collect();
        Bench { hz_rows, dem, layout, graph, cfg, rounds, n: code.n }
    }

    #[test]
    fn layout_tiles_detectors_and_classifies_every_coupler() {
        let b = bench();
        assert_eq!(b.layout.n_checks, 36);
        assert_eq!(b.layout.layers, b.rounds + 1);
        assert!(b.layout.exposure.iter().all(|&e| e > 0.0 && e < 0.2));
        // data faults couple their three checks in one layer; measurement faults couple a
        // check with itself across consecutive layers
        assert!(b.layout.pairs.iter().any(|p| p.lag == 0));
        assert!(b.layout.pairs.iter().any(|p| p.lag == 1 && p.ci == p.cj));
        for e in 0..b.dem.n_err() {
            let k = b.dem.dets(e).len();
            assert_eq!(b.layout.mech_pairs(e).len(), k * (k - 1) / 2);
        }
    }

    #[test]
    fn mechanism_classes_tile_the_data_faults() {
        let b = bench_with(coincidence_cfg());
        let l = &b.layout;
        // every data fault flips its three checks in one layer: one class per data qubit
        assert_eq!(l.n_cls(), b.n);
        assert!(l.cls_span.iter().all(|&s| s == 1) && l.cls_sup.is_empty());
        for e in 0..b.dem.n_err() {
            assert_eq!(l.mech_cls[e] != NO_CLASS, b.dem.dets(e).len() >= 3);
        }
        for k in 0..l.n_cls() {
            assert_eq!(l.cls_pattern(k).len(), 3);
            // nominal coincidence rate: about the data fault probability, never zero
            let pi = l.cls_pi_q32[k] as f64 / 4294967296.0;
            assert!(pi > 5.0e-4 && pi < 2.0e-2, "class {k}: nominal rate {pi}");
            assert!(l.cls_sampled(k, 0) && l.cls_sampled(k, l.layers - 1));
        }
        // the pairwise monitor builds no class tables
        assert_eq!(bench().layout.n_cls(), 0);
    }

    /// The coincidence monitor must not pay for its sensitivity on a stationary stream:
    /// re-weighting stays rare and the decoder fed the live weights is not worse.
    #[test]
    fn coincidence_monitor_is_harmless_on_a_stationary_stream() {
        let b = bench_with(coincidence_cfg());
        let dec_cfg = DecoderConfig {
            bp: BpConfig::plain(48, 160),
            smc: SmcConfig { k_cols: 256, walkers: 64, beta_steps: 16, ..SmcConfig::default() },
            esc: EscalationConfig::default(),
        };
        let dec = CosetDecoder::new(&b.graph, dec_cfg);
        let mut work = DecodeWork::default();
        let nominal_en: Vec<i32> = b.layout.llr_nom.iter().map(|&l| energy_from_llr(l)).collect();
        let (mut touched, mut blocks, mut static_fail, mut live_fail) = (0u32, 0u32, 0u32, 0u32);
        for stream in 0..8u64 {
            let mut st = StreamState::new(&b.layout);
            for blk in 0..150u64 {
                let (dets, truth) = b.dem.sample(&b.dem.probs, 700 + stream, blk);
                let rep = st.ingest_block(&b.layout, &b.cfg, &dets);
                touched += (rep.modified_mechanisms > 0) as u32;
                blocks += 1;
                let id = (stream * 1000 + blk) as u32;
                static_fail += (dec.decode(&mut work, &b.layout.llr_nom, &nominal_en, &dets, id).obs != truth) as u32;
                live_fail += (dec.decode(&mut work, &st.llr, &st.en, &dets, id).obs != truth) as u32;
            }
        }
        eprintln!("[coincidence stationary] re-weighted blocks {touched}/{blocks}; failures static={static_fail} live={live_fail}");
        assert!(touched * 4 <= blocks, "false surgery in {touched}/{blocks} stationary blocks");
        assert!(live_fail <= static_fail + static_fail / 10 + 3, "stationary penalty: {live_fail} vs {static_fail}");
    }

    #[test]
    fn stationary_stream_is_left_at_the_nominal_model() {
        let b = bench();
        let mut touched = 0u32;
        let mut blocks = 0u32;
        for stream in 0..8u64 {
            let mut st = StreamState::new(&b.layout);
            for blk in 0..150u64 {
                let (dets, _) = b.dem.sample(&b.dem.probs, 500 + stream, blk);
                let rep = st.ingest_block(&b.layout, &b.cfg, &dets);
                touched += (rep.modified_mechanisms > 0) as u32;
                blocks += 1;
            }
        }
        assert!(touched * 50 <= blocks, "false surgery in {touched}/{blocks} stationary blocks");
    }

    /// Mid-run dropout burst: from global round 500 on, the six data qubits of one check are
    /// maximally mixed.  The monitor must flag the region, convert the incident mechanisms to
    /// (near-)erasures, and the decoder fed the live weights must out-survive the static one.
    #[test]
    fn dropout_burst_is_converted_to_erasures_and_survived() {
        dropout_burst(&bench(), "pairwise", 64);
    }

    /// Same burst under the coincidence monitor.  Six maximally mixed qubits around one check
    /// interfere, so the coincidence rate of each is below 1/2: the mechanisms are driven to
    /// within two nats of an erasure rather than to `w = 0`.
    #[test]
    fn dropout_burst_is_survived_by_the_coincidence_monitor() {
        dropout_burst(&bench_with(coincidence_cfg()), "coincidence", 128);
    }

    fn dropout_burst(b: &Bench, label: &str, erased_llr: i32) {
        let layers = b.layout.layers;
        let onset_block = 500 / b.rounds;
        let total_blocks = onset_block + 60;
        let dec_cfg = DecoderConfig {
            bp: BpConfig::plain(48, 160),
            smc: SmcConfig { k_cols: 256, walkers: 64, beta_steps: 16, ..SmcConfig::default() },
            esc: EscalationConfig::default(),
        };
        let dec = CosetDecoder::new(&b.graph, dec_cfg);
        let mut work = DecodeWork::default();
        let nominal_en: Vec<i32> = b.layout.llr_nom.iter().map(|&l| energy_from_llr(l)).collect();
        let (mut static_fail, mut live_fail, mut post_blocks) = (0u32, 0u32, 0u32);
        let (mut pre_static, mut pre_live, mut pre_blocks) = (0u32, 0u32, 0u32);
        let mut latencies = Vec::new();
        let mut erase_lat = Vec::new();
        let mut erased_ok = 0u32;
        let streams = 16u64;
        for stream in 0..streams {
            // the six data qubits of one Z check: a spatially compact hot cluster
            let hot_set: Vec<usize> = b.hz_rows[(5 * stream as usize + 1) % b.hz_rows.len()].clone();
            // phenomenological ordering: data(q, t) = t * n + q
            let mut hot_probs = b.dem.probs.clone();
            for t in 0..layers {
                for &q in &hot_set {
                    hot_probs[t * b.n + q] = 0.5;
                }
            }
            let mut st = StreamState::new(&b.layout);
            let mut detected: Option<u64> = None;
            let mut erased_at: Option<u64> = None;
            for blk in 0..total_blocks as u64 {
                let hot = blk as usize >= onset_block;
                let (dets, truth) = b.dem.sample(if hot { &hot_probs } else { &b.dem.probs }, 9000 + stream, blk);
                let rep = st.ingest_block(&b.layout, &b.cfg, &dets);
                if hot && detected.is_none() && rep.hot_checks > 0 {
                    detected = Some((blk - onset_block as u64 + 1) * layers as u64);
                }
                let s = dec.decode(&mut work, &b.layout.llr_nom, &nominal_en, &dets, (stream * 1000 + blk) as u32);
                let l = dec.decode(&mut work, &st.llr, &st.en, &dets, (stream * 1000 + blk) as u32);
                if hot && blk as usize >= onset_block + 2 {
                    static_fail += (s.obs != truth) as u32;
                    live_fail += (l.obs != truth) as u32;
                    post_blocks += 1;
                } else if !hot {
                    pre_static += (s.obs != truth) as u32;
                    pre_live += (l.obs != truth) as u32;
                    pre_blocks += 1;
                }
                let erased = hot_set.iter().filter(|&&q| (0..layers).all(|t| st.llr[t * b.n + q] <= erased_llr)).count();
                if hot && erased_at.is_none() && erased == hot_set.len() {
                    erased_at = Some((blk - onset_block as u64 + 1) * layers as u64);
                }
            }
            latencies.push(detected.unwrap_or(u64::MAX));
            erase_lat.push(erased_at.unwrap_or(u64::MAX));
            erased_ok += erased_at.is_some() as u32;
        }
        erase_lat.sort_unstable();
        latencies.sort_unstable();
        let p90 = latencies[latencies.len() * 9 / 10];
        eprintln!(
            "[tls {label}] detection latency layers median={} p90={p90}; erasure latency layers median={} p90={} ({erased_ok}/{streams} streams); \\
             post-onset failures static={static_fail} live={live_fail} of {post_blocks}; \\
             pre-onset failures static={pre_static} live={pre_live} of {pre_blocks}",
            latencies[latencies.len() / 2],
            erase_lat[erase_lat.len() / 2],
            erase_lat[erase_lat.len() * 9 / 10]
        );
        assert!(p90 <= 2 * layers as u64, "hot checks must be flagged within two blocks: p90 = {p90} layers");
        assert!(erased_ok * 10 >= streams as u32 * 9, "hot-qubit mechanisms erased in only {erased_ok}/{streams} streams");
        assert!(static_fail >= 20, "the drift must actually hurt the static decoder ({static_fail})");
        assert!(live_fail * 2 <= static_fail, "live weights {live_fail} vs static {static_fail} failures");
        assert!(pre_live <= pre_static + 2, "no penalty before the onset: {pre_live} vs {pre_static}");
    }
}
