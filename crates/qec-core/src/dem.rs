//! Stim detector-error-model (`.dem`) and bit-packed shot (`.b8`) loaders, plus a
//! pure-Rust phenomenological space-time DEM used where no Stim file is available.

use crate::bb::CssCode;
use crate::f2::SparseF2;
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// A detector error model: independent fault mechanisms, each flipping a set of detectors
/// and a set of logical observables.
#[derive(Clone, Debug)]
pub struct Dem {
    pub n_det: usize,
    pub n_obs: usize,
    pub probs: Vec<f64>,
    pub err_ptr: Vec<u32>,
    pub err_det: Vec<u32>,
    pub err_obs: Vec<u64>,
    /// `(check index, detector layer, kind)` annotations; NaN when absent.
    pub det_coords: Vec<[f64; 3]>,
    /// Mechanisms that flip observables but no detector (undetectable; excluded from `H`).
    pub undetectable: Vec<(f64, u64)>,
}

struct ParseState {
    det_offset: u64,
    coord_offset: [f64; 3],
    max_det: i64,
    max_obs: i64,
    mechs: Vec<(f64, Vec<u32>, u64)>,
    coords: HashMap<u32, [f64; 3]>,
}

fn parse_args(tok: &str) -> Result<(String, Vec<f64>)> {
    match tok.find('(') {
        None => Ok((tok.to_string(), Vec::new())),
        Some(open) => {
            let close = tok.rfind(')').ok_or_else(|| anyhow!("unbalanced parenthesis in '{tok}'"))?;
            let mut vals = Vec::new();
            for part in tok[open + 1..close].split(',') {
                let part = part.trim();
                if !part.is_empty() {
                    vals.push(part.parse::<f64>().with_context(|| format!("bad number '{part}'"))?);
                }
            }
            Ok((tok[..open].to_string(), vals))
        }
    }
}

fn process(lines: &[String], state: &mut ParseState) -> Result<()> {
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].as_str();
        // Rejoin "name(a, b, c)" split by whitespace inside the parentheses.
        let (head, rest) = match line.find(')') {
            Some(close) if line.find('(').is_some_and(|o| o < close) => {
                (line[..=close].replace(' ', ""), line[close + 1..].trim())
            }
            _ => {
                let mut it = line.splitn(2, char::is_whitespace);
                (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").trim())
            }
        };
        let (name, args) = parse_args(&head)?;
        match name.as_str() {
            "error" => {
                ensure!(args.len() == 1, "error instruction needs one probability: '{line}'");
                let mut dets: Vec<u32> = Vec::new();
                let mut obs = 0u64;
                for t in rest.split_whitespace() {
                    if t == "^" {
                        continue;
                    }
                    if let Some(d) = t.strip_prefix('D') {
                        let id = d.parse::<u64>()? + state.det_offset;
                        state.max_det = state.max_det.max(id as i64);
                        dets.push(id as u32);
                    } else if let Some(l) = t.strip_prefix('L') {
                        let id = l.parse::<u64>()?;
                        ensure!(id < 64, "at most 64 logical observables are supported");
                        state.max_obs = state.max_obs.max(id as i64);
                        obs ^= 1u64 << id;
                    } else {
                        bail!("unknown DEM target '{t}'");
                    }
                }
                state.mechs.push((args[0], dets, obs));
            }
            "detector" => {
                for t in rest.split_whitespace() {
                    let d = t.strip_prefix('D').ok_or_else(|| anyhow!("bad detector target '{t}'"))?;
                    let id = d.parse::<u64>()? + state.det_offset;
                    state.max_det = state.max_det.max(id as i64);
                    let mut c = [f64::NAN; 3];
                    for (k, slot) in c.iter_mut().enumerate() {
                        if k < args.len() {
                            *slot = args[k] + state.coord_offset[k];
                        }
                    }
                    state.coords.insert(id as u32, c);
                }
            }
            "logical_observable" => {
                for t in rest.split_whitespace() {
                    let l = t.strip_prefix('L').ok_or_else(|| anyhow!("bad observable target '{t}'"))?;
                    state.max_obs = state.max_obs.max(l.parse::<i64>()?);
                }
            }
            "shift_detectors" => {
                for (k, a) in args.iter().enumerate().take(3) {
                    state.coord_offset[k] += a;
                }
                let n: u64 = rest.split_whitespace().next().unwrap_or("0").parse()?;
                state.det_offset += n;
            }
            "repeat" => {
                let count: u64 = rest.trim_end_matches('{').trim().parse()?;
                let mut depth = 1usize;
                let mut j = i + 1;
                while j < lines.len() {
                    if lines[j].ends_with('{') {
                        depth += 1;
                    } else if lines[j] == "}" {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    j += 1;
                }
                ensure!(j < lines.len(), "unterminated repeat block");
                for _ in 0..count {
                    process(&lines[i + 1..j], state)?;
                }
                i = j;
            }
            other => bail!("unsupported DEM instruction '{other}'"),
        }
        i += 1;
    }
    Ok(())
}

impl Dem {
    pub fn n_err(&self) -> usize {
        self.probs.len()
    }

    #[inline]
    pub fn dets(&self, e: usize) -> &[u32] {
        &self.err_det[self.err_ptr[e] as usize..self.err_ptr[e + 1] as usize]
    }

    pub fn n_edges(&self) -> usize {
        self.err_det.len()
    }

    /// Builds a DEM, XOR-merging duplicated targets and combining mechanisms with identical
    /// symptoms via `p <- p1 (1 - p2) + p2 (1 - p1)`.
    pub fn from_mechanisms(
        n_det: usize,
        n_obs: usize,
        mechs: Vec<(f64, Vec<u32>, u64)>,
        det_coords: Vec<[f64; 3]>,
    ) -> Result<Dem> {
        ensure!(n_obs <= 64, "at most 64 logical observables are supported");
        let mut index: HashMap<(Vec<u32>, u64), usize> = HashMap::new();
        let mut probs = Vec::new();
        let mut sigs: Vec<(Vec<u32>, u64)> = Vec::new();
        let mut undetectable = Vec::new();
        for (p, dets, obs) in mechs {
            ensure!((0.0..=1.0).contains(&p), "probability {p} out of range");
            let row = SparseF2::from_rows(n_det.max(1), &[dets]);
            let dets: Vec<u32> = row.row(0).to_vec();
            if dets.is_empty() {
                if obs != 0 {
                    undetectable.push((p, obs));
                }
                continue;
            }
            let key = (dets, obs);
            match index.get(&key) {
                Some(&j) => {
                    let q: f64 = probs[j];
                    probs[j] = q * (1.0 - p) + p * (1.0 - q);
                }
                None => {
                    index.insert(key.clone(), probs.len());
                    probs.push(p);
                    sigs.push(key);
                }
            }
        }
        let mut err_ptr = Vec::with_capacity(probs.len() + 1);
        let mut err_det = Vec::new();
        let mut err_obs = Vec::with_capacity(probs.len());
        err_ptr.push(0u32);
        for (dets, obs) in &sigs {
            err_det.extend_from_slice(dets);
            err_ptr.push(err_det.len() as u32);
            err_obs.push(*obs);
        }
        let mut coords = det_coords;
        coords.resize(n_det, [f64::NAN; 3]);
        Ok(Dem { n_det, n_obs, probs, err_ptr, err_det, err_obs, det_coords: coords, undetectable })
    }

    pub fn parse(text: &str) -> Result<Dem> {
        let lines: Vec<String> = text
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let mut state = ParseState {
            det_offset: 0,
            coord_offset: [0.0; 3],
            max_det: -1,
            max_obs: -1,
            mechs: Vec::new(),
            coords: HashMap::new(),
        };
        process(&lines, &mut state)?;
        let n_det = (state.max_det + 1) as usize;
        let n_obs = (state.max_obs + 1) as usize;
        let mut coords = vec![[f64::NAN; 3]; n_det];
        for (d, c) in state.coords {
            coords[d as usize] = c;
        }
        Dem::from_mechanisms(n_det, n_obs, state.mechs, coords)
    }

    pub fn load(path: &Path) -> Result<Dem> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading DEM {}", path.display()))?;
        Dem::parse(&text).with_context(|| format!("parsing DEM {}", path.display()))
    }

    /// Serializes back to Stim's flattened text format.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for e in 0..self.n_err() {
            out.push_str(&format!("error({:e})", self.probs[e]));
            for &d in self.dets(e) {
                out.push_str(&format!(" D{d}"));
            }
            for o in 0..self.n_obs {
                if (self.err_obs[e] >> o) & 1 == 1 {
                    out.push_str(&format!(" L{o}"));
                }
            }
            out.push('\n');
        }
        for (d, c) in self.det_coords.iter().enumerate() {
            if !c[0].is_nan() {
                out.push_str(&format!("detector({}, {}, {}) D{d}\n", c[0], c[1], c[2]));
            }
        }
        out
    }

    /// Mechanism-by-detector incidence (`n_err x n_det`).
    pub fn by_error(&self) -> SparseF2 {
        SparseF2 { rows: self.n_err(), cols: self.n_det, ptr: self.err_ptr.clone(), idx: self.err_det.clone() }
    }

    /// The detector check matrix `H` (`n_det x n_err`).
    pub fn check_matrix(&self) -> SparseF2 {
        self.by_error().transpose()
    }

    /// Samples one shot: every mechanism fires independently with probability `probs[e]`
    /// (which may differ from the model's own priors, e.g. to inject drift).  Returns the
    /// bit-packed detection events and the observable mask.  Deterministic in `(seed, shot)`.
    pub fn sample(&self, probs: &[f64], seed: u64, shot: u64) -> (Vec<u64>, u64) {
        let mut dets = vec![0u64; self.n_det.div_ceil(64).max(1)];
        let mut obs = 0u64;
        let key = crate::fixed::mix64(seed ^ shot.wrapping_mul(0xA24B_AED4_963E_E407));
        for (e, &p) in probs.iter().enumerate() {
            let u = (crate::fixed::mix64(key ^ (e as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64
                / (1u64 << 53) as f64;
            if u < p {
                for &d in self.dets(e) {
                    dets[d as usize >> 6] ^= 1 << (d & 63);
                }
                obs ^= self.err_obs[e];
            }
        }
        (dets, obs)
    }

    /// Phenomenological space-time DEM of a CSS code under X noise: `rounds` noisy Z-syndrome
    /// extractions followed by one perfect readout.  Returns the DEM together with the
    /// explicit 2-cells (trivial cycles): X stabilizers per time slice and the time-like
    /// plaquettes `{data(q, t), data(q, t+1), meas(c, t) for c in checks(q)}`.
    pub fn phenomenological(code: &CssCode, rounds: usize, p_data: f64, p_meas: f64) -> Result<(Dem, Vec<Vec<u32>>)> {
        let n = code.n;
        let m = code.hz.rows;
        let hz_t = code.hz.transpose();
        let layers = rounds + 1;
        let det = |c: usize, t: usize| (t * m + c) as u32;
        let mut mechs = Vec::new();
        for t in 0..layers {
            for q in 0..n {
                let dets = hz_t.row(q).iter().map(|&c| det(c as usize, t)).collect();
                let obs = (0..code.k).fold(0u64, |a, j| a | ((code.lz.get(j, q) as u64) << j));
                mechs.push((p_data, dets, obs));
            }
        }
        for t in 0..rounds {
            for c in 0..m {
                mechs.push((p_meas, vec![det(c, t), det(c, t + 1)], 0u64));
            }
        }
        let expect = mechs.len();
        let mut coords = Vec::with_capacity(layers * m);
        for t in 0..layers {
            for c in 0..m {
                coords.push([c as f64, t as f64, if t == rounds { 1.0 } else { 0.0 }]);
            }
        }
        let dem = Dem::from_mechanisms(layers * m, code.k, mechs, coords)?;
        ensure!(dem.n_err() == expect, "phenomenological mechanisms unexpectedly merged");
        let data = |q: usize, t: usize| (t * n + q) as u32;
        let meas = |c: usize, t: usize| (layers * n + t * m + c) as u32;
        let mut cells = Vec::new();
        for t in 0..layers {
            for g in 0..code.hx.rows {
                cells.push(code.hx.row(g).iter().map(|&q| data(q as usize, t)).collect());
            }
        }
        for t in 0..rounds {
            for q in 0..n {
                let mut cell = vec![data(q, t), data(q, t + 1)];
                cell.extend(hz_t.row(q).iter().map(|&c| meas(c as usize, t)));
                cells.push(cell);
            }
        }
        Ok((dem, cells))
    }
}

/// Bit-packed shots: `shots` rows of `width` bits, little-endian within `u64` words
/// (bit `k` of a shot is bit `k % 8` of byte `k / 8` in Stim's `b8` format).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotBits {
    pub width: usize,
    pub shots: usize,
    pub wps: usize,
    pub data: Vec<u64>,
}

impl ShotBits {
    pub fn zeros(shots: usize, width: usize) -> Self {
        let wps = width.div_ceil(64).max(1);
        Self { width, shots, wps, data: vec![0u64; shots * wps] }
    }

    pub fn from_b8_bytes(bytes: &[u8], width: usize) -> Result<Self> {
        let bps = width.div_ceil(8);
        ensure!(bps > 0, "zero-width shots");
        ensure!(bytes.len().is_multiple_of(bps), "b8 length {} is not a multiple of {} bytes/shot", bytes.len(), bps);
        let shots = bytes.len() / bps;
        let mut out = Self::zeros(shots, width);
        for s in 0..shots {
            let src = &bytes[s * bps..(s + 1) * bps];
            let dst = &mut out.data[s * out.wps..(s + 1) * out.wps];
            for (k, &b) in src.iter().enumerate() {
                dst[k >> 3] |= (b as u64) << ((k & 7) * 8);
            }
            if !width.is_multiple_of(64) {
                dst[out.wps - 1] &= (1u64 << (width % 64)) - 1;
            }
        }
        Ok(out)
    }

    pub fn load_b8(path: &Path, width: usize) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading shots {}", path.display()))?;
        Self::from_b8_bytes(&bytes, width)
    }

    pub fn to_b8_bytes(&self) -> Vec<u8> {
        let bps = self.width.div_ceil(8);
        let mut out = vec![0u8; self.shots * bps];
        for s in 0..self.shots {
            let src = self.shot(s);
            for k in 0..bps {
                out[s * bps + k] = (src[k >> 3] >> ((k & 7) * 8)) as u8;
            }
        }
        out
    }

    #[inline]
    pub fn shot(&self, s: usize) -> &[u64] {
        &self.data[s * self.wps..(s + 1) * self.wps]
    }

    #[inline]
    pub fn shot_mut(&mut self, s: usize) -> &mut [u64] {
        &mut self.data[s * self.wps..(s + 1) * self.wps]
    }

    #[inline]
    pub fn get(&self, s: usize, bit: usize) -> bool {
        (self.data[s * self.wps + (bit >> 6)] >> (bit & 63)) & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bb::bb72;

    const SAMPLE: &str = "\
# a tiny Stim-style model
error(0.01) D0 D1
error(0.02) D1 D2 L0
error(0.03) D0 D1
detector(3, 0, 0) D0
repeat 2 {
    error(0.125) D3 ^ D4 L1
    detector(1, 5, 0) D3
    shift_detectors(0, 1, 0) 2
}
error(0.5) L0
";

    #[test]
    fn parses_merges_and_expands_repeat_blocks() {
        let dem = Dem::parse(SAMPLE).unwrap();
        assert_eq!(dem.n_det, 7);
        assert_eq!(dem.n_obs, 2);
        assert_eq!(dem.n_err(), 4);
        // identical symptoms {D0, D1} merge: 0.01 * 0.97 + 0.03 * 0.99
        assert!((dem.probs[0] - (0.01 * 0.97 + 0.03 * 0.99)).abs() < 1e-15);
        assert_eq!(dem.dets(1), &[1, 2]);
        assert_eq!(dem.err_obs[1], 1);
        assert_eq!(dem.dets(2), &[3, 4]);
        assert_eq!(dem.dets(3), &[5, 6]);
        assert_eq!(dem.err_obs[3], 2);
        assert_eq!(dem.det_coords[3], [1.0, 5.0, 0.0]);
        assert_eq!(dem.det_coords[5], [1.0, 6.0, 0.0]);
        assert_eq!(dem.undetectable, vec![(0.5, 1)]);
    }

    #[test]
    fn text_round_trip_is_lossless() {
        let dem = Dem::parse(SAMPLE).unwrap();
        let again = Dem::parse(&dem.to_text()).unwrap();
        assert_eq!(again.err_det, dem.err_det);
        assert_eq!(again.err_obs, dem.err_obs);
        for (a, b) in again.probs.iter().zip(dem.probs.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn b8_round_trip_and_bit_order() {
        // 11 bits/shot -> 2 bytes/shot; bit 0 is the LSB of byte 0, bit 8 the LSB of byte 1
        let bytes = [0b0000_0101u8, 0b0000_0010, 0xFF, 0x07];
        let shots = ShotBits::from_b8_bytes(&bytes, 11).unwrap();
        assert_eq!(shots.shots, 2);
        assert!(shots.get(0, 0) && !shots.get(0, 1) && shots.get(0, 2) && shots.get(0, 9));
        assert!((0..11).all(|b| shots.get(1, b)));
        assert_eq!(shots.to_b8_bytes(), bytes);
        assert!(ShotBits::from_b8_bytes(&bytes[..3], 11).is_err());
    }

    #[test]
    fn phenomenological_cells_are_trivial_cycles() {
        let code = bb72().build().unwrap();
        let (dem, cells) = Dem::phenomenological(&code, 3, 0.01, 0.01).unwrap();
        assert_eq!(dem.n_det, 4 * 36);
        assert_eq!(dem.n_err(), 4 * 72 + 3 * 36);
        for cell in &cells {
            let mut acc = vec![0u8; dem.n_det];
            let mut obs = 0u64;
            for &e in cell {
                for &d in dem.dets(e as usize) {
                    acc[d as usize] ^= 1;
                }
                obs ^= dem.err_obs[e as usize];
            }
            assert!(acc.iter().all(|&x| x == 0) && obs == 0);
        }
    }
}
