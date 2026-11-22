#!/usr/bin/env python3
"""PRISM-QEC Stim oracle (prism.qec.sota.v1).

Deterministic circuit-level datasets for six codes in three families and reference BP-OSD /
BP-LSD baselines decoded on the exact same .b8 shot files.

  codes      bb72 [[72,12,6]], bb90 [[90,8,10]], bb144_gross [[144,12,12]]  (bivariate bicycle,
             depth-7 CNOT schedule of Bravyi et al., Nature 627, 778 (2024))
             surface_d5, surface_d7  (Stim surface_code:rotated_memory_z)
             hgp_hamming [[58,16,3]]  (hypergraph product of the [7,4,3] Hamming code with itself)
  noise      uniform circuit-level depolarizing: DEPOLARIZE2(p) after every CNOT, DEPOLARIZE1(p)
             on idle data, reset to the orthogonal state and measurement flip with probability p
  seeds      sha256("<domain>:<code>:<p_ppm>:<regime>:<part>"); domains CALIB_2026 (calibration)
             and BLIND_HOLDOUT_2026 (evaluation) never share a seed

Subcommands
  selfcheck  code parameters, noiseless silence, seed determinism
  calib      CALIB_2026 datasets (bb72 + surface_d5 at p = 0.002)
  campaign   BLIND_HOLDOUT_2026 stationary matrix + drift experiments N1 / N2
  baseline   resumable multi-process BP-OSD / BP-LSD decode of one dataset
  baselines  the same for every dataset below a root, within a wall-clock budget
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

DOMAIN_CALIB = "CALIB_2026"
DOMAIN_BLIND = "BLIND_HOLDOUT_2026"
NOISE_POINTS = (0.001, 0.002, 0.003, 0.004)
SX_SCHEDULE = ["idle", 1, 4, 3, 5, 0, 2]
SZ_SCHEDULE = [3, 5, 0, 1, 2, 4, "idle"]
DORMANT_P = 1e-7
BB_SPECS = {
    "bb72": {"l": 6, "m": 6, "a": [("x", 3), ("y", 1), ("y", 2)], "b": [("y", 3), ("x", 1), ("x", 2)],
             "n": 72, "k": 12, "d": 6},
    "bb90": {"l": 15, "m": 3, "a": [("x", 9), ("y", 1), ("y", 2)], "b": [("x", 0), ("x", 2), ("x", 7)],
             "n": 90, "k": 8, "d": 10},
    "bb144_gross": {"l": 12, "m": 6, "a": [("x", 3), ("y", 1), ("y", 2)], "b": [("y", 3), ("x", 1), ("x", 2)],
                    "n": 144, "k": 12, "d": 12},
}
CODE_IDS = ("bb72", "bb90", "bb144_gross", "surface_d5", "surface_d7", "hgp_hamming")
BASELINE_CFG = {"bp_method": "minimum_sum", "max_iter": 100, "ms_scaling_factor": 0.625,
                "osd_method": "osd_cs", "osd_order": 7, "lsd_method": "lsd_cs", "lsd_order": 0}


# ----------------------------------------------------------------------------- GF(2) helpers
def gf2_rref(mat):
    m = (np.array(mat, dtype=np.uint8) & 1).copy()
    rows, cols = m.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            m[[r, p]] = m[[p, r]]
        elim = np.nonzero(m[:, c])[0]
        elim = elim[elim != r]
        m[elim] ^= m[r]
        pivots.append(c)
        r += 1
    return m, pivots


def gf2_rank(mat):
    return len(gf2_rref(mat)[1])


def gf2_nullspace(mat):
    mat = np.array(mat, dtype=np.uint8) & 1
    rref, pivots = gf2_rref(mat)
    cols = mat.shape[1]
    pset = set(pivots)
    free = [c for c in range(cols) if c not in pset]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            if rref[r, f]:
                basis[i, p] = 1
    return basis


def css_logical_z(hx, hz):
    """Basis of ker(HX) / rowspace(HZ): the Z-type logical operators."""
    ker = gf2_nullspace(hx)
    span = hz.copy()
    rank = gf2_rank(span)
    out = []
    for v in ker:
        trial = np.vstack([span, v[None, :]])
        r2 = gf2_rank(trial)
        if r2 > rank:
            span, rank = trial, r2
            out.append(v)
    return np.array(out, dtype=np.uint8)


# ----------------------------------------------------------------------------- codes
def bb_code(code_id):
    spec = BB_SPECS[code_id]
    l, m = spec["l"], spec["m"]
    eye_l, eye_m = np.eye(l, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    x = np.kron(np.roll(eye_l, 1, axis=1), eye_m)
    y = np.kron(eye_l, np.roll(eye_m, 1, axis=1))

    def mono(term):
        var, power = term
        base = x if var == "x" else y
        return (np.linalg.matrix_power(base.astype(np.int64), power) % 2).astype(np.uint8)

    a_terms = [mono(t) for t in spec["a"]]
    b_terms = [mono(t) for t in spec["b"]]
    a = (a_terms[0] + a_terms[1] + a_terms[2]) % 2
    b = (b_terms[0] + b_terms[1] + b_terms[2]) % 2
    hx = np.hstack([a, b]).astype(np.uint8)
    hz = np.hstack([b.T, a.T]).astype(np.uint8)
    return finish_css(code_id, "bb", hx, hz, spec["d"], spec["d"],
                      {"a_terms": a_terms, "b_terms": b_terms, "expect": (spec["n"], spec["k"])})


def hgp_hamming_code():
    h = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
    m, n = h.shape
    hx = np.hstack([np.kron(h, np.eye(n, dtype=np.uint8)), np.kron(np.eye(m, dtype=np.uint8), h.T)])
    hz = np.hstack([np.kron(np.eye(n, dtype=np.uint8), h), np.kron(h.T, np.eye(m, dtype=np.uint8))])
    return finish_css("hgp_hamming", "hgp", hx.astype(np.uint8), hz.astype(np.uint8), 3, 3, {"expect": (58, 16)})


def finish_css(code_id, family, hx, hz, d, rounds, extra):
    assert not ((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2).any(), "CSS orthogonality violated"
    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    assert (n, k) == extra["expect"], (code_id, n, k)
    lz = css_logical_z(hx, hz)
    assert lz.shape[0] == k
    code = {"code_id": code_id, "family": family, "hx": hx, "hz": hz, "lz": lz, "n": n, "k": k, "d": d,
            "rounds": rounds}
    code.update(extra)
    return code


def get_code(code_id):
    if code_id in BB_SPECS:
        return bb_code(code_id)
    if code_id == "hgp_hamming":
        return hgp_hamming_code()
    if code_id.startswith("surface_d"):
        d = int(code_id[len("surface_d"):])
        return {"code_id": code_id, "family": "surface", "n": d * d, "k": 1, "d": d, "rounds": d}
    raise ValueError(code_id)


def bb_neighbours(code):
    n2 = code["n"] // 2
    a_terms, b_terms = code["a_terms"], code["b_terms"]
    nbs_x, nbs_z = [], []
    for i in range(n2):
        row = [int(np.nonzero(a_terms[t][i, :])[0][0]) for t in range(3)]
        row += [n2 + int(np.nonzero(b_terms[t][i, :])[0][0]) for t in range(3)]
        nbs_x.append(row)
    for i in range(n2):
        row = [int(np.nonzero(b_terms[t][:, i])[0][0]) for t in range(3)]
        row += [n2 + int(np.nonzero(a_terms[t][:, i])[0][0]) for t in range(3)]
        nbs_z.append(row)
    return nbs_x, nbs_z


# ----------------------------------------------------------------------------- circuits
class BbTemplate:
    """Bivariate-bicycle memory-Z experiment (Z-check detectors, logical-Z observables),
    pre-assembled per round so drift variants only splice in their own noise instructions.

    hot:       round -> {data qubit: X-flip probability} added to the end-of-cycle idle of that
               qubit (TLS dropout / cosmic-ray heating).
    xt_pairs:  data-qubit pairs carrying a correlated XX channel right after CNOT step 3 of
               every cycle, i.e. conditioned on the syndrome schedule (dormant at DORMANT_P).
    xt_active: round -> {pair index: probability}.
    """

    def __init__(self, code, rounds, p, xt_pairs=None):
        self.code, self.rounds, self.p = code, rounds, p
        self.xt_pairs = xt_pairs or []
        n = code["n"]
        n2 = n // 2
        self.n2 = n2
        nbs_x, nbs_z = bb_neighbours(code)
        xq = list(range(0, n2))
        dq = list(range(n2, n2 + n))
        zq = list(range(n2 + n, 2 * n))
        data = lambda i: n2 + i  # noqa: E731
        head = stim.Circuit()
        head.append("R", dq)
        head.append("X_ERROR", dq, p)
        head.append("R", zq)
        head.append("X_ERROR", zq, p)
        head.append("TICK")
        self.head = head
        # part A: reset X ancillas ... CNOT step 3 (identical in every round)
        a = stim.Circuit()
        a.append("RX", xq)
        a.append("Z_ERROR", xq, p)
        pairs, used = [], set()
        for i in range(n2):
            dqb = data(nbs_z[i][SZ_SCHEDULE[0]])
            pairs += [dqb, zq[i]]
            used.add(dqb)
        a.append("CX", pairs)
        a.append("DEPOLARIZE2", pairs, p)
        idle = [q for q in dq if q not in used]
        if idle:
            a.append("DEPOLARIZE1", idle, p)
        a.append("TICK")

        def cnot_step(circ, t):
            pr = []
            for i in range(n2):
                pr += [xq[i], data(nbs_x[i][SX_SCHEDULE[t]])]
            for i in range(n2):
                pr += [data(nbs_z[i][SZ_SCHEDULE[t]]), zq[i]]
            circ.append("CX", pr)
            circ.append("DEPOLARIZE2", pr, p)

        for t in (1, 2, 3):
            cnot_step(a, t)
            if t < 3:
                a.append("TICK")
        self.part_a = a
        # part B: CNOT steps 4, 5, Z measurement, step 6, idle (up to the hot-channel slot)
        b = stim.Circuit()
        b.append("TICK")
        for t in (4, 5):
            cnot_step(b, t)
            b.append("TICK")
        b.append("M", zq, p)
        pairs, used = [], set()
        for i in range(n2):
            dqb = data(nbs_x[i][SX_SCHEDULE[6]])
            pairs += [xq[i], dqb]
            used.add(dqb)
        b.append("CX", pairs)
        b.append("DEPOLARIZE2", pairs, p)
        idle = [q for q in dq if q not in used]
        if idle:
            b.append("DEPOLARIZE1", idle, p)
        b.append("TICK")
        b.append("DEPOLARIZE1", dq, p)
        self.part_b = b
        # part C: X measurement, Z ancilla reset, detectors (first round differs)
        self.part_c = []
        for first in (True, False):
            cc = stim.Circuit()
            cc.append("MX", xq, p)
            cc.append("R", zq)
            cc.append("X_ERROR", zq, p)
            for i in range(n2):
                cur = stim.target_rec(i - 2 * n2)
                if first:
                    cc.append("DETECTOR", [cur], [i, 0, 0])
                else:
                    cc.append("DETECTOR", [cur, stim.target_rec(i - 4 * n2)], [i, 0, 0])
            cc.append("TICK")
            self.part_c.append(cc)
        tail = stim.Circuit()
        tail.append("M", dq, p)
        for i in range(n2):
            recs = [stim.target_rec(int(q) - n) for q in np.nonzero(code["hz"][i])[0]]
            recs.append(stim.target_rec(i - n - 2 * n2))
            tail.append("DETECTOR", recs, [i, rounds, 1])
        for j in range(code["lz"].shape[0]):
            tail.append("OBSERVABLE_INCLUDE", [stim.target_rec(int(q) - n) for q in np.nonzero(code["lz"][j])[0]], j)
        self.tail = tail
        self.data = data

    def assemble(self, hot=None, xt_active=None):
        hot = hot or {}
        xt_active = xt_active or {}
        c = self.head.copy()
        for r in range(self.rounds):
            c += self.part_a
            act = xt_active.get(r, {})
            for pi, (qa, qb) in enumerate(self.xt_pairs):
                c.append("CORRELATED_ERROR", [stim.target_x(self.data(qa)), stim.target_x(self.data(qb))],
                         float(act.get(pi, DORMANT_P)))
            c += self.part_b
            for qi, ph in sorted(hot.get(r, {}).items()):
                c.append("X_ERROR", [self.data(qi)], float(ph))
            part_c = self.part_c[0 if r == 0 else 1]
            # detector layer coordinate: rewrite (check, 0, 0) -> (check, r, 0)
            if r == 0:
                c += part_c
            else:
                shifted = stim.Circuit()
                for inst in part_c:
                    if inst.name == "DETECTOR":
                        args = inst.gate_args_copy()
                        shifted.append("DETECTOR", inst.targets_copy(), [args[0], r, 0])
                    else:
                        shifted.append(inst)
                c += shifted
        c += self.tail
        return c


def build_bb_memory_z(code, rounds, p, hot=None, xt_pairs=None, xt_active=None):
    return BbTemplate(code, rounds, p, xt_pairs).assemble(hot, xt_active)


def edge_colouring(h):
    """Greedy proper edge colouring of the bipartite Tanner graph: list of CNOT layers."""
    layers = []
    for c in range(h.shape[0]):
        for q in np.nonzero(h[c])[0]:
            for layer in layers:
                if all(c != c2 and q != q2 for c2, q2 in layer):
                    layer.append((c, int(q)))
                    break
            else:
                layers.append([(c, int(q))])
    return layers


def build_css_memory_z(code, rounds, p):
    """Generic CSS memory-Z experiment: per round all Z checks are extracted, then all X
    checks (always a valid schedule).  Z-check detectors and logical-Z observables only."""
    hx, hz, lz = code["hx"], code["hz"], code["lz"]
    n, mx, mz = hx.shape[1], hx.shape[0], hz.shape[0]
    dq = list(range(n))
    zq = list(range(n, n + mz))
    xq = list(range(n + mz, n + mz + mx))
    z_layers, x_layers = edge_colouring(hz), edge_colouring(hx)
    c = stim.Circuit()
    c.append("R", dq)
    c.append("X_ERROR", dq, p)
    c.append("TICK")
    meas_z, n_meas = [], 0
    for r in range(rounds):
        c.append("R", zq)
        c.append("X_ERROR", zq, p)
        for layer in z_layers:
            pairs = [t for chk, q in layer for t in (dq[q], zq[chk])]
            c.append("CX", pairs)
            c.append("DEPOLARIZE2", pairs, p)
            c.append("TICK")
        c.append("M", zq, p)
        meas_z.append(list(range(n_meas, n_meas + mz)))
        n_meas += mz
        c.append("RX", xq)
        c.append("Z_ERROR", xq, p)
        for layer in x_layers:
            pairs = [t for chk, q in layer for t in (xq[chk], dq[q])]
            c.append("CX", pairs)
            c.append("DEPOLARIZE2", pairs, p)
            c.append("TICK")
        c.append("MX", xq, p)
        n_meas += mx
        c.append("DEPOLARIZE1", dq, p)
        for i in range(mz):
            cur = meas_z[r][i] - n_meas
            if r == 0:
                c.append("DETECTOR", [stim.target_rec(cur)], [i, r, 0])
            else:
                c.append("DETECTOR", [stim.target_rec(cur), stim.target_rec(meas_z[r - 1][i] - n_meas)], [i, r, 0])
        c.append("TICK")
    c.append("M", dq, p)
    data_meas0 = n_meas
    n_meas += n
    for i in range(mz):
        recs = [stim.target_rec(data_meas0 + int(q) - n_meas) for q in np.nonzero(hz[i])[0]]
        recs.append(stim.target_rec(meas_z[rounds - 1][i] - n_meas))
        c.append("DETECTOR", recs, [i, rounds, 1])
    for j in range(lz.shape[0]):
        recs = [stim.target_rec(data_meas0 + int(q) - n_meas) for q in np.nonzero(lz[j])[0]]
        c.append("OBSERVABLE_INCLUDE", recs, j)
    return c


def build_circuit(code, p, **kw):
    rounds = code["rounds"]
    if code["family"] == "bb":
        return build_bb_memory_z(code, rounds, p, **kw)
    assert not kw, "drift injection is implemented for the bivariate-bicycle circuits"
    if code["family"] == "hgp":
        return build_css_memory_z(code, rounds, p)
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=code["d"], rounds=rounds,
                                  after_clifford_depolarization=p, after_reset_flip_probability=p,
                                  before_measure_flip_probability=p, before_round_data_depolarization=p)


# ----------------------------------------------------------------------------- DEM helpers
def site_coordinates(circuit, family):
    """detector -> (check index, layer, kind).  Surface-code detectors carry (x, y, t)."""
    coords = circuit.get_detector_coordinates()
    n_det = circuit.num_detectors
    if family != "surface":
        return [(int(coords[d][0]), int(coords[d][1]), int(coords[d][2])) for d in range(n_det)]
    sites = sorted({(coords[d][0], coords[d][1]) for d in range(n_det)})
    index = {s: i for i, s in enumerate(sites)}
    return [(index[(coords[d][0], coords[d][1])], int(coords[d][2]), 0) for d in range(n_det)]


def dem_text(dem, sites):
    lines = [str(inst) for inst in dem.flattened() if inst.type == "error"]
    lines += [f"detector({c}, {l}, {k}) D{d}" for d, (c, l, k) in enumerate(sites)]
    return "\n".join(lines) + "\n"


def dem_to_matrices(dem):
    """Returns (H csc uint8 [n_det x n_err], L csc uint8 [n_obs x n_err], priors float64)."""
    n_det, n_obs = dem.num_detectors, dem.num_observables
    sig_index, probs, cols_d, cols_o = {}, [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        pr = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        if not dets:
            continue
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        if key in sig_index:
            j = sig_index[key]
            probs[j] = probs[j] * (1 - pr) + pr * (1 - probs[j])
        else:
            sig_index[key] = len(probs)
            probs.append(pr)
            cols_d.append(key[0])
            cols_o.append(key[1])
    n_err = len(probs)
    hr, hc, lr, lc = [], [], [], []
    for j in range(n_err):
        for d in cols_d[j]:
            hr.append(d)
            hc.append(j)
        for o in cols_o[j]:
            lr.append(o)
            lc.append(j)
    h = sp.csc_matrix((np.ones(len(hr), dtype=np.uint8), (hr, hc)), shape=(n_det, n_err))
    lmat = sp.csc_matrix((np.ones(len(lr), dtype=np.uint8), (lr, lc)), shape=(n_obs, n_err))
    return h, lmat, np.array(probs, dtype=np.float64)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(domain, code_id, p, regime, *parts):
    text = ":".join([domain, code_id, str(int(round(p * 1e6))), regime] + [str(x) for x in parts]).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def write_b8(path, bits):
    np.packbits(np.asarray(bits, dtype=np.bool_), axis=1, bitorder="little").tofile(path)


def read_b8(path, width, lo=0, hi=None):
    nbytes = (width + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size % nbytes == 0, (path, raw.size, nbytes)
    raw = raw.reshape(-1, nbytes)[lo:hi]
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :width]


def write_dataset(out, code, circuit, dets, obs, meta_extra):
    out.mkdir(parents=True, exist_ok=True)
    dem = circuit.detector_error_model(flatten_loops=True)
    sites = site_coordinates(circuit, code["family"])
    (out / "circuit.stim").write_text(str(circuit) + "\n")
    (out / "model.dem").write_text(dem_text(dem, sites))
    write_b8(out / "dets.b8", dets)
    write_b8(out / "obs.b8", obs)
    h, _, _ = dem_to_matrices(dem)
    meta = {
        "schema": "prism.qec.sota.v1.dataset", "name": out.name, "code": code["code_id"], "family": code["family"],
        "n": code["n"], "k": code["k"], "d": code["d"], "rounds": code["rounds"], "basis": "memory_z",
        "shots": int(dets.shape[0]), "num_detectors": dem.num_detectors, "num_observables": dem.num_observables,
        "num_error_mechanisms": int(h.shape[1]), "num_edges": int(h.nnz),
        "checks_per_layer": 1 + max(s[0] for s in sites), "detector_layers": 1 + max(s[1] for s in sites),
        "stim_version": stim.__version__,
        "files": {f: sha256_file(out / f) for f in ("circuit.stim", "model.dem", "dets.b8", "obs.b8")},
    }
    meta.update(meta_extra)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[oracle] {out.name}: domain={meta['seed_domain']} shots={meta['shots']} det={meta['num_detectors']} "
          f"err={meta['num_error_mechanisms']} edges={meta['num_edges']}", flush=True)


# ----------------------------------------------------------------------------- generation
def gen_stationary(root, code_id, p, shots, domain):
    code = get_code(code_id)
    name = f"{code_id}_p{p:.4f}_stationary"
    out = Path(root) / "datasets" / name
    if (out / "meta.json").is_file():
        return name
    circuit = build_circuit(code, p)
    seed = derive_seed(domain, code_id, p, "stationary", "sample")
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    write_dataset(out, code, circuit, dets, obs,
                  {"kind": "stationary", "regime": "stationary", "p": p, "seed_domain": domain, "stim_seed": seed})
    return name


def cosmic_cluster(code, rng, target):
    n2 = code["n"] // 2
    zc = int(rng.integers(0, n2))
    cluster, frontier, seen = [], [zc], {zc}
    while frontier and len(cluster) < target:
        c = frontier.pop(0)
        for q in np.nonzero(code["hz"][c])[0]:
            if int(q) not in cluster and len(cluster) < target:
                cluster.append(int(q))
            for c2 in np.nonzero(code["hz"][:, q])[0]:
                if int(c2) not in seen:
                    seen.add(int(c2))
                    frontier.append(int(c2))
    return sorted(cluster)


def gen_drift(root, code_id, experiment, traces, domain, p=0.0015, total_rounds=500, onset_round=100):
    """Non-stationary stream: every trace is `blocks` consecutive memory blocks whose noise
    changes at global round `onset_round` without telling any decoder (all receive the
    nominal DEM).

      tls      N1: two random data qubits' idle fault channel jumps to X-flip probability 0.25
               and stays there.
      xtburst  N2: three schedule-conditioned crosstalk pairs activate (correlated XX with
               probability 0.06 after CNOT step 3) and a cosmic-ray impact multiplies the idle
               error of a compact cluster of n/9 data qubits by 20, relaxing with a 60-round
               time constant (cut off after four time constants).
    """
    code = get_code(code_id)
    n, n2, rounds = code["n"], code["n"] // 2, code["rounds"]
    name = f"{code_id}_p{p:.4f}_drift_{experiment}"
    out = Path(root) / "datasets" / name
    if (out / "meta.json").is_file():
        return name
    blocks = (total_rounds + rounds - 1) // rounds
    onset_block = onset_round // rounds
    pairs = [(i, n2 + i) for i in range(n2)]
    template = BbTemplate(code, rounds, p, pairs)
    nominal = template.assemble()
    n_det, n_obs = nominal.num_detectors, nominal.num_observables
    dets = np.zeros((traces * blocks, n_det), dtype=np.bool_)
    obs = np.zeros((traces * blocks, n_obs), dtype=np.bool_)
    pre = traces * onset_block
    if pre:
        s = nominal.compile_detector_sampler(seed=derive_seed(domain, code_id, p, experiment, "nominal"))
        d0, o0 = s.sample(pre, separate_observables=True)
        for tr in range(traces):
            dets[tr * blocks: tr * blocks + onset_block] = d0[tr * onset_block:(tr + 1) * onset_block]
            obs[tr * blocks: tr * blocks + onset_block] = o0[tr * onset_block:(tr + 1) * onset_block]
    truth = []
    for tr in range(traces):
        rng = np.random.default_rng(derive_seed(domain, code_id, p, experiment, "trace", tr))
        if experiment == "tls":
            hot_q = sorted(int(q) for q in rng.choice(n, size=2, replace=False))
            xt_idx, tau, p_hot, p_xt = [], None, 0.25, 0.0
        elif experiment == "xtburst":
            hot_q = cosmic_cluster(code, rng, max(6, n // 9))
            xt_idx = sorted(int(i) for i in rng.choice(len(pairs), size=3, replace=False))
            tau, p_hot, p_xt = 60.0, 20.0 * p * 2.0 / 3.0, 0.06
        else:
            raise ValueError(experiment)
        truth.append({"trace": tr, "hot_data_qubits": hot_q, "xt_pairs": [list(pairs[i]) for i in xt_idx],
                      "p_hot": p_hot, "p_xt": p_xt, "decay_tau_rounds": tau})
        steady = None
        for b in range(onset_block, blocks):
            hot, act = {}, {}
            for r in range(rounds):
                g = b * rounds + r
                if g < onset_round:
                    continue
                over = tau is not None and g - onset_round >= 4.0 * tau
                scale = 1.0 if tau is None else (0.0 if over else float(np.exp(-(g - onset_round) / tau)))
                if scale > 0.0:
                    hot[r] = {q: p_hot * scale for q in hot_q}
                if xt_idx:
                    act[r] = {i: p_xt for i in xt_idx}
            first_steady = onset_block + 1 if tau is None else (onset_round + int(4.0 * tau)) // rounds + 1
            if b >= first_steady:
                if steady is None:
                    steady = (b, template.assemble(hot, act))
                continue
            circ = template.assemble(hot, act)
            s = circ.compile_detector_sampler(seed=derive_seed(domain, code_id, p, experiment, "blk", tr, b))
            d1, o1 = s.sample(1, separate_observables=True)
            dets[tr * blocks + b], obs[tr * blocks + b] = d1[0], o1[0]
        if steady is not None:
            b0, circ = steady
            s = circ.compile_detector_sampler(seed=derive_seed(domain, code_id, p, experiment, "steady", tr))
            d1, o1 = s.sample(blocks - b0, separate_observables=True)
            dets[tr * blocks + b0:(tr + 1) * blocks] = d1
            obs[tr * blocks + b0:(tr + 1) * blocks] = o1
    write_dataset(out, code, nominal, dets, obs, {
        "kind": "drift_" + experiment, "regime": experiment, "p": p, "seed_domain": domain,
        "stim_seed": derive_seed(domain, code_id, p, experiment, "nominal"),
        "traces": traces, "blocks_per_trace": blocks, "onset_round": onset_round, "onset_block": onset_block,
        "onset_round_in_block": onset_round % rounds, "total_rounds": blocks * rounds,
        "xt_candidate_pairs": [list(x) for x in pairs]})
    (out / "drift_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    return name


# ----------------------------------------------------------------------------- baselines
def _decode_chunk(args):
    ds_dir, decoder, lo, hi, cfg, out_path = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ldpc import BpOsdDecoder
    from ldpc.bplsd_decoder import BpLsdDecoder
    ds = Path(ds_dir)
    dem = stim.DetectorErrorModel.from_file(str(ds / "model.dem"))
    h, lmat, priors = dem_to_matrices(dem)
    dets = read_b8(ds / "dets.b8", dem.num_detectors, lo, hi)
    priors_c = np.clip(priors, 1e-12, 0.5 - 1e-12)
    weights = np.log((1 - priors_c) / priors_c)
    common = dict(error_channel=list(priors_c), bp_method=cfg["bp_method"], max_iter=cfg["max_iter"],
                  ms_scaling_factor=cfg["ms_scaling_factor"], schedule="parallel")
    if decoder == "bposd":
        dec = BpOsdDecoder(h, osd_method=cfg["osd_method"], osd_order=cfg["osd_order"], **common)
    else:
        dec = BpLsdDecoder(h, lsd_method=cfg["lsd_method"], lsd_order=cfg["lsd_order"], **common)
    lcsr, hcsr = lmat.tocsr(), h.tocsr()
    nshots = hi - lo
    pred = np.zeros((nshots, lmat.shape[0]), dtype=np.uint8)
    lat = np.zeros(nshots, dtype=np.uint64)
    conv = np.zeros(nshots, dtype=np.uint8)
    valid = np.zeros(nshots, dtype=np.uint8)
    wsum = np.zeros(nshots, dtype=np.float32)
    margin = np.zeros(nshots, dtype=np.float32)
    for i in range(nshots):
        syn = np.ascontiguousarray(dets[i], dtype=np.uint8)
        t0 = time.perf_counter_ns()
        e = dec.decode(syn)
        lat[i] = time.perf_counter_ns() - t0
        e = np.asarray(e, dtype=np.uint8)
        pred[i] = (lcsr @ e) & 1
        conv[i] = 1 if dec.converge else 0
        valid[i] = 0 if (((hcsr @ e) & 1) ^ syn).any() else 1
        wsum[i] = float(weights @ e)
        # soft gap: smallest posterior log-likelihood-ratio magnitude on the decoded faults and
        # the faults BP was least sure about (the BP soft output the OSD stage starts from)
        llr = np.abs(np.asarray(dec.log_prob_ratios, dtype=np.float64))
        sup = np.nonzero(e)[0]
        margin[i] = float(llr[sup].min()) if sup.size else float(llr.min())
    np.savez(out_path, lo=lo, pred=pred, lat=lat, conv=conv, valid=valid, wsum=wsum, margin=margin)
    return lo


STAGE_SHOTS = 50000
MIN_ERRORS = 50


def cell_shots(ds, meta):
    """Statistical power rule, fixed in advance: a stationary cell uses the first 50,000 shots
    if BP-OSD makes at least 50 logical errors on them, otherwise all 100,000.  Drift datasets
    always use every shot.  Returns None while the BP-OSD first stage is still running."""
    if meta["kind"] != "stationary" or meta["shots"] <= STAGE_SHOTS:
        return meta["shots"]
    rule = ds / "baselines" / "shot_rule.json"
    return json.loads(rule.read_text())["shots"] if rule.is_file() else None


def chunk_jobs(meta, shots):
    """Chunk boundaries; the first stage always ends exactly at STAGE_SHOTS so its chunks are
    reused unchanged when a cell is extended."""
    chunk = int(min(5000, max(100, 2.5e6 / meta["num_error_mechanisms"])))
    edges = [0, min(shots, STAGE_SHOTS)] + ([shots] if shots > STAGE_SHOTS else [])
    jobs = []
    for a, b in zip(edges[:-1], edges[1:]):
        jobs += [(lo, min(b, lo + chunk)) for lo in range(a, b, chunk)]
    return jobs


def decode_missing(ds, decoder, jobs, out, workers, deadline):
    todo = [(lo, hi) for lo, hi in jobs if not (out / f"chunk_{lo:07d}.npz").is_file()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool:
        pending = set()
        it = iter(todo)
        while True:
            while len(pending) < workers and time.time() < deadline:
                nxt = next(it, None)
                if nxt is None:
                    break
                lo, hi = nxt
                pending.add(pool.submit(_decode_chunk, (str(ds), decoder, lo, hi, BASELINE_CFG,
                                                        str(out / f"chunk_{lo:07d}.tmp.npz"))))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                lo = f.result()
                os.replace(out / f"chunk_{lo:07d}.tmp.npz", out / f"chunk_{lo:07d}.npz")
    return all((out / f"chunk_{lo:07d}.npz").is_file() for lo, _ in jobs)


def run_baseline(ds_dir, decoder, workers, deadline):
    """Decodes every missing chunk until `deadline` (time.time()); returns True when complete."""
    ds = Path(ds_dir)
    meta = json.loads((ds / "meta.json").read_text())
    base = ds / "baselines"
    if (base / f"{decoder}.summary.json").is_file():
        return True
    out = base / decoder
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shots = cell_shots(ds, meta)
    if shots is None:
        if decoder != "bposd":
            return False  # the rule is set by the BP-OSD first stage
        jobs = chunk_jobs(meta, STAGE_SHOTS)
        if not decode_missing(ds, decoder, jobs, out, workers, deadline):
            print(f"[baseline] {meta['name']} {decoder}: first stage incomplete ({time.time() - t0:.0f}s)", flush=True)
            return False
        truth = read_b8(ds / "obs.b8", meta["num_observables"], 0, STAGE_SHOTS)
        errors = 0
        for lo, hi in jobs:
            errors += int((np.load(out / f"chunk_{lo:07d}.npz")["pred"] != truth[lo:hi]).any(axis=1).sum())
        shots = STAGE_SHOTS if errors >= MIN_ERRORS else meta["shots"]
        (base / "shot_rule.json").write_text(json.dumps(
            {"rule": "50000 shots if BP-OSD errors >= 50 on them, else all shots", "bposd_errors_first_stage": errors,
             "shots": shots}, indent=1) + "\n")
    jobs = chunk_jobs(meta, shots)
    complete = decode_missing(ds, decoder, jobs, out, workers, deadline)
    have = sum((out / f"chunk_{lo:07d}.npz").is_file() for lo, _ in jobs)
    print(f"[baseline] {meta['name']} {decoder}: {have}/{len(jobs)} chunks of {shots} shots "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not complete:
        return False
    merge_baseline(ds, decoder, meta, out, jobs, workers, shots)
    return True


def merge_baseline(ds, decoder, meta, out, jobs, workers, shots):
    n_obs = meta["num_observables"]
    pred = np.zeros((shots, n_obs), dtype=np.uint8)
    lat = np.zeros(shots, dtype=np.uint64)
    conv = np.zeros(shots, dtype=np.uint8)
    valid = np.zeros(shots, dtype=np.uint8)
    wsum = np.zeros(shots, dtype=np.float32)
    margin = np.zeros(shots, dtype=np.float32)
    for lo, hi in jobs:
        z = np.load(out / f"chunk_{lo:07d}.npz")
        pred[lo:hi], lat[lo:hi], conv[lo:hi] = z["pred"], z["lat"], z["conv"]
        valid[lo:hi], wsum[lo:hi], margin[lo:hi] = z["valid"], z["wsum"], z["margin"]
    truth = read_b8(ds / "obs.b8", n_obs, 0, shots)
    fail = (pred != truth).any(axis=1)
    base = ds / "baselines"
    write_b8(base / f"{decoder}.pred.b8", pred)
    lat.astype("<u8").tofile(base / f"{decoder}.lat_ns.u64")
    conv.tofile(base / f"{decoder}.bpconv.u8")
    wsum.astype("<f4").tofile(base / f"{decoder}.weight.f32")
    margin.astype("<f4").tofile(base / f"{decoder}.margin.f32")
    lat_sorted = np.sort(lat)

    def q(x):
        return int(lat_sorted[min(len(lat_sorted) - 1, int(x * len(lat_sorted)))])

    summary = {
        "schema": "prism.qec.sota.v1.baseline", "dataset": meta["name"], "decoder": decoder,
        "config": BASELINE_CFG, "shots": int(shots), "logical_failures": int(fail.sum()),
        "logical_error_rate": float(fail.mean()), "bp_converged_fraction": float(conv.mean()),
        "syndrome_valid_fraction": float(valid.mean()),
        "latency_ns": {"mean": float(lat.mean()), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
                       "p999": q(0.999), "max": int(lat_sorted[-1])},
        "workers": workers, "ldpc_version": __import__("ldpc").__version__,
    }
    (base / f"{decoder}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for leftover in out.iterdir():
        leftover.unlink()
    out.rmdir()
    print(json.dumps({k: summary[k] for k in ("dataset", "decoder", "shots", "logical_failures",
                                                "logical_error_rate", "bp_converged_fraction", "latency_ns")}))


def run_baselines(root, workers, max_seconds, only):
    deadline = time.time() + max_seconds
    names = sorted(p.name for p in (Path(root) / "datasets").iterdir() if (p / "meta.json").is_file())
    remaining = 0
    for decoder in ("bposd", "bplsd"):
        for name in names:
            if only and not any(tok in name for tok in only):
                continue
            if time.time() >= deadline:
                remaining += 1
                continue
            if not run_baseline(Path(root) / "datasets" / name, decoder, workers, deadline):
                remaining += 1
    print(f"[baselines] {'COMPLETE' if remaining == 0 else f'INCOMPLETE ({remaining} dataset/decoder pairs left)'}")
    return remaining == 0


# ----------------------------------------------------------------------------- plans
def write_manifest(root, domain):
    root = Path(root)
    names = sorted(p.name for p in (root / "datasets").iterdir() if (p / "meta.json").is_file())
    manifest = {"schema": "prism.qec.sota.v1.oracle_manifest", "seed_domain": domain, "datasets": names,
                "baseline_config": BASELINE_CFG, "stim_version": stim.__version__}
    (root / "oracle_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def run_calib(root, shots):
    for code_id in ("bb72", "surface_d5"):
        gen_stationary(root, code_id, 0.002, shots, DOMAIN_CALIB)
    gen_drift(root, "bb72", "tls", max(8, shots // 2000), DOMAIN_CALIB)
    gen_drift(root, "bb72", "xtburst", max(8, shots // 2000), DOMAIN_CALIB)
    write_manifest(root, DOMAIN_CALIB)


def run_campaign(root, shots, traces72, traces144, only, domain):
    assert domain != DOMAIN_CALIB, "the calibration domain is generated by `calib`"
    for code_id in CODE_IDS:
        for p in NOISE_POINTS:
            if only and not any(tok in f"{code_id}_p{p:.4f}_stationary" for tok in only):
                continue
            gen_stationary(root, code_id, p, shots, domain)
    for experiment in ("tls", "xtburst"):
        for code_id, traces in (("bb72", traces72), ("bb144_gross", traces144)):
            if only and not any(tok in f"{code_id}_p0.0015_drift_{experiment}" for tok in only):
                continue
            gen_drift(root, code_id, experiment, traces, domain)
    write_manifest(root, domain)


def selfcheck():
    assert derive_seed(DOMAIN_CALIB, "bb72", 0.002, "stationary", "sample") != \
        derive_seed(DOMAIN_BLIND, "bb72", 0.002, "stationary", "sample")
    for code_id in CODE_IDS:
        code = get_code(code_id)
        circ = build_circuit(code, 0.001)
        a = circ.compile_detector_sampler(seed=7).sample(64)
        b = circ.compile_detector_sampler(seed=7).sample(64)
        assert (a == b).all(), "seeded sampling is not deterministic"
        clean = build_circuit(code, 0.0).compile_detector_sampler(seed=1).sample(16, append_observables=True)
        assert not clean.any(), "noiseless circuit produced detection events or observable flips"
        dem = circ.detector_error_model(flatten_loops=True)
        sites = site_coordinates(circ, code["family"])
        assert len(set((c, l) for c, l, _ in sites)) == len(sites), "two detectors share a (check, layer) site"
        h, _, _ = dem_to_matrices(dem)
        print(f"[selfcheck] {code_id}: [[{code['n']},{code['k']},{code['d']}]] rounds={code['rounds']} "
              f"detectors={dem.num_detectors} errors={h.shape[1]} observables={dem.num_observables} OK")
    print("[selfcheck] PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    c = sub.add_parser("calib")
    c.add_argument("--root", required=True)
    c.add_argument("--shots", type=int, default=20000)
    g = sub.add_parser("campaign")
    g.add_argument("--root", required=True)
    g.add_argument("--shots", type=int, default=100000)
    g.add_argument("--traces72", type=int, default=400)
    g.add_argument("--traces144", type=int, default=400)
    g.add_argument("--only", nargs="*", default=[])
    g.add_argument("--domain", default=DOMAIN_BLIND,
                   help="seed domain; anything but the default is a pipeline smoke run on unrelated seeds")
    b = sub.add_parser("baseline")
    b.add_argument("--dataset", required=True)
    b.add_argument("--decoder", choices=("bposd", "bplsd"), required=True)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--max-seconds", type=float, default=480.0)
    bs = sub.add_parser("baselines")
    bs.add_argument("--root", required=True)
    bs.add_argument("--workers", type=int, default=6)
    bs.add_argument("--max-seconds", type=float, default=480.0)
    bs.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    if args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "calib":
        run_calib(args.root, args.shots)
    elif args.cmd == "campaign":
        run_campaign(args.root, args.shots, args.traces72, args.traces144, args.only, args.domain)
    elif args.cmd == "baseline":
        run_baseline(args.dataset, args.decoder, args.workers, time.time() + args.max_seconds)
    elif args.cmd == "baselines":
        run_baselines(args.root, args.workers, args.max_seconds, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
