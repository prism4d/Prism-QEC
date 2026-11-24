<!-- PRISM_QEC_AUTOGEN_README -->
# PRISM-QEC

**Deterministic GPU-Accelerated Quantum Error Correction, Non-Equilibrium Syndrome Decoding & Hardware Verification (`sm_120`)**  
**Author / Architect:** Ididia J. Serfaty (`is@delfictus.com`) — Delfictus IO Inc.

---

## 1. Overview

`PRISM-QEC` is a deterministic, zero-proxy Rust + CUDA (`sm_120` NVIDIA Blackwell) architecture engineered for high-throughput quantum error correction (QEC) decoding, fault-tolerant state verification, and hardware-governed benchmark gates.

- **Current Version:** `v0.1.0`
- **Source Footprint:** `68` source/build files (`10,296` lines of Rust, `1,803` lines of CUDA C++/headers)
- **Pre-Closeout Tree Lineage:** `de9961734c367f2d1174c13019679bf829db67d8` (`HEAD` prior to closeout: `8161c2e18b9a7fa65879dba01078e35611986c4b`)
- **Latest Closeout & Gate Ledger:** [`docs/CLOSEOUT_STATUS_20261124.md`](docs/CLOSEOUT_STATUS_20261124.md) · [`docs/QEC_CLOSEOUT_LEDGER.json`](docs/QEC_CLOSEOUT_LEDGER.json)

---

## 2. Workspace Crate Topology

| Crate | Version | Targets | Path | Rust Files (LOC) | Role / Summary |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `qec-core` | `0.1.0` | `lib` | `crates/qec-core` | 7 (`1,902` LOC) | Core F2 cellular chain-complex, Bivariate Bicycle qLDPC, and dual-Merkle primitives. |
| `qec-hodge-bp` | `0.1.0` | `lib` | `crates/qec-hodge-bp` | 4 (`1,004` LOC) | Wavefront normalized min-sum belief propagation coupled inline with the matrix-free |
| `qec-smc-coset` | `0.1.0` | `lib` | `crates/qec-smc-coset` | 5 (`1,919` LOC) | Spanning-forest manifold pivot, Nishimori twisted SMC over coset coordinates, marginal |
| `qec-neuro-dem` | `0.1.0` | `lib` | `crates/qec-neuro-dem` | 6 (`2,367` LOC) | Neuromorphic defect-spike monitor: per-detector Bayesian online changepoint detection, |
| `qec-cuda` | `0.1.0` | `custom-build,lib,test` | `crates/qec-cuda` | 4 (`1,062` LOC) | SM120 (RTX 5080) execution of the PRISM-QEC decoder: one static VRAM arena, a |
| `prism-qec-cli` | `0.1.0` | `bin` | `crates/prism-qec-cli` | 5 (`1,982` LOC) | `prism-qec`: end-to-end CLI of the PRISM-QEC decoder stack. |

---

## 3. CUDA `sm_120` Device Kernel Inventory

| Kernel Entrypoint (`__global__`) | Source File | Target Architecture |
| :--- | :--- | :--- |
| `qec_neuro_update` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_bp_decode` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_fast_advance` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_slow_gate` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_slow_advance` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_f2_matmul_popc` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_f2_matmul_b1` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |
| `qec_smc_decode` | `cuda/qec_sm120_kernels.cu` | `sm_120` (Blackwell) |

---

## 4. Executables & Reproducible Build

### Executable Targets
- **`prism-qec`** (crate `prism-qec-cli`, entrypoint `crates/prism-qec-cli/src/main.rs`): `cargo build --release --bin prism-qec`

### Host & Toolchain Requirements
- **OS:** Linux x86_64 (verified on Ubuntu 24.04 / PrismOS)
- **GPU:** NVIDIA GeForce RTX 5080 (`sm_120` Blackwell, 16 GB GDDR7) with CUDA 13.x (`nvcc`)
- **Compiler:** Stable Rust (`cargo` / `rustc`, deterministic release profile)

### Build & Verification Commands
```bash
# Compile release binaries
cargo build --release --workspace

# Run pure-CPU unit & contract tests without touching GPU 0 (safe alongside PRISM-4D)
CUDA_VISIBLE_DEVICES="" cargo test --release --workspace --lib

# Execute live RTX 5080 hardware gates ONLY when NVML reports 0 competing compute processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 5. Gate Verification & Latest Closeout Snapshot

| Artifact | Size (B) | Integrity | Key Metrics | SHA-256 (prefix) |
| :--- | ---: | :--- | :--- | :--- |
| `.harness/logs/gate1_algebra.log` | 6,212 | `OK` | verified | `13ee4d5c7397d02b…` |
| `.harness/logs/gate1_clippy.log` | 157 | `OK` | verified | `4dfd82f7ed5b0e9f…` |
| `.harness/logs/gate2_clippy.log` | 157 | `OK` | verified | `0463a2a11a1ceec6…` |
| `.harness/logs/gate2_full_cpu.log` | 7,851 | `OK` | verified | `a0aab0b0bf772413…` |
| `.harness/logs/gate3_release_build.log` | 216 | `OK` | verified | `e0e9daf07486fd19…` |
| `.harness/logs/gate4_gpu_tests.log` | 7,350 | `OK` | verified | `da1de325a542dc3f…` |
| `.harness/state/gate_0.passed` | 143 | `OK` | verified | `658b052f164ce401…` |
| `.harness/state/gate_1.passed` | 143 | `OK` | verified | `0db12bdf6811ec55…` |
| `.harness/state/gate_2.passed` | 143 | `OK` | verified | `249647f87fdb5f38…` |
| `.harness/state/gate_3.passed` | 143 | `OK` | verified | `74913730a60cc506…` |
| `.harness/state/gate_4.passed` | 143 | `OK` | verified | `aabc18fba07f0ec1…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/meta.json` | 867 | `OK` | `shots=100000` | `565b70720a929402…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/bplsd.summary.json` | 614 | `OK` | `logical_error_rate=2e-05`, `shots=100000` | `2fd70e64eac05151…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/bposd.summary.json` | 619 | `OK` | `logical_error_rate=1e-05`, `shots=100000` | `ed5691da8f88c2cf…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/shot_rule.json` | 124 | `OK` | `shots=100000` | `b7c1e7c7e9eb242d…` |

---

## 6. Intellectual Property & License

Copyright (c) 2026 Delfictus IO Inc. (Ididia J. Serfaty). All rights reserved. Proprietary and confidential unless explicitly licensed otherwise in writing.
