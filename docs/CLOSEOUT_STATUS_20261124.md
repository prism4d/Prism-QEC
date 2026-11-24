# PRISM-QEC Post-Tmux Closeout & Gate Status (20261124T043808Z)

## 1. Operational Disposition
- **Repository**: `/mnt/storage/prism-qec`
- **Pre-Closeout Commit / Tree**: `8161c2e18b9a7fa65879dba01078e35611986c4b` (tree `de9961734c367f2d1174c13019679bf829db67d8`)
- **GPU Arbitration**: Host `tmux` session deactivated intentionally to eliminate RTX 5080 (`sm_120`) VRAM/SM contention with the concurrent `PRISM-4D` campaign. Closeout executed strictly under `CUDA_VISIBLE_DEVICES=""`.
- **Artifact Integrity Audit**: `1801` gate/benchmark/log artifacts hashed (`0` mid-write truncations detected).
- **Flattened Executable Source**: `PRISM_QEC_EXECUTABLE_SRC_FLATTENED.txt` (`68` files, `23,562` LOC, `sha256:1e2b79c3426ec155c07561fb95f545016b42293e612e2f92c3fd5ee25e362e1b`)
- **Results Archive**: `prism-qec-results-snapshot-20261124T043808Z.tar.zst` (`169,205,023` B, `sha256:808327efff0f34d217ffe5749e04b748a68e771c897240efe80387bdbc8a32f9`)

## 2. Audited Gate & Benchmark Artifacts
| Artifact Path | Size (B) | Modified (UTC) | Integrity | Key Extracted Metrics | SHA-256 (prefix) |
| :--- | ---: | :--- | :--- | :--- | :--- |
| `.harness/logs/gate1_algebra.log` | 6,212 | `2026-11-23T22:18:01Z` | `OK` | — | `13ee4d5c7397d02b…` |
| `.harness/logs/gate1_clippy.log` | 157 | `2026-11-23T22:17:43Z` | `OK` | — | `4dfd82f7ed5b0e9f…` |
| `.harness/logs/gate2_clippy.log` | 157 | `2026-11-23T22:18:01Z` | `OK` | — | `0463a2a11a1ceec6…` |
| `.harness/logs/gate2_full_cpu.log` | 7,851 | `2026-11-23T22:18:13Z` | `OK` | — | `a0aab0b0bf772413…` |
| `.harness/logs/gate3_release_build.log` | 216 | `2026-11-23T22:18:17Z` | `OK` | — | `e0e9daf07486fd19…` |
| `.harness/logs/gate4_gpu_tests.log` | 7,350 | `2026-11-23T20:31:30Z` | `OK` | — | `da1de325a542dc3f…` |
| `.harness/state/gate_0.passed` | 143 | `2026-11-23T22:18:17Z` | `OK` | — | `658b052f164ce401…` |
| `.harness/state/gate_1.passed` | 143 | `2026-11-23T22:18:01Z` | `OK` | — | `0db12bdf6811ec55…` |
| `.harness/state/gate_2.passed` | 143 | `2026-11-23T22:18:13Z` | `OK` | — | `249647f87fdb5f38…` |
| `.harness/state/gate_3.passed` | 143 | `2026-11-23T22:18:17Z` | `OK` | — | `74913730a60cc506…` |
| `.harness/state/gate_4.passed` | 143 | `2026-11-23T20:31:30Z` | `OK` | — | `aabc18fba07f0ec1…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/meta.json` | 867 | `2026-11-22T08:07:30Z` | `OK` | `shots=100000` | `565b70720a929402…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/bplsd.summary.json` | 614 | `2026-11-22T12:46:21Z` | `OK` | `logical_error_rate=2e-05`, `shots=100000` | `2fd70e64eac05151…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/bposd.summary.json` | 619 | `2026-11-22T08:44:34Z` | `OK` | `logical_error_rate=1e-05`, `shots=100000` | `ed5691da8f88c2cf…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0010_stationary/baselines/shot_rule.json` | 124 | `2026-11-22T08:24:00Z` | `OK` | `shots=100000` | `b7c1e7c7e9eb242d…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_tls/meta.json` | 2,737 | `2026-11-23T00:58:26Z` | `OK` | `shots=16800` | `1bc97b976882e89f…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_tls/baselines/bplsd.summary.json` | 651 | `2026-11-23T01:20:27Z` | `OK` | `logical_error_rate=0.013571428571428571`, `shots=16800` | `7b7422d7d071fb95…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_tls/baselines/bposd.summary.json` | 653 | `2026-11-23T01:08:23Z` | `OK` | `logical_error_rate=0.004404761904761904`, `shots=16800` | `a4f509466ae10f8c…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_xtburst/meta.json` | 2,794 | `2026-11-23T01:00:23Z` | `OK` | `shots=16800` | `e9084ef7cb0b6ce9…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_xtburst/baselines/bplsd.summary.json` | 657 | `2026-11-23T01:21:41Z` | `OK` | `logical_error_rate=0.09886904761904762`, `shots=16800` | `8745aa87882c2a88…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift2_xtburst/baselines/bposd.summary.json` | 657 | `2026-11-23T01:18:33Z` | `OK` | `logical_error_rate=0.05255952380952381`, `shots=16800` | `0a785b244c06c1c4…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift_tls/meta.json` | 2,635 | `2026-11-22T08:07:53Z` | `OK` | `shots=16800` | `e392033950914a1a…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift_tls/baselines/bposd.summary.json` | 653 | `2026-11-22T08:51:02Z` | `OK` | `logical_error_rate=5.9523809523809524e-05`, `shots=16800` | `8d43f90f88a8da96…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift_xtburst/meta.json` | 2,647 | `2026-11-22T08:12:25Z` | `OK` | `shots=16800` | `8780803fb422c723…` |
| `runs/bb144_sota_campaign/datasets/bb144_gross_p0.0015_drift_xtburst/baselines/bposd.summary.json` | 638 | `2026-11-22T08:59:25Z` | `OK` | `logical_error_rate=0.0`, `shots=16800` | `64e77a1a7de1d340…` |
