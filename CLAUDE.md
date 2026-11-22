# PRISM-QEC Sovereign Architecture & Execution Contract (`prism.qec.v1`)

## 0. Air-Gap, Process & RTX 5080 Contention Laws (Mechanically Enforced by `0555` Hooks)
1. **Strict Repository Air-Gap (`QEC-SEG-01`)**:
   - Workspace root is strictly `/mnt/storage/prism-qec` on branch `main`.
   - You MUST NEVER read, write, symlink, grep, or mention any sibling repository, data vault, or molecular run tree. `git worktree`, `git submodule`, and nested `Cargo.toml` layouts are forbidden.
2. **Isolated Tmux Control Plane (`QEC-SEG-02`)**:
   - Never invoke bare `tmux` (reserved for sibling GPU workloads) or any other tmux socket, and never run `pkill`, `killall`, `kill -9`, `nohup`, or `sleep >= 3`.
   - Route all builds, Python/Stim evaluations, and GPU benchmarks exclusively through `./harness/qec-ctl.sh`.
3. **RTX 5080 Live Contention Mutex (`QEC-SEG-03`)**:
   - Default session environment sets `CUDA_VISIBLE_DEVICES=""` and bounds CPU builds to 6 threads.
   - Offline CUDA kernel compilation (`nvcc -arch=sm_120 --cubin` / `--ptx --ptxas-options=-v`) uses **0 bytes of VRAM** and MAY run at any time (Gate 3).
   - Live RTX 5080 execution MUST run **exclusively** via `./harness/qec-ctl.sh gpu-exec <label> <cmd...>` (Gate 4), which atomically verifies zero external GPU processes, `VRAM < 1200 MiB`, and `GPU util < 10%` before exposing `CUDA_VISIBLE_DEVICES=0`. If `gpu-exec` reports `[GPU-LEASE-DENIED]`, finish all Gate 0..3 work first and wait or notify the operator—never bypass `gpu-exec`.
4. **Anti-Piecemeal Unified Delivery Law**:
   - `todo!()`, `unimplemented!()`, `allow(dead_code)`, `examples/`, mock/hardcoded benchmark tables, and global `cudaDeviceSynchronize()` in `.cu`/`.cuh` are blocked.
   - Build and wire ALL crates (`qec-core`, `qec-hodge-bp`, `qec-smc-coset`, `qec-neuro-dem`, `qec-cuda`, `prism-qec-cli`) + `pytools/stim_bb_oracle.py` end-to-end before claiming benchmark closure.

## 1. Scientific & Systems Architecture
Read `docs/PRISM_QEC_ARCHITECTURE.md` in full before writing code. All four innovations must be implemented genuinely and verified against `Stim` circuit-level Bivariate Bicycle codes (`[[72, 12, 6]]` and `[[144, 12, 12]]` Gross code) and head-to-head `BP-OSD` / `BP-LSD` baselines from `/mnt/storage/prism-qec/.venv/bin/python`.
