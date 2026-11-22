# PRISM-QEC: Causal Hodge-Twisted Thermodynamic Coset Decoding on SM120

## 1. Flat Crate Topology
All production Rust code lives in flat workspace crates wired into `crates/prism-qec-cli` (`bin: prism-qec`):
1. `crates/qec-core`:
   - Bivariate Bicycle (BB) group-algebra constructor over $\mathbb{F}_2[x, y] / (x^\ell - 1, y^m - 1)$ generating exact IBM `[[72, 12, 6]]` (`l=6, m=6`) and `[[144, 12, 12]]` Gross (`l=12, m=6`, $A = x^3 + y + y^2$, $B = y^3 + x + x^2$) parity matrices $(H_X, H_Z)$, logical bases $(L_X, L_Z)$, and space-time circuit Detector Error Models (`Stim` `.dem` + `.b8` binary shot/observable streams).
   - 3-term cellular chain complex $C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0$ with load-time assertion $\partial_1 \circ \partial_2 = 0 \pmod 2$.
   - Dual `SHA-256 + BLAKE3` Merkle seal (`chronicle_merkle_seal.json`) over all benchmark inputs, Stim seeds, DEM files, shot files, and `qec_frontier_receipt.json`.
2. `crates/qec-hodge-bp`:
   - Wavefront normalized min-sum / belief-propagation coupled every $m$ iterations with **Matrix-Free Coexact Curl Projection**:
     $$\ell^{(t + 1/2)} = \ell^{(t)} - \gamma W_t^{-1} \partial_2 \mathrm{diag}(\partial_2^T W_t^{-1} \partial_2)^{-1} \partial_2^T \ell^{(t)}, \quad W_t = \mathrm{diag}(\tanh(|\ell^{(t)}|/2) + \epsilon)$$
   - Provably preserves syndrome boundary $\partial_1$ and 1st cohomology $\mathcal{H}^1 = \ker \partial_1 / \mathrm{im}\,\partial_2$ while damping symmetric stabilizer cycles (`im d2`) that trap standard BP.
3. `crates/qec-smc-coset`:
   - Triggered ONLY on residual frustrated clusters where fast-path $\partial_1 \hat{e} \oplus s \neq 0$.
   - **Exact Manifold Pivot**: Fast local GF(2) spanning-forest solver finds $e_{\text{piv}}$ satisfying $\partial_1 e_{\text{piv}} = s$ identically, parameterizing all valid syndrome solutions without hard constraints as $e(g, \ell) = e_{\text{piv}} \oplus \partial_2 g \oplus L \ell$.
   - **Nishimori Twisted SMC ($\beta: 0 \to 1$)**: Evolves $W \in \{64, 128, 256\}$ parallel walkers under exact Nishimori energy $H_N(g, \ell) = \sum_i w_i [e(g, \ell)]_i$ ($w_i = \ln((1-p_i)/p_i)$) with cluster-flip Gibbs/Metropolis-Hastings mutations guided by Hodge-BP soft beliefs.
   - **Marginal Coset Free-Energy Gaps**: Computes $\Delta F_j = -\ln(\sum_{w:\ell_{w,j}=1} \tilde{\omega}_w / \sum_{w:\ell_{w,j}=0} \tilde{\omega}_w)$ across all $k=12$ logical qubits simultaneously ($O(W)$, avoiding $2^{12}=4096$ coset enumeration).
   - **Thermodynamically Certified Heralded Logical Erasure (`HLE`)**: Computes Vehtari's **PSIS Pareto $\hat{k}$** on terminal log-weights BEFORE resampling; flags `HLE = true` iff $\hat{k} > 0.7$ or $\min_j |\Delta F_j| < \tau_{\text{cert}}$.
4. `crates/qec-neuro-dem`:
   - Sparse neuromorphic defect-spike monitor running online **Bayesian Online Changepoint Detection (`BOCPD`)** per detector + **Schedule-Conditioned Transfer Entropy (`do-TIDE`)** per coupler pair.
   - Performs live in-memory/L2 `DEM` weight surgery: sudden TLS hot-qubit spikes set incident $w_i \leftarrow 0$ ($p_i \to 0.5$ known erasure) within 5–10 rounds; correlated crosstalk activates hyperedge weights without offline recalibration.
5. `crates/qec-cuda` + `cuda/qec_sm120_kernels.cu`:
   - Single startup static VRAM arena + `CU_MEMHOSTALLOC_DEVICEMAP` zero-copy syndrome ring + `GB203` persisting L2 pin for `DEM` weights.
   - Asymmetric Green Contexts `[28, 56]` SMs (84 SMs total on RTX 5080) + `CU_GRAPH_COND_TYPE_WHILE` (`host_syncs == 0`):
     - 28-SM fast path executes persistent Wavefront Hodge-BP + `BOCPD` update.
     - Device-side residual check sets conditional handle `1` iff $\partial_1 \hat{e} \oplus s \neq 0$, waking the 56-SM Nishimori Twisted-SMC solver in-device with zero CPU roundtrips.
   - **Exact $\mathbb{F}_2$ Bit-Matrix Engine**: Implements bit-packed `u64` `__popcll(a & b) & 1` AND Blackwell `.b1` Tensor-Core `mma.sync.aligned.m16n8k256.s32.b1.b1.and.popc` (taking `acc & 1` for exact $\mathbb{F}_2$ parity multiplication, never conflating `xor.popc` Hamming weight with `and.popc` mod-2 dot product), selecting the fastest verified `sm_120` path via microbenchmark receipt.
6. `pytools/stim_bb_oracle.py`:
   - Uses `$QEC_ROOT/.venv/bin/python` (`stim`, `ldpc`, `pymatching`, `scipy`) to generate deterministic circuit-level BB72 & BB144 syndrome datasets (stationary $p \in \{0.001, 0.002, 0.003\}$ + mid-run round-500 TLS/cosmic-ray burst traces) and runs reference `BP-OSD` and `BP-LSD` baselines on the exact same shot files for falsifiable head-to-head comparison.

## 2. Five Mandatory Referee Gates (`./harness/qec-ctl.sh verify-gate 0..4`)
- **Gate 0**: Air-gap segregation, `0555` immutable hooks, and `.venv` (`stim`, `ldpc`, `pymatching`) health.
- **Gate 1**: `clippy -D warnings` + `qec-core` & `qec-hodge-bp` algebraic tests ($\partial_1 \circ \partial_2 = 0$ on BB72/BB144; proof that Hodge curl projection preserves $\partial_1$ & $\mathcal{H}^1$ and breaks symmetric trapping loops).
- **Gate 2**: Full CPU workspace parity (`qec-smc-coset` exact $2^k$ partition-function match on small degenerate codes, PSIS $\hat{k}$ HLE dominance over heuristic gap norms, `BOCPD + do-TIDE` online survival under injected TLS/burst drift).
- **Gate 3**: Offline `nvcc -arch=sm_120 --ptxas-options=-v` kernel build (**0 bytes local/stack spill**, zero `cudaDeviceSynchronize()`) + release CLI build (0 VRAM required).
- **Gate 4**: Live RTX 5080 execution via `qec-ctl.sh gpu-exec` verifying `[28, 56]` Green Contexts, `host_syncs == 0`, persisting L2 residency, GPU-vs-CPU parity, and head-to-head BB144 campaign receipt (`runs/bb144_sota_campaign/qec_frontier_receipt.json` + dual `SHA-256 + BLAKE3` `chronicle_merkle_seal.json`) beating static `BP-OSD`/`BP-LSD` on tail latency, HLE post-selection frontier, and non-stationary burst survival.
