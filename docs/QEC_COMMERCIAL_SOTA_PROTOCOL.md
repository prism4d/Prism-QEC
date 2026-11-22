# PRISM-QEC Commercial SOTA Evaluation & Statistical Proof Protocol (`prism.qec.sota.v1`)

## 1. Algorithmic Fix for `BB72` (`w_stab == d == 6`) Fast-Path False Convergence
1. **Velocity-Space Rayleigh Gauge Damping (Zero Fixed-Point Shift)**:
   - NEVER project static log-likelihood ratios $\ell^{(t)}$ onto $\mathrm{im}\,\partial_2$ — subtracting coexact components from static beliefs shifts the Bayes/Nishimori fixed point away from prior $\ell^{(0)}$ and pushes weight-3 faults into weight-3 logical complements on $d = 6$ codes.
   - Apply coexact projection **strictly to inter-iteration velocity** $\Delta \ell^{(t)} = \ell_{\text{BP}}^{(t)} - \ell^{(t-1)}$ restricted to the frustrated subgraph $\mathcal{F}_t$:
     $$\ell^{(t)} = \ell_{\text{BP}}^{(t)} - \eta \, \Pi_{\mathrm{im}\,\partial_2, \mathcal{F}_t}\!\left(\ell_{\text{BP}}^{(t)} - \ell^{(t-1)}\right)$$
   - Every exact BP fixed point ($\Delta \ell^{(t)} = 0$) experiences **zero bias**, while 2-cycle gauge trapping oscillations dissipate energy monotonically.
2. **Uncertainty-Gated Fast-Path Escalation (`syndrome == 0` Is Not Sufficient)**:
   - Even when $\partial_1 \hat{e}_{\text{BP}} \oplus s == 0$, the fast path MUST escalate $\hat{e}_{\text{BP}}$ directly as the exact manifold pivot ($e_{\text{piv}} = \hat{e}_{\text{BP}}$, zero Gaussian/forest setup cost!) into **Nishimori Twisted SMC** whenever ANY of:
     a. Convergence occurred after the unperturbed prefix ($t > T_{\text{clean}}$);
     b. Boundary LLR confidence margin $\min_{i \in \mathcal{B}(\hat{e}_{\text{BP}})} |\ell_i^{(t)}| < \tau_{\text{margin}}$;
     c. Local 1-plaquette $\mathbb{F}_2$ gauge descent ($\min_g H_N(\hat{e}_{\text{BP}} \oplus \partial_2 g)$) lowers Nishimori energy or reveals a competing valley within $\Delta H_N < \tau_{\text{gap}}$.
   - **Hard Gate**: Fast-path false-convergence rate on `BB72` MUST drop below `0.10 * total_logical_errors`, and overall `PRISM-QEC` LER on `BB72` MUST NOT regress against `BP-OSD` (`p_mcnemar_worse > 0.05` or `prism_errors <= bposd_errors`).

## 2. Zero-Tuning Hyperparameter Lock & Blind Seed Quarantine
1. **Single Immutable Config (`CanonicalDecoderConfig::FROZEN_V1`)**:
   - All dimensionless solver parameters (`T_clean`, `T_max`, Rayleigh damping $\eta$, escalation margin $\tau_{\text{margin}}$, gap $\tau_{\text{gap}}$, walker count $W$, annealing schedule) MUST live in a single compile-time `const FROZEN_V1` struct whose SHA-256 is recorded in every receipt.
   - Calibration may inspect ONLY `seed_domain = "CALIB_2026"` (`BB72` + `Surface_d5` at $p = 0.002$).
   - **Zero per-code, per-distance, or per-$p$ hyperparameter branching or CLI tuning flags** are permitted on blind evaluation runs (`seed_domain = "BLIND_HOLDOUT_2026"`, deterministic SHA-256 seed derivation from `(code_id, p_ppm, regime)`).

## 3. Multi-Family, Multi-Distance, Multi-Noise Matrix & Statistical Power
Evaluate `PRISM-QEC` (`FROZEN_V1`), `BP-OSD` (`ldpc` combination-sweep `osd_order=7`), and `BP-LSD` (`ldpc`) on the **exact same blind `.b8` syndrome streams**:
1. **Code Families**:
   - **Bivariate Bicycle qLDPC (Bravyi et al. 2024)**:
     - `bb72`: `[[72, 12, 6]]` ($l=6, m=6, A=x^3+y+y^2, B=y^3+x+x^2$)
     - `bb90`: `[[90, 8, 10]]` ($l=15, m=3, A=x^9+y+y^2, B=1+x^2+x^7$) — *unseen geometry during calibration*
     - `bb144_gross`: `[[144, 12, 12]]` ($l=12, m=6, A=x^3+y+y^2, B=y^3+x+x^2$)
   - **Topological 2D Surface Code (`Stim` circuit-level `surface_code:rotated_memory_z`)**:
     - `surface_d5` ($d=5$) and `surface_d7` ($d=7$, unseen distance)
   - **Hypergraph Product (`HGP`) qLDPC**:
     - `hgp_hamming` (`[7,4,3] x [7,4,3]` classical Hamming HGP, `[[85, 16, 3]]` or `[15,7,5]` HGP — unseen check-degree family)
2. **Noise Points & Regimes**:
   - Stationary circuit-level / DEM noise across **$p \in \{0.001, 0.002, 0.003, 0.004\}$** (at least 3 noise points per code).
   - **Statistical Power Law**: Every stationary benchmark point MUST run until **`shots >= 50_000` AND (`logical_errors >= 50` OR `shots >= 100_000`)** (no 2,000-shot toy runs!).
   - Every cell MUST report: `shots`, `prism_errors`, `bposd_errors`, `bplsd_errors`, exact **95% Clopper-Pearson intervals**, and **paired discordant counts** `(prism_wins_bposd_fails, prism_fails_bposd_wins)` + exact two-sided **McNemar $p$-value**.

## 4. Mandatory Non-Stationary (`BOCPD + do-TIDE`) & HLE Frontier Proofs (Measured, Not Speculative)
1. **Non-Stationary Drift Benchmark (`neuro_dem_evaluation` in receipt)**:
   - **Experiment N1 (TLS Hot-Qubit Dropout)**: On `bb72` and `bb144_gross` ($p_{\text{base}} = 0.0015$, `>= 50,000` shots across contiguous 500-round streams), 2 random physical fault mechanisms jump to $p_{\text{TLS}} = 0.25$ at round 100 without telling any decoder. Compare `BP-OSD` (static DEM), `PRISM-QEC` (`neuro-dem` OFF), and `PRISM-QEC` (`BOCPD + do-TIDE` ON). Require statistically significant LER reduction (`>= 3x` lower LER than static `BP-OSD`, `mcnemar_p < 1e-6`).
   - **Experiment N2 (Schedule-Correlated Coupler Crosstalk + Cosmic-Ray Burst)**: Inject correlated pair faults conditioned on active syndrome schedule + $20\times$ transient burst; measure `do-TIDE` edge precision/recall and LER gain vs. `BOCPD`-only and static `BP-OSD`.
2. **Heralded Logical Erasure (`hle_pareto_frontier` in receipt)**:
   - On `bb144_gross`, report retained LER at exact abort budgets `0.1%`, `0.5%`, `1.0%`, `2.0%` comparing `PSIS k_hat + min_j |Delta F_j|` against `BP-OSD` soft-gap post-selection on the exact same shots.
3. **SM120 RTX 5080 Hardware Proof (`gpu_sm120_telemetry` in receipt, Gate 4)**:
   - `[28, 56]` SM Green Contexts, `host_syncs_in_decode_loop == 0`, persisting L2 residency, `p50_us`, `p99_us`, `p999_us` per-shot latency, `shots_per_sec` speedup vs. `BP-OSD`, and GPU-vs-CPU parity check.
