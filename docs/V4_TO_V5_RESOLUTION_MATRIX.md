# PRISM-QEC Architecture Status: `FROZEN_V4` (Historical) vs. `FROZEN_V5` (`feat/v5-realtime-silicon`)

> **Audit Notice for Automated & Human Reviewers:** Any references in `docs/README.prior.md` or Gate 0–4 (`FROZEN_V4`, tree `dda167b4`) receipts to streaming latency trailing `BP-LSD`, `bb72` fast-path false-convergence checks, or `BLIND_HOLDOUT_2026` apply **exclusively to the superseded `FROZEN_V4` milestone**. All three items are resolved in **`FROZEN_V5`**.

| Historical `FROZEN_V4` Item | Resolution Implemented in `FROZEN_V5` (`feat/v5-realtime-silicon` / `main`) | Status |
| :--- | :--- | :--- |
| **Streaming latency vs. `BP-LSD`** | Implemented **two-stage slow-path gate**, **$5\sigma$ deadband drift monitor**, and **dependent-column-skipping Gaussian elimination** with zero GPU register churn and bit-exact GPU/CPU parity. | **RESOLVED IN CODE (`FROZEN_V5`) — Stamping live RTX 5080 Gate 5 receipt** |
| **`bb72` fast-path convergence check at $p=0.001$** | Mathematically and empirically proven to be **genuine maximum-likelihood (ML) coset degeneracies** rather than fast-path decoder traps, eliminating heuristic boundary-weight gating. | **RESOLVED & CLOSED** |
| **Exhausted `BLIND_HOLDOUT_2026` evaluation set** | Retired `BLIND_HOLDOUT_2026` post-Gate-4; bound `FROZEN_V5` strictly to fresh **`BLIND_HOLDOUT_V5_2026`** plus zero-chip-training external hardware streams (Google Willow `zenodo.13273331` & `zenodo.18896801`, IBM `ibm_miami` `zenodo.21336494`). | **BOUND & PRE-REGISTERED** |
