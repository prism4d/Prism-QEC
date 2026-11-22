// PRISM-QEC SM120 parameter block shared by the kernels (qec_sm120_kernels.cu), the host
// shim (crates/qec-cuda/native/qec_cuda_host.cpp) and, field for field, the Rust mirror
// (crates/qec-cuda/src/params.rs).  Plain C layout: 8-byte fields first, then 4-byte fields.
//
// Every device pointer addresses the single static VRAM arena allocated at start-up, except
// `syn`, which is the device mapping of the page-locked CU_MEMHOSTALLOC_DEVICEMAP syndrome
// ring written directly by the host.
#ifndef QEC_SM120_PARAMS_CUH
#define QEC_SM120_PARAMS_CUH

#define QEC_PARAMS_MAGIC 0x5052534D51454331ULL /* "PRSMQEC1" */
#define QEC_R_MAX 32      /* BOCPD run-length window */
#define QEC_HISTORY 32    /* spike-history ring per stream */
#define QEC_EN_BITS 9     /* Nishimori weight bit-planes */
#define QEC_BUCKETS 1024  /* reliability buckets of the suspect sort */
#define QEC_NO_DETECTOR 0xFFFFFFFFu
#define QEC_MAX_WALKERS 256
#define QEC_BP_THREADS 256

typedef struct QecParams {
    unsigned long long magic;
    // ---- static model: Tanner graph, 2-cells, logical masks ------------------------------
    unsigned long long chk_ptr;    // u32[n_chk + 1]
    unsigned long long chk_var;    // u32[n_edge]
    unsigned long long var_ptr;    // u32[n_var + 1]
    unsigned long long var_edge;   // u32[n_edge]
    unsigned long long var_chk;    // u32[n_edge]
    unsigned long long cell_ptr;   // u32[n_cell + 1]
    unsigned long long cell_var;   // u32[cell nnz]
    unsigned long long vcell_ptr;  // u32[n_var + 1]
    unsigned long long vcell;      // u32[cell nnz]
    unsigned long long obs;        // u64[n_var]
    unsigned long long eps_site;   // i32[n_var]
    unsigned long long tanh_tab;   // u16[tanh_len]
    unsigned long long logistic;   // u32[2 * twist_clip + 1]
    unsigned long long metropolis; // u32[metro_len]
    // ---- static monitor tables -------------------------------------------------------------
    unsigned long long det_of;      // u32[layers * n_checks]
    unsigned long long exposure;    // f64[n_chk]
    unsigned long long pairs;       // u32[n_pairs]: ci | cj << 12 | lag << 24
    unsigned long long pair_nom_te; // i64[n_pairs]
    unsigned long long mech_pair_ptr; // u32[n_var + 1]
    unsigned long long mech_pair;   // u32[mech pair nnz]
    unsigned long long mech_size;   // u8[n_var]
    unsigned long long llr_nom;     // i32[n_var]
    unsigned long long p_nom_q12;   // i32[n_var]
    unsigned long long log_q16;     // i32[256]
    unsigned long long llr_of_p;    // i32[2049]
    // ---- per-stream monitor state ----------------------------------------------------------
    unsigned long long bo_r;        // f64[streams * n_checks * QEC_R_MAX]
    unsigned long long bo_b;        // f64[streams * n_checks * QEC_R_MAX]
    unsigned long long bo_a;        // u32[streams * n_checks * QEC_R_MAX]
    unsigned long long hot;         // u8[streams * n_checks]
    unsigned long long trans;       // u8[streams * n_checks]
    unsigned long long counts;      // u8[streams * n_pairs * 8]
    unsigned long long hist_bits;   // u64[streams * QEC_HISTORY * wc]
    unsigned long long hist_layer;  // u8[streams * QEC_HISTORY]
    unsigned long long hist_pos;    // u32[streams * 2]: head, len
    unsigned long long cur_bits;    // u64[streams * wc]
    unsigned long long pair_rate;   // i32[streams * n_pairs]
    unsigned long long explained;   // i32[streams * n_pairs]
    unsigned long long llr;         // i32[streams * n_var]   live prior LLRs (the L2-pinned DEM)
    unsigned long long en;          // i32[streams * n_var]   live Nishimori weights
    // ---- per-stream fast-path state --------------------------------------------------------
    unsigned long long c2v;         // i16[streams * n_edge]
    unsigned long long bp;          // i32[streams * n_var]
    unsigned long long post;        // i32[streams * n_var]
    unsigned long long snap;        // i32[streams * n_var]
    unsigned long long winv;        // i32[streams * n_var]
    unsigned long long ycell;       // i64[streams * n_cell]
    unsigned long long unsat_flag;  // u8[streams * n_chk]
    // ---- shot stream and outputs -----------------------------------------------------------
    unsigned long long syn;         // u64[ring_shots * ws]  zero-copy DEVICEMAP syndrome ring
    unsigned long long out;         // QecShotOut[n_shots]
    unsigned long long rec_shot;    // u32[n_shots]   slow-path queue: shot of each record
    unsigned long long rec_list;    // u32[n_shots * k_cols]  suspect list of each record
    unsigned long long rec_en;      // i32[n_shots * k_cols]  Nishimori weights of the suspects (as decoded)
    unsigned long long ctrl;        // u32[16] device control block (see QEC_CTRL_*)
    // ---- per-block slow-path scratch (smc_blocks blocks) -----------------------------------
    unsigned long long s_pos;       // i32[smc_blocks * n_var]
    unsigned long long s_mat;       // u64[smc_blocks * n_chk * wm]
    unsigned long long s_pivcol;    // i32[smc_blocks * n_chk]
    unsigned long long s_enrow;     // i32[smc_blocks * n_chk]
    unsigned long long s_planes;    // u64[smc_blocks * QEC_EN_BITS * wr]
    unsigned long long s_epiv;      // u64[smc_blocks * wr]
    unsigned long long s_free;      // u32[smc_blocks * k_cols]
    unsigned long long s_col;       // u64[smc_blocks * k_cols * wr]
    unsigned long long s_rowt;      // u64[smc_blocks * n_chk * wz]
    unsigned long long s_enfree;    // i32[smc_blocks * k_cols]
    unsigned long long s_scol;      // i32[smc_blocks * k_cols]
    unsigned long long s_theta;     // i32[smc_blocks * k_cols]
    unsigned long long s_cycobs;    // u64[smc_blocks * k_cols]
    unsigned long long s_zq;        // u64[smc_blocks * wz]
    unsigned long long w_rows;      // u64[smc_blocks * 2 * walkers * wr]
    unsigned long long w_z;         // u64[smc_blocks * 2 * walkers * wz]
    unsigned long long w_i32;       // i32[smc_blocks * 2 * walkers * 2]  energy, theta_sum
    unsigned long long w_obs;       // u64[smc_blocks * 2 * walkers]
    unsigned long long w_acc;       // i64[smc_blocks * walkers]
    unsigned long long w_anc;       // u32[smc_blocks * walkers]
    unsigned long long w_wt;        // u64[smc_blocks * walkers]
    unsigned long long w_lw;        // f64[smc_blocks * (2 * walkers + 128)]  log-weights + PSIS scratch
    unsigned long long seed_smc;
    double hazard;
    double hot_on;
    double hot_off;
    double tau_cert;
    double khat_max;
    // ---- sizes -----------------------------------------------------------------------------
    unsigned int n_chk, n_var, n_edge, n_cell, n_obs;
    unsigned int n_checks, layers, n_pairs;
    unsigned int streams, steps, n_shots, stream_major;
    unsigned int ws, wc, wr, wm, wz;
    unsigned int k_cols, walkers, smc_blocks;
    // ---- frozen decoder parameters ---------------------------------------------------------
    int t_max, t_clean, alpha_q8, eta_q8, margin;
    int tau_margin, tau_gap, escalate;
    int beta_steps, sweeps, quench_sweeps, resample_q8;
    int tanh_max, twist_clip, metro_len, en_one;
    int neuro_on, n_min, g_threshold, g_threshold_hot, count_cap;
    int f2_engine; // 0: __popcll(a & b) & 1, 1: .b1 and.popc tensor-core path
    int steps_per_launch; // fast-loop time steps per graph launch (0: all of them)
} QecParams;

// Device control block (u32[16]).
#define QEC_CTRL_STEP 0        /* current time step of the fast-path loop            */
#define QEC_CTRL_NREC 1        /* slow-path records produced so far                  */
#define QEC_CTRL_CURSOR 2      /* next slow-path record to process                   */
#define QEC_CTRL_FAST_ITERS 3  /* WHILE iterations executed by the fast-path loop    */
#define QEC_CTRL_SLOW_ITERS 4  /* WHILE iterations executed by the slow-path loop    */
#define QEC_CTRL_GATE 5        /* value the residual gate wrote to the slow handle   */
#define QEC_CTRL_INTEGRITY 6   /* walker states failing the F2 integrity check       */
#define QEC_CTRL_WAKES 7       /* launches in which the gate woke the slow path      */

// Per-shot decoder record.
typedef struct QecShotOut {
    unsigned long long obs;          // final predicted observables
    unsigned long long bp_obs;
    unsigned long long piv_obs;
    unsigned long long q_obs;
    unsigned long long best_obs;
    unsigned long long mode_obs;
    unsigned long long marginal_obs;
    unsigned long long damp_mask;    // bit t-1 set iff iteration t applied a nonzero gauge correction
    double khat;
    double min_abs_df;
    double ess;
    double mode_free_energy;
    int bp_iters;
    int bp_unsat;
    int damped_steps;
    int rank;
    int n_free;
    int piv_energy;
    int best_energy;
    int energy_gap;
    int bp_min_abs;
    int resamples;
    int modified_mechanisms;
    int hot_checks;
    unsigned char bp_converged;
    unsigned char esc_reason;
    unsigned char smc_ran;
    unsigned char solvable;
    unsigned char hle;
    unsigned char pad[3];
} QecShotOut;

#endif
