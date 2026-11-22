// PRISM-QEC SM120 kernels: neuromorphic monitor, velocity-damped min-sum fast path,
// Nishimori twisted SMC slow path, graph control kernels and the exact F2 bit-matrix engine.
//
// Every decision is integer arithmetic mirrored statement for statement from the Rust CPU
// reference (crates/qec-hodge-bp, qec-smc-coset, qec-neuro-dem); the BOCPD filter uses IEEE
// doubles with explicitly non-contracted operations.  No global device synchronization is
// used anywhere: control flow is carried by CUDA graph conditional handles.
#include <cuda_runtime.h>
#include "qec_sm120_params.cuh"

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;

#define P_U32(f) ((const u32*)(P->f))
#define P_I32(f) ((int*)(P->f))
#define P_U64(f) ((u64*)(P->f))
#define P_U8(f) ((unsigned char*)(P->f))
#define QEC_INT_MAX 0x7FFFFFFF
#define QEC_POST_CLIP (1 << 24) /* saturation of the damped beliefs (see fixed.rs) */
#define QEC_DF_CAP 60.0
#define QEC_PAIR_VETO (-1)
#define QEC_PAIR_MASKED (-2)

// ------------------------------------------------------------------------------------------
// shared helpers
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ u64 mix64(u64 z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ u64 rng_u64(u64 seed, u32 shot, u32 walker, u32 counter)
{
    u64 k = mix64(seed ^ ((((u64)shot << 32) | (u64)walker) + 0x9E3779B97F4A7C15ULL));
    return mix64(k ^ ((u64)counter * 0xD6E8FEB86659FD93ULL + 0x2545F4914F6CDD1DULL));
}

__device__ __forceinline__ u32 shot_of(const QecParams* P, u32 stream, u32 step)
{
    return P->stream_major ? stream * P->steps + step : step * P->streams + stream;
}

__device__ __forceinline__ int get_bit(const u64* w, u32 i)
{
    return (int)((w[i >> 6] >> (i & 63)) & 1ULL);
}

__device__ __forceinline__ i64 shr_sym(i64 x, int s)
{
    return x >= 0 ? (x >> s) : -((-x) >> s);
}

__device__ __forceinline__ int energy_from_llr(int llr, int en_one)
{
    int shift = 64 / en_one;
    int x = llr + shift / 2;
    int q = x >= 0 ? x / shift : -((-x + shift - 1) / shift);
    return q < 0 ? 0 : (q > 511 ? 511 : q);
}

// ------------------------------------------------------------------------------------------
// neuromorphic monitor: BOCPD per check, transfer-entropy counters per coupler pair, surgery
// ------------------------------------------------------------------------------------------
__device__ __noinline__ double bocpd_update(double* r, double* b, u32* a, int spike, double e, double h)
{
    double total = 0.0;
    for (int k = 0; k < QEC_R_MAX; ++k) {
        double q = (1.0 + b[k]) / (1.0 + b[k] + e);
        double acc = 1.0, sq = q;
        u32 ex = 1u + a[k];
        while (ex > 0u) {
            if (ex & 1u) acc = __dmul_rn(acc, sq);
            sq = __dmul_rn(sq, sq);
            ex >>= 1;
        }
        double pred = spike ? (1.0 - acc) : acc;
        r[k] = __dmul_rn(r[k], pred);
        total = __dadd_rn(total, r[k]);
    }
    double grow = 1.0 - h;
    double w_last = r[QEC_R_MAX - 1], w_prev = r[QEC_R_MAX - 2];
    u32 a_tail = a[QEC_R_MAX - 1];
    double b_tail = b[QEC_R_MAX - 1];
    for (int k = QEC_R_MAX - 1; k >= 1; --k) {
        r[k] = __dmul_rn(r[k - 1], grow);
        a[k] = a[k - 1] + (u32)spike;
        b[k] = b[k - 1] + e;
    }
    r[QEC_R_MAX - 1] = __dadd_rn(r[QEC_R_MAX - 1], __dmul_rn(w_last, grow));
    if (w_last >= w_prev) {
        a[QEC_R_MAX - 1] = a_tail + (u32)spike;
        b[QEC_R_MAX - 1] = b_tail + e;
    }
    r[0] = __dmul_rn(h, total);
    a[0] = 0u;
    b[0] = 0.0;
    double norm = 0.0;
    for (int k = 0; k < QEC_R_MAX; ++k) norm = __dadd_rn(norm, r[k]);
    double mean = 0.0;
    for (int k = 0; k < QEC_R_MAX; ++k) {
        r[k] = r[k] / norm;
        mean = __dadd_rn(mean, __dmul_rn(r[k], (1.0 + (double)a[k]) / (1.0 + b[k])));
    }
    return mean;
}

__device__ __forceinline__ int bocpd_hot_run(const double* r, const double* b, const u32* a, double thr)
{
    int best = 1;
    double best_v = 0.0;
    for (int k = 0; k < QEC_R_MAX; ++k) {
        if ((1.0 + (double)a[k]) / (1.0 + b[k]) >= thr && r[k] > best_v) {
            best_v = r[k];
            best = k;
        }
    }
    return best;
}

__device__ __forceinline__ void pair_bump(unsigned char* cnt, int bin, u32 cap)
{
    u32 total = 0;
    for (int k = 0; k < 8; ++k) total += cnt[k];
    if (total >= cap) {
        for (int k = 0; k < 8; ++k) cnt[k] >>= 1;
    }
    cnt[bin] += 1;
}

__device__ __forceinline__ int pair_sampled(const QecParams* P, u32 ci, u32 cj, u32 lag, u32 tl)
{
    const u32* det_of = P_U32(det_of);
    u32 nc = P->n_checks;
    if (tl < 1u) return 0;
    u32 src = lag == 0u ? tl : tl - 1u;
    return det_of[tl * nc + cj] != QEC_NO_DETECTOR && det_of[(tl - 1u) * nc + cj] != QEC_NO_DETECTOR
        && det_of[src * nc + ci] != QEC_NO_DETECTOR;
}

__device__ __noinline__ i64 pair_g_statistic(const unsigned char* cnt, const int* logt)
{
    i64 g = 0;
    for (int c = 0; c < 2; ++c) {
        int n00 = cnt[(c << 1)], n01 = cnt[(c << 1) | 1], n10 = cnt[4 | (c << 1)], n11 = cnt[4 | (c << 1) | 1];
        int n_c = n00 + n01 + n10 + n11;
        if (n00) g += (i64)n00 * ((i64)logt[n00] + logt[n_c] - logt[n00 + n01] - logt[n00 + n10]);
        if (n01) g += (i64)n01 * ((i64)logt[n01] + logt[n_c] - logt[n00 + n01] - logt[n01 + n11]);
        if (n10) g += (i64)n10 * ((i64)logt[n10] + logt[n_c] - logt[n10 + n11] - logt[n00 + n10]);
        if (n11) g += (i64)n11 * ((i64)logt[n11] + logt[n_c] - logt[n10 + n11] - logt[n01 + n11]);
    }
    return 2 * g;
}

__device__ __forceinline__ u64 isqrt_u64(u64 x)
{
    if (x < 2ULL) return x;
    u64 r = (u64)sqrt((double)x);
    while (r * r > x) --r;
    while ((r + 1ULL) * (r + 1ULL) <= x) ++r;
    return r;
}

__device__ __noinline__ int pair_rate_q12(const unsigned char* cnt)
{
    i64 n = 0;
    for (int k = 0; k < 8; ++k) n += cnt[k];
    if (n == 0) return 0;
    i64 n11 = (i64)cnt[5] + cnt[7];
    i64 nx = (i64)cnt[1] + cnt[3] + n11;
    i64 ny = (i64)cnt[4] + cnt[6] + n11;
    i64 num = 4 * (n * n11 - nx * ny);
    i64 den = n * (n - 2 * nx - 2 * ny + 4 * n11);
    if (num <= 0) return 0;
    if (den <= 0 || num >= den) return 2048;
    u64 t_q24 = (((u64)(den - num)) << 24) / (u64)den;
    return (int)((4096 - (i64)isqrt_u64(t_q24)) / 2);
}

__device__ __forceinline__ int pair_masked(const unsigned char* cnt)
{
    u32 n = 0;
    for (int k = 0; k < 8; ++k) n += cnt[k];
    u32 n11 = (u32)cnt[5] + cnt[7];
    u32 nx = (u32)cnt[1] + cnt[3] + n11;
    u32 ny = (u32)cnt[4] + cnt[6] + n11;
    int fair_x = 20u * nx >= 7u * n && 20u * nx <= 13u * n;
    int fair_y = 20u * ny >= 7u * n && 20u * ny <= 13u * n;
    return n > 0u && (fair_x || fair_y);
}

extern "C" __global__ void qec_neuro_update(const QecParams* P)
{
    __shared__ int sh_cnt[2];
    const u32 tid = threadIdx.x, nt = blockDim.x, s = blockIdx.x;
    if (!P->neuro_on) return;
    u32* ctrl = (u32*)P->ctrl;
    const u32 shot = shot_of(P, s, ctrl[QEC_CTRL_STEP]);
    if (shot >= P->n_shots) return;
    const u32 nc = P->n_checks, wc = P->wc, np = P->n_pairs, nv = P->n_var;
    const u64* syn = P_U64(syn) + (u64)shot * P->ws;
    const u32* det_of = P_U32(det_of);
    const double* exposure = (const double*)P->exposure;
    double* bo_r = (double*)P->bo_r + (u64)s * nc * QEC_R_MAX;
    double* bo_b = (double*)P->bo_b + (u64)s * nc * QEC_R_MAX;
    u32* bo_a = (u32*)P->bo_a + (u64)s * nc * QEC_R_MAX;
    unsigned char* hot = P_U8(hot) + (u64)s * nc;
    unsigned char* trans = P_U8(trans) + (u64)s * nc;
    unsigned char* counts = P_U8(counts) + (u64)s * np * 8;
    u64* hist_bits = P_U64(hist_bits) + (u64)s * QEC_HISTORY * wc;
    unsigned char* hist_layer = P_U8(hist_layer) + (u64)s * QEC_HISTORY;
    u32* hist_pos = (u32*)P->hist_pos + (u64)s * 2;
    u64* cur = P_U64(cur_bits) + (u64)s * wc;
    const u32* pairs = P_U32(pairs);
    for (u32 l = 0; l < P->layers; ++l) {
        for (u32 w = tid; w < wc; w += nt) cur[w] = 0ULL;
        __syncthreads();
        // phase 1: changepoint filters
        for (u32 c = tid; c < nc; c += nt) {
            trans[c] = 0;
            u32 d = det_of[l * nc + c];
            if (d == QEC_NO_DETECTOR) continue;
            int x = get_bit(syn, d);
            if (x) atomicOr(&cur[c >> 6], 1ULL << (c & 63));
            double m = bocpd_update(bo_r + (u64)c * QEC_R_MAX, bo_b + (u64)c * QEC_R_MAX, bo_a + (u64)c * QEC_R_MAX, x,
                                    exposure[d], P->hazard);
            int flip = (hot[c] == 0 && m >= P->hot_on) || (hot[c] == 1 && m <= P->hot_off);
            if (flip) {
                hot[c] ^= 1;
                int run = 1;
                if (hot[c] == 1) {
                    run = bocpd_hot_run(bo_r + (u64)c * QEC_R_MAX, bo_b + (u64)c * QEC_R_MAX,
                                        bo_a + (u64)c * QEC_R_MAX, P->hot_on);
                    run = run < 1 ? 1 : (run > QEC_HISTORY - 2 ? QEC_HISTORY - 2 : run);
                }
                trans[c] = (unsigned char)run;
            }
        }
        __syncthreads();
        // phase 2: coupler pair counters (regime replay from the history ring on a transition)
        const u32 head = hist_pos[0], hlen = hist_pos[1];
        for (u32 i = tid; i < np; i += nt) {
            const u32 pc = pairs[i];
            const u32 ci = pc & 0xFFFu, cj = (pc >> 12) & 0xFFFu, lag = pc >> 24;
            unsigned char* cnt = counts + (u64)i * 8;
            u32 r = trans[ci] > trans[cj] ? trans[ci] : trans[cj];
            if (r > 0u) {
                for (int k = 0; k < 8; ++k) cnt[k] = 0;
                for (u32 k = r - 1u; k >= 1u; --k) {
                    u32 slot_t = (head + QEC_HISTORY - (k - 1u)) % QEC_HISTORY;
                    if (k + 1u > hlen || !pair_sampled(P, ci, cj, lag, hist_layer[slot_t])) continue;
                    u32 slot_p = (head + QEC_HISTORY - k) % QEC_HISTORY;
                    const u64* tgt = hist_bits + (u64)slot_t * wc;
                    const u64* prev = hist_bits + (u64)slot_p * wc;
                    int x = lag == 0u ? get_bit(tgt, ci) : get_bit(prev, ci);
                    pair_bump(cnt, (get_bit(tgt, cj) << 2) | (get_bit(prev, cj) << 1) | x, (u32)P->count_cap);
                }
            }
            if (pair_sampled(P, ci, cj, lag, l)) {
                const u64* prev = hist_bits + (u64)head * wc;
                int x = lag == 0u ? get_bit(cur, ci) : get_bit(prev, ci);
                pair_bump(cnt, (get_bit(cur, cj) << 2) | (get_bit(prev, cj) << 1) | x, (u32)P->count_cap);
            }
        }
        __syncthreads();
        // phase 3: push the layer into the history ring
        if (tid == 0u) {
            u32 h = (head + 1u) % QEC_HISTORY;
            for (u32 w = 0; w < wc; ++w) hist_bits[(u64)h * wc + w] = cur[w];
            hist_layer[h] = (unsigned char)l;
            hist_pos[0] = h;
            hist_pos[1] = hlen + 1u > QEC_HISTORY ? QEC_HISTORY : hlen + 1u;
        }
        __syncthreads();
    }
    // ---- surgery
    int* pair_rate = P_I32(pair_rate) + (u64)s * np;
    int* explained = P_I32(explained) + (u64)s * np;
    const i64* nom_te = (const i64*)P->pair_nom_te;
    const int* logt = P_I32(log_q16);
    if (tid == 0u) {
        sh_cnt[0] = 0;
        sh_cnt[1] = 0;
    }
    for (u32 i = tid; i < np; i += nt) {
        const u32 pc = pairs[i];
        const u32 ci = pc & 0xFFFu, cj = (pc >> 12) & 0xFFFu;
        const unsigned char* cnt = counts + (u64)i * 8;
        i64 n = 0;
        for (int k = 0; k < 8; ++k) n += cnt[k];
        i64 excess = pair_g_statistic(cnt, logt) - 2 * n * nom_te[i];
        int both_hot = hot[ci] == 1 && hot[cj] == 1;
        i64 thr = (i64)(both_hot ? P->g_threshold_hot : P->g_threshold) * 65536;
        int active = n >= (i64)P->n_min && excess >= thr;
        pair_rate[i] = active ? pair_rate_q12(cnt) : (pair_masked(cnt) ? QEC_PAIR_MASKED : QEC_PAIR_VETO);
        explained[i] = 0;
    }
    __syncthreads();
    const u32* mp_ptr = P_U32(mech_pair_ptr);
    const u32* mp = P_U32(mech_pair);
    const unsigned char* msize = P_U8(mech_size);
    const int* llr_nom = P_I32(llr_nom);
    const int* p_nom = P_I32(p_nom_q12);
    const int* llr_of_p = P_I32(llr_of_p);
    int* llr = P_I32(llr) + (u64)s * nv;
    int* en = P_I32(en) + (u64)s * nv;
    for (int phase = 0; phase < 2; ++phase) {
        int modified = 0;
        for (u32 e = tid; e < nv; e += nt) {
            const u32 lo = mp_ptr[e], hi = mp_ptr[e + 1];
            const int big = msize[e] >= 3;
            if (lo == hi || big != (phase == 0)) continue;
            int p = QEC_INT_MAX, veto = 0;
            for (u32 q = lo; q < hi; ++q) {
                int r = pair_rate[mp[q]];
                if (r == QEC_PAIR_VETO) {
                    veto = 1;
                } else if (r != QEC_PAIR_MASKED) {
                    int cand = big ? r : r - explained[mp[q]];
                    p = cand < p ? cand : p;
                }
            }
            int value = llr_nom[e];
            if (!veto && p != QEC_INT_MAX && p > p_nom[e]) {
                int cand = llr_of_p[p > 2048 ? 2048 : p];
                value = cand < value ? cand : value;
                if (big) {
                    for (u32 q = lo; q < hi; ++q) atomicMax(&explained[mp[q]], p);
                }
            }
            llr[e] = value;
            en[e] = energy_from_llr(value, P->en_one);
            modified += value != llr_nom[e];
        }
        atomicAdd(&sh_cnt[0], modified);
        __syncthreads();
    }
    int hot_local = 0;
    for (u32 c = tid; c < nc; c += nt) hot_local += hot[c];
    atomicAdd(&sh_cnt[1], hot_local);
    __syncthreads();
    if (tid == 0u) {
        QecShotOut* out = (QecShotOut*)P->out + shot;
        out->modified_mechanisms = sh_cnt[0];
        out->hot_checks = sh_cnt[1];
    }
}

// ------------------------------------------------------------------------------------------
// fast path: velocity-damped normalized min-sum (flooding; identical to the lazy CPU schedule)
// ------------------------------------------------------------------------------------------
__device__ __noinline__ int bp_count_unsat(const QecParams* P, const u64* syn, const int* post, unsigned char* uflag,
                                           int* sh_unsat)
{
    const u32 tid = threadIdx.x, nt = blockDim.x;
    const u32* chk_ptr = P_U32(chk_ptr);
    const u32* chk_var = P_U32(chk_var);
    if (tid == 0u) *sh_unsat = 0;
    __syncthreads();
    int local = 0;
    for (u32 c = tid; c < P->n_chk; c += nt) {
        int par = get_bit(syn, c);
        for (u32 e = chk_ptr[c]; e < chk_ptr[c + 1]; ++e) par ^= post[chk_var[e]] < 0;
        uflag[c] = (unsigned char)par;
        local += par;
    }
    atomicAdd(sh_unsat, local);
    __syncthreads();
    return *sh_unsat;
}

extern "C" __global__ void qec_bp_decode(const QecParams* P)
{
    __shared__ int sh_i[6];
    __shared__ u64 sh_obs;
    __shared__ u32 sh_hist[QEC_BUCKETS + 1];
    const u32 tid = threadIdx.x, nt = blockDim.x, s = blockIdx.x;
    u32* ctrl = (u32*)P->ctrl;
    const u32 shot = shot_of(P, s, ctrl[QEC_CTRL_STEP]);
    if (shot >= P->n_shots) return;
    const u32 n_chk = P->n_chk, n_var = P->n_var, n_edge = P->n_edge, n_cell = P->n_cell;
    const u64* syn = P_U64(syn) + (u64)shot * P->ws;
    const u32* chk_ptr = P_U32(chk_ptr);
    const u32* chk_var = P_U32(chk_var);
    const u32* var_ptr = P_U32(var_ptr);
    const u32* var_edge = P_U32(var_edge);
    const u32* var_chk = P_U32(var_chk);
    const u32* cell_ptr = P_U32(cell_ptr);
    const u32* cell_var = P_U32(cell_var);
    const u32* vcell_ptr = P_U32(vcell_ptr);
    const u32* vcell = P_U32(vcell);
    const u64* obs = P_U64(obs);
    const int* eps_site = P_I32(eps_site);
    const unsigned short* tanh_tab = (const unsigned short*)P->tanh_tab;
    const int* llr = P_I32(llr) + (u64)s * n_var;
    const int* en = P_I32(en) + (u64)s * n_var;
    short* c2v = (short*)P->c2v + (u64)s * n_edge;
    int* bp = P_I32(bp) + (u64)s * n_var;
    int* post = P_I32(post) + (u64)s * n_var;
    int* snap = P_I32(snap) + (u64)s * n_var;
    int* winv = P_I32(winv) + (u64)s * n_var;
    i64* ycell = (i64*)P->ycell + (u64)s * n_cell;
    unsigned char* uflag = P_U8(unsat_flag) + (u64)s * n_chk;
    const int msg_clip = 2047;

    for (u32 e = tid; e < n_edge; e += nt) c2v[e] = 0;
    for (u32 v = tid; v < n_var; v += nt) {
        bp[v] = llr[v];
        post[v] = llr[v];
    }
    __syncthreads();
    int unsat = bp_count_unsat(P, syn, post, uflag, &sh_i[0]);
    int converged = unsat == 0, iters = 0, damped_steps = 0, snapped = 0;
    u64 damp_mask = 0ULL;
    if (!converged) {
        for (int it = 1; it <= P->t_max; ++it) {
            iters = it;
            // ---- check phase
            for (u32 c = tid; c < n_chk; c += nt) {
                int sign = get_bit(syn, c);
                int min1 = msg_clip, min2 = msg_clip;
                u32 arg = 0xFFFFFFFFu;
                for (u32 e = chk_ptr[c]; e < chk_ptr[c + 1]; ++e) {
                    int m = post[chk_var[e]] - (int)c2v[e];
                    m = m < -msg_clip ? -msg_clip : (m > msg_clip ? msg_clip : m);
                    sign ^= m < 0;
                    int a = m < 0 ? -m : m;
                    if (a < min1) {
                        min2 = min1;
                        min1 = a;
                        arg = e;
                    } else if (a < min2) {
                        min2 = a;
                    }
                }
                for (u32 e = chk_ptr[c]; e < chk_ptr[c + 1]; ++e) {
                    int m = post[chk_var[e]] - (int)c2v[e];
                    m = m < -msg_clip ? -msg_clip : (m > msg_clip ? msg_clip : m);
                    int mag = ((e == arg ? min2 : min1) * P->alpha_q8) >> 8;
                    int neg = sign ^ (m < 0);
                    c2v[e] = (short)(neg ? -mag : mag);
                }
            }
            __syncthreads();
            // ---- variable phase: plain min-sum proposal
            for (u32 v = tid; v < n_var; v += nt) {
                int acc = llr[v];
                for (u32 q = var_ptr[v]; q < var_ptr[v + 1]; ++q) acc += (int)c2v[var_edge[q]];
                bp[v] = acc;
            }
            if (tid == 0u) sh_i[1] = 0;
            __syncthreads();
            const int damp = P->eta_q8 != 0 && it > P->t_clean && n_cell > 0u;
            if (damp && !snapped) {
                for (u32 v = tid; v < n_var; v += nt) snap[v] = post[v];
                snapped = 1;
            }
            if (damp) {
                for (u32 v = tid; v < n_var; v += nt) {
                    int a = bp[v] < 0 ? -bp[v] : bp[v];
                    a = a > P->tanh_max ? P->tanh_max : a;
                    winv[v] = (1 << 24) / ((int)tanh_tab[a] + eps_site[v]);
                }
                __syncthreads();
                for (u32 f = tid; f < n_cell; f += nt) {
                    i64 curl = 0, diag = 0;
                    int active = 0;
                    for (u32 q = cell_ptr[f]; q < cell_ptr[f + 1]; ++q) {
                        const u32 v = cell_var[q];
                        curl += (i64)(bp[v] - post[v]);
                        diag += (i64)winv[v];
                        int a = bp[v] < 0 ? -bp[v] : bp[v];
                        int hotv = a < P->margin;
                        for (u32 k = var_ptr[v]; k < var_ptr[v + 1] && !hotv; ++k) hotv = uflag[var_chk[k]];
                        active |= hotv;
                    }
                    ycell[f] = active ? (curl << 16) / diag : 0;
                }
                __syncthreads();
                int any = 0;
                for (u32 v = tid; v < n_var; v += nt) {
                    i64 acc = 0;
                    for (u32 q = vcell_ptr[v]; q < vcell_ptr[v + 1]; ++q) acc += ycell[vcell[q]];
                    int corr = (int)shr_sym((i64)P->eta_q8 * (i64)winv[v] * acc, 24);
                    any |= corr != 0;
                    int nv = bp[v] - corr;
                    post[v] = nv < -QEC_POST_CLIP ? -QEC_POST_CLIP : (nv > QEC_POST_CLIP ? QEC_POST_CLIP : nv);
                }
                if (any) atomicOr(&sh_i[1], 1);
            } else {
                for (u32 v = tid; v < n_var; v += nt) post[v] = bp[v];
            }
            __syncthreads();
            damped_steps += sh_i[1];
            damp_mask |= (u64)(sh_i[1] != 0) << ((it - 1) & 63);
            unsat = bp_count_unsat(P, syn, post, uflag, &sh_i[0]);
            if (unsat == 0) {
                converged = 1;
                break;
            }
        }
    }
    if (!snapped) {
        for (u32 v = tid; v < n_var; v += nt) snap[v] = post[v];
    }
    if (tid == 0u) {
        sh_obs = 0ULL;
        sh_i[2] = 0;
        sh_i[3] = QEC_INT_MAX;
    }
    __syncthreads();
    // ---- hard-decision observables and escalation triggers (b), (c)
    u64 obs_local = 0ULL;
    int flags = 0;
    for (u32 v = tid; v < n_var; v += nt) {
        if (post[v] >= 0) continue;
        obs_local ^= obs[v];
        if (!(converged && P->escalate)) continue;
        if (-post[v] < P->tau_margin) flags |= 2;
        for (u32 q = vcell_ptr[v]; q < vcell_ptr[v + 1]; ++q) {
            const u32 f = vcell[q];
            int delta = 0;
            for (u32 k = cell_ptr[f]; k < cell_ptr[f + 1]; ++k) {
                const u32 u = cell_var[k];
                int a = post[u] < 0 ? -post[u] : post[u];
                if (a < P->tau_margin) flags |= 2;
                delta += post[u] < 0 ? -en[u] : en[u];
            }
            if (delta < P->tau_gap) flags |= 4;
        }
    }
    atomicXor(&sh_obs, obs_local);
    if (flags) atomicOr(&sh_i[2], flags);
    __syncthreads();
    int esc = 0;
    if (converged && P->escalate) esc = sh_i[2] | (iters > P->t_clean ? 1 : 0);
    const int slow = !converged || esc != 0;
    const int* order = converged ? post : snap;
    if (slow) {
        // reliability histogram and the smallest belief magnitude
        for (u32 b = tid; b <= QEC_BUCKETS; b += nt) sh_hist[b] = 0u;
        __syncthreads();
        int amin = QEC_INT_MAX;
        for (u32 v = tid; v < n_var; v += nt) {
            int o = order[v];
            int a = o < 0 ? -o : o;
            amin = a < amin ? a : amin;
            int b = (o >> 3) + 512;
            b = b < 0 ? 0 : (b > QEC_BUCKETS - 1 ? QEC_BUCKETS - 1 : b);
            atomicAdd(&sh_hist[b + 1], 1u);
        }
        atomicMin(&sh_i[3], amin);
        __syncthreads();
    }
    if (tid == 0u) {
        QecShotOut* out = (QecShotOut*)P->out + shot;
        out->obs = sh_obs;
        out->bp_obs = sh_obs;
        out->piv_obs = 0ULL;
        out->q_obs = 0ULL;
        out->best_obs = 0ULL;
        out->mode_obs = 0ULL;
        out->marginal_obs = 0ULL;
        out->damp_mask = damp_mask;
        out->khat = 0.0;
        out->min_abs_df = 1.0 / 0.0;
        out->ess = 0.0;
        out->mode_free_energy = 0.0;
        out->bp_iters = iters;
        out->bp_unsat = unsat;
        out->damped_steps = damped_steps;
        out->rank = 0;
        out->n_free = 0;
        out->piv_energy = 0;
        out->best_energy = 0;
        out->energy_gap = 0;
        out->bp_min_abs = slow ? sh_i[3] : 0;
        out->resamples = 0;
        if (!P->neuro_on) {
            out->modified_mechanisms = 0;
            out->hot_checks = 0;
        }
        out->bp_converged = (unsigned char)converged;
        out->esc_reason = (unsigned char)esc;
        out->smc_ran = (unsigned char)slow;
        out->solvable = 1;
        out->hle = 0;
        if (slow) {
            // stable counting sort of the K least reliable faults, ties by index
            const u32 k_cols = P->k_cols;
            const u32 slot = atomicAdd(&ctrl[QEC_CTRL_NREC], 1u);
            u32* rec_list = (u32*)P->rec_list + (u64)slot * k_cols;
            int* rec_en = P_I32(rec_en) + (u64)slot * k_cols;
            ((u32*)P->rec_shot)[slot] = shot;
            for (u32 b = 0; b < QEC_BUCKETS; ++b) sh_hist[b + 1] += sh_hist[b];
            for (u32 v = 0; v < n_var; ++v) {
                int b = (order[v] >> 3) + 512;
                b = b < 0 ? 0 : (b > QEC_BUCKETS - 1 ? QEC_BUCKETS - 1 : b);
                const u32 at = sh_hist[b]++;
                if (at < k_cols) {
                    rec_list[at] = v;
                    rec_en[at] = en[v];
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// graph control kernels (one thread each)
// ------------------------------------------------------------------------------------------
extern "C" __global__ void qec_fast_advance(const QecParams* P, cudaGraphConditionalHandle h_fast)
{
    u32* ctrl = (u32*)P->ctrl;
    const u32 step = ctrl[QEC_CTRL_STEP] + 1u;
    ctrl[QEC_CTRL_STEP] = step;
    ctrl[QEC_CTRL_FAST_ITERS] += 1u;
    const int launch_done = P->steps_per_launch > 0 && (step % (u32)P->steps_per_launch) == 0u;
    cudaGraphSetConditional(h_fast, (step < P->steps && !launch_done) ? 1u : 0u);
}

extern "C" __global__ void qec_slow_gate(const QecParams* P, cudaGraphConditionalHandle h_slow)
{
    // device-side residual check: wake the slow path iff some shot of this launch left a
    // frustrated residual (d1 e_hat (+) s != 0) or was escalated
    u32* ctrl = (u32*)P->ctrl;
    const u32 wake = ctrl[QEC_CTRL_CURSOR] < ctrl[QEC_CTRL_NREC] ? 1u : 0u;
    ctrl[QEC_CTRL_GATE] = wake;
    ctrl[QEC_CTRL_WAKES] += wake;
    cudaGraphSetConditional(h_slow, wake);
}

extern "C" __global__ void qec_slow_advance(const QecParams* P, cudaGraphConditionalHandle h_slow)
{
    u32* ctrl = (u32*)P->ctrl;
    const u32 nrec = ctrl[QEC_CTRL_NREC];
    u32 cursor = ctrl[QEC_CTRL_CURSOR] + P->smc_blocks;
    const u32 more = cursor < nrec ? 1u : 0u;
    if (!more) cursor = nrec;
    ctrl[QEC_CTRL_CURSOR] = cursor;
    ctrl[QEC_CTRL_SLOW_ITERS] += 1u;
    cudaGraphSetConditional(h_slow, more);
}

// ------------------------------------------------------------------------------------------
// exact F2 bit-matrix engine: C[i][j] = popcount(A_i & B_j) & 1
// ------------------------------------------------------------------------------------------
// Tensor-core path: one warp computes a 16 x 8 tile over a 256-bit slice of the inner
// dimension with mma.m16n8k256 .b1 and.popc; the s32 accumulators chain across slices and the
// F2 product is the low bit.  Lane layout (g = lane >> 2, t = lane & 3), measured on the
// device register by register: lane holds the 64-bit slice t of A rows g and g + 8 and of B
// column g.  A registers interleave the two rows: a0 = row g bits [0, 32), a1 = row g + 8 bits
// [0, 32), a2 = row g bits [32, 64), a3 = row g + 8 bits [32, 64).  B registers are the low and
// high half of the slice.  D entries are (g, 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1).
__device__ __forceinline__ void b1_and_popc(u64 a_lo, u64 a_hi, u64 b, int& d0, int& d1, int& d2, int& d3)
{
    u32 a0 = (u32)a_lo, a1 = (u32)a_hi, a2 = (u32)(a_lo >> 32), a3 = (u32)(a_hi >> 32);
    u32 b0 = (u32)b, b1 = (u32)(b >> 32);
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// C is bit-packed: row i occupies wn = ceil(n / 64) words.
extern "C" __global__ void qec_f2_matmul_popc(const u64* A, const u64* B, u64* C, int m, int n, int kw)
{
    const int wn = (n + 63) / 64;
    const u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (u64)m * wn) return;
    const int i = (int)(idx / wn), w = (int)(idx % wn);
    const u64* a = A + (u64)i * kw;
    u64 word = 0ULL;
    for (int bit = 0; bit < 64 && w * 64 + bit < n; ++bit) {
        const u64* b = B + (u64)(w * 64 + bit) * kw;
        int par = 0;
        for (int q = 0; q < kw; ++q) par ^= __popcll(a[q] & b[q]);
        word |= (u64)(par & 1) << bit;
    }
    C[idx] = word;
}

// Requires m % 16 == 0, n % 8 == 0 and kw % 4 == 0; one warp per 16 x 8 tile.
extern "C" __global__ void qec_f2_matmul_b1(const u64* A, const u64* B, u64* C, int m, int n, int kw)
{
    const int wn = (n + 63) / 64;
    const int tiles_j = n / 8;
    const u64 tile = (u64)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (tile >= (u64)(m / 16) * tiles_j) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int i0 = (int)(tile / tiles_j) * 16, j0 = (int)(tile % tiles_j) * 8;
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int c = 0; c < kw / 4; ++c) {
        b1_and_popc(A[(u64)(i0 + g) * kw + 4 * c + t], A[(u64)(i0 + g + 8) * kw + 4 * c + t],
                    B[(u64)(j0 + g) * kw + 4 * c + t], d0, d1, d2, d3);
    }
    const int ja = j0 + 2 * t, jb = j0 + 2 * t + 1;
    const u64 lo = ((u64)(d0 & 1) << (ja & 63)), lo2 = ((u64)(d1 & 1) << (jb & 63));
    const u64 hi = ((u64)(d2 & 1) << (ja & 63)), hi2 = ((u64)(d3 & 1) << (jb & 63));
    if (lo) atomicOr(&C[(u64)(i0 + g) * wn + (ja >> 6)], lo);
    if (lo2) atomicOr(&C[(u64)(i0 + g) * wn + (jb >> 6)], lo2);
    if (hi) atomicOr(&C[(u64)(i0 + g + 8) * wn + (ja >> 6)], hi);
    if (hi2) atomicOr(&C[(u64)(i0 + g + 8) * wn + (jb >> 6)], hi2);
}

// ------------------------------------------------------------------------------------------
// slow path: spanning-forest pivot, quench, Nishimori twisted SMC, certificate
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ int plane_dot(const u64* planes, u32 wr, const u64* a, const u64* b)
{
    u32 acc = 0u;
    for (int bit = 0; bit < QEC_EN_BITS; ++bit) {
        const u64* pl = planes + (u64)bit * wr;
        u32 c = 0u;
        for (u32 w = 0; w < wr; ++w) c += (u32)__popcll(a[w] & b[w] & pl[w]);
        acc += c << bit;
    }
    return (int)acc;
}

struct SmcView {
    const u64* col;
    const u64* planes;
    const int* enfree;
    const int* scol;
    u32 wr;
};

__device__ __forceinline__ int delta_energy(const SmcView& V, u32 t, const u64* rows, int zt)
{
    return (zt ? -V.enfree[t] : V.enfree[t]) + V.scol[t] - 2 * plane_dot(V.planes, V.wr, V.col + (u64)t * V.wr, rows);
}

// Zhang-Stephens generalized-Pareto shape of the weight tail (PSIS k-hat); `ws` is scratch of
// 2 * W + 128 doubles whose first W entries hold the log-weights (destroyed).
__device__ __noinline__ double psis_khat(double* ws, int s)
{
    double a = (double)s / 5.0, b3 = 3.0 * sqrt((double)s);
    const int m = (int)ceil(a < b3 ? a : b3);
    if (m < 5 || m >= s) return 1.0 / 0.0;
    for (int i = 1; i < s; ++i) {
        double key = ws[i];
        int j = i - 1;
        while (j >= 0 && ws[j] > key) {
            ws[j + 1] = ws[j];
            --j;
        }
        ws[j + 1] = key;
    }
    double* x = ws + s;
    double* bb = ws + 2 * s;
    double* len = ws + 2 * s + 64;
    const double top = ws[s - 1];
    const double cutoff = exp(ws[s - m - 1] - top);
    for (int i = 0; i < m; ++i) x[i] = exp(ws[s - m + i] - top) - cutoff;
    if (x[m - 1] <= 1e-300 || x[m - 1] - x[0] <= 1e-12 * x[m - 1]) return 0.0;
    const double floor_v = x[m - 1] * 1e-12;
    for (int i = 0; i < m; ++i) x[i] = x[i] > floor_v ? x[i] : floor_v;
    const double nf = (double)m;
    const int mm = 30 + (int)sqrt(nf);
    int qi = (int)floor(nf / 4.0 + 0.5);
    qi = qi < 1 ? 1 : qi;
    const double quartile = x[qi - 1], x_max = x[m - 1];
    for (int j = 0; j < mm; ++j) {
        double bj = (1.0 - sqrt((double)mm / ((double)j + 0.5))) / (3.0 * quartile) + 1.0 / x_max;
        double kj = 0.0;
        for (int i = 0; i < m; ++i) kj += log1p(-bj * x[i]);
        kj /= nf;
        bb[j] = bj;
        len[j] = nf * (log(-bj / kj) - kj - 1.0);
    }
    const double eps = 10.0 * 2.220446049250313e-16;
    double total = 0.0;
    for (int j = 0; j < mm; ++j) {
        double sj = 0.0;
        for (int l = 0; l < mm; ++l) sj += exp(len[l] - len[j]);
        double wj = 1.0 / sj;
        if (wj >= eps) total += wj;
    }
    double b_post = 0.0;
    for (int j = 0; j < mm; ++j) {
        double sj = 0.0;
        for (int l = 0; l < mm; ++l) sj += exp(len[l] - len[j]);
        double wj = 1.0 / sj;
        if (wj >= eps) b_post += bb[j] * wj / total;
    }
    double k_post = 0.0;
    for (int i = 0; i < m; ++i) k_post += log1p(-b_post * x[i]);
    k_post /= nf;
    double k = (nf * k_post + 10.0 * 0.5) / (nf + 10.0);
    return isfinite(k) ? k : 1.0 / 0.0;
}

extern "C" __global__ void qec_smc_decode(const QecParams* P)
{
    __shared__ int sh_piv[2];
    __shared__ int sh_i[8];
    __shared__ u64 sh_obs;
    const u32 tid = threadIdx.x, nt = blockDim.x, blk = blockIdx.x;
    u32* ctrl = (u32*)P->ctrl;
    const u32 rec = ctrl[QEC_CTRL_CURSOR] + blk;
    if (rec >= ctrl[QEC_CTRL_NREC]) return;
    const u32 shot = ((const u32*)P->rec_shot)[rec];
    const u32 n_chk = P->n_chk, n_var = P->n_var, k_cols = P->k_cols, n_obs = P->n_obs;
    const u32 wr = P->wr, wm = P->wm, wz = P->wz, nw = P->walkers;
    const u32* list = (const u32*)P->rec_list + (u64)rec * k_cols;
    const int* rec_en = P_I32(rec_en) + (u64)rec * k_cols;
    const u64* syn = P_U64(syn) + (u64)shot * P->ws;
    const u32* chk_ptr = P_U32(chk_ptr);
    const u32* chk_var = P_U32(chk_var);
    const u64* obs = P_U64(obs);
    int* pos = P_I32(s_pos) + (u64)blk * n_var;
    u64* mat = P_U64(s_mat) + (u64)blk * n_chk * wm;
    int* pivcol = P_I32(s_pivcol) + (u64)blk * n_chk;
    int* enrow = P_I32(s_enrow) + (u64)blk * n_chk;
    u64* planes = P_U64(s_planes) + (u64)blk * QEC_EN_BITS * wr;
    u64* epiv = P_U64(s_epiv) + (u64)blk * wr;
    u32* freep = (u32*)P->s_free + (u64)blk * k_cols;
    u64* col = P_U64(s_col) + (u64)blk * k_cols * wr;
    u64* rowt = P_U64(s_rowt) + (u64)blk * n_chk * wz;
    int* enfree = P_I32(s_enfree) + (u64)blk * k_cols;
    int* scol = P_I32(s_scol) + (u64)blk * k_cols;
    int* theta = P_I32(s_theta) + (u64)blk * k_cols;
    u64* cycobs = P_U64(s_cycobs) + (u64)blk * k_cols;
    u64* zq = P_U64(s_zq) + (u64)blk * wz;
    u64* w_rows = P_U64(w_rows) + (u64)blk * 2 * nw * wr;
    u64* w_z = P_U64(w_z) + (u64)blk * 2 * nw * wz;
    int* w_i32 = P_I32(w_i32) + (u64)blk * 2 * nw * 2;
    u64* w_obs = P_U64(w_obs) + (u64)blk * 2 * nw;
    i64* w_acc = (i64*)P->w_acc + (u64)blk * nw;
    u32* w_anc = (u32*)P->w_anc + (u64)blk * nw;
    u64* w_wt = P_U64(w_wt) + (u64)blk * nw;
    double* w_lw = (double*)P->w_lw + (u64)blk * (2 * nw + 128);
    const u32* logistic = P_U32(logistic);
    const u32* metro = P_U32(metropolis);
    QecShotOut* out = (QecShotOut*)P->out + shot;

    // ---- suspect positions and the augmented matrix [H_K | s]
    for (u32 v = tid; v < n_var; v += nt) pos[v] = -1;
    __syncthreads();
    for (u32 j = tid; j < k_cols; j += nt) pos[list[j]] = (int)j;
    __syncthreads();
    for (u32 c = tid; c < n_chk; c += nt) {
        u64* row = mat + (u64)c * wm;
        for (u32 q = 0; q < wm; ++q) row[q] = 0ULL;
        for (u32 e = chk_ptr[c]; e < chk_ptr[c + 1]; ++e) {
            int j = pos[chk_var[e]];
            if (j >= 0) row[j >> 6] |= 1ULL << (j & 63);
        }
        if (get_bit(syn, c)) row[k_cols >> 6] |= 1ULL << (k_cols & 63);
        pivcol[c] = -1;
    }
    if (tid == 0u) {
        sh_piv[0] = QEC_INT_MAX;
        sh_piv[1] = QEC_INT_MAX;
    }
    __syncthreads();
    // ---- Gauss-Jordan in reliability order, lowest unused row as pivot
    for (u32 j = 0; j < k_cols; ++j) {
        const u32 w = j >> 6;
        const u64 m = 1ULL << (j & 63);
        for (u32 r = tid; r < n_chk; r += nt) {
            if (pivcol[r] < 0 && (mat[(u64)r * wm + w] & m)) {
                atomicMin(&sh_piv[j & 1u], (int)r);
                break;
            }
        }
        __syncthreads();
        const int pr = sh_piv[j & 1u];
        if (pr != QEC_INT_MAX) {
            const u64* prow = mat + (u64)pr * wm;
            for (u32 r = tid; r < n_chk; r += nt) {
                u64* row = mat + (u64)r * wm;
                if ((int)r != pr && (row[w] & m)) {
                    for (u32 q = 0; q < wm; ++q) row[q] ^= prow[q];
                }
            }
        }
        if (tid == 0u) {
            if (pr != QEC_INT_MAX) pivcol[pr] = (int)j;
            sh_piv[(j + 1u) & 1u] = QEC_INT_MAX;
        }
        __syncthreads();
    }
    // ---- consistency, forest solution, weight bit-planes
    for (u32 q = tid; q < QEC_EN_BITS * wr; q += nt) planes[q] = 0ULL;
    for (u32 q = tid; q < wr; q += nt) epiv[q] = 0ULL;
    for (u32 j = tid; j < k_cols; j += nt) pos[j] = 1;
    if (tid == 0u) {
        sh_i[0] = 1; // solvable
        sh_i[1] = 0; // energy of the forest solution
        sh_i[2] = 0; // rank
        sh_obs = 0ULL;
    }
    __syncthreads();
    {
        int en_local = 0, rank_local = 0, bad = 0;
        u64 obs_local = 0ULL;
        for (u32 r = tid; r < n_chk; r += nt) {
            const int aug = (int)((mat[(u64)r * wm + (k_cols >> 6)] >> (k_cols & 63)) & 1ULL);
            if (pivcol[r] < 0) {
                bad |= aug;
                enrow[r] = 0;
                continue;
            }
            ++rank_local;
            pos[pivcol[r]] = 0;
            const int e = rec_en[pivcol[r]];
            enrow[r] = e;
            for (int bit = 0; bit < QEC_EN_BITS; ++bit) {
                if ((e >> bit) & 1) atomicOr(&planes[(u64)bit * wr + (r >> 6)], 1ULL << (r & 63));
            }
            if (aug) {
                atomicOr(&epiv[r >> 6], 1ULL << (r & 63));
                obs_local ^= obs[list[pivcol[r]]];
                en_local += e;
            }
        }
        if (bad) atomicAnd(&sh_i[0], 0);
        atomicAdd(&sh_i[1], en_local);
        atomicAdd(&sh_i[2], rank_local);
        atomicXor(&sh_obs, obs_local);
    }
    __syncthreads();
    const int solvable = sh_i[0], en_piv = sh_i[1], rank = sh_i[2];
    const u64 obs_piv = sh_obs;
    const u32 n_free = k_cols - (u32)rank;
    if (tid == 0u) {
        u32 nf = 0u;
        for (u32 j = 0; j < k_cols; ++j) {
            if (pos[j]) freep[nf++] = j;
        }
        out->rank = rank;
        out->n_free = (int)n_free;
        out->solvable = (unsigned char)solvable;
        if (!solvable) {
            out->hle = 1;
            out->min_abs_df = 0.0;
            out->khat = 1.0 / 0.0;
        }
    }
    __syncthreads();
    if (!solvable) return;
    // ---- fundamental cycles: chord columns, their weights and logical action
    for (u32 t = tid; t < n_free; t += nt) {
        const u32 fp = freep[t], w = fp >> 6;
        const u64 m = 1ULL << (fp & 63);
        u64* ct = col + (u64)t * wr;
        for (u32 q = 0; q < wr; ++q) ct[q] = 0ULL;
        int ssum = 0;
        u64 o = obs[list[fp]];
        for (u32 r = 0; r < n_chk; ++r) {
            if (mat[(u64)r * wm + w] & m) {
                ct[r >> 6] |= 1ULL << (r & 63);
                ssum += enrow[r];
                o ^= obs[list[pivcol[r]]];
            }
        }
        enfree[t] = rec_en[fp];
        scol[t] = ssum;
        cycobs[t] = o;
    }
    for (u32 r = tid; r < n_chk; r += nt) {
        u64* rt = rowt + (u64)r * wz;
        for (u32 q = 0; q < wz; ++q) rt[q] = 0ULL;
        for (u32 t = 0; t < n_free; ++t) {
            const u32 fp = freep[t];
            if ((mat[(u64)r * wm + (fp >> 6)] >> (fp & 63)) & 1ULL) rt[t >> 6] |= 1ULL << (t & 63);
        }
    }
    __syncthreads();
    SmcView V;
    V.col = col;
    V.planes = planes;
    V.enfree = enfree;
    V.scol = scol;
    V.wr = wr;
    // ---- every walker performs the same deterministic quench on its own state
    u32 cur = 0u;
    u64* rows = w_rows + ((u64)cur * nw + tid) * wr;
    u64* z = w_z + ((u64)cur * nw + tid) * wz;
    int energy = en_piv, theta_sum = 0;
    u64 wobs = obs_piv;
    i64 acc = 0;
    const int active_walker = tid < nw;
    if (active_walker) {
        for (u32 q = 0; q < wr; ++q) rows[q] = epiv[q];
        for (u32 q = 0; q < wz; ++q) z[q] = 0ULL;
        for (int sweep = 0; sweep < P->quench_sweeps; ++sweep) {
            int moved = 0;
            for (u32 t = 0; t < n_free; ++t) {
                const int zt = get_bit(z, t);
                const int de = delta_energy(V, t, rows, zt);
                if (de < 0) {
                    const u64* ct = col + (u64)t * wr;
                    for (u32 q = 0; q < wr; ++q) rows[q] ^= ct[q];
                    z[t >> 6] ^= 1ULL << (t & 63);
                    energy += de;
                    wobs ^= cycobs[t];
                    moved = 1;
                }
            }
            if (!moved) break;
        }
        if (tid == 0u) {
            for (u32 q = 0; q < wz; ++q) zq[q] = z[q];
            out->piv_obs = obs_piv;
            out->piv_energy = en_piv;
            out->q_obs = wobs;
        }
    }
    __syncthreads();
    // ---- twist field at the quenched state
    if (active_walker) {
        for (u32 t = tid; t < n_free; t += nw) {
            int th = delta_energy(V, t, rows, get_bit(z, t));
            theta[t] = th < -P->twist_clip ? -P->twist_clip : (th > P->twist_clip ? P->twist_clip : th);
        }
    }
    __syncthreads();
    // ---- exact draw from the twisted reference
    if (active_walker) {
        for (u32 t = 0; t < n_free; ++t) {
            const u32 u = (u32)(rng_u64(P->seed_smc, shot, tid, t) >> 32);
            if (u < logistic[theta[t] + P->twist_clip]) {
                energy += delta_energy(V, t, rows, get_bit(z, t));
                const u64* ct = col + (u64)t * wr;
                for (u32 q = 0; q < wr; ++q) rows[q] ^= ct[q];
                z[t >> 6] ^= 1ULL << (t & 63);
                theta_sum += theta[t];
                wobs ^= cycobs[t];
            }
        }
    }
    if (tid == 0u) sh_i[3] = 0; // resampling events
    // ---- anneal beta = b / B with adaptive integer systematic resampling
    const i64 b_total = (i64)P->beta_steps;
    for (int b = 1; b <= P->beta_steps; ++b) {
        if (active_walker) acc += (i64)(energy - theta_sum);
        if (b < P->beta_steps && P->resample_q8 > 0) {
            if (active_walker) {
                w_acc[tid] = acc;
                w_i32[((u64)cur * nw + tid) * 2] = energy;
                w_i32[((u64)cur * nw + tid) * 2 + 1] = theta_sum;
                w_obs[(u64)cur * nw + tid] = wobs;
            }
            __syncthreads();
            if (tid == 0u) {
                i64 lo = w_acc[0];
                for (u32 i = 1; i < nw; ++i) lo = w_acc[i] < lo ? w_acc[i] : lo;
                u64 s1 = 0ULL, s2 = 0ULL, total = 0ULL;
                for (u32 i = 0; i < nw; ++i) {
                    const i64 d = w_acc[i] - lo;
                    const u64 wt = d < (i64)P->metro_len ? (u64)metro[d] : 0ULL;
                    w_wt[i] = wt;
                    total += wt;
                    s1 += wt >> 16;
                    s2 += (wt >> 16) * (wt >> 16);
                }
                int doit = 256ULL * s1 * s1 < (u64)P->resample_q8 * (u64)nw * s2;
                if (doit) {
                    const u64 u0 = rng_u64(P->seed_smc, shot, 0xFFFFFFFFu, (u32)b) % total;
                    u32 i = 0u;
                    u64 cum = w_wt[0];
                    for (u32 j = 0; j < nw; ++j) {
                        const u64 target = u0 + (u64)j * total;
                        while (cum * (u64)nw <= target) {
                            ++i;
                            cum += w_wt[i];
                        }
                        w_anc[j] = i;
                    }
                    sh_i[3] += 1;
                }
                sh_i[4] = doit;
            }
            __syncthreads();
            if (sh_i[4]) {
                const u32 nxt = cur ^ 1u;
                if (active_walker) {
                    const u32 a = w_anc[tid];
                    const u64* src_r = w_rows + ((u64)cur * nw + a) * wr;
                    const u64* src_z = w_z + ((u64)cur * nw + a) * wz;
                    u64* dst_r = w_rows + ((u64)nxt * nw + tid) * wr;
                    u64* dst_z = w_z + ((u64)nxt * nw + tid) * wz;
                    for (u32 q = 0; q < wr; ++q) dst_r[q] = src_r[q];
                    for (u32 q = 0; q < wz; ++q) dst_z[q] = src_z[q];
                    energy = w_i32[((u64)cur * nw + a) * 2];
                    theta_sum = w_i32[((u64)cur * nw + a) * 2 + 1];
                    wobs = w_obs[(u64)cur * nw + a];
                    acc = 0;
                    rows = dst_r;
                    z = dst_z;
                }
                cur = nxt;
                __syncthreads();
            }
        }
        if (active_walker) {
            for (int sweep = 0; sweep < P->sweeps; ++sweep) {
                const u32 base = (u32)(b * P->sweeps + sweep) << 12;
                for (u32 t = 0; t < n_free; ++t) {
                    const int zt = get_bit(z, t);
                    const int de = delta_energy(V, t, rows, zt);
                    const int dth = (zt != get_bit(zq, t)) ? -theta[t] : theta[t];
                    const i64 d = (b_total - (i64)b) * (i64)dth + (i64)b * (i64)de;
                    int accept = d <= 0;
                    if (!accept && d < (i64)P->metro_len) {
                        accept = (u32)(rng_u64(P->seed_smc, shot, tid, base | t) >> 32) < metro[d];
                    }
                    if (accept) {
                        const u64* ct = col + (u64)t * wr;
                        for (u32 q = 0; q < wr; ++q) rows[q] ^= ct[q];
                        z[t >> 6] ^= 1ULL << (t & 63);
                        energy += de;
                        theta_sum += dth;
                        wobs ^= cycobs[t];
                    }
                }
            }
        }
    }
    if (active_walker) {
        w_acc[tid] = acc;
        w_i32[((u64)cur * nw + tid) * 2] = energy;
        w_obs[(u64)cur * nw + tid] = wobs;
    }
    __syncthreads();
    // ---- F2 integrity check of every walker: rows == e_piv (+) (R_free * z), recomputed with
    // the selected bit-matrix engine
    int bad_bits = 0;
    if (P->f2_engine == 1 && (nt & 31u) == 0u && (nw & 7u) == 0u && (wz & 3u) == 0u) {
        const u32 warp = tid >> 5, lane = tid & 31u, g = lane >> 2, tq = lane & 3u;
        const u32 tiles_w = nw / 8u, tiles_r = n_chk / 16u;
        for (u32 tile = warp; tile < tiles_r * tiles_w; tile += nt / 32u) {
            const u32 r0 = (tile / tiles_w) * 16u, w0 = (tile % tiles_w) * 8u;
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (u32 c = 0; c < wz / 4u; ++c) {
                b1_and_popc(rowt[(u64)(r0 + g) * wz + 4u * c + tq], rowt[(u64)(r0 + g + 8u) * wz + 4u * c + tq],
                            w_z[((u64)cur * nw + w0 + g) * wz + 4u * c + tq], d0, d1, d2, d3);
            }
            const u32 wa = w0 + 2u * tq, wb = wa + 1u;
            const u64* ra = w_rows + ((u64)cur * nw + wa) * wr;
            const u64* rb = w_rows + ((u64)cur * nw + wb) * wr;
            bad_bits += ((d0 & 1) ^ get_bit(epiv, r0 + g)) != get_bit(ra, r0 + g);
            bad_bits += ((d1 & 1) ^ get_bit(epiv, r0 + g)) != get_bit(rb, r0 + g);
            bad_bits += ((d2 & 1) ^ get_bit(epiv, r0 + g + 8u)) != get_bit(ra, r0 + g + 8u);
            bad_bits += ((d3 & 1) ^ get_bit(epiv, r0 + g + 8u)) != get_bit(rb, r0 + g + 8u);
        }
        // rows beyond the last full 16-row tile
        if (active_walker) {
            for (u32 r = tiles_r * 16u; r < n_chk; ++r) {
                int par = 0;
                for (u32 q = 0; q < wz; ++q) par ^= __popcll(rowt[(u64)r * wz + q] & z[q]);
                bad_bits += ((par & 1) ^ get_bit(epiv, r)) != get_bit(rows, r);
            }
        }
    } else if (active_walker) {
        for (u32 r = 0; r < n_chk; ++r) {
            int par = 0;
            for (u32 q = 0; q < wz; ++q) par ^= __popcll(rowt[(u64)r * wz + q] & z[q]);
            bad_bits += ((par & 1) ^ get_bit(epiv, r)) != get_bit(rows, r);
        }
    }
    if (bad_bits) atomicAdd(&ctrl[QEC_CTRL_INTEGRITY], (u32)bad_bits);
    // ---- certificate (integer weights for every decision)
    if (tid != 0u) return;
    i64 lo = w_acc[0];
    for (u32 i = 1; i < nw; ++i) lo = w_acc[i] < lo ? w_acc[i] : lo;
    u64 total = 0ULL;
    double sq = 0.0;
    const double scale = 1.0 / ((double)P->beta_steps * (double)P->en_one);
    for (u32 i = 0; i < nw; ++i) {
        const i64 d = w_acc[i] - lo;
        const u64 wt = d < (i64)P->metro_len ? (u64)metro[d] : 0ULL;
        w_wt[i] = wt;
        total += wt;
        sq += (double)wt * (double)wt;
        w_lw[i] = -((double)d) * scale;
    }
    const u64* fobs = w_obs + (u64)cur * nw;
    const int* fen = w_i32 + (u64)cur * nw * 2;
    u64 mode_obs = 0ULL, mode_mass = 0ULL;
    for (u32 i = 0; i < nw; ++i) {
        int seen = 0;
        for (u32 j = 0; j < i && !seen; ++j) seen = fobs[j] == fobs[i];
        if (seen) continue;
        u64 mass = 0ULL;
        for (u32 j = 0; j < nw; ++j) {
            if (fobs[j] == fobs[i]) mass += w_wt[j];
        }
        if (i == 0u || mass > mode_mass) {
            mode_mass = mass;
            mode_obs = fobs[i];
        }
    }
    u64 marginal = 0ULL;
    double min_abs = 1.0 / 0.0;
    for (u32 j = 0; j < n_obs; ++j) {
        u64 p1 = 0ULL;
        for (u32 i = 0; i < nw; ++i) {
            if ((fobs[i] >> j) & 1ULL) p1 += w_wt[i];
        }
        const u64 p0 = total - p1;
        double df;
        if (p1 == 0ULL) {
            df = QEC_DF_CAP;
        } else if (p0 == 0ULL) {
            df = -QEC_DF_CAP;
        } else {
            df = -log((double)p1 / (double)p0);
            df = df < -QEC_DF_CAP ? -QEC_DF_CAP : (df > QEC_DF_CAP ? QEC_DF_CAP : df);
        }
        if (p1 > p0) marginal |= 1ULL << j;
        const double ad = df < 0.0 ? -df : df;
        min_abs = ad < min_abs ? ad : min_abs;
    }
    int best_energy = fen[0];
    u64 best_obs = fobs[0];
    int inside = QEC_INT_MAX, outside = QEC_INT_MAX;
    for (u32 i = 0; i < nw; ++i) {
        const int e = fen[(u64)i * 2];
        if (e < best_energy) {
            best_energy = e;
            best_obs = fobs[i];
        }
        if (fobs[i] == mode_obs) {
            inside = e < inside ? e : inside;
        } else {
            outside = e < outside ? e : outside;
        }
    }
    const double khat = psis_khat(w_lw, (int)nw);
    out->obs = mode_obs;
    out->mode_obs = mode_obs;
    out->marginal_obs = marginal;
    out->best_obs = best_obs;
    out->best_energy = best_energy;
    out->energy_gap = (inside != QEC_INT_MAX && outside != QEC_INT_MAX) ? outside - inside : QEC_INT_MAX;
    out->khat = khat;
    out->min_abs_df = min_abs;
    out->ess = sq > 0.0 ? ((double)total * (double)total) / sq : 0.0;
    out->mode_free_energy = mode_mass > 0ULL ? -log((double)mode_mass / (double)total) : QEC_DF_CAP;
    out->resamples = sh_i[3];
    out->hle = (unsigned char)(khat > P->khat_max || min_abs < P->tau_cert);
}
