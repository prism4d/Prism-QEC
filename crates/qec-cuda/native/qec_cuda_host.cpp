// PRISM-QEC SM120 host shim: CUDA driver API only.
//
//  * one static VRAM arena allocated at start-up (bump allocation, no later allocations),
//  * a CU_MEMHOSTALLOC_DEVICEMAP page-locked syndrome ring written by the host and read by
//    the device through its mapped pointer (zero copy),
//  * two green contexts carved out of the device's SMs (fast path / slow path),
//  * the decode graph: a CU_GRAPH_COND_TYPE_WHILE loop for the fast path, a device-side
//    residual gate, and a CU_GRAPH_COND_TYPE_WHILE loop for the slow path,
//  * a persisting-L2 access-policy window over the live DEM weights,
//  * accounting of every blocking host call, so `host_syncs_in_decode_loop` is measured.
#include <cuda.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "qec_sm120_params.cuh"

extern "C" {

typedef struct QecRtInfo {
    unsigned long long arena_bytes, arena_used, ring_bytes, vram_total;
    unsigned long long l2_max_persist, l2_max_window, l2_limit, l2_window_bytes;
    unsigned long long host_syncs_total, host_syncs_in_loop, launches;
    unsigned int sm_total, sm_fast, sm_slow, sm_align, sm_min_partition;
    int split_mode;         // 1..: which SM split request the driver accepted, 0: none
    int graph_mode;         // 1: single graph, 2: two graphs chained by an event, 0: none
    int l2_limit_status;    // CUresult of cuCtxSetLimit(PERSISTING_L2_CACHE_SIZE)
    int l2_attr_status;     // CUresult of the kernel-node access-policy attribute
    int l2_attr_readback;   // 1 iff the attribute read back PERSISTING with the same window
    int cc_major, cc_minor;
    int last_error;
    char name[96];
} QecRtInfo;

struct QecRt {
    CUdevice dev;
    CUcontext primary;
    CUgreenCtx g_fast, g_slow;
    CUcontext c_fast, c_slow;
    CUstream s_fast, s_slow;
    CUlibrary lib;
    CUkernel k_neuro, k_bp, k_fadv, k_gate, k_smc, k_sadv, k_popc, k_b1;
    CUdeviceptr arena;
    void* ring_host;
    CUdeviceptr ring_dev;
    CUgraph graph_a, graph_b;
    CUgraphExec exec_a, exec_b;
    CUevent chain;
    CUgraphConditionalHandle h_fast, h_slow;
    unsigned long long arg_params;
    void* args_p[1];
    void* args_fast[2];
    void* args_slow[2];
    int in_loop;
    QecRtInfo info;
};

size_t qec_params_sizeof(void) { return sizeof(QecParams); }
size_t qec_shot_out_sizeof(void) { return sizeof(QecShotOut); }
size_t qec_params_offsetof(int which)
{
    switch (which) {
    case 0: return offsetof(QecParams, magic);
    case 1: return offsetof(QecParams, obs);
    case 2: return offsetof(QecParams, llr);
    case 3: return offsetof(QecParams, rec_en);
    case 4: return offsetof(QecParams, w_lw);
    case 5: return offsetof(QecParams, seed_smc);
    case 6: return offsetof(QecParams, khat_max);
    case 7: return offsetof(QecParams, n_chk);
    case 8: return offsetof(QecParams, k_cols);
    case 9: return offsetof(QecParams, t_max);
    case 10: return offsetof(QecParams, steps_per_launch);
    case 11: return offsetof(QecParams, arm_state);
    case 12: return offsetof(QecParams, blk_sigma_q8);
    case 13: return offsetof(QecParams, h_rank);
    case 14: return offsetof(QecParams, tau_ambig);
    case 15: return offsetof(QecParams, llr_deadband);
    default: return (size_t)-1;
    }
}
size_t qec_shot_out_offsetof(int which)
{
    switch (which) {
    case 0: return offsetof(QecShotOut, khat);
    case 1: return offsetof(QecShotOut, bp_iters);
    case 2: return offsetof(QecShotOut, hot_checks);
    case 3: return offsetof(QecShotOut, bp_converged);
    case 4: return offsetof(QecShotOut, hle);
    case 5: return offsetof(QecShotOut, amb_gap);
    case 6: return offsetof(QecShotOut, smc_sampled);
    default: return (size_t)-1;
    }
}

static int fail(QecRt* rt, CUresult r, const char* what, char* err, size_t errlen)
{
    const char* name = NULL;
    cuGetErrorName(r, &name);
    if (err && errlen) snprintf(err, errlen, "%s failed: %s (%d)", what, name ? name : "?", (int)r);
    if (rt) rt->info.last_error = (int)r;
    return (int)r ? (int)r : -1;
}

#define QEC_TRY(call, what)                                    \
    do {                                                       \
        CUresult _r = (call);                                  \
        if (_r != CUDA_SUCCESS) return fail(rt, _r, what, err, errlen); \
    } while (0)

static void sync_account(QecRt* rt)
{
    rt->info.host_syncs_total += 1;
    if (rt->in_loop) rt->info.host_syncs_in_loop += 1;
}

static CUresult split_sms(QecRt* rt, const CUdevResource* sm, CUdevResource* parts)
{
    // The fast path wants 28 SMs and the slow path 56.  Compute architecture 9.0+ aligns SM
    // groups to 8 by default, so smaller co-scheduling units are requested explicitly.
    static const unsigned cosched[] = {28, 4, 2, 1};
    CUdevResource rem;
    for (unsigned i = 0; i < sizeof(cosched) / sizeof(cosched[0]); ++i) {
        CU_DEV_SM_RESOURCE_GROUP_PARAMS gp[2];
        memset(gp, 0, sizeof(gp));
        gp[0].smCount = 28;
        gp[0].coscheduledSmCount = cosched[i];
        gp[1].smCount = 56;
        gp[1].coscheduledSmCount = cosched[i];
        memset(&rem, 0, sizeof(rem));
        if (cuDevSmResourceSplit(parts, 2, sm, &rem, 0, gp) == CUDA_SUCCESS) {
            rt->info.split_mode = 1 + (int)i;
            return CUDA_SUCCESS;
        }
    }
    {
        CU_DEV_SM_RESOURCE_GROUP_PARAMS gp[2];
        memset(gp, 0, sizeof(gp));
        gp[0].smCount = 28;
        gp[0].flags = CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
        gp[1].smCount = 56;
        gp[1].flags = CU_DEV_SM_RESOURCE_GROUP_BACKFILL;
        memset(&rem, 0, sizeof(rem));
        if (cuDevSmResourceSplit(parts, 2, sm, &rem, 0, gp) == CUDA_SUCCESS) {
            rt->info.split_mode = 5;
            return CUDA_SUCCESS;
        }
    }
    {
        // last resort: whatever one >= 28-SM group and the remainder the driver will give
        unsigned nb = 1;
        memset(&rem, 0, sizeof(rem));
        CUresult r = cuDevSmResourceSplitByCount(parts, &nb, sm, &rem,
                                                 CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, 28);
        if (r == CUDA_SUCCESS && nb == 1 && rem.type == CU_DEV_RESOURCE_TYPE_SM) {
            parts[1] = rem;
            rt->info.split_mode = 6;
            return CUDA_SUCCESS;
        }
        return r == CUDA_SUCCESS ? CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION : r;
    }
}

void qec_rt_destroy(QecRt* rt)
{
    if (!rt) return;
    if (rt->primary) cuCtxSetCurrent(rt->primary);
    if (rt->exec_a) cuGraphExecDestroy(rt->exec_a);
    if (rt->exec_b) cuGraphExecDestroy(rt->exec_b);
    if (rt->graph_a) cuGraphDestroy(rt->graph_a);
    if (rt->graph_b) cuGraphDestroy(rt->graph_b);
    if (rt->chain) cuEventDestroy(rt->chain);
    if (rt->s_fast) cuStreamDestroy(rt->s_fast);
    if (rt->s_slow) cuStreamDestroy(rt->s_slow);
    if (rt->lib) cuLibraryUnload(rt->lib);
    if (rt->ring_host) cuMemFreeHost(rt->ring_host);
    if (rt->arena) cuMemFree(rt->arena);
    if (rt->g_fast) cuGreenCtxDestroy(rt->g_fast);
    if (rt->g_slow) cuGreenCtxDestroy(rt->g_slow);
    if (rt->primary) cuDevicePrimaryCtxRelease(rt->dev);
    delete rt;
}

int qec_rt_create(QecRt** out, const void* cubin, size_t arena_bytes, size_t ring_bytes, char* err, size_t errlen)
{
    QecRt* rt = new QecRt();
    memset(rt, 0, sizeof(*rt));
    *out = NULL;
#define QEC_TRY_NEW(call, what)                                \
    do {                                                       \
        CUresult _r = (call);                                  \
        if (_r != CUDA_SUCCESS) {                              \
            int code = fail(rt, _r, what, err, errlen);        \
            qec_rt_destroy(rt);                                \
            return code;                                       \
        }                                                      \
    } while (0)
    QEC_TRY_NEW(cuInit(0), "cuInit");
    QEC_TRY_NEW(cuDeviceGet(&rt->dev, 0), "cuDeviceGet");
    cuDeviceGetName(rt->info.name, sizeof(rt->info.name), rt->dev);
    cuDeviceGetAttribute(&rt->info.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, rt->dev);
    cuDeviceGetAttribute(&rt->info.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, rt->dev);
    {
        size_t total = 0;
        cuDeviceTotalMem(&total, rt->dev);
        rt->info.vram_total = total;
        int v = 0;
        cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, rt->dev);
        rt->info.l2_max_persist = (unsigned long long)v;
        cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, rt->dev);
        rt->info.l2_max_window = (unsigned long long)v;
    }
    QEC_TRY_NEW(cuDevicePrimaryCtxSetFlags(rt->dev, CU_CTX_SCHED_SPIN), "cuDevicePrimaryCtxSetFlags");
    QEC_TRY_NEW(cuDevicePrimaryCtxRetain(&rt->primary, rt->dev), "cuDevicePrimaryCtxRetain");
    QEC_TRY_NEW(cuCtxSetCurrent(rt->primary), "cuCtxSetCurrent");
    // ---- green contexts
    CUdevResource sm, parts[2];
    memset(&sm, 0, sizeof(sm));
    memset(parts, 0, sizeof(parts));
    QEC_TRY_NEW(cuDeviceGetDevResource(rt->dev, &sm, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource(SM)");
    rt->info.sm_total = sm.sm.smCount;
    rt->info.sm_align = sm.sm.smCoscheduledAlignment;
    rt->info.sm_min_partition = sm.sm.minSmPartitionSize;
    QEC_TRY_NEW(split_sms(rt, &sm, parts), "cuDevSmResourceSplit");
    CUdevResourceDesc d_fast = NULL, d_slow = NULL;
    QEC_TRY_NEW(cuDevResourceGenerateDesc(&d_fast, &parts[0], 1), "cuDevResourceGenerateDesc(fast)");
    QEC_TRY_NEW(cuDevResourceGenerateDesc(&d_slow, &parts[1], 1), "cuDevResourceGenerateDesc(slow)");
    QEC_TRY_NEW(cuGreenCtxCreate(&rt->g_fast, d_fast, rt->dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate(fast)");
    QEC_TRY_NEW(cuGreenCtxCreate(&rt->g_slow, d_slow, rt->dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate(slow)");
    QEC_TRY_NEW(cuCtxFromGreenCtx(&rt->c_fast, rt->g_fast), "cuCtxFromGreenCtx(fast)");
    QEC_TRY_NEW(cuCtxFromGreenCtx(&rt->c_slow, rt->g_slow), "cuCtxFromGreenCtx(slow)");
    QEC_TRY_NEW(cuGreenCtxStreamCreate(&rt->s_fast, rt->g_fast, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate(fast)");
    QEC_TRY_NEW(cuGreenCtxStreamCreate(&rt->s_slow, rt->g_slow, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate(slow)");
    {
        // what the contexts actually own, read back from the driver
        CUdevResource rb;
        memset(&rb, 0, sizeof(rb));
        QEC_TRY_NEW(cuGreenCtxGetDevResource(rt->g_fast, &rb, CU_DEV_RESOURCE_TYPE_SM), "cuGreenCtxGetDevResource(fast)");
        rt->info.sm_fast = rb.sm.smCount;
        memset(&rb, 0, sizeof(rb));
        QEC_TRY_NEW(cuGreenCtxGetDevResource(rt->g_slow, &rb, CU_DEV_RESOURCE_TYPE_SM), "cuGreenCtxGetDevResource(slow)");
        rt->info.sm_slow = rb.sm.smCount;
    }
    // ---- kernels (context-independent library kernels)
    QEC_TRY_NEW(cuLibraryLoadData(&rt->lib, cubin, NULL, NULL, 0, NULL, NULL, 0), "cuLibraryLoadData");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_neuro, rt->lib, "qec_neuro_update"), "kernel qec_neuro_update");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_bp, rt->lib, "qec_bp_decode"), "kernel qec_bp_decode");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_fadv, rt->lib, "qec_fast_advance"), "kernel qec_fast_advance");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_gate, rt->lib, "qec_slow_gate"), "kernel qec_slow_gate");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_smc, rt->lib, "qec_smc_decode"), "kernel qec_smc_decode");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_sadv, rt->lib, "qec_slow_advance"), "kernel qec_slow_advance");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_popc, rt->lib, "qec_f2_matmul_popc"), "kernel qec_f2_matmul_popc");
    QEC_TRY_NEW(cuLibraryGetKernel(&rt->k_b1, rt->lib, "qec_f2_matmul_b1"), "kernel qec_f2_matmul_b1");
    // ---- memory: one arena, one device-mapped ring
    QEC_TRY_NEW(cuMemAlloc(&rt->arena, arena_bytes), "cuMemAlloc(arena)");
    rt->info.arena_bytes = arena_bytes;
    QEC_TRY_NEW(cuMemHostAlloc(&rt->ring_host, ring_bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE),
                "cuMemHostAlloc(DEVICEMAP ring)");
    QEC_TRY_NEW(cuMemHostGetDevicePointer(&rt->ring_dev, rt->ring_host, 0), "cuMemHostGetDevicePointer");
    rt->info.ring_bytes = ring_bytes;
    QEC_TRY_NEW(cuEventCreate(&rt->chain, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    // ---- persisting L2 budget for the fast-path context
    {
        size_t want = (size_t)rt->info.l2_max_persist;
        cuCtxPushCurrent(rt->c_fast);
        CUresult r = cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, want);
        size_t got = 0;
        if (r == CUDA_SUCCESS) cuCtxGetLimit(&got, CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        CUcontext popped;
        cuCtxPopCurrent(&popped);
        if (r != CUDA_SUCCESS) {
            r = cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, want);
            if (r == CUDA_SUCCESS) cuCtxGetLimit(&got, CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        }
        rt->info.l2_limit_status = (int)r;
        rt->info.l2_limit = got;
    }
    *out = rt;
    return 0;
#undef QEC_TRY_NEW
}

void qec_rt_info(QecRt* rt, QecRtInfo* out) { *out = rt->info; }
void* qec_rt_ring_host(QecRt* rt) { return rt->ring_host; }
unsigned long long qec_rt_ring_dev(QecRt* rt) { return (unsigned long long)rt->ring_dev; }
void qec_rt_arena_reset(QecRt* rt) { rt->info.arena_used = 0; }

// Bump allocation inside the static arena (64-byte aligned); 0 when exhausted.
unsigned long long qec_rt_arena_alloc(QecRt* rt, size_t bytes)
{
    size_t start = (size_t)((rt->info.arena_used + 63ULL) & ~63ULL);
    if (start + bytes > rt->info.arena_bytes) return 0ULL;
    rt->info.arena_used = start + bytes;
    return (unsigned long long)rt->arena + start;
}

int qec_rt_upload(QecRt* rt, unsigned long long dptr, const void* src, size_t bytes, char* err, size_t errlen)
{
    cuCtxSetCurrent(rt->primary);
    sync_account(rt);
    QEC_TRY(cuMemcpyHtoD((CUdeviceptr)dptr, src, bytes), "cuMemcpyHtoD");
    return 0;
}

int qec_rt_download(QecRt* rt, void* dst, unsigned long long dptr, size_t bytes, char* err, size_t errlen)
{
    cuCtxSetCurrent(rt->primary);
    sync_account(rt);
    QEC_TRY(cuMemcpyDtoH(dst, (CUdeviceptr)dptr, bytes), "cuMemcpyDtoH");
    return 0;
}

int qec_rt_zero(QecRt* rt, unsigned long long dptr, size_t bytes, char* err, size_t errlen)
{
    cuCtxSetCurrent(rt->primary);
    sync_account(rt);
    QEC_TRY(cuMemsetD8((CUdeviceptr)dptr, 0, bytes), "cuMemsetD8");
    return 0;
}

static CUresult add_kernel(CUgraph g, CUgraphNode* node, const CUgraphNode* deps, size_t ndeps, CUkernel k,
                           CUcontext ctx, unsigned grid, unsigned block, void** args)
{
    CUgraphNodeParams np;
    memset(&np, 0, sizeof(np));
    np.type = CU_GRAPH_NODE_TYPE_KERNEL;
    np.kernel.func = NULL;
    np.kernel.kern = k;
    np.kernel.ctx = ctx;
    np.kernel.gridDimX = grid;
    np.kernel.gridDimY = 1;
    np.kernel.gridDimZ = 1;
    np.kernel.blockDimX = block;
    np.kernel.blockDimY = 1;
    np.kernel.blockDimZ = 1;
    np.kernel.kernelParams = args;
    return cuGraphAddNode(node, g, deps, NULL, ndeps, &np);
}

static CUresult add_while(CUgraph g, CUgraphNode* node, const CUgraphNode* deps, size_t ndeps, CUcontext ctx,
                          CUgraphConditionalHandle h, CUgraph* body)
{
    CUgraphNodeParams np;
    memset(&np, 0, sizeof(np));
    np.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    np.conditional.handle = h;
    np.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
    np.conditional.size = 1;
    np.conditional.ctx = ctx;
    CUresult r = cuGraphAddNode(node, g, deps, NULL, ndeps, &np);
    if (r == CUDA_SUCCESS) *body = np.conditional.phGraph_out[0];
    return r;
}

static void drop_graphs(QecRt* rt)
{
    if (rt->exec_a) cuGraphExecDestroy(rt->exec_a);
    if (rt->exec_b) cuGraphExecDestroy(rt->exec_b);
    if (rt->graph_a) cuGraphDestroy(rt->graph_a);
    if (rt->graph_b) cuGraphDestroy(rt->graph_b);
    rt->exec_a = rt->exec_b = NULL;
    rt->graph_a = rt->graph_b = NULL;
    rt->info.graph_mode = 0;
}

// Fast-path loop (context `c_fast`): WHILE { monitor -> damped min-sum -> advance }.
static CUresult build_fast(QecRt* rt, CUgraph g, CUgraphNode* loop, unsigned streams, unsigned long long l2_base,
                           size_t l2_bytes)
{
    CUresult r = cuGraphConditionalHandleCreate(&rt->h_fast, g, rt->c_fast, 1, CU_GRAPH_COND_ASSIGN_DEFAULT);
    if (r != CUDA_SUCCESS) return r;
    CUgraph body = NULL;
    r = add_while(g, loop, NULL, 0, rt->c_fast, rt->h_fast, &body);
    if (r != CUDA_SUCCESS) return r;
    rt->args_fast[0] = &rt->arg_params;
    rt->args_fast[1] = &rt->h_fast;
    CUgraphNode n_neuro, n_bp, n_adv;
    r = add_kernel(body, &n_neuro, NULL, 0, rt->k_neuro, rt->c_fast, streams, 128, rt->args_p);
    if (r != CUDA_SUCCESS) return r;
    r = add_kernel(body, &n_bp, &n_neuro, 1, rt->k_bp, rt->c_fast, streams, QEC_BP_THREADS, rt->args_p);
    if (r != CUDA_SUCCESS) return r;
    r = add_kernel(body, &n_adv, &n_bp, 1, rt->k_fadv, rt->c_fast, 1, 1, rt->args_fast);
    if (r != CUDA_SUCCESS) return r;
    // persisting L2 residency of the live DEM weights read by the fast path
    rt->info.l2_window_bytes = 0;
    rt->info.l2_attr_readback = 0;
    if (l2_bytes > 0) {
        size_t window = l2_bytes < (size_t)rt->info.l2_max_window ? l2_bytes : (size_t)rt->info.l2_max_window;
        CUkernelNodeAttrValue v, rb;
        memset(&v, 0, sizeof(v));
        v.accessPolicyWindow.base_ptr = (void*)l2_base;
        v.accessPolicyWindow.num_bytes = window;
        v.accessPolicyWindow.hitRatio = 1.0f;
        v.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
        v.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;
        CUresult a = cuGraphKernelNodeSetAttribute(n_bp, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &v);
        rt->info.l2_attr_status = (int)a;
        if (a == CUDA_SUCCESS) {
            rt->info.l2_window_bytes = window;
            memset(&rb, 0, sizeof(rb));
            if (cuGraphKernelNodeGetAttribute(n_bp, CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &rb) == CUDA_SUCCESS) {
                rt->info.l2_attr_readback = rb.accessPolicyWindow.hitProp == CU_ACCESS_PROPERTY_PERSISTING
                    && rb.accessPolicyWindow.base_ptr == v.accessPolicyWindow.base_ptr
                    && rb.accessPolicyWindow.num_bytes == window;
            }
        }
    }
    return CUDA_SUCCESS;
}

// Slow path (context `c_slow`): residual gate -> WHILE { coset SMC -> advance }.
static CUresult build_slow(QecRt* rt, CUgraph g, const CUgraphNode* deps, size_t ndeps, unsigned smc_blocks,
                           unsigned walkers)
{
    CUresult r = cuGraphConditionalHandleCreate(&rt->h_slow, g, rt->c_slow, 0, CU_GRAPH_COND_ASSIGN_DEFAULT);
    if (r != CUDA_SUCCESS) return r;
    rt->args_slow[0] = &rt->arg_params;
    rt->args_slow[1] = &rt->h_slow;
    CUgraphNode n_gate, loop, n_smc, n_adv;
    r = add_kernel(g, &n_gate, deps, ndeps, rt->k_gate, rt->c_slow, 1, 1, rt->args_slow);
    if (r != CUDA_SUCCESS) return r;
    CUgraph body = NULL;
    r = add_while(g, &loop, &n_gate, 1, rt->c_slow, rt->h_slow, &body);
    if (r != CUDA_SUCCESS) return r;
    r = add_kernel(body, &n_smc, NULL, 0, rt->k_smc, rt->c_slow, smc_blocks, walkers, rt->args_p);
    if (r != CUDA_SUCCESS) return r;
    return add_kernel(body, &n_adv, &n_smc, 1, rt->k_sadv, rt->c_slow, 1, 1, rt->args_slow);
}

// Builds the decode graph for a parameter block already resident in the arena.
// `force_mode`: 0 tries the single multi-context graph and falls back to two event-chained
// graphs; 1 or 2 insists on that topology.
int qec_rt_build_graph(QecRt* rt, unsigned long long params_dptr, unsigned streams, unsigned smc_blocks,
                       unsigned walkers, unsigned long long l2_base, size_t l2_bytes, int force_mode, char* err,
                       size_t errlen)
{
    cuCtxSetCurrent(rt->primary);
    drop_graphs(rt);
    rt->arg_params = params_dptr;
    rt->args_p[0] = &rt->arg_params;
    CUresult r = CUDA_ERROR_UNKNOWN;
    if (force_mode != 2) {
        // mode 1: one graph, fast loop in the 28-SM context, gate + slow loop in the 56-SM one
        CUgraphNode loop_a;
        r = cuGraphCreate(&rt->graph_a, 0);
        if (r == CUDA_SUCCESS) r = build_fast(rt, rt->graph_a, &loop_a, streams, l2_base, l2_bytes);
        if (r == CUDA_SUCCESS) r = build_slow(rt, rt->graph_a, &loop_a, 1, smc_blocks, walkers);
        if (r == CUDA_SUCCESS) r = cuGraphInstantiate(&rt->exec_a, rt->graph_a, 0);
        if (r == CUDA_SUCCESS) {
            rt->info.graph_mode = 1;
            return 0;
        }
        drop_graphs(rt);
        if (force_mode == 1) return fail(rt, r, "single multi-context graph", err, errlen);
    }
    // mode 2: two graphs, each inside its own green context, chained by an event
    CUgraphNode loop_a;
    QEC_TRY(cuGraphCreate(&rt->graph_a, 0), "cuGraphCreate(fast)");
    QEC_TRY(build_fast(rt, rt->graph_a, &loop_a, streams, l2_base, l2_bytes), "fast-path graph");
    QEC_TRY(cuGraphCreate(&rt->graph_b, 0), "cuGraphCreate(slow)");
    QEC_TRY(build_slow(rt, rt->graph_b, NULL, 0, smc_blocks, walkers), "slow-path graph");
    cuCtxPushCurrent(rt->c_fast);
    r = cuGraphInstantiate(&rt->exec_a, rt->graph_a, 0);
    CUcontext popped;
    cuCtxPopCurrent(&popped);
    if (r != CUDA_SUCCESS) return fail(rt, r, "cuGraphInstantiate(fast)", err, errlen);
    cuCtxPushCurrent(rt->c_slow);
    r = cuGraphInstantiate(&rt->exec_b, rt->graph_b, 0);
    cuCtxPopCurrent(&popped);
    if (r != CUDA_SUCCESS) return fail(rt, r, "cuGraphInstantiate(slow)", err, errlen);
    rt->info.graph_mode = 2;
    return 0;
}

// Enqueues one decode launch.  Nothing here blocks: from this call until qec_rt_wait() the
// decode loop runs on the device, and any blocking host call is counted against it.
int qec_rt_launch(QecRt* rt, char* err, size_t errlen)
{
    rt->in_loop = 1;
    rt->info.launches += 1;
    if (rt->info.graph_mode == 1) {
        QEC_TRY(cuGraphLaunch(rt->exec_a, rt->s_fast), "cuGraphLaunch");
    } else if (rt->info.graph_mode == 2) {
        QEC_TRY(cuGraphLaunch(rt->exec_a, rt->s_fast), "cuGraphLaunch(fast)");
        QEC_TRY(cuEventRecord(rt->chain, rt->s_fast), "cuEventRecord");
        QEC_TRY(cuStreamWaitEvent(rt->s_slow, rt->chain, 0), "cuStreamWaitEvent");
        QEC_TRY(cuGraphLaunch(rt->exec_b, rt->s_slow), "cuGraphLaunch(slow)");
    } else {
        return fail(rt, CUDA_ERROR_NOT_INITIALIZED, "launch without a graph", err, errlen);
    }
    return 0;
}

// Terminal wait: the decode loop has been left to the device; this is the single completion
// wait after it.
int qec_rt_wait(QecRt* rt, char* err, size_t errlen)
{
    rt->in_loop = 0;
    sync_account(rt);
    QEC_TRY(cuStreamSynchronize(rt->s_fast), "cuStreamSynchronize(fast)");
    if (rt->info.graph_mode == 2) {
        QEC_TRY(cuStreamSynchronize(rt->s_slow), "cuStreamSynchronize(slow)");
    }
    return 0;
}

// Runs one F2 engine kernel `reps` times in the slow-path context; returns mean ns per call.
int qec_rt_f2_bench(QecRt* rt, int engine, unsigned long long a, unsigned long long b, unsigned long long c, int m,
                    int n, int kw, int reps, double* ns_per_call, char* err, size_t errlen)
{
    cuCtxPushCurrent(rt->c_slow);
    CUfunction fn = NULL;
    CUresult r = cuKernelGetFunction(&fn, engine == 1 ? rt->k_b1 : rt->k_popc);
    if (r == CUDA_SUCCESS) {
        void* args[6] = {&a, &b, &c, &m, &n, &kw};
        const int wn = (n + 63) / 64;
        unsigned grid, block;
        if (engine == 1) {
            block = 128;
            grid = (unsigned)(((size_t)(m / 16) * (size_t)(n / 8) + 3) / 4);
        } else {
            block = 256;
            grid = (unsigned)(((size_t)m * (size_t)wn + 255) / 256);
        }
        sync_account(rt);
        r = cuStreamSynchronize(rt->s_slow);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps && r == CUDA_SUCCESS; ++i) {
            if (engine == 1) r = cuMemsetD8Async((CUdeviceptr)c, 0, (size_t)m * wn * 8, rt->s_slow);
            if (r == CUDA_SUCCESS) r = cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, rt->s_slow, args, NULL);
        }
        if (r == CUDA_SUCCESS) {
            sync_account(rt);
            r = cuStreamSynchronize(rt->s_slow);
        }
        auto t1 = std::chrono::steady_clock::now();
        *ns_per_call = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)(reps > 0 ? reps : 1);
    }
    CUcontext popped;
    cuCtxPopCurrent(&popped);
    if (r != CUDA_SUCCESS) return fail(rt, r, engine == 1 ? "f2 b1 engine" : "f2 popc engine", err, errlen);
    return 0;
}

} // extern "C"
