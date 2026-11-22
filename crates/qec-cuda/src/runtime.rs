//! Safe wrapper over the driver-API host shim: the static VRAM arena, the device-mapped
//! syndrome ring, the two green contexts and the conditional decode graph.

use crate::params::{QecParams, QecShotOut};
use anyhow::{anyhow, bail, ensure, Result};
use std::os::raw::{c_char, c_int, c_void};

/// SM120 kernels compiled offline by `build.rs`.
pub static CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qec_sm120_kernels.cubin"));
/// `ptxas -v` report of that compilation.
pub static PTXAS_LOG: &str = include_str!(concat!(env!("OUT_DIR"), "/ptxas.log"));

#[repr(C)]
struct RawRt {
    _private: [u8; 0],
}

/// Runtime facts measured by the shim (never assumed).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RtInfo {
    pub arena_bytes: u64,
    pub arena_used: u64,
    pub ring_bytes: u64,
    pub vram_total: u64,
    pub l2_max_persist: u64,
    pub l2_max_window: u64,
    pub l2_limit: u64,
    pub l2_window_bytes: u64,
    pub host_syncs_total: u64,
    pub host_syncs_in_loop: u64,
    pub launches: u64,
    pub sm_total: u32,
    pub sm_fast: u32,
    pub sm_slow: u32,
    pub sm_align: u32,
    pub sm_min_partition: u32,
    pub split_mode: i32,
    pub graph_mode: i32,
    pub l2_limit_status: i32,
    pub l2_attr_status: i32,
    pub l2_attr_readback: i32,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub last_error: i32,
    pub name: [c_char; 96],
}

impl RtInfo {
    pub fn device_name(&self) -> String {
        self.name.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect()
    }
}

extern "C" {
    fn qec_params_sizeof() -> usize;
    fn qec_shot_out_sizeof() -> usize;
    fn qec_params_offsetof(which: c_int) -> usize;
    fn qec_shot_out_offsetof(which: c_int) -> usize;
    fn qec_rt_create(out: *mut *mut RawRt, cubin: *const c_void, arena: usize, ring: usize, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_destroy(rt: *mut RawRt);
    fn qec_rt_info(rt: *mut RawRt, out: *mut RtInfo);
    fn qec_rt_ring_host(rt: *mut RawRt) -> *mut c_void;
    fn qec_rt_ring_dev(rt: *mut RawRt) -> u64;
    fn qec_rt_arena_reset(rt: *mut RawRt);
    fn qec_rt_arena_alloc(rt: *mut RawRt, bytes: usize) -> u64;
    fn qec_rt_upload(rt: *mut RawRt, dptr: u64, src: *const c_void, bytes: usize, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_download(rt: *mut RawRt, dst: *mut c_void, dptr: u64, bytes: usize, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_zero(rt: *mut RawRt, dptr: u64, bytes: usize, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_build_graph(
        rt: *mut RawRt,
        params: u64,
        streams: u32,
        smc_blocks: u32,
        walkers: u32,
        l2_base: u64,
        l2_bytes: usize,
        force_mode: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn qec_rt_launch(rt: *mut RawRt, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_wait(rt: *mut RawRt, err: *mut c_char, errlen: usize) -> c_int;
    fn qec_rt_f2_bench(
        rt: *mut RawRt,
        engine: c_int,
        a: u64,
        b: u64,
        c: u64,
        m: c_int,
        n: c_int,
        kw: c_int,
        reps: c_int,
        ns: *mut f64,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

/// Offsets of probe fields as the Rust mirror sees them, in the shim's probe order.
fn rust_param_offsets() -> [usize; 11] {
    use std::mem::offset_of;
    [
        offset_of!(QecParams, magic),
        offset_of!(QecParams, obs),
        offset_of!(QecParams, llr),
        offset_of!(QecParams, rec_en),
        offset_of!(QecParams, w_lw),
        offset_of!(QecParams, seed_smc),
        offset_of!(QecParams, khat_max),
        offset_of!(QecParams, n_chk),
        offset_of!(QecParams, k_cols),
        offset_of!(QecParams, t_max),
        offset_of!(QecParams, steps_per_launch),
    ]
}

fn rust_out_offsets() -> [usize; 5] {
    use std::mem::offset_of;
    [
        offset_of!(QecShotOut, khat),
        offset_of!(QecShotOut, bp_iters),
        offset_of!(QecShotOut, hot_checks),
        offset_of!(QecShotOut, bp_converged),
        offset_of!(QecShotOut, hle),
    ]
}

/// Verifies that the Rust mirrors agree with the C compiler on size and field offsets.
pub fn verify_layout() -> Result<()> {
    // SAFETY: pure functions of the shim, no device access.
    unsafe {
        ensure!(
            qec_params_sizeof() == std::mem::size_of::<QecParams>(),
            "QecParams size: C {} vs Rust {}",
            qec_params_sizeof(),
            std::mem::size_of::<QecParams>()
        );
        ensure!(
            qec_shot_out_sizeof() == std::mem::size_of::<QecShotOut>(),
            "QecShotOut size: C {} vs Rust {}",
            qec_shot_out_sizeof(),
            std::mem::size_of::<QecShotOut>()
        );
        for (i, &off) in rust_param_offsets().iter().enumerate() {
            ensure!(qec_params_offsetof(i as c_int) == off, "QecParams probe {i}: C {} vs Rust {off}", qec_params_offsetof(i as c_int));
        }
        for (i, &off) in rust_out_offsets().iter().enumerate() {
            ensure!(qec_shot_out_offsetof(i as c_int) == off, "QecShotOut probe {i}: C {} vs Rust {off}", qec_shot_out_offsetof(i as c_int));
        }
    }
    Ok(())
}

pub struct Runtime {
    raw: *mut RawRt,
    ring_words: usize,
}

fn err_text(buf: &[c_char]) -> String {
    buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect()
}

macro_rules! shim {
    ($self:ident, $f:ident ( $($arg:expr),* )) => {{
        let mut err = [0 as c_char; 256];
        // SAFETY: `raw` is a live runtime handle owned by `self`; buffers outlive the call.
        let code = unsafe { $f($self.raw, $($arg,)* err.as_mut_ptr(), err.len()) };
        if code != 0 {
            bail!("{}", err_text(&err));
        }
    }};
}

impl Runtime {
    /// Creates the runtime on device 0: arena, ring, green contexts, kernels.
    pub fn new(arena_bytes: usize, ring_bytes: usize) -> Result<Self> {
        verify_layout()?;
        let mut raw: *mut RawRt = std::ptr::null_mut();
        let mut err = [0 as c_char; 256];
        // SAFETY: the shim copies nothing it keeps beyond the call except the library image,
        // which is a `'static` byte slice.
        let code = unsafe {
            qec_rt_create(&mut raw, CUBIN.as_ptr() as *const c_void, arena_bytes, ring_bytes, err.as_mut_ptr(), err.len())
        };
        if code != 0 || raw.is_null() {
            return Err(anyhow!("SM120 runtime unavailable: {}", err_text(&err)));
        }
        Ok(Self { raw, ring_words: ring_bytes / 8 })
    }

    pub fn info(&self) -> RtInfo {
        // SAFETY: `RtInfo` is plain data fully written by the shim.
        unsafe {
            let mut info = std::mem::zeroed::<RtInfo>();
            qec_rt_info(self.raw, &mut info);
            info
        }
    }

    pub fn arena_reset(&mut self) {
        // SAFETY: live handle.
        unsafe { qec_rt_arena_reset(self.raw) }
    }

    /// Bump-allocates `bytes` of the static arena (uninitialised).
    pub fn alloc(&mut self, bytes: usize) -> Result<u64> {
        // SAFETY: live handle.
        let p = unsafe { qec_rt_arena_alloc(self.raw, bytes.max(8)) };
        ensure!(p != 0, "static VRAM arena exhausted ({} bytes requested, {} of {} used)", bytes, self.info().arena_used, self.info().arena_bytes);
        Ok(p)
    }

    pub fn alloc_zeroed(&mut self, bytes: usize) -> Result<u64> {
        let p = self.alloc(bytes)?;
        shim!(self, qec_rt_zero(p, bytes.max(8)));
        Ok(p)
    }

    /// Allocates and fills a device array.
    pub fn upload<T: Copy>(&mut self, data: &[T]) -> Result<u64> {
        let bytes = std::mem::size_of_val(data);
        let p = self.alloc(bytes)?;
        if bytes > 0 {
            shim!(self, qec_rt_upload(p, data.as_ptr() as *const c_void, bytes));
        }
        Ok(p)
    }

    pub fn write<T: Copy>(&mut self, dptr: u64, data: &[T]) -> Result<()> {
        shim!(self, qec_rt_upload(dptr, data.as_ptr() as *const c_void, std::mem::size_of_val(data)));
        Ok(())
    }

    pub fn read<T: Copy + Default>(&mut self, dptr: u64, n: usize) -> Result<Vec<T>> {
        let mut out = vec![T::default(); n];
        shim!(self, qec_rt_download(out.as_mut_ptr() as *mut c_void, dptr, n * std::mem::size_of::<T>()));
        Ok(out)
    }

    /// Host view of the page-locked, device-mapped syndrome ring.
    pub fn ring(&mut self) -> &mut [u64] {
        // SAFETY: the ring is `ring_words * 8` bytes of page-locked host memory owned by the
        // runtime and aligned to a page; the borrow is tied to `self`.
        unsafe { std::slice::from_raw_parts_mut(qec_rt_ring_host(self.raw) as *mut u64, self.ring_words) }
    }

    pub fn ring_dev(&self) -> u64 {
        // SAFETY: live handle.
        unsafe { qec_rt_ring_dev(self.raw) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_graph(&mut self, params: u64, streams: u32, smc_blocks: u32, walkers: u32, l2_base: u64, l2_bytes: usize, force_mode: i32) -> Result<()> {
        shim!(self, qec_rt_build_graph(params, streams, smc_blocks, walkers, l2_base, l2_bytes, force_mode));
        Ok(())
    }

    /// Enqueues one decode launch without blocking.
    pub fn launch(&mut self) -> Result<()> {
        shim!(self, qec_rt_launch());
        Ok(())
    }

    /// The single terminal completion wait of a launch.
    pub fn wait(&mut self) -> Result<()> {
        shim!(self, qec_rt_wait());
        Ok(())
    }

    /// Mean wall time (ns) of one F2 engine kernel call (`engine` 0: popcount, 1: `.b1`).
    #[allow(clippy::too_many_arguments)]
    pub fn f2_bench(&mut self, engine: i32, a: u64, b: u64, c: u64, m: i32, n: i32, kw: i32, reps: i32) -> Result<f64> {
        let mut ns = 0.0f64;
        shim!(self, qec_rt_f2_bench(engine, a, b, c, m, n, kw, reps, &mut ns));
        Ok(ns)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: `raw` was created by `qec_rt_create` and is destroyed exactly once.
        unsafe { qec_rt_destroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_mirrors_match_the_c_layout() {
        verify_layout().unwrap();
        assert_eq!(std::mem::size_of::<QecShotOut>() % 8, 0);
    }

    #[test]
    fn embedded_cubin_carries_every_kernel_and_no_spill() {
        assert!(CUBIN.len() > 4096 && &CUBIN[..4] == b"\x7fELF");
        for name in [
            "qec_neuro_update",
            "qec_bp_decode",
            "qec_fast_advance",
            "qec_slow_gate",
            "qec_smc_decode",
            "qec_slow_advance",
            "qec_f2_matmul_popc",
            "qec_f2_matmul_b1",
        ] {
            assert!(CUBIN.windows(name.len()).any(|w| w == name.as_bytes()), "kernel {name} missing from the cubin");
            assert!(PTXAS_LOG.contains(&format!("Compiling entry function '{name}' for 'sm_120'")));
        }
        assert_eq!(env!("QEC_KERNEL_ENTRIES"), "8");
        for line in PTXAS_LOG.lines().filter(|l| l.contains("spill stores")) {
            assert!(line.contains(" 0 bytes spill stores"), "spill reported: {line}");
        }
    }
}
