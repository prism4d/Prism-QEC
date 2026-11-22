//! Flat CSR views of the chain complex used by the message-passing kernels (CPU and SM120).

use qec_core::complex::ChainComplex;

/// Edge `e` of the Tanner graph is the `e`-th entry of the check-major CSR of `d1`.
#[derive(Clone, Debug)]
pub struct TannerGraph {
    pub n_chk: usize,
    pub n_var: usize,
    pub n_edge: usize,
    pub n_cell: usize,
    /// check -> edges (`chk_ptr[c]..chk_ptr[c + 1]`), `chk_var[e]` is the variable of edge `e`.
    pub chk_ptr: Vec<u32>,
    pub chk_var: Vec<u32>,
    /// variable -> incident edge ids and the check of each.
    pub var_ptr: Vec<u32>,
    pub var_edge: Vec<u32>,
    pub var_chk: Vec<u32>,
    /// 2-cell -> boundary variables (`d2^T`).
    pub cell_ptr: Vec<u32>,
    pub cell_var: Vec<u32>,
    /// variable -> 2-cells containing it (`d2`).
    pub vcell_ptr: Vec<u32>,
    pub vcell: Vec<u32>,
    /// logical observable mask per variable.
    pub obs: Vec<u64>,
    pub n_obs: usize,
}

impl TannerGraph {
    pub fn from_complex(cx: &ChainComplex) -> Self {
        let n_chk = cx.n0();
        let n_var = cx.n1();
        let n_edge = cx.d1.nnz();
        let chk_ptr = cx.d1.ptr.clone();
        let chk_var = cx.d1.idx.clone();
        let mut var_ptr = vec![0u32; n_var + 1];
        for &v in &chk_var {
            var_ptr[v as usize + 1] += 1;
        }
        for v in 0..n_var {
            var_ptr[v + 1] += var_ptr[v];
        }
        let mut cursor = var_ptr.clone();
        let mut var_edge = vec![0u32; n_edge];
        let mut var_chk = vec![0u32; n_edge];
        for c in 0..n_chk {
            for e in chk_ptr[c] as usize..chk_ptr[c + 1] as usize {
                let v = chk_var[e] as usize;
                let slot = cursor[v] as usize;
                var_edge[slot] = e as u32;
                var_chk[slot] = c as u32;
                cursor[v] += 1;
            }
        }
        Self {
            n_chk,
            n_var,
            n_edge,
            n_cell: cx.n2(),
            chk_ptr,
            chk_var,
            var_ptr,
            var_edge,
            var_chk,
            cell_ptr: cx.d2_t.ptr.clone(),
            cell_var: cx.d2_t.idx.clone(),
            vcell_ptr: cx.d2.ptr.clone(),
            vcell: cx.d2.idx.clone(),
            obs: cx.obs.clone(),
            n_obs: cx.n_obs,
        }
    }

    #[inline]
    pub fn chk_edges(&self, c: usize) -> std::ops::Range<usize> {
        self.chk_ptr[c] as usize..self.chk_ptr[c + 1] as usize
    }

    #[inline]
    pub fn var_slots(&self, v: usize) -> std::ops::Range<usize> {
        self.var_ptr[v] as usize..self.var_ptr[v + 1] as usize
    }

    #[inline]
    pub fn cell_vars(&self, f: usize) -> &[u32] {
        &self.cell_var[self.cell_ptr[f] as usize..self.cell_ptr[f + 1] as usize]
    }

    #[inline]
    pub fn var_cells(&self, v: usize) -> &[u32] {
        &self.vcell[self.vcell_ptr[v] as usize..self.vcell_ptr[v + 1] as usize]
    }

    /// Syndrome `d1 * e` of a bit-packed error.
    pub fn syndrome_of(&self, err: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; self.n_chk.div_ceil(64).max(1)];
        for c in 0..self.n_chk {
            let mut p = 0u64;
            for e in self.chk_edges(c) {
                let v = self.chk_var[e] as usize;
                p ^= (err[v >> 6] >> (v & 63)) & 1;
            }
            out[c >> 6] |= p << (c & 63);
        }
        out
    }

    /// Logical observable mask `L * e` of a bit-packed error.
    pub fn observables_of(&self, err: &[u64]) -> u64 {
        let mut acc = 0u64;
        for v in 0..self.n_var {
            if (err[v >> 6] >> (v & 63)) & 1 == 1 {
                acc ^= self.obs[v];
            }
        }
        acc
    }
}
