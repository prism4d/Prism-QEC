//! Exact F2 linear algebra: bit-packed dense matrices and sorted-index sparse matrices.

/// Parity of the F2 inner product of two bit-packed vectors: `popcount(a & b) & 1`.
#[inline]
pub fn parity_and(a: &[u64], b: &[u64]) -> u8 {
    let mut acc = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc ^= (x & y).count_ones();
    }
    (acc & 1) as u8
}

#[inline]
pub fn words_for(bits: usize) -> usize {
    bits.div_ceil(64)
}

#[inline]
pub fn get_bit(words: &[u64], i: usize) -> bool {
    (words[i >> 6] >> (i & 63)) & 1 == 1
}

#[inline]
pub fn set_bit(words: &mut [u64], i: usize) {
    words[i >> 6] |= 1u64 << (i & 63);
}

#[inline]
pub fn flip_bit(words: &mut [u64], i: usize) {
    words[i >> 6] ^= 1u64 << (i & 63);
}

pub fn weight(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

/// Dense row-major bit-packed F2 matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMat {
    pub rows: usize,
    pub cols: usize,
    pub wpr: usize,
    pub data: Vec<u64>,
}

impl BitMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let wpr = words_for(cols).max(1);
        Self { rows, cols, wpr, data: vec![0u64; rows * wpr] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i);
        }
        m
    }

    pub fn from_rows(cols: usize, rows: &[Vec<u32>]) -> Self {
        let mut m = Self::zeros(rows.len(), cols);
        for (i, r) in rows.iter().enumerate() {
            for &c in r {
                m.flip(i, c as usize);
            }
        }
        m
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.wpr..(i + 1) * self.wpr]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [u64] {
        &mut self.data[i * self.wpr..(i + 1) * self.wpr]
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> bool {
        (self.data[i * self.wpr + (j >> 6)] >> (j & 63)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize) {
        self.data[i * self.wpr + (j >> 6)] |= 1u64 << (j & 63);
    }

    #[inline]
    pub fn flip(&mut self, i: usize, j: usize) {
        self.data[i * self.wpr + (j >> 6)] ^= 1u64 << (j & 63);
    }

    /// `row[dst] ^= row[src]`.
    pub fn xor_row(&mut self, src: usize, dst: usize) {
        debug_assert_ne!(src, dst);
        let w = self.wpr;
        let (a, b) = if src < dst {
            let (lo, hi) = self.data.split_at_mut(dst * w);
            (&lo[src * w..(src + 1) * w], &mut hi[..w])
        } else {
            let (lo, hi) = self.data.split_at_mut(src * w);
            (&hi[..w], &mut lo[dst * w..(dst + 1) * w])
        };
        for (d, s) in b.iter_mut().zip(a.iter()) {
            *d ^= *s;
        }
    }

    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let w = self.wpr;
        for k in 0..w {
            self.data.swap(a * w + k, b * w + k);
        }
    }

    pub fn row_support(&self, i: usize) -> Vec<u32> {
        (0..self.cols).filter(|&j| self.get(i, j)).map(|j| j as u32).collect()
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.get(i, j) {
                    t.set(j, i);
                }
            }
        }
        t
    }

    /// F2 product `self * other` computed as popcount parities against `other^T` rows.
    pub fn mul(&self, other: &BitMat) -> BitMat {
        assert_eq!(self.cols, other.rows);
        let ot = other.transpose();
        let mut out = BitMat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            let ri = self.row(i);
            for j in 0..other.cols {
                if parity_and(ri, ot.row(j)) == 1 {
                    out.set(i, j);
                }
            }
        }
        out
    }

    /// F2 matrix-vector product with a bit-packed vector of `cols` bits.
    pub fn mul_vec(&self, v: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; words_for(self.rows).max(1)];
        for i in 0..self.rows {
            if parity_and(self.row(i), v) == 1 {
                set_bit(&mut out, i);
            }
        }
        out
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&w| w == 0)
    }

    /// In-place reduced row echelon form restricted to the first `col_limit` columns.
    /// Returns the pivot column of each of the leading `rank` rows.
    pub fn rref_in_place(&mut self, col_limit: usize) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut r = 0usize;
        for c in 0..col_limit.min(self.cols) {
            if r >= self.rows {
                break;
            }
            let mut p = None;
            for i in r..self.rows {
                if self.get(i, c) {
                    p = Some(i);
                    break;
                }
            }
            let Some(p) = p else { continue };
            self.swap_rows(r, p);
            for i in 0..self.rows {
                if i != r && self.get(i, c) {
                    self.xor_row(r, i);
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }

    pub fn rank(&self) -> usize {
        let mut m = self.clone();
        m.rref_in_place(self.cols).len()
    }

    /// Basis (as rows) of the right null space `{v : self * v = 0}`.
    pub fn nullspace(&self) -> BitMat {
        let mut m = self.clone();
        let pivots = m.rref_in_place(self.cols);
        let mut is_pivot = vec![false; self.cols];
        for &p in &pivots {
            is_pivot[p] = true;
        }
        let free: Vec<usize> = (0..self.cols).filter(|&c| !is_pivot[c]).collect();
        let mut out = BitMat::zeros(free.len(), self.cols);
        for (k, &f) in free.iter().enumerate() {
            out.set(k, f);
            for (r, &p) in pivots.iter().enumerate() {
                if m.get(r, f) {
                    out.set(k, p);
                }
            }
        }
        out
    }

    /// Stacks `other` below `self`.
    pub fn vstack(&self, other: &BitMat) -> BitMat {
        assert_eq!(self.cols, other.cols);
        let mut out = BitMat::zeros(self.rows + other.rows, self.cols);
        out.data[..self.data.len()].copy_from_slice(&self.data);
        out.data[self.data.len()..].copy_from_slice(&other.data);
        out
    }

    pub fn to_bytes_dense(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[i * self.cols + j] = self.get(i, j) as u8;
            }
        }
        out
    }
}

/// Sparse F2 matrix in CSR form with strictly increasing column indices per row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseF2 {
    pub rows: usize,
    pub cols: usize,
    pub ptr: Vec<u32>,
    pub idx: Vec<u32>,
}

impl SparseF2 {
    /// Builds from row supports; repeated indices cancel mod 2.
    pub fn from_rows(cols: usize, rows: &[Vec<u32>]) -> Self {
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        let mut idx = Vec::new();
        ptr.push(0u32);
        for r in rows {
            let mut s = r.clone();
            s.sort_unstable();
            let mut k = 0;
            while k < s.len() {
                let mut run = 1;
                while k + run < s.len() && s[k + run] == s[k] {
                    run += 1;
                }
                if run % 2 == 1 {
                    assert!((s[k] as usize) < cols, "column index out of range");
                    idx.push(s[k]);
                }
                k += run;
            }
            ptr.push(idx.len() as u32);
        }
        Self { rows: rows.len(), cols, ptr, idx }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[u32] {
        &self.idx[self.ptr[i] as usize..self.ptr[i + 1] as usize]
    }

    pub fn nnz(&self) -> usize {
        self.idx.len()
    }

    pub fn transpose(&self) -> Self {
        let mut counts = vec![0u32; self.cols + 1];
        for &c in &self.idx {
            counts[c as usize + 1] += 1;
        }
        for c in 0..self.cols {
            counts[c + 1] += counts[c];
        }
        let ptr = counts.clone();
        let mut idx = vec![0u32; self.idx.len()];
        let mut cursor = counts;
        for r in 0..self.rows {
            for &c in self.row(r) {
                idx[cursor[c as usize] as usize] = r as u32;
                cursor[c as usize] += 1;
            }
        }
        Self { rows: self.cols, cols: self.rows, ptr, idx }
    }

    pub fn to_dense(&self) -> BitMat {
        let mut m = BitMat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for &c in self.row(i) {
                m.set(i, c as usize);
            }
        }
        m
    }

    /// `self * v` for a bit-packed vector `v` of `cols` bits.
    pub fn mul_vec(&self, v: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; words_for(self.rows).max(1)];
        for i in 0..self.rows {
            let mut p = false;
            for &c in self.row(i) {
                p ^= get_bit(v, c as usize);
            }
            if p {
                set_bit(&mut out, i);
            }
        }
        out
    }

    /// True iff `self * other == 0 (mod 2)`; `other` is given by its transpose in CSR form
    /// (`other_t.row(j)` lists the rows of column `j` of `other`).
    pub fn product_is_zero(&self, other_t: &SparseF2) -> bool {
        assert_eq!(self.cols, other_t.cols);
        let self_t = self.transpose();
        let mut acc = vec![0u8; self.rows];
        for j in 0..other_t.rows {
            let mut touched = Vec::new();
            for &mid in other_t.row(j) {
                for &r in self_t.row(mid as usize) {
                    acc[r as usize] ^= 1;
                    touched.push(r as usize);
                }
            }
            let mut bad = false;
            for &r in &touched {
                bad |= acc[r] != 0;
                acc[r] = 0;
            }
            if bad {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 11
    }

    fn random_mat(rows: usize, cols: usize, seed: u64) -> BitMat {
        let mut s = seed;
        let mut m = BitMat::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if lcg(&mut s) & 3 == 0 {
                    m.set(i, j);
                }
            }
        }
        m
    }

    #[test]
    fn nullspace_is_annihilated_and_rank_nullity_holds() {
        for seed in 1..6u64 {
            let m = random_mat(23, 70, seed);
            let ns = m.nullspace();
            assert_eq!(ns.rows + m.rank(), 70);
            assert!(m.mul(&ns.transpose()).is_zero());
            assert_eq!(ns.rank(), ns.rows);
        }
    }

    #[test]
    fn popcount_product_matches_naive_product() {
        let a = random_mat(17, 130, 11);
        let b = random_mat(130, 29, 12);
        let c = a.mul(&b);
        for i in 0..17 {
            for j in 0..29 {
                let mut acc = false;
                for k in 0..130 {
                    acc ^= a.get(i, k) & b.get(k, j);
                }
                assert_eq!(c.get(i, j), acc);
            }
        }
    }

    #[test]
    fn sparse_transpose_and_zero_product() {
        let rows = vec![vec![0u32, 1], vec![1, 2], vec![0, 2]];
        let d1 = SparseF2::from_rows(3, &rows);
        assert_eq!(d1.transpose().transpose(), d1);
        // the all-ones vector is a cycle of the triangle graph
        let cyc_t = SparseF2::from_rows(3, &[vec![0u32, 1, 2]]);
        assert!(d1.product_is_zero(&cyc_t));
        let not_cyc_t = SparseF2::from_rows(3, &[vec![0u32, 1]]);
        assert!(!d1.product_is_zero(&not_cyc_t));
    }

    #[test]
    fn repeated_indices_cancel_mod_two() {
        let m = SparseF2::from_rows(5, &[vec![3, 1, 3, 4, 4, 4]]);
        assert_eq!(m.row(0), &[1, 4]);
    }
}
