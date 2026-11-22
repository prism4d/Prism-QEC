//! Chronicle seal: dual SHA-256 + BLAKE3 Merkle trees over a set of files.
//!
//! Leaf   `= H(0x00 || path || 0x00 || H(file bytes))`
//! Node   `= H(0x01 || left || right)`; an unpaired node is promoted unchanged.
//! Leaves are ordered by their path relative to the sealed root directory.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SEAL_SCHEMA: &str = "prism.qec.v1.chronicle_merkle_seal";
pub const SEAL_FILE: &str = "chronicle_merkle_seal.json";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SealLeaf {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub blake3: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChronicleSeal {
    pub schema: String,
    pub leaf_count: usize,
    pub merkle_root_sha256: String,
    pub merkle_root_blake3: String,
    pub sealed_utc: String,
    pub leaves: Vec<SealLeaf>,
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn unhex32(s: &str) -> Result<[u8; 32]> {
    ensure!(s.len() == 64, "digest must be 64 hex characters");
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).context("bad hex digit")?;
    }
    Ok(out)
}

/// Streams a file through both hash functions.
pub fn hash_file(path: &Path) -> Result<(u64, [u8; 32], [u8; 32])> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut sha = Sha256::new();
    let mut b3 = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
        b3.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, sha.finalize().into(), *b3.finalize().as_bytes()))
}

fn sha_leaf(path: &str, digest: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([0u8]);
    h.update(path.as_bytes());
    h.update([0u8]);
    h.update(digest);
    h.finalize().into()
}

fn b3_leaf(path: &str, digest: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&[0u8]);
    h.update(path.as_bytes());
    h.update(&[0u8]);
    h.update(digest);
    *h.finalize().as_bytes()
}

fn fold_root(mut level: Vec<[u8; 32]>, node: impl Fn(&[u8; 32], &[u8; 32]) -> [u8; 32]) -> [u8; 32] {
    if level.is_empty() {
        return [0u8; 32];
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 { node(&pair[0], &pair[1]) } else { pair[0] });
        }
        level = next;
    }
    level[0]
}

/// Both Merkle roots of a leaf list (leaves must already be in canonical path order).
pub fn merkle_roots(leaves: &[SealLeaf]) -> Result<(String, String)> {
    let mut sha_level = Vec::with_capacity(leaves.len());
    let mut b3_level = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        sha_level.push(sha_leaf(&leaf.path, &unhex32(&leaf.sha256)?));
        b3_level.push(b3_leaf(&leaf.path, &unhex32(&leaf.blake3)?));
    }
    let sha_root = fold_root(sha_level, |l, r| {
        let mut h = Sha256::new();
        h.update([1u8]);
        h.update(l);
        h.update(r);
        h.finalize().into()
    });
    let b3_root = fold_root(b3_level, |l, r| {
        let mut h = blake3::Hasher::new();
        h.update(&[1u8]);
        h.update(l);
        h.update(r);
        *h.finalize().as_bytes()
    });
    Ok((hex(&sha_root), hex(&b3_root)))
}

fn collect_files(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_files(&path, root, out)?;
        } else if ft.is_file() {
            if path.parent() == Some(root) && path.file_name().and_then(|s| s.to_str()) == Some(SEAL_FILE) {
                continue;
            }
            out.push(path);
        }
    }
    Ok(())
}

fn rel_path(path: &Path, root: &Path) -> Result<String> {
    let rel = path.strip_prefix(root).context("sealed file escapes the root directory")?;
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    Ok(parts.join("/"))
}

/// Hashes every regular file below `root` (excluding the seal itself) into a seal.
pub fn seal_directory(root: &Path, sealed_utc: &str) -> Result<ChronicleSeal> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    ensure!(!files.is_empty(), "nothing to seal under {}", root.display());
    let mut leaves = Vec::with_capacity(files.len());
    for f in &files {
        let (bytes, sha, b3) = hash_file(f)?;
        leaves.push(SealLeaf { path: rel_path(f, root)?, bytes, sha256: hex(&sha), blake3: hex(&b3) });
    }
    leaves.sort_by(|a, b| a.path.cmp(&b.path));
    let (merkle_root_sha256, merkle_root_blake3) = merkle_roots(&leaves)?;
    Ok(ChronicleSeal {
        schema: SEAL_SCHEMA.to_string(),
        leaf_count: leaves.len(),
        merkle_root_sha256,
        merkle_root_blake3,
        sealed_utc: sealed_utc.to_string(),
        leaves,
    })
}

/// Seals `root` and writes `chronicle_merkle_seal.json` into it.
pub fn write_seal(root: &Path, sealed_utc: &str) -> Result<ChronicleSeal> {
    let seal = seal_directory(root, sealed_utc)?;
    let text = serde_json::to_string_pretty(&seal)? + "\n";
    std::fs::write(root.join(SEAL_FILE), text)?;
    Ok(seal)
}

/// Re-hashes every sealed file and recomputes both roots; fails on any divergence,
/// including files added to or removed from the sealed directory.
pub fn verify_seal(root: &Path) -> Result<ChronicleSeal> {
    let text = std::fs::read_to_string(root.join(SEAL_FILE)).context("reading seal")?;
    let seal: ChronicleSeal = serde_json::from_str(&text).context("parsing seal")?;
    ensure!(seal.schema == SEAL_SCHEMA, "unexpected seal schema '{}'", seal.schema);
    let fresh = seal_directory(root, &seal.sealed_utc)?;
    if fresh.leaves != seal.leaves {
        for (a, b) in fresh.leaves.iter().zip(seal.leaves.iter()) {
            if a != b {
                bail!("seal mismatch at '{}' (recorded '{}')", a.path, b.path);
            }
        }
        bail!("sealed file set changed: {} files now, {} recorded", fresh.leaves.len(), seal.leaves.len());
    }
    ensure!(fresh.merkle_root_sha256 == seal.merkle_root_sha256, "SHA-256 Merkle root mismatch");
    ensure!(fresh.merkle_root_blake3 == seal.merkle_root_blake3, "BLAKE3 Merkle root mismatch");
    Ok(seal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qec_merkle_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        dir
    }

    #[test]
    fn known_answer_digests() {
        let dir = scratch("kat");
        std::fs::write(dir.join("abc.txt"), b"abc").unwrap();
        let (n, sha, b3) = hash_file(&dir.join("abc.txt")).unwrap();
        assert_eq!(n, 3);
        assert_eq!(hex(&sha), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&b3), "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn seal_verifies_and_detects_tampering() {
        let dir = scratch("tamper");
        std::fs::write(dir.join("a.bin"), [1u8, 2, 3]).unwrap();
        std::fs::write(dir.join("sub/b.bin"), [4u8; 100]).unwrap();
        std::fs::write(dir.join("c.json"), b"{}").unwrap();
        let seal = write_seal(&dir, "2026-01-01T00:00:00Z").unwrap();
        assert_eq!(seal.leaf_count, 3);
        assert_eq!(seal.merkle_root_sha256.len(), 64);
        assert_eq!(seal.merkle_root_blake3.len(), 64);
        assert_ne!(seal.merkle_root_sha256, seal.merkle_root_blake3);
        assert_eq!(verify_seal(&dir).unwrap(), seal);
        // flipping one byte breaks the seal
        std::fs::write(dir.join("sub/b.bin"), [5u8; 100]).unwrap();
        assert!(verify_seal(&dir).is_err());
        std::fs::write(dir.join("sub/b.bin"), [4u8; 100]).unwrap();
        assert!(verify_seal(&dir).is_ok());
        // adding a file breaks the seal
        std::fs::write(dir.join("extra.txt"), b"x").unwrap();
        assert!(verify_seal(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn root_depends_on_leaf_order_and_path() {
        let leaf = |p: &str, b: u8| SealLeaf {
            path: p.to_string(),
            bytes: 1,
            sha256: hex(&[b; 32]),
            blake3: hex(&[b ^ 0xFF; 32]),
        };
        let a = merkle_roots(&[leaf("x", 1), leaf("y", 2), leaf("z", 3)]).unwrap();
        let b = merkle_roots(&[leaf("y", 2), leaf("x", 1), leaf("z", 3)]).unwrap();
        let c = merkle_roots(&[leaf("x", 1), leaf("y", 2), leaf("w", 3)]).unwrap();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
