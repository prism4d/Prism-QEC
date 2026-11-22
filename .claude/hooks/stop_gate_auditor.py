#!/usr/bin/env python3
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/storage/prism-qec")
STATE = ROOT / ".harness" / "state"

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

if payload.get("stop_hook_active"):
    sys.exit(0)


def tree_hash() -> str:
    files = []
    for sub in ("crates", "cuda", "pytools", "Cargo.toml", "Cargo.lock"):
        p = ROOT / sub
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    files.sort(key=lambda x: str(x.relative_to(ROOT)))
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(ROOT)).encode()
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"  " + rel + b"\n")
    return h.hexdigest()


def rtx5080_idle() -> bool:
    pg = subprocess.run(
        ["pgrep", "-f", "(nhs_rt_full|prism_mesh_cells|dstw_export_wt_pathb|prism_neqfep|prism_cavity_persistence)"],
        capture_output=True, text=True,
    )
    if pg.returncode == 0 and pg.stdout.strip():
        return False
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if smi.returncode == 0 and smi.stdout.strip():
        parts = [int(x.strip()) for x in smi.stdout.splitlines()[0].split(",")]
        return parts[0] <= 1200 and parts[1] <= 10
    return False


# 1. Orphan Rust module check
for rs in (ROOT / "crates").rglob("*.rs"):
    rel = str(rs.relative_to(ROOT))
    if "/bin/" in rel or "/tests/" in rel:
        continue
    stem = rs.stem
    if stem in ("lib", "main", "mod", "build"):
        continue
    hit = subprocess.run(
        ["grep", "-rnE", rf"\bmod\s+{re.escape(stem)}\b", str(ROOT / "crates")],
        capture_output=True, text=True,
    )
    if hit.returncode != 0:
        sys.stderr.write(f"[STOP BLOCKED — ORPHAN MODULE]: '{rel}' is not wired via 'mod {stem};'.\n")
        sys.exit(2)

# 2. Segregation & cargo check
cur_sha = tree_hash()
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
env["CARGO_BUILD_JOBS"] = "6"
seg = subprocess.run([str(ROOT / "harness" / "verify_segregation.sh")], env=env, capture_output=True, text=True)
if seg.returncode != 0:
    sys.stderr.write(f"[STOP BLOCKED — SEGREGATION FAIL]:\n{seg.stderr}\n")
    sys.exit(2)

chk = subprocess.run(
    ["cargo", "check", "--workspace", "--all-targets"],
    cwd=str(ROOT), env=env, capture_output=True, text=True,
)
if chk.returncode != 0:
    tail = "\n".join(chk.stderr.splitlines()[-35:])
    sys.stderr.write(f"[STOP BLOCKED — CARGO CHECK FAILED]:\n{tail}\n")
    sys.exit(2)
(STATE / "last_clean_tree").write_text(cur_sha + "\n")

# 3. Mandatory Full Gate Enforcement (Gates 0..3 ALWAYS required; Gate 4 required whenever RTX 5080 is idle!)
required_gates = [0, 1, 2, 3] + ([4] if rtx5080_idle() else [])
missing_or_stale = []
for g in required_gates:
    stamp = STATE / f"gate_{g}.passed"
    if not stamp.is_file():
        missing_or_stale.append(f"gate_{g}(missing)")
    elif g >= 1:
        s_data = json.loads(stamp.read_text())
        if s_data.get("tree_sha256") != cur_sha:
            missing_or_stale.append(f"gate_{g}(stale)")

if missing_or_stale:
    sys.stderr.write(
        f"[STOP BLOCKED — UNSTAMPED/STALE COMMERCIAL SOTA GATES: {', '.join(missing_or_stale)}]: "
        "Read  and run  "
        "to completion before stopping (no 2,000-shot partial CPU handoffs permitted).\n"
    )
    sys.exit(2)

sys.exit(0)
