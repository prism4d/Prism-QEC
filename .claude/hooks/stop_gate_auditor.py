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
        sys.stderr.write(f"[STOP BLOCKED — ORPHAN MODULE]: '{rel}' is never wired via 'mod {stem};'.\n")
        sys.exit(2)

# 2. Segregation & cargo check when tree changes
cur_sha = tree_hash()
last_sha = (STATE / "last_clean_tree").read_text().strip() if (STATE / "last_clean_tree").is_file() else ""
if cur_sha != last_sha:
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

# 3. Verify any claimed gate in assistant message matches current tree_sha256
last_msg = payload.get("last_assistant_message") or ""
for g in range(5):
    if re.search(rf"gate\s+{g}\s+(is\s+)?(complete|passed|done|verified|stamped)", last_msg, re.I):
        stamp = STATE / f"gate_{g}.passed"
        if not stamp.is_file():
            sys.stderr.write(f"[STOP BLOCKED]: Claimed Gate {g} passed, but {stamp} is missing.\n")
            sys.exit(2)
        if g >= 1:
            s_data = json.loads(stamp.read_text())
            if s_data.get("tree_sha256") != cur_sha:
                sys.stderr.write(f"[STOP BLOCKED]: Gate {g} stamp is stale vs current tree ({cur_sha[:12]}). Re-run verify-gate {g}.\n")
                sys.exit(2)

sys.exit(0)
