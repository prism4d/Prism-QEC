#!/usr/bin/env python3
import json
import os
import re
import sys

ROOT = "/mnt/storage/prism-qec"

FORBIDDEN_PAT = re.compile(
    r"(prism-4d|prism_4d|prism-nhs|prism_nhs|nhs_rt_full|slipstream|sdst|"
    r"neurospatial|prismview|quarantine_7rpz|7rpz|mrtx1133|kras_g12d|ptp1b|"
    r"stat3_6tlc|tead4_8a8r|/mnt/storage/(runs|inputs|weights|PRISM_ENGINE)|/mnt/\.vault)",
    re.IGNORECASE,
)

READ_ALLOWED_OUTSIDE = (
    "/tmp/qec_",
    "/usr/",
    "/etc/",
    "/lib",
    "/home/diddy/.cargo/",
    "/home/diddy/.rustup/",
)


def block(msg: str) -> None:
    sys.stderr.write(f"[PRISM-QEC SOVEREIGN GUARD — BLOCKED]: {msg}\n")
    sys.exit(2)


try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)

tool = payload.get("tool_name", "")
inp = payload.get("tool_input", {}) or {}


def check_path(raw: str, write_mode: bool) -> None:
    if not raw:
        return
    if FORBIDDEN_PAT.search(raw):
        block(f"Path '{raw}' references a forbidden external repository/dataset (QEC-SEG-01).")
    resolved = os.path.realpath(os.path.expanduser(raw))
    if FORBIDDEN_PAT.search(resolved):
        block(f"Resolved path '{resolved}' escapes into a forbidden workspace (QEC-SEG-01).")
    if write_mode:
        imm = (
            os.path.join(ROOT, ".claude"),
            os.path.join(ROOT, "harness"),
            os.path.join(ROOT, "CLAUDE.md"),
            os.path.join(ROOT, ".git"),
        )
        if any(resolved == p or resolved.startswith(p + "/") for p in imm):
            block(f"Guardrail path '{resolved}' is immutable (HAR-05).")
        if not (resolved.startswith(ROOT + "/") or resolved.startswith("/tmp/qec_")):
            block(f"Write target '{resolved}' escapes QEC workspace '{ROOT}'.")
    else:
        if resolved == ROOT or resolved.startswith(ROOT + "/"):
            return
        if any(resolved.startswith(p) for p in READ_ALLOWED_OUTSIDE):
            return
        block(f"Read/search target '{resolved}' escapes QEC workspace '{ROOT}' (QEC-SEG-01).")


if tool in ("Read", "Glob", "Grep"):
    check_path(inp.get("file_path") or inp.get("path") or "", write_mode=False)
    sys.exit(0)

if tool in ("Write", "Edit", "MultiEdit"):
    raw_fp = inp.get("file_path") or ""
    check_path(raw_fp, write_mode=True)
    fp = os.path.realpath(os.path.expanduser(raw_fp))
    base = os.path.basename(fp)

    if any(s in fp for s in ("/examples/", "/benches/", "/mocks/")):
        block(f"Standalone island '{fp}' is forbidden. Implement directly in canonical crates/modules.")
    if re.search(r"(?i)(audit|status_board|handoff|tracker).*\.(md|txt|json)$", base):
        block(f"Tracker/handoff file '{base}' is forbidden. Deliver compiled, gate-verified code.")
    if tool == "Write" and not os.path.exists(fp):
        if re.search(r"(?i)(_v\d+|_new|_old|_bak|_copy|_mock|_stub|_tmp)\.(rs|cu|cuh|py|toml)$", base):
            block(f"Versioned/mock filename '{base}' is forbidden. Edit canonical modules in place.")
        if fp.startswith(ROOT + "/crates/") and base == "Cargo.toml":
            rel = fp[len(ROOT) + 1:].split("/")
            if len(rel) != 3:
                block(f"Nested Cargo.toml '{fp}' is forbidden. Keep flat crates/<name>/Cargo.toml.")

    added = []
    if tool == "Write":
        old_s = set(open(fp, errors="ignore").read().splitlines()) if os.path.exists(fp) else set()
        added = [ln for ln in (inp.get("content") or "").splitlines() if ln not in old_s]
    elif tool == "Edit":
        old_s = set((inp.get("old_string") or "").splitlines())
        added = [ln for ln in (inp.get("new_string") or "").splitlines() if ln not in old_s]
    elif tool == "MultiEdit":
        for ed in inp.get("edits") or []:
            old_s = set((ed.get("old_string") or "").splitlines())
            added.extend(ln for ln in (ed.get("new_string") or "").splitlines() if ln not in old_s)

    blob = "\n".join(added)
    if FORBIDDEN_PAT.search(blob):
        block("Source edit mentions forbidden sibling-repo/molecular identifiers (QEC-SEG-01).")
    if "/tests/" not in fp and fp.endswith((".rs", ".cu", ".cuh", ".py")) and blob:
        if re.search(r"allow\s*\(\s*(dead_code|unused)|\b(todo|unimplemented)!\s*\(|\"NOT_BUILT\"", blob):
            block("Stubs (, , ) are forbidden.")
        if fp.endswith((".cu", ".cuh")) and re.search(r"\bcudaDeviceSynchronize\s*\(", blob):
            block("Global cudaDeviceSynchronize() in CUDA kernels is forbidden; use stream/graph events.")
    sys.exit(0)

if tool == "Bash":
    cmd = inp.get("command") or ""
    if FORBIDDEN_PAT.search(cmd):
        block("Command references forbidden external repository/dataset paths (QEC-SEG-01).")
    if re.search(r"\bgit\s+(worktree|submodule|clone|init)\b", cmd):
        block("git worktree/submodule/clone/init are forbidden. Operate strictly in this flat repo.")
    if re.search(r"(\.claude/|harness/|CLAUDE\.md)", cmd) and re.search(
        r"\b(rm|mv|cp|chmod|chown|sed\s+-i|tee|truncate|unlink)\b|>", cmd
    ):
        block("Mutating .claude/, harness/, or CLAUDE.md is forbidden (HAR-05).")
    if re.search(r"(^|[;&|]\s*)tmux\b", cmd) and not re.search(r"tmux\s+-L\s+qec_socket\b", cmd):
        block("Bare tmux or foreign sockets are forbidden. Use './harness/qec-ctl.sh' exclusively.")
    if re.search(r"\b(pkill|killall|fuser\s+-k)\b|kill\s+-9|\b(nohup|disown)\b", cmd):
        block("Untracked backgrounding or blind process kills are forbidden. Use './harness/qec-ctl.sh'.")
    if re.search(r"(^|[;&|]\s*)sleep\s+([3-9]|[1-9][0-9]+)\b", cmd):
        block("Lazy sleep >= 3s is forbidden. Use './harness/qec-ctl.sh exec-sync'.")
    if re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*[\"']?[0-9]", cmd) and "qec-ctl.sh" not in cmd:
        block("Direct GPU exposure is forbidden (QEC-SEG-03). Run GPU workloads ONLY via './harness/qec-ctl.sh gpu-exec' so RTX 5080 contention is checked.")
    if re.search(r"\.harness/state/gate_[0-9]+\.passed", cmd) and "verify-gate" not in cmd:
        block("Gate stamps can ONLY be minted by './harness/qec-ctl.sh verify-gate <0..4>'.")
    sys.exit(0)
