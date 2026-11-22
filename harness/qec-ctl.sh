#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/storage/prism-qec"
SOCK="qec_socket"
SESS="qec_main"
CONF="$ROOT/.harness/tmux.conf"
LOG_DIR="$ROOT/.harness/logs"
RUN_DIR="$ROOT/.harness/run"
STATE_DIR="$ROOT/.harness/state"
LOCK_FILE="/var/tmp/prism_rtx5080_qec.lock"

export CARGO_TARGET_DIR="$ROOT/target"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-6}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$STATE_DIR"

tmx() { tmux -L "$SOCK" -f "$CONF" "$@"; }

ensure_tmux() {
  if ! tmx has-session -t "$SESS" 2>/dev/null; then
    CUDA_VISIBLE_DEVICES="" tmx new-session -d -s "$SESS" -n "claude" -c "$ROOT"
    CUDA_VISIBLE_DEVICES="" tmx new-window -t "$SESS:1" -n "build" -c "$ROOT"
    CUDA_VISIBLE_DEVICES="" tmx new-window -t "$SESS:2" -n "gpu_bench" -c "$ROOT"
    CUDA_VISIBLE_DEVICES="" tmx new-window -t "$SESS:3" -n "stim_oracle" -c "$ROOT"
    CUDA_VISIBLE_DEVICES="" tmx new-window -t "$SESS:4" -n "audit" -c "$ROOT"
    tmx select-window -t "$SESS:claude"
  fi
}

compute_tree_hash() {
  python3 - "$ROOT" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
files = []
for sub in ("crates", "cuda", "pytools", "Cargo.toml", "Cargo.lock"):
    p = root / sub
    if p.is_file():
        files.append(p)
    elif p.is_dir():
        files.extend(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
files.sort(key=lambda x: str(x.relative_to(root)))
h = hashlib.sha256()
for f in files:
    rel = str(f.relative_to(root)).encode()
    h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode() + b"  " + rel + b"\n")
print(h.hexdigest())
PY
}

check_rtx5080_idle() {
  # 1. Refuse immediately if any sibling molecular/engine GPU binary is alive
  if pgrep -f '(nhs_rt_full|prism_mesh_cells|dstw_export_wt_pathb|prism_neqfep|prism_cavity_persistence|prism-sbdd)' >/dev/null 2>&1; then
    echo "[GPU-LEASE-DENIED] Sibling engine workload is actively running on RTX 5080. Defer Gate 4 until GPU is idle." >&2
    return 1
  fi
  # 2. Refuse if any compute PID already holds /dev/nvidia0
  COMP_APPS="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)"
  if [[ -n "$COMP_APPS" ]]; then
    echo "[GPU-LEASE-DENIED] Active compute process(es) on RTX 5080: $COMP_APPS" >&2
    return 1
  fi
  # 3. Refuse if VRAM > 1200 MiB or GPU utilization > 10%
  read -r USED_MIB GPU_UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ',')
  if (( USED_MIB > 1200 || GPU_UTIL > 10 )); then
    echo "[GPU-LEASE-DENIED] RTX 5080 busy (used=${USED_MIB} MiB, util=${GPU_UTIL}%)." >&2
    return 1
  fi
  return 0
}

run_in_window() {
  local win="$1" gpu_mode="$2" label="$3"
  shift 3
  ensure_tmux
  local safe_label="$(echo "$label" | tr -cd 'a-zA-Z0-9_-')"
  local stamp="qec_${safe_label}_$$_$(date +%s%N)"
  local out_log="$LOG_DIR/${safe_label}.log"
  local rc_file="$RUN_DIR/${stamp}.rc"
  local runner="$RUN_DIR/${stamp}.sh"

  {
    echo '#!/usr/bin/env bash'
    if [[ "$gpu_mode" == "gpu" ]]; then
      echo 'export CUDA_VISIBLE_DEVICES="0"'
    else
      echo 'export CUDA_VISIBLE_DEVICES=""'
    fi
    printf 'export CARGO_TARGET_DIR=%q CARGO_BUILD_JOBS="6" RAYON_NUM_THREADS="6"\n' "$ROOT/target"
    printf 'cd %q\n' "$ROOT"
    printf 'echo %q | tee %q\n' ">>> [QEC-CTL:$safe_label ($gpu_mode)] $*" "$out_log"
    printf '( '; printf '%q ' "$@"; printf ') 2>&1 | tee -a %q\n' "$out_log"
    printf 'rc=${PIPESTATUS[0]}; echo "$rc" > %q\n' "$rc_file"
    printf 'tmux -L %q wait-for -S %q\n' "$SOCK" "$stamp"
  } > "$runner"
  chmod 700 "$runner"
  tmx send-keys -t "$SESS:$win" "bash '$runner'" C-m
  tmx wait-for "$stamp"
  local rc="$(cat "$rc_file" 2>/dev/null || echo 1)"
  rm -f "$rc_file" "$runner"
  echo "=== [QEC-CTL:$safe_label] EXIT=$rc (tail of $out_log) ==="
  tail -n 60 "$out_log"
  return "$rc"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  init-tmux)
    ensure_tmux
    tmx list-windows -t "$SESS"
    ;;
  status)
    ensure_tmux
    echo "=== TMUX ($SOCK / $SESS) ==="
    tmx list-windows -t "$SESS"
    echo "=== RTX 5080 LEASE STATUS ==="
    if check_rtx5080_idle 2>/dev/null; then
      echo "  RTX 5080: IDLE (eligible for gpu-exec / Gate 4)"
    else
      echo "  RTX 5080: RESERVED BY EXTERNAL WORKLOAD (CPU + offline nvcc Gate 0..3 open)"
    fi
    echo "=== GATE STAMPS ==="
    ls -1 "$STATE_DIR"/gate_*.passed 2>/dev/null || echo "  (none)"
    ;;
  exec-sync)
    label="${1:?Usage: qec-ctl.sh exec-sync <label> <cmd...>}"
    shift
    run_in_window "build" "cpu" "$label" "$@"
    ;;
  gpu-exec)
    label="${1:?Usage: qec-ctl.sh gpu-exec <label> <cmd...>}"
    shift
    exec 9>"$LOCK_FILE"
    flock -n 9 || { echo "[GPU-LEASE-DENIED] Another qec gpu-exec holds $LOCK_FILE" >&2; exit 1; }
    check_rtx5080_idle
    run_in_window "gpu_bench" "gpu" "$label" "$@"
    ;;
  verify-gate)
    gate="${1:?Usage: qec-ctl.sh verify-gate <0|1|2|3|4>}"
    CUDA_VISIBLE_DEVICES="" "$ROOT/harness/verify_segregation.sh"
    SLOP="$(grep -rnIE --exclude-dir=target --exclude-dir=.venv '(todo!\(|unimplemented!\(|allow\s*\(\s*(dead_code|unused))' "$ROOT/crates" "$ROOT/cuda" 2>/dev/null || true)"
    [[ -z "$SLOP" ]] || { echo "[GATE-FAIL] Stubs/dead_code detected:\n$SLOP" >&2; exit 1; }

    case "$gate" in
      0)
        "$ROOT/.venv/bin/python" -c "import stim, pymatching, ldpc, scipy, numpy; print('QEC Python stack OK:', stim.__version__)"
        ;;
      1)
        run_in_window "build" "cpu" "gate1_clippy" cargo clippy --workspace --all-targets -- -D warnings
        run_in_window "build" "cpu" "gate1_algebra" cargo test --workspace --lib
        ;;
      2)
        for req_crate in qec-core qec-hodge-bp qec-smc-coset qec-neuro-dem prism-qec-cli; do
          [[ -f "$ROOT/crates/$req_crate/Cargo.toml" ]] || { echo "[GATE-2-FAIL] Missing unified crate $req_crate" >&2; exit 1; }
        done
        run_in_window "build" "cpu" "gate2_clippy" cargo clippy --workspace --all-targets -- -D warnings
        run_in_window "build" "cpu" "gate2_full_cpu" cargo test --workspace --release
        ;;
      3)
        [[ -f "$ROOT/crates/qec-cuda/Cargo.toml" ]] || { echo "[GATE-3-FAIL] Missing crates/qec-cuda" >&2; exit 1; }
        CU_FILES=("$ROOT"/cuda/*.cu)
        [[ -e "${CU_FILES[0]}" ]] || { echo "[GATE-3-FAIL] Missing cuda/*.cu SM120 kernel source" >&2; exit 1; }
        for cu in "${CU_FILES[@]}"; do
          PTX_LOG="$LOG_DIR/ptxas_$(basename "$cu" .cu).log"
          nvcc -arch=sm_120 --cubin --ptxas-options=-v "$cu" -o "/tmp/qec_$(basename "$cu" .cu).cubin" >"$PTX_LOG" 2>&1
          cat "$PTX_LOG"
          if grep -qE '[1-9][0-9]* bytes spill stores' "$PTX_LOG"; then
            echo "[GATE-3-FAIL] Register spill detected in $cu!" >&2
            exit 1
          fi
        done
        run_in_window "build" "cpu" "gate3_release_build" cargo build --workspace --release
        ;;
      4)
        [[ -f "$STATE_DIR/gate_3.passed" ]] || { echo "[GATE-4-FAIL] Stamp Gate 3 before running Gate 4 on GPU." >&2; exit 1; }
        "$ROOT/harness/qec-ctl.sh" gpu-exec "gate4_gpu_tests" cargo test --workspace --release -- --include-ignored
        RECEIPT="$ROOT/runs/bb144_sota_campaign/qec_frontier_receipt.json"
        SEAL="$ROOT/runs/bb144_sota_campaign/chronicle_merkle_seal.json"
        [[ -f "$RECEIPT" && -f "$SEAL" ]] || { echo "[GATE-4-FAIL] Missing $RECEIPT or $SEAL" >&2; exit 1; }
        python3 - "$RECEIPT" "$SEAL" <<'PY'
import json, sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text())
s = json.loads(Path(sys.argv[2]).read_text())

errs = []
# 1. Hardware & Persistent SM120 Graph Proof
if r.get("green_ctx_sms") != [28, 56]:
    errs.append(f"green_ctx_sms={r.get('green_ctx_sms')} != [28, 56]")
if r.get("host_syncs_in_decode_loop") != 0:
    errs.append("host_syncs_in_decode_loop != 0")
gpu_tel = r.get("gpu_sm120_telemetry") or {}
for k in ("p50_us", "p99_us", "p999_us", "shots_per_sec", "speedup_vs_bposd"):
    if not isinstance(gpu_tel.get(k), (int, float)) or gpu_tel[k] <= 0:
        errs.append(f"missing/invalid gpu_sm120_telemetry.{k}")

# 2. Frozen Zero-Tuning Hyperparameter & Blind Seed Proof
cfg_lock = r.get("config_lock") or {}
if not str(cfg_lock.get("config_id", "")).startswith("FROZEN_V") or not cfg_lock.get("config_sha256"):
    errs.append("missing config_lock FROZEN_V1 + config_sha256")
if cfg_lock.get("eval_seed_domain") != "BLIND_HOLDOUT_2026" or cfg_lock.get("per_code_tuning") is not False:
    errs.append("must enforce BLIND_HOLDOUT_2026 with per_code_tuning == false")

# 3. Multi-Code, Multi-Noise Statistical Power & BB72 Non-Regression
cells = r.get("stationary_matrix") or []
codes_seen = {c.get("code_id") for c in cells}
for req_code in ("bb72", "bb90", "bb144_gross", "surface_d5", "surface_d7", "hgp_hamming"):
    if req_code not in codes_seen:
        errs.append(f"stationary_matrix missing required code family '{req_code}'")
if len(cells) < 18:
    errs.append(f"stationary_matrix has only {len(cells)} cells (need >= 18 multi-code x multi-p cells)")

for c in cells:
    cid = f"{c.get('code_id')}@p={c.get('p')}"
    shots = int(c.get("shots", 0))
    p_err = int(c.get("prism_errors", -1))
    o_err = int(c.get("bposd_errors", -1))
    if shots < 50000:
        errs.append(f"{cid}: shots={shots} < 50,000 statistical floor")
    if "clopper_pearson_95" not in c or "mcnemar_p_value" not in c:
        errs.append(f"{cid}: missing clopper_pearson_95 or paired mcnemar_p_value")
    if c.get("code_id") == "bb72":
        fp_wrong = int(c.get("fast_path_false_converged", 999999))
        if p_err > 0 and fp_wrong > max(3, int(0.12 * p_err)):
            errs.append(f"{cid}: BB72 fast_path_false_converged={fp_wrong}/{p_err} still regressing")
        if p_err > o_err and float(c.get("mcnemar_p_value", 0.0)) < 0.05:
            errs.append(f"{cid}: statistically significant BB72 regression vs BP-OSD ({p_err} vs {o_err})")

# 4. Measured Non-Stationary Neuro-DEM (BOCPD + do-TIDE) & HLE Pareto Frontier
neuro = r.get("neuro_dem_evaluation") or {}
for exp_key in ("tls_hot_qubit_dropout", "correlated_crosstalk_and_burst"):
    ex = neuro.get(exp_key) or {}
    if int(ex.get("shots", 0)) < 50000:
        errs.append(f"neuro_dem_evaluation.{exp_key} shots < 50,000")
    if not (float(ex.get("prism_neuro_on_ler", 1.0)) * 3.0 <= float(ex.get("bposd_static_ler", 0.0))):
        errs.append(f"neuro_dem_evaluation.{exp_key} failed >=3x LER improvement over static BP-OSD")

hle = r.get("hle_pareto_frontier") or []
if len(hle) < 4:
    errs.append("hle_pareto_frontier must report >= 4 abort-rate operating points (0.1%, 0.5%, 1.0%, 2.0%)")

# 5. Dual Merkle Seal
if len(s.get("merkle_root_sha256", "")) != 64 or len(s.get("merkle_root_blake3", "")) != 64:
    errs.append("invalid dual SHA-256 + BLAKE3 Merkle roots")

if errs:
    sys.stderr.write("[GATE-4 COMMERCIAL SOTA AUDIT FAILED]:
  - " + "
  - ".join(errs) + "
")
    sys.exit(1)
print("[GATE-4 COMMERCIAL SOTA RECEIPT & DUAL MERKLE SEAL VERIFIED]")
PY
        ;;
      *)
        echo "Unknown gate '$gate' (expected 0..4)" >&2; exit 1 ;;
    esac

    tsha="$(compute_tree_hash)"
    python3 - "$gate" "$tsha" "$STATE_DIR/gate_${gate}.passed" "$STATE_DIR/last_clean_tree" <<'PY'
import datetime, json, sys
from pathlib import Path
gate, tsha, out_p, clean_p = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
out_p.write_text(json.dumps({
    "gate": gate,
    "tree_sha256": tsha,
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2) + "\n")
clean_p.write_text(tsha + "\n")
PY
    echo "[GATE-$gate-PASSED] tree_sha256=$tsha"
    ;;
  *)
    echo "Usage: ./harness/qec-ctl.sh {init-tmux|status|exec-sync|gpu-exec|verify-gate <0..4>}"
    ;;
esac
