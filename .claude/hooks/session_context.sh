#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/storage/prism-qec"
STAMPS="$(ls -1 "$ROOT/.harness/state"/gate_*.passed 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ' || true)"
[[ -n "$STAMPS" ]] || STAMPS="none"
echo "[PRISM-QEC SOVEREIGN HARNESS ACTIVE]"
echo "- Isolated Repo: $ROOT (100% air-gapped from all sibling repos & molecular vaults)"
echo "- Tmux Control: ./harness/qec-ctl.sh (socket: qec_socket, session: qec_main)"
echo "- Python QEC Env: $ROOT/.venv/bin/python (stim, ldpc, pymatching, sinter, scipy)"
echo "- GPU Arbiter: Default CUDA_VISIBLE_DEVICES=\"\"; offline nvcc -arch=sm_120 allowed anytime (Gate 3); live RTX 5080 gated via './harness/qec-ctl.sh gpu-exec' (Gate 4)"
echo "- Stamped Gates: ${STAMPS}"
