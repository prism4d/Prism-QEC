#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/storage/prism-qec"
cd "$ROOT"

fail() { echo "[QEC-SEGREGATION-FAIL] $*" >&2; exit 1; }

TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ "$TOP" == "$ROOT" ]] || fail "Git top-level ($TOP) != $ROOT"
[[ -d "$ROOT/.git" && ! -L "$ROOT/.git" ]] || fail ".git must be a standalone directory, never a linked worktree"

if [[ "${CUDA_VISIBLE_DEVICES-UNSET}" != "" ]]; then
  fail "Default CUDA_VISIBLE_DEVICES must be empty string (\"\"), got '${CUDA_VISIBLE_DEVICES-UNSET}'"
fi

HITS="$(grep -rnIE   --exclude-dir=target   --exclude-dir=.git   --exclude-dir=.harness   --exclude-dir=.venv   --exclude-dir=.claude   --exclude-dir=harness   --exclude=CLAUDE.md   '(prism-4d|prism_4d|prism-nhs|nhs_rt_full|slipstream|neurospatial|prismview|quarantine_7rpz|mrtx1133|kras_g12d|path\s*=\s*"\.\./)'   "$ROOT" 2>/dev/null || true)"
[[ -z "$HITS" ]] || fail "Cross-repo reference or parent path traversal found:\n$HITS"

while IFS= read -r -d '' link; do
  t="$(realpath -m "$link")"
  case "$t" in
    "$ROOT"/*|"$ROOT") ;;
    *) fail "Symlink escape outside $ROOT: $link -> $t" ;;
  esac
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o -path "$ROOT/.venv" -prune -o -type l -print0)

for prot in   "$ROOT/.claude/settings.json"   "$ROOT/.claude/hooks/pre_tool_guard.py"   "$ROOT/.claude/hooks/stop_gate_auditor.py"   "$ROOT/.claude/hooks/session_context.sh"   "$ROOT/harness/qec-ctl.sh"   "$ROOT/harness/verify_segregation.sh"   "$ROOT/CLAUDE.md"; do
  if [[ -e "$prot" && -w "$prot" ]]; then
    fail "Protected guardrail file is writable ($prot)."
  fi
done

echo "[QEC-SEGREGATION-OK] Standalone git jail, symlink boundary, CPU default mask, and 0555 guardrails verified."
