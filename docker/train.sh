#!/usr/bin/env bash

echo "argv0=$0"
echo "argv1=$1"
echo "argv2=$2"

set -euo pipefail

PY="$1";  shift || true       #   first arg = user script
DRY="${1:-}"                  #   second arg = --dryrun (optional)

# ─────────────────  safety guard  ───────────────────
if [[ "$PY" == "$0" ]]; then
  echo "Error: PY points to train.sh itself"
  exit 1
fi

# ───────────────  resource limits  ────────────────
if [[ "$DRY" =~ (--dryrun|--dry-run) ]]; then
  TIMEOUT="${DRYRUN_TIMEOUT_S:-600}"
else
  TIMEOUT="${TRAIN_TIMEOUT_S:-7200}"
fi   # seconds
MEM_GB="${MEMORY_LIMIT_GB:-8}"
PIDS_LIMIT="${PIDS_LIMIT:-512}"

ulimit -t  "$TIMEOUT"                  # CPU-seconds
ulimit -v  $((MEM_GB * 1024 * 1024))   # address-space KB
ulimit -n  "$PIDS_LIMIT"               # open files / pids

# ───────────────  resource runtime diagnostics  ────────────────
echo "Runtime Resource Diagnostics:"
echo "TIMEOUT       = $TIMEOUT seconds"
echo "MEMORY_LIMIT  = ${MEM_GB} GB"
echo "PIDS_LIMIT    = ${PIDS_LIMIT}"
echo "---------- ulimit -a ----------"
ulimit -a
echo "---------- free -h (if available) ----------"
echo "Mem cgroup limit: $(cat /sys/fs/cgroup/memory.max)"
echo "Mem in use     : $(cat /sys/fs/cgroup/memory.current)"

# ───────────────  run user script  ────────────────
timeout --signal=KILL "${TIMEOUT}s" python "$PY" $DRY
python_rc=$?

#   If user code crashed, propagate failure to container exit-code
if [[ $python_rc -ne 0 ]]; then
  echo "User script returned error code $python_rc"
  exit $python_rc
fi

# ───────────────  skip hashing during dry-run  ────────────────
if [[ "$DRY" == "--dryrun" ]]; then
  echo "Dry-run: skipping artefact manifest"
  exit 0
fi

# ───────────────  hash expected artefacts  ────────────────
MODEL_DIR="$(dirname "$PY")"                 # …/outputs/<DATE>/<CHALLENGE>/<Q>/<MODEL>
MODEL_NAME="$(basename "$MODEL_DIR")"        # <MODEL>

pushd "$MODEL_DIR" >/dev/null
shopt -s nullglob
sha256sum *_state.pt *_model.pkl *_preproc.pkl > "${MODEL_NAME}_manifest.sha256"
shopt -u nullglob
echo "Hash manifest written → ${MODEL_NAME}_manifest.sha256"
popd >/dev/null
