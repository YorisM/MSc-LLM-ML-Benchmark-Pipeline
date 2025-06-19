#!/usr/bin/env bash

echo "argv0=$0"
echo "argv1=$1"
echo "argv2=$2"

set -euo pipefail

PY="$1";  shift || true                         #   first arg = user script
DRY="$(echo "${1:-}" | tr -d '\r\n' | xargs)"   # second arg = --dryrun / --dry-run (trim CR/LF & spaces)

# ─────────────────  safety guard  ───────────────────
if [[ "$PY" == "$0" ]]; then
  echo "Error: PY points to train.sh itself"
  exit 1
fi

# ───────────────  resource limits  ────────────────
if [[ "$DRY" =~ (--dryrun|--dry-run) ]]; then
    TIMEOUT="${DRYRUN_TIMEOUT_S}"
else
    TIMEOUT="${TRAIN_TIMEOUT_S}"
fi

# Default to 0 (= unlimited) if the env-var is missing
: "${MEMORY_LIMIT_GB:=0}"
: "${PIDS_LIMIT:=0}"

ulimit -t "$TIMEOUT"                      # CPU-seconds

# virtual-memory limit: only apply when > 0
if (( MEMORY_LIMIT_GB > 0 )); then
    ulimit -v $((MEMORY_LIMIT_GB * 1024 * 1024))
fi

# open-files / processes limit: only apply when > 0
if (( PIDS_LIMIT > 0 )); then
    ulimit -n "$PIDS_LIMIT"
fi

# ───────────────  run user script  ────────────────
timeout --signal=KILL "${TIMEOUT}s" python "$PY" $DRY
python_rc=$?

#   If user code crashed, propagate failure to container exit-code
if [[ $python_rc -ne 0 ]]; then
  echo "User script returned error code $python_rc"
  exit $python_rc
fi

# ───────────────  skip hashing during dry-run  ────────────────
if [[ "$DRY" =~ (--dryrun|--dry-run) ]]; then
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
