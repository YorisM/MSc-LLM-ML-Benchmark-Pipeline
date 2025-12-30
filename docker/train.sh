#!/usr/bin/env bash

echo "argv0=$0"
echo "argv1=$1"
echo "argv2=$2"

set -euo pipefail

# Outputs trace to stderr -- usefull for debugging
# export BASH_XTRACEFD=2   # send `set -x` trace to stderr
# set -x                   # print each command as it runs
# exec 1>&2

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

# ─────────────── hash expected artefacts ────────────────
MODEL_DIR="$(dirname "$PY")"
pushd "$MODEL_DIR" >/dev/null
shopt -s nullglob

# Find a reference artefact to derive the stem (prefer *_model.pkl)
ref=""
for cand in *_model.pkl *_state.pt *_preproc.pkl; do
  [[ -e "$cand" ]] || continue
  ref="$cand"
  break
done

if [[ -z "$ref" ]]; then
  echo "No artefacts found to derive manifest name in $MODEL_DIR" >&2
  exit 1
fi

# Derive the stem from the reference artefact filename
stem="$ref"
case "$ref" in
  *_model.pkl)   stem="${ref%_model.pkl}" ;;
  *_state.pt)    stem="${ref%_state.pt}" ;;
  *_preproc.pkl) stem="${ref%_preproc.pkl}" ;;
  *_loaderspec.json) stem="${ref%__loaderspec.json}" ;;
  *)             stem="${ref%.*}" ;;
esac

# Manifest is "<stem>_manifest.sha256"
MANIFEST="${stem}_manifest.sha256"

# Only hash artefacts that share this stem
files=( "${stem}_state.pt" "${stem}_model.pkl" "${stem}_preproc.pkl" "${stem}_loaderspec.json" )
have=()
for f in "${files[@]}"; do
  [[ -e "$f" ]] && have+=("$f")
done

# Require ALL artefacts are present (except state.pt)
req=( "${stem}_loaderspec.json" "${stem}_model.pkl" "${stem}_preproc.pkl")
for f in "${req[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing required artefact: $f" >&2
    exit 1
  fi
done

if ((${#have[@]}==0)); then
  echo "No artefacts matching ${stem}_* to hash" >&2
  exit 1
fi

sha256sum "${have[@]}" > "$MANIFEST"
shopt -u nullglob
echo "Hash manifest written → $MANIFEST"

popd >/dev/null

# docker build -f docker/Dockerfile.sandbox -t llm-sandbox:latest .