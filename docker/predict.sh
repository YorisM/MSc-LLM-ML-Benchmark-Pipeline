#!/usr/bin/env bash
# Usage:  predict.sh <artefact_dir>/<something>_model.pkl
set -euo pipefail

# Outputs trace to stderr
export BASH_XTRACEFD=2
set -x
exec 3>&1 # Keep a clean stdout channel for machine-readable JSON
exec 1>&2 # Send all human logs to stderr

# ───────── manifest check ─────────
ART="$1"
MODEL_DIR="$(dirname "$ART")"
BASENAME="$(basename "$ART")"

# Derive stem from artefact basename
stem="$BASENAME"
case "$BASENAME" in
  *_model.pkl)   stem="${BASENAME%_model.pkl}" ;;
  *_state.pt)    stem="${BASENAME%_state.pt}" ;;
  *_preproc.pkl) stem="${BASENAME%_preproc.pkl}" ;;
  *_loaderspec.json) stem="${BASENAME%_loaderspec.json}" ;;
  *) echo "Unrecognized artefact filename: $BASENAME" >&2; exit 1 ;;
esac

MANIFEST="$MODEL_DIR/${stem}_manifest.sha256"
echo "Using manifest: $MANIFEST"

pushd "$MODEL_DIR" >/dev/null

# Require core artefacts to exist and be covered by the manifest (pre-eval integrity)
req=( "${stem}_model.pkl" "${stem}_preproc.pkl" "${stem}_loaderspec.json" )

for f in "${req[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required artefact: ${MODEL_DIR}/${f}" >&2
    exit 1
  fi
  if ! grep -qE "[[:space:]]${f}$" "$(basename "$MANIFEST")"; then
    echo "Manifest does not list required artefact: ${f}" >&2
    echo "Re-run training (train.sh) to regenerate the manifest with required artefacts." >&2
    exit 1
  fi
done

sha256sum -c "$(basename "$MANIFEST")"
popd >/dev/null

# Determine CHALLENGE / QUESTION
# If they are not provided via environment, fall back to path-based inference
if [ -z "${CHALLENGE:-}" ] || [ -z "${QUESTION:-}" ]; then
  QUESTION_DIR=$(dirname "$MODEL_DIR")
  CHALLENGE_DIR=$(dirname "$QUESTION_DIR")
  CHALLENGE=${CHALLENGE:-$(basename "$CHALLENGE_DIR")}
  QUESTION=${QUESTION:-$(basename "$QUESTION_DIR")}
fi

echo "--- evaluation entrypoint ---"
echo "Challenge : $CHALLENGE"
echo "Question  : $QUESTION"
echo "Model dir : $MODEL_DIR"
echo "Artefact  : $ART"
echo "------------------------------"

# ───────── sandbox resource limits ─────────────────────
TIMEOUT="${EVALUATE_TIMEOUT_S:-1800}"

# Default to 0 (= unlimited) if the env-var is missing
MEM_GB="${MEMORY_LIMIT_GB:=0}"
PIDS_LIMIT="${PIDS_LIMIT:=0}"

ulimit -t "$TIMEOUT"                      # CPU-seconds

# virtual-memory limit: only apply when > 0
if (( MEMORY_LIMIT_GB > 0 )); then
    ulimit -v $((MEMORY_LIMIT_GB * 1024 * 1024))
fi

# open-files / processes limit: only apply when > 0
if (( PIDS_LIMIT > 0 )); then
    ulimit -n "$PIDS_LIMIT"
fi

# ───────── invoke Python helper (one-liner keeps image small) ─────────
timeout --signal=KILL "${TIMEOUT}s" python - "$ART" "$CHALLENGE" <<'PY'
import sys, os, importlib, json, time, logging, inspect, traceback, torch
import pandas as pd
import numpy as np

model_path, challenge = sys.argv[1], sys.argv[2]
model_dir  = os.path.dirname(model_path)

OUT = os.fdopen(3, "w", closefd=False)  # FD 3 is original stdout

def emit(obj: dict):
    print(json.dumps(obj, allow_nan=False), file=OUT, flush=True)

def to_jsonable(x):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu()
            if x.ndim == 0:
                return x.item()
            return x.tolist()
    except Exception:
        pass

    # numpy
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()

    # containers
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # plain python scalars
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    # fallback: stringify weird objects
    return str(x)

try:
    # 1) import evaluator
    module_name = f"challenges.{challenge}.evaluate_{challenge.lower()}"
    eval_mod = importlib.import_module(module_name)

    load_test  = getattr(eval_mod, f"load_{challenge}_test")
    evaluate_f = getattr(eval_mod, f"evaluate_{challenge}")

    # 2) build test set
    default_tag = getattr(eval_mod, "DEFAULT_TAG", None)
    sig_names   = load_test.__code__.co_varnames

    t0 = time.perf_counter()
    if "model_path" in sig_names and "tag" in sig_names:
        test_set = load_test(model_path=model_path, tag=default_tag)
    elif "model_path" in sig_names:
        test_set = load_test(model_path=model_path)
    elif "tag" in sig_names:
        test_set = load_test(tag=default_tag)
    else:
        test_set = load_test()
    t_load = time.perf_counter() - t0

    # 3) evaluate
    t0 = time.perf_counter()
    metrics = evaluate_f(model_path, test_set)
    t_eval  = time.perf_counter() - t0

    payload = {"ok": True, "load_s": round(t_load, 2), "eval_s": round(t_eval, 2)}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            payload[k] = to_jsonable(v)
    else:
        payload["metric"] = float(metrics)

    emit(payload)

except Exception as e:
    emit({
        "ok": False,
        "error": str(e),
        "traceback": traceback.format_exc(),
    })
    sys.exit(1)
PY

# ───────── final hash check (integrity after write-backs) ────────────
pushd "$MODEL_DIR" >/dev/null
sha256sum -c "$(basename "$MANIFEST")"
popd >/dev/null


# docker build -f docker/Dockerfile.sandbox -t llm-sandbox:latest .