#!/usr/bin/env bash
# Usage:  predict.sh <artefact_dir>/<something>_model.pkl
set -euo pipefail

ART="$1"                              # any artefact that lives in <MODEL_DIR>
MODEL_DIR="$(dirname "$ART")"         # .../outputs/<DATE>/<CHALLENGE>/<Q>/<MODEL>
MODEL_NAME="$(basename "$MODEL_DIR")"

# ───────── manifest check (before we burn time) ─────────
MANIFEST="$MODEL_DIR/${MODEL_NAME}_manifest.sha256"
sha256sum -c "$MANIFEST"

# ───────── peel path components we need ────────────────
QUESTION_DIR="$(dirname "$MODEL_DIR")"
CHALLENGE_DIR="$(dirname "$QUESTION_DIR")"
CHALLENGE="$(basename "$CHALLENGE_DIR")"         # FOURTOPS | TRACKFORMERS | …
QUESTION="$(basename "$QUESTION_DIR")"

echo "--- evaluation entrypoint ---"
echo "Challenge : $CHALLENGE"
echo "Question  : $QUESTION"
echo "Model dir : $MODEL_DIR"
echo "Artefact  : $ART"
echo "------------------------------"

# ───────── sandbox resource limits ─────────────────────
TIMEOUT="${EVALUATE_TIMEOUT_S:-1800}"
MEM_GB="${MEMORY_LIMIT_GB:-8}"
PIDS_LIMIT="${PIDS_LIMIT:-512}"

ulimit -t "$TIMEOUT"
ulimit -v $((MEM_GB*1024*1024))
ulimit -n "$PIDS_LIMIT"

# ───────── invoke Python helper (one-liner keeps image small) ─────────
timeout --signal=KILL "${TIMEOUT}s" python - "$ART" "$CHALLENGE" <<'PY'
import sys, os, importlib, json, time, logging
import torch, pandas as pd, numpy as np              # used by some loaders

model_path, challenge = sys.argv[1], sys.argv[2]
model_dir  = os.path.dirname(model_path)

# ---------------------------------------------------------------------
# 1.  Dynamically import the challenge evaluator module
#     Expected file:  challenges/<CHALLENGE>/evaluate_<challenge>.py
# ---------------------------------------------------------------------
module_name = f"challenges.{challenge}.evaluate_{challenge.lower()}"
try:
    eval_mod = importlib.import_module(module_name)
except ModuleNotFoundError as e:
    raise RuntimeError(f"Could not import evaluator module {module_name}") from e

# mandatory symbols inside the evaluator module
try:
    load_test  = getattr(eval_mod, f"load_{challenge}_test")
    evaluate_f = getattr(eval_mod, f"evaluate_{challenge}")
except AttributeError as e:
    raise RuntimeError(f"Evaluator module {module_name} is missing required symbols") from e

# ---------------------------------------------------------------------
# 2.  Build the test set - the individual loader decides what to return
# ---------------------------------------------------------------------
t0 = time.perf_counter()
test_set = load_test(getattr(eval_mod, "DEFAULT_TAG", None)) \
           if "tag" in load_test.__code__.co_varnames else load_test()
t_load = time.perf_counter() - t0

# ---------------------------------------------------------------------
# 3.  Run evaluation
# ---------------------------------------------------------------------
t0 = time.perf_counter()
metrics = evaluate_f(model_path, test_set)
t_eval  = time.perf_counter() - t0

# ---------------------------------------------------------------------
# 4.  Emit JSON so the host runner can parse it
# ---------------------------------------------------------------------
print(json.dumps({
    "load_s" : round(t_load,  2),
    "eval_s" : round(t_eval,  2),
    **({k: float(v) for k, v in metrics.items()} if isinstance(metrics, dict) else {"metric": float(metrics)})
}, allow_nan=False))
PY

# ───────── final hash check (integrity after write-backs) ────────────
sha256sum -c "$MANIFEST"
