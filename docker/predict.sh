#!/usr/bin/env bash
# Usage:  predict.sh <artefact_dir>/<model_or_scripted>.pt|.pkl

set -euo pipefail
ART="$1"                                       # path to any artefact inside <MODEL> dir
MODEL_DIR="$(dirname "$ART")"                  # .../outputs/<DATE>/<CHALLENGE>/<Q>/<MODEL>
MODEL_NAME="$(basename "$MODEL_DIR")"

# ────────── hash-check (BEFORE GPU alloc) ──────────
MANIFEST="$MODEL_DIR/${MODEL_NAME}_manifest.sha256"
sha256sum -c "$MANIFEST"

# ────────── pull challenge + question names ────────
#   .../outputs/<DATE>/<CHALLENGE>/<QUESTION>/<MODEL>
QUESTION_DIR="$(dirname "$MODEL_DIR")"
CHALLENGE_DIR="$(dirname "$QUESTION_DIR")"
CHALLENGE="$(basename "$CHALLENGE_DIR")"        # e.g. FOURTOPS
QUESTION="$(basename "$QUESTION_DIR")"          # e.g. Q1

echo "Challenge : $CHALLENGE"
echo "Question  : $QUESTION"
echo "Model dir : $MODEL_DIR"
echo "Artefact  : $ART"

# ────────── resources ─────────────
TIMEOUT="${EVALUATE_TIMEOUT_S:-1800}"
MEM_GB="${MEMORY_LIMIT_GB:-8}"
PIDS_LIMIT="${PIDS_LIMIT:-512}"

ulimit -t "$TIMEOUT"
ulimit -v $((MEM_GB*1024*1024))
ulimit -n "$PIDS_LIMIT"

# ────────── invoke Python one-liner ────────────────
timeout --signal=KILL "${TIMEOUT}s" python - <<'PY'

import sys, importlib, json, torch, os, pandas as pd, numpy as np
# ------------------------------------------------------------------
artefact_path = sys.argv[1]
model_dir     = os.path.dirname(artefact_path)
challenge     = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))  # <CHALLENGE>

# --------- dynamic import of evaluator ----------------------------------------
eval_mod      = importlib.import_module("evaluator")
eval_fn_name  = f"evaluate_{challenge}"
if not hasattr(eval_mod, eval_fn_name):
    raise RuntimeError(f"Evaluator function '{eval_fn_name}' not found in evaluator.py")
evaluate = getattr(eval_mod, eval_fn_name)

# --------- locate model / preproc artefacts -----------------------------------
# prefer pickled model; fall back to TorchScript
try:
    model_pkl = next(p for p in os.listdir(model_dir) if p.endswith("_model.pkl"))
except StopIteration:
    raise RuntimeError("No *_model.pkl found in model dir")
model_path = os.path.join(model_dir, model_pkl)

# --------- load test set (convention: ./challenges/<CHALLENGE>/data) ----------
data_root  = f"./challenges/{challenge}/data"
X_test = pd.read_csv(os.path.join(data_root, "X_test.csv"), dtype=np.float32).to_numpy(copy=False)
y_test = pd.read_csv(os.path.join(data_root, "Y_test.csv"), dtype=np.int64).to_numpy(copy=False).ravel()

test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                         torch.from_numpy(y_test))
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2048)

# --------- evaluate & print JSON ----------------------------------------------
fpr, tpr, auc, acc = evaluate(model_path, test_loader)   # evaluator decides device etc.
print(json.dumps({"auc": auc, "acc": acc}))
PY "$ART"

# ────────── final manifest check (post-eval) ───────
sha256sum -c "$MANIFEST"
