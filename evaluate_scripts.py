# evaluate_scripts.py

# Imports
import os, sys, importlib, importlib.util, glob, logging, torch, csv, time, argparse, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_curve, auc

# - - - - - TODO - - - - - 
#   fix test_FOURTOPS_outputs
#   generalize evaluation for multiple challenges
# - - - - - - - - - - - - -

def _apply_preproc(preproc, x: torch.Tensor) -> torch.Tensor:
    if callable(preproc):
        return preproc(x)
    if hasattr(preproc, "transform"):
        return preproc.transform(x)
    raise TypeError("Pre-processor is neither callable nor has .transform()")

def _mount_llm_script(model_dir: str) -> None:
    """
    Import the LLM-generated script that lives next to the artefacts and
    register it *also* as sys.modules['__main__'] so that
    __main__.MyPreprocessor can be resolved during unpickling.
    Safe to call more than once per process.
    """
    # find the script_<model>_*.py file
    script_path = next(
        f for f in os.listdir(model_dir)
        if f.startswith("script_") and f.endswith(".py")
    )
    script_path = os.path.join(model_dir, script_path)

    # If we already loaded *this* script, nothing to do
    if "__main__" in sys.modules and getattr(sys.modules["__main__"], "__file__", None) == script_path:
        return

    spec = importlib.util.spec_from_file_location("llm_script", script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)           # type: ignore[attr-defined]

    sys.modules["llm_script"] = mod        # real name
    sys.modules["__main__"]   = mod        # alias used inside pickle

def _initialize_artefacts(model_path: str):
    model_dir = os.path.dirname(model_path)

    # ❶ Make sure MyPreprocessor lives in sys.modules['__main__']
    _mount_llm_script(model_dir)

    # ❷ Now unpickle safely
    with open(model_path.replace("_model.pkl", "_preproc.pkl"), "rb") as f:
        preproc = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, preproc

def find_FOURTOPS_models(date_str):
    root = os.path.join("outputs", date_str, "FOURTOPS")
    candidates = []
    for qid in os.listdir(root):
        qpath = os.path.join(root, qid)
        if not os.path.isdir(qpath) or qid == "Failed Dry‑run Scripts":
            continue
        if not os.path.isdir(qpath) or qid == "StaticFail":
            continue
        for m in os.listdir(qpath):
            mpath = os.path.join(qpath, m)
            if not os.path.isdir(mpath):
                continue
            # look for any *_scripted.pt
            for pt in glob.glob(os.path.join(mpath, "*_model.pkl")):
                candidates.append((qid, m, pt))
    logging.info(f"Found FOURTOP model candidates: {candidates}")
    return candidates

def load_FOURTOPS_test():
    X = pd.read_csv('./challenges/FOURTOPS/data/X_test.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y = pd.read_csv('./challenges/FOURTOPS/data/Y_test.csv',
                          dtype=np.int64).to_numpy(copy=False).ravel()
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    test_ds = TensorDataset(X, Y)
    return (DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0))

def evaluate_FOURTOPS(model_path: str, test_loader) \
    -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    PARAMETERS
    model_path  : path to `<MODEL>_model.pkl`
    test_loader : DataLoader yielding (features, labels)

    RETURNS
    fpr, tpr : np.ndarray
    auc      : float
    acc      : float
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating %s on %s", model_path, device)

    model, preproc = _initialize_artefacts(model_path)
    model = model.to(device)

    # Determine output mode
    it  = iter(test_loader)
    xb0, yb0 = next(it)
    xb0, yb0 = xb0.to(device), yb0.to(device)
    xb0      = _apply_preproc(preproc, xb0)      # ← fixed: xb0 not xb
    out0     = model(xb0).detach()

    mn, mx = out0.min().item(), out0.max().item()
    if 0.0 <= mn <= mx <= 1.0:
        mode = "prob"
    else:
        mode = "logit"

    if mode == "prob":
        if out0.ndim == 2 and out0.size(1) == 2:          # p(bg), p(sig)
            act = lambda o: o[:, 1]
        else:                                             # already (N,)
            act = lambda o: o.squeeze()
    else:  # raw logits
        if out0.ndim == 2 and out0.size(1) == 1:
            act = lambda o: torch.sigmoid(o).squeeze(1)
        elif out0.ndim == 2 and out0.size(1) == 2:
            act = lambda o: torch.softmax(o, 1)[:, 1]
        elif out0.ndim == 1:
            act = lambda o: torch.sigmoid(o)
        else:
            raise RuntimeError(f"Unexpected output shape {tuple(out0.shape)}")

    logging.info("Output mode: %s; activation chosen.", mode)

    # Collect Predictions
    all_probs  = act(out0).cpu().numpy().tolist()
    all_labels = yb0.cpu().numpy().tolist()

    with torch.no_grad():
        for xb, yb in it:
            xb, yb = xb.to(device), yb.to(device)
            xb = _apply_preproc(preproc, xb)
            probs = act(model(xb))
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    # Metrics
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    acc         = accuracy_score(all_labels,
                                 (np.array(all_probs) >= 0.5).astype(int))

    logging.info("Evaluation finished: AUC %.4f  ACC %.4f", roc_auc, acc)
    return fpr, tpr, roc_auc, acc

def test_FOURTOPS_outputs(date_str: str):
    """
    Walks ./outputs/{date_str}/FOURTOPS/ and, for every question sub-folder
    (except “Failed Dry-run Scripts”), ensures each model folder contains a file
    that matches the seven wildcard patterns below.

    Returns dict:  {question_id -> {model_name -> list_of_missing_patterns}}
    """

    root = os.path.join("outputs", date_str, "FOURTOPS")
    results: dict[str, dict[str, list[str]]] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder {root!r} does not exist")

    PATTERNS = [
        "*_model.pth",
        "*_scripted.pt",
        "*_preproc.pt",
        "*_loss.png",
        "*_accuracy.png",
        "*_ROC.png",
        "*_manifest.sha256"
    ]

    for qid in os.listdir(root):                                    # question folders
        qpath = os.path.join(root, qid)
        if not os.path.isdir(qpath) or qid == "Failed Dry-run Scripts":
            continue
        if not os.path.isdir(qpath) or qid == "StaticFail":
            continue

        results[qid] = {}
        for model_name in os.listdir(qpath):                        # model folders
            mpath = os.path.join(qpath, model_name)
            if not os.path.isdir(mpath) or model_name == "Failed Dry-run Scripts":
                continue
            if not os.path.isdir(mpath) or model_name == "StaticFail":
                continue

            # check each glob-pattern
            missing = [
                pat for pat in PATTERNS
                if len(glob.glob(os.path.join(mpath, pat))) == 0
            ]

            results[qid][model_name] = missing
            if missing:
                logging.warning("Date %s - %s - Model %s - MISSING: %s", date_str, qid, model_name, ", ".join(missing))
            else:
                logging.info("Date %s - %s - Model %s all present",  date_str, qid, model_name)
    return results


challenge_evaluators = {
    "FOURTOPS": dict(
        find_models=find_FOURTOPS_models,
        load_test=load_FOURTOPS_test,
        evaluate=evaluate_FOURTOPS,
        test_outputs=test_FOURTOPS_outputs
    ),

    # future ones go here...
}

def evaluate_results(input_dir):
    # derive date and challenge from path: ./outputs/<date>/<challenge>/<question>/
    parts      = Path(input_dir).parts
    date_str, challenge = parts[-3], parts[-2]

    if challenge not in challenge_evaluators:
        logging.error("No evaluator defined for challenge %s", challenge)
        return

    # 1) load test data
    test_loader = challenge_evaluators[challenge]["load_test"]()

    # 2) find scripted models
    candidates  = challenge_evaluators[challenge]["find_models"](date_str)

    # 3) run all evaluate_FOURTOPS (or whatever) and stash results
    eval_results = []   # will hold tuples (qid, model_name, pt_path, fpr, tpr, auc, acc)
    for qid, model_name, pt_path in candidates:
        fpr, tpr, roc_auc, acc = challenge_evaluators[challenge]["evaluate"](pt_path, test_loader)
        eval_results.append((qid, model_name, pt_path, fpr, tpr, roc_auc, acc))
        logging.info("Evaluated %s/%s → acc=%.3f auc=%.3f", qid, model_name, acc, roc_auc)

    # 4) now that all models have been evaluated, check for missing outputs once
    missing = challenge_evaluators[challenge]["test_outputs"](date_str)

    # 5) write out your summary.csv, including missing‐files in the last column
    summary_path = os.path.join("outputs", date_str, challenge, "summary.csv")
    with open(summary_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["question","model","accuracy","auc","path","missing_files"])
        for qid, model_name, pt_path, fpr, tpr, roc_auc, acc in eval_results:
            miss_list = missing.get(qid, {}).get(model_name, [])
            writer.writerow([
                qid,
                model_name,
                f"{acc:.4f}",
                f"{roc_auc:.4f}",
                pt_path,
                ",".join(miss_list)
            ])

            # save ROC plot
            outdir   = os.path.dirname(pt_path)
            basename = os.path.splitext(os.path.basename(pt_path))[0]
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(f"{qid}-{model_name} ROC")
            plt.legend()
            plt.savefig(os.path.join(outdir, f"{basename}_ROC.png"))
            plt.close()

    logging.info("Summary written to %s", summary_path)

def main(date_str="17-04"):
    test_loader = load_FOURTOPS_test()
    scripts = find_FOURTOPS_models(date_str)

    summary_path = f"outputs/{date_str}/FOURTOPS/summary.csv"
    with open(summary_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["question","model","accuracy","auc","scripted_pt"])

        for qid, model_name, input_dir in scripts:
            fpr, tpr, roc_auc, acc = evaluate_FOURTOPS(input_dir, test_loader)
            writer.writerow([qid, model_name, f"{acc:.4f}", f"{roc_auc:.4f}", input_dir])

            # save ROC plot right next to the .pt
            outdir = os.path.dirname(input_dir)
            basename = os.path.splitext(os.path.basename(input_dir))[0]
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{qid}-{model_name} ROC")
            plt.legend()
            plt.savefig(os.path.join(outdir, f"{basename}_ROC.png"))
            plt.close()

            logging.info("Evaluated %s/%s → acc=%.3f auc=%.3f",
                         qid, model_name, acc, roc_auc)

    logging.info("Summary written to %s", summary_path)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=time.strftime("%d-%m"))
    args = p.parse_args()
    main(args.date)