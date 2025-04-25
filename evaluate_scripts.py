# evaluate_scripts.py

# Imports
import os, glob, logging, torch, csv, time, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_curve, auc

# - - - - - TODO - - - - - 
#   fix test_FOURTOPS_outputs
#   generalize evaluation for multiple challenges
# - - - - - - - - - - - - -

def find_FOURTOPS_models(date_str):
    root = os.path.join("outputs", date_str, "FOURTOPS")
    candidates = []
    for qid in os.listdir(root):
        qpath = os.path.join(root, qid)
        if not os.path.isdir(qpath) or qid == "Failed Dry‑run Scripts":
            continue
        for m in os.listdir(qpath):
            mpath = os.path.join(qpath, m)
            if not os.path.isdir(mpath):
                continue
            # look for any *_scripted.pt
            for pt in glob.glob(os.path.join(mpath, "*_scripted.pt")):
                candidates.append((qid, m, pt))
    logging.info(f"Found FOURTOP model candidates: {candidates}")
    return candidates

def load_FOURTOPS_test(batch_size=512):
    X = pd.read_csv("challenges/FOURTOPS/data/X_test.csv").values
    Y = pd.read_csv("challenges/FOURTOPS/data/Y_test.csv").values.squeeze()
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.long)
    ds = TensorDataset(Xt, Yt)
    logging.debug("Loaded FOURTOP test data tensors")
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def evaluate_FOURTOPS(pt_path, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"pt_path: {pt_path}")

    # Load model & preproc
    model        = torch.jit.load(pt_path, map_location=device).eval()
    preproc_path = pt_path.replace("_scripted.pt", "_preproc.pt")
    logging.info(f"preproc_path: {preproc_path}")
    preproc      = torch.jit.load(preproc_path, map_location=device).eval()

    # Peek at first batch to choose activation once
    it = iter(test_loader)
    xb0, yb0 = next(it)
    xb0, yb0 = xb0.to(device), yb0.to(device)
    xb0 = preproc(xb0)
    out0 = model(xb0)

    mn, mx = out0.min().item(), out0.max().item()
    if mn >= 0.0 and mx <= 1.0:
        mode = "already-probabilities"
    else:
        mode = "raw-logits"

    # Decide head-type & activation function
    if mode == "already-probabilities":
        if out0.ndim == 2 and out0.size(1) == 2:
            act = lambda o: o[:,1]             # two-prob columns
            head = "2-column probabilities → pick col-1"
        else:
            act = lambda o: o.squeeze()       # already 0–1
            head = "single-probabilities"
    else:  # raw logits
        if out0.ndim == 2 and out0.size(1) == 1:
            act = lambda o: torch.sigmoid(o).squeeze(1)
            head = "single-logit → sigmoid"
        elif out0.ndim == 2 and out0.size(1) == 2:
            act = lambda o: torch.softmax(o,1)[:,1]
            head = "two-logit → softmax"
        elif out0.ndim == 1:
            act = lambda o: torch.sigmoid(o)
            head = "1D logits → sigmoid"
        else:
            raise RuntimeError(f"Unexpected output shape {tuple(out0.shape)}")

    logging.info(f"Detected mode: {mode}; using activation: {head}")

    # Now process first batch and the rest
    all_labels = yb0.cpu().numpy().tolist()
    all_probs  = act(out0).cpu().detach().numpy().tolist()

    # Continue with remaining batches
    with torch.no_grad():
        for xb, yb in it:
            xb, yb = xb.to(device), yb.to(device)
            xb = preproc(xb)
            out = model(xb)

            probs = act(out)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    # Compute ROC / AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    # Compute accuracy at 0.5 threshold
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)

    logging.info(f"Eval complete: AUC={roc_auc:.4f}, Acc={acc:.4f}")
    return fpr, tpr, roc_auc, acc

def test_FOURTOPS_outputs(date_str: str):
    """
    Walks ./outputs/{date_str}/FOURTOPS/ and, for every question sub-folder
    (except “Failed Dry-run Scripts”), ensures each model folder contains a file
    that matches the six wildcard patterns below.

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
    ]

    for qid in os.listdir(root):                                    # question folders
        qpath = os.path.join(root, qid)
        if not os.path.isdir(qpath) or qid == "Failed Dry-run Scripts":
            continue

        results[qid] = {}
        for model_name in os.listdir(qpath):                        # model folders
            mpath = os.path.join(qpath, model_name)
            if not os.path.isdir(mpath) or model_name == "Failed Dry-run Scripts":
                continue

            # check each glob-pattern
            missing = [
                pat for pat in PATTERNS
                if len(glob.glob(os.path.join(mpath, pat))) == 0
            ]

            results[qid][model_name] = missing
            if missing:
                logging.warning("Date %s — %s — Model %s — MISSING: %s", date_str, qid, model_name, ", ".join(missing))
            else:
                logging.info("Date %s — %s — Model %s all present",  date_str, qid, model_name)
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