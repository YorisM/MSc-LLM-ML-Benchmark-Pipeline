# evaluate_scripts.py

import os, glob, logging, csv, time, argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from challenges.FOURTOPS.evaluate_fourtops import load_FOURTOPS_test, evaluate_FOURTOPS
from challenges.TRACKFORMERS.evaluate_trackformers import load_TRACKFORMERS_test, evaluate_TRACKFORMERS
from utils.utils import append_to_response_json

def _iter_input_dir(input_dir: str | Path) -> Iterator[Path]:
    """
    Yield every .../outputs/<DATE>/<CHALLENGE>/<QUESTION>/ directory that
    lives under *input_dir*, regardless of whether the caller hands us
    
        ─ outputs/<DATE>/
        ─ outputs/<DATE>/<CHALLENGE>/
        ─ outputs/<DATE>/<CHALLENGE>/<QUESTION>/
    
    Any other depth raises ValueError.
    """

    root = Path(input_dir).resolve()
    try:
        out_idx = root.parts.index("outputs")
    except ValueError:
        raise ValueError(f"{root} is not inside an 'outputs' tree")

    # relative path depth after “outputs/”
    depth = len(root.parts) - out_idx - 1

    if depth == 1:                       # outputs/<DATE>
        for ch in root.iterdir():
            if ch.is_dir():
                yield from _iter_input_dir(ch)

    elif depth == 2:                     # outputs/<DATE>/<CHALLENGE>
        for q in root.iterdir():
            if q.is_dir():
                yield from _iter_input_dir(q)

    elif depth == 3:                     # outputs/<DATE>/<CHALLENGE>/<QUESTION>
        yield root

    else:
        raise ValueError("Path depth must be 1-3 under 'outputs/'")

def find_models(date_str: str, challenge: str) -> List[Tuple[str, str, str]]:
    """
    Walks ./outputs/<date>/<challenge>/** and returns a list of
    (question_id, model_folder_name, <path-to-model.pkl>) tuples.
    """

    root = os.path.join("outputs", date_str, challenge)
    candidates: List[Tuple[str, str, str]] = []

    if not os.path.isdir(root):
        logging.warning("Folder %s does not exist", root)
        return candidates
    
    skip = {"Failed Dry-run Scripts", "StaticFail"}

    for qid in os.listdir(root):
        qpath = os.path.join(root, qid)
        if qid in skip or not os.path.isdir(qpath):
            continue
        
        for m in os.listdir(qpath):
            mpath = os.path.join(qpath, m)
            if not os.path.isdir(mpath):
                continue

            for pkl in glob.glob(os.path.join(mpath, "*_model.pkl")):
                candidates.append((qid, m, pkl))

    logging.info(f"Found {challenge} model candidates: {candidates}")
    return candidates

def verify_outputs(date_str: str, challenge: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Verify that each model folder under ./outputs/<date>/<challenge>/… 
    contains all artefacts defined in *patterns*.
    Returns {question_id -> {model_name -> [missing_patterns]}}.
    """

    root = os.path.join("outputs", date_str, challenge)
    results: dict[str, dict[str, list[str]]] = {}
    skip = {"Failed Dry-run Scripts", "StaticFail"}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder {root!r} does not exist")

    PATTERNS = [
        "*_model.pkl",
        "*_state.pt",
        "*_preproc.pkl",
        "*_loss.png",
        "*_accuracy.png",
        "*_manifest.sha256"
    ]

    for qid in os.listdir(root):
        qpath = os.path.join(root, qid)
        if qid in skip or not os.path.isdir(qpath):
            continue

        results[qid] = {}
        for model_name in os.listdir(qpath):
            mpath = os.path.join(qpath, model_name)
            if model_name in skip or not os.path.isdir(mpath):
                continue

            missing = [
                pat for pat in PATTERNS
                if not glob.glob(os.path.join(mpath, pat))
            ]
            results[qid][model_name] = missing

            results[qid][model_name] = missing
            if missing:
                logging.warning("Date %s - %s - Model %s - MISSING: %s", date_str, qid, model_name, ", ".join(missing))
            else:
                logging.info("Date %s - %s - Model %s all present",  date_str, qid, model_name)
    return results

challenge_evaluators = {
    "FOURTOPS": dict(
        find_models = find_models,
        load_test   = load_FOURTOPS_test,
        evaluate    = evaluate_FOURTOPS,
        test_outputs= verify_outputs
    ),

    "TRACKFORMERS": dict(
        find_models = find_models,
        load_test   = load_TRACKFORMERS_test,
        evaluate    = evaluate_TRACKFORMERS,
        test_outputs= verify_outputs
    )
}

def evaluate_results(input_dir):
    # derive date and challenge from input dir
    for q_dir in _iter_input_dir(input_dir):
        parts = q_dir.parts
        date_str, challenge = parts[-3], parts[-2]

        if challenge not in challenge_evaluators:
            logging.error("No evaluator defined for challenge %s", challenge)
            continue

    # 1) load test data
    test_loader = challenge_evaluators[challenge]["load_test"]()

    # 2) find scripted models
    candidates  = challenge_evaluators[challenge]["find_models"](date_str, challenge)

    # 3) run all evaluate_<challenge> fuinctions and stash results + write JSON block
    eval_results = []   # will hold metrics
    for qid, model_name, pt_path in candidates:
        model_dir = os.path.dirname(pt_path) 
        t0 = time.perf_counter() # start timer

        try:
            metrics = challenge_evaluators[challenge]["evaluate"](pt_path, test_loader)
            eval_ok = True
        except Exception as e:
            logging.error("Evaluation failed for %s: %s", pt_path, e)
            metrics = {}
            eval_ok = False
        eval_time = time.perf_counter() - t0

        # Write Evaluation block response JSON
        try:
            json_file = next(p for p in os.listdir(model_dir)
                            if p.startswith("response_") and p.endswith(".json"))
            json_path = os.path.join(model_dir, json_file)

            append_to_response_json(
                json_path,
                "Evaluation",
                {
                    "passed": eval_ok,
                    "metrics": metrics,
                    "runtime_s": round(eval_time, 2)
                }
            )
        except StopIteration:
            logging.warning("No response_…json found in %s; skipping append", model_dir)
        
        eval_results.append((qid, model_name, pt_path, metrics))

    # 4) now that all models have been evaluated, check for missing outputs once
    missing = challenge_evaluators[challenge]["test_outputs"](date_str, challenge)

    # 5) write out your summary.csv, including missing‐files in the last column
    summary_path = os.path.join("outputs", date_str, challenge, "summary.csv")
    with open(summary_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["question","model","path","metrics_json","missing_files"])
        for qid, model_name, pt_path, metrics in eval_results:
            miss_list = missing.get(qid, {}).get(model_name, [])
            writer.writerow([
                qid,
                model_name,
                pt_path,
                json.dumps(metrics, allow_nan = False),
                ",".join(miss_list)
            ])

    logging.info("Summary written to %s", summary_path)

def main(date_str: str, challenge: str):
    input_root = os.path.join("outputs", date_str, challenge)
    if not os.path.isdir(input_root):
        logging.error("Folder %s does not exist", input_root)
        return
    evaluate_results(input_root)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date",      default=time.strftime("%d-%m"))
    p.add_argument("--challenge", "-c", default="FOURTOPS",
                   choices=["FOURTOPS", "TRACKFORMERS"])
    args = p.parse_args()
    main(args.date, args.challenge)