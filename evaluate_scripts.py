# evaluate_scripts.py

import os, glob, logging, time, argparse, json
from pathlib import Path
from typing import List, Tuple, Dict, Iterator
from tqdm import tqdm
from collections import defaultdict
from challenges.FOURTOPS.evaluate_fourtops import load_FOURTOPS_test, evaluate_FOURTOPS
from challenges.TRACKFORMERS.evaluate_trackformers import load_TRACKFORMERS_test, evaluate_TRACKFORMERS
from utils.utils import append_to_response_json, _NpEncoder, iter_input_dir, SKIP_DIRS

def _find_models(input_dir: str | Path) -> List[Tuple[str, str, str]]:
    """
    Walk <input_dir> (any depth allowed) and return a list of
    (question_id, model_folder, path-to-model.pkl).
    """

    candidates: List[Tuple[str, str, str]] = []

    for q_dir in iter_input_dir(input_dir):
        qid = q_dir.name
        for m_dir in q_dir.iterdir():
            if not (m_dir.is_dir() and m_dir.name not in SKIP_DIRS):
                continue
            for pkl in m_dir.glob("*_model.pkl"):
                candidates.append((qid, m_dir.name, str(pkl)))

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
        find_models = _find_models,
        load_test   = load_FOURTOPS_test,
        evaluate    = evaluate_FOURTOPS,
        test_outputs= verify_outputs
    ),

    "TRACKFORMERS": dict(
        find_models = _find_models,
        load_test   = load_TRACKFORMERS_test,
        evaluate    = evaluate_TRACKFORMERS,
        test_outputs= verify_outputs
    )
}

def evaluate_results(input_dir: str | Path):
    """
    Evluates every model that lives under input_dir
    """

    input_dir = Path(input_dir).resolve()
    challenges: Dict[Path, List[Path]] = defaultdict(list)

    # 1) gather question dirs and group them by challenge root
    for q_dir in iter_input_dir(input_dir):
        try:
            out_idx = q_dir.parts.index("outputs")
        except ValueError:
            continue
        
        challenge_root = Path(*q_dir.parts[: out_idx + 3])
        challenges[challenge_root].append(q_dir)


    # 2) process each challenge one at a time
    for challenge_root, q_dirs in challenges.items():
        date_str  = challenge_root.parts[-2]
        challenge = challenge_root.parts[-1]

        if challenge not in challenge_evaluators:
            logging.error("No evaluator defined for challenge %s - skipping.", challenge)
            continue
    
        # 3) find all (qid, model, pkl) for this challenge
        find_models = challenge_evaluators[challenge]["find_models"]
        allowed_qids = {qd.name for qd in q_dirs}
        candidates   = [
            tup for tup in find_models(challenge_root)
            if tup[0] in allowed_qids
        ]

        logging.info("Found %d model candidates for challenge %s: %s", len(candidates), challenge, candidates)

        # 4) evaluate each candidate
        q_summaries: dict[str, dict[str, dict]] = {}
        for qid, model_name, pt_path in tqdm(candidates, desc=f"{challenge} models evaluation",
                                            leave=False):
            
            model_dir = Path(pt_path).parent
            t0 = time.perf_counter() # start timer

            try:
                test_loader = challenge_evaluators[challenge]["load_test"]
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
        
        # Write Summary block response JSON
        q_summaries.setdefault(qid, {})[model_name] = metrics

        # 5) now that all models have been evaluated, check for missing outputs once
        missing = challenge_evaluators[challenge]["test_outputs"](date_str, challenge)
        missing = {qid: m for qid, m in missing.items() if qid in allowed_qids}

    # 6)  write <Q>_summary.json files
    for qid, models_metrics in q_summaries.items():
        out_path = challenge_root / qid / f"{qid}_summary.json"
        for model_name, metrics in models_metrics.items():
            metrics["missing_files"] = missing.get(qid, {}).get(model_name, [])
        out_path.write_text(
            json.dumps(models_metrics, cls=_NpEncoder, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logging.info("Wrote summary to %s", out_path)

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