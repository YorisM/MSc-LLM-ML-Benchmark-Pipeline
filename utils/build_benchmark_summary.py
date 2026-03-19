# ./utils/build_benchmark_summary.py

# Build/merge a single global benchmark summary JSON from your per-attempt response_*.json files
#
# Example Usage:
# python utils/build_benchmark_summary.py --input "outputs/01-05(2)" --input "outputs/01-06" --input "outputs/01-10" --output benchmark_summary.json


from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FNAME_RE = re.compile(r"^response_(?P<stem>.+)\.json$", re.IGNORECASE)

# Helpers
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: Path, obj: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def sum_nonnull(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return float(sum(vals))

def parse_model_and_attempt_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse: response_<MODEL>_HH:MM_<ATTEMPT>.json
    We parse from the right so model may contain underscores.
    """
    m = FNAME_RE.match(filename)
    if not m:
        return None, None
    stem = m.group("stem")

    if "_" not in stem:
        return stem, None

    left, attempt_s = stem.rsplit("_", 1)
    attempt = to_int(attempt_s)

    if "_" not in left:
        return left, attempt

    model_part, _time_part = left.rsplit("_", 1)
    return model_part, attempt

def infer_run_challenge_question(json_path: Path) -> Tuple[str, str, str]:
    """
    Infer outputs/<RUN>/<CHALLENGE>/<QID>/... from path parts.
    """
    parts = list(json_path.parts)
    try:
        i = next(idx for idx, p in enumerate(parts) if p.lower() == "outputs")
        run_id = parts[i + 1]
        challenge = parts[i + 2]
        qid = parts[i + 3]
        return run_id, challenge, qid
    except Exception:
        raise ValueError(f"Cannot infer run_id/challenge/qid from path: {json_path}")

def detect_metric(challenge: str, eval_metrics: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """
    ONLY:
      - FOURTOPS -> auc
      - TRACKFORMERS -> FitAccuracy (supports a few key spellings)
    """
    ch = challenge.upper()
    if ch == "FOURTOPS":
        return "auc", to_float(eval_metrics.get("auc", None))

    if ch == "TRACKFORMERS":
        for k in ["FitAccuracy", "fit_accuracy", "fitaccuracy", "Fit_Accuracy", "fitAccuracy"]:
            if k in eval_metrics:
                return "FitAccuracy", to_float(eval_metrics[k])
        return "FitAccuracy", None

    # fallback (shouldn't happen in your repo)
    for k, v in eval_metrics.items():
        fv = to_float(v)
        if fv is not None:
            return k, fv
    return "metric", None

def stage_from_flags(
    pylint_pass: Optional[bool],
    bandit_pass: Optional[bool],
    dryrun_pass: Optional[bool],
    eval_pass: Optional[bool],
) -> str:
    """
    You said failures only happen at static analysis or dry-run.
    Still, we keep a robust classifier.
    """
    if pylint_pass is False or bandit_pass is False:
        return "static_failed"
    if dryrun_pass is False:
        return "dryrun_failed"
    if eval_pass is True:
        return "success"
    return "unknown"

def epochs_actual_from_losses(train_loss: Any, val_loss: Any) -> Optional[int]:
    """
    Return actual epoch count from loss arrays.
    If lists exist but empty -> treat as missing (None).
    """
    if isinstance(train_loss, list) and len(train_loss) > 0:
        return len(train_loss)
    if isinstance(val_loss, list) and len(val_loss) > 0:
        return len(val_loss)
    return None

def iter_response_jsons(inputs: List[Path]) -> Iterable[Path]:
    for p in inputs:
        if p.is_file():
            if p.name.lower().startswith("response_") and p.suffix.lower() == ".json":
                yield p
        elif p.is_dir():
            yield from p.rglob("response_*.json")


# Data model
@dataclass
class Attempt:
    run_id: str
    challenge: str
    qid: str
    question_key: str  # e.g. "FOURTOPS/Q1"
    model: str         # openrouter id (canonical)
    attempt_index: int # from filename

    # stage / checks
    stage: str
    pylint_pass: Optional[bool]
    bandit_pass: Optional[bool]

    # generation
    cost_usd: Optional[float]
    generation_s: Optional[float]
    response_tokens: Optional[int]
    outer_attempt: Optional[int]
    inner_attempt: Optional[int]

    # training (only meaningful for success)
    training_time_s: Optional[float]
    epochs_planned: Optional[int]
    epochs_actual: Optional[int]
    train_loss: Optional[List[float]]
    val_loss: Optional[List[float]]

    # evaluation (only for success)
    metric_name: str
    metric_value: Optional[float]
    eval_runtime_s: Optional[float]

    # source pointer
    source_json: str

def parse_attempt(json_path: Path) -> Attempt:
    data = load_json(json_path)
    run_id, challenge, qid = infer_run_challenge_question(json_path)
    question_key = f"{challenge}/{qid}"

    # Model (canonical): OpenRouter id in LLMGeneration.model
    llm = data.get("LLMGeneration", {}) if isinstance(data.get("LLMGeneration", {}), dict) else {}
    model = llm.get("model", None)

    # fallback to filename-derived model if needed
    model_from_fname, attempt_from_fname = parse_model_and_attempt_from_filename(json_path.name)
    if model is None:
        model = model_from_fname or "UNKNOWN_MODEL"

    attempt_index = attempt_from_fname
    if attempt_index is None:
        # As a last resort, try Outer_attempt
        attempt_index = to_int(llm.get("Outer_attempt", None)) or 10**9

    # passes
    pylint_pass = safe_get(data, ["StaticChecks", "PyLint", "passed"], default=None)
    bandit_pass = safe_get(data, ["StaticChecks", "Bandit", "passed"], default=None)
    dryrun_pass = safe_get(data, ["DryRun", "passed"], default=None)
    eval_pass = safe_get(data, ["Evaluation", "passed"], default=None)

    stage = stage_from_flags(pylint_pass, bandit_pass, dryrun_pass, eval_pass)

    # generation fields
    cost_usd = to_float(llm.get("cost_usd", None))
    gen_ms = to_float(llm.get("generation_ms", None))
    generation_s = (gen_ms / 1000.0) if gen_ms is not None else None
    response_tokens = to_int(llm.get("response_tokens", None))

    outer_attempt = to_int(llm.get("Outer_attempt", None))
    inner_attempt = to_int(llm.get("Inner attempt", None)) or to_int(llm.get("Inner_attempt", None))

    # training fields
    training_time_s = to_float(safe_get(data, ["Training", "resources", "training_time_s"], default=None))
    epochs_planned = to_int(safe_get(data, ["Training", "metrics", "epochs"], default=None))
    train_loss = safe_get(data, ["Training", "metrics", "train_loss"], default=None)
    val_loss = safe_get(data, ["Training", "metrics", "val_loss"], default=None)

    # normalise losses: only keep if list; else None
    train_loss_list = train_loss if isinstance(train_loss, list) else None
    val_loss_list = val_loss if isinstance(val_loss, list) else None

    epochs_actual = epochs_actual_from_losses(train_loss_list, val_loss_list)

    # If losses are empty lists, treat as "no info": store None (we’ll omit arrays later)
    if isinstance(train_loss_list, list) and len(train_loss_list) == 0:
        train_loss_list = None
    if isinstance(val_loss_list, list) and len(val_loss_list) == 0:
        val_loss_list = None

    # evaluation fields
    eval_metrics = safe_get(data, ["Evaluation", "metrics"], default={})
    if not isinstance(eval_metrics, dict):
        eval_metrics = {}
    metric_name, metric_value = detect_metric(challenge, eval_metrics)
    eval_runtime_s = to_float(safe_get(data, ["Evaluation", "runtime_s"], default=None))

    return Attempt(
        run_id=run_id,
        challenge=challenge,
        qid=qid,
        question_key=question_key,
        model=model,
        attempt_index=int(attempt_index),

        stage=stage,
        pylint_pass=pylint_pass,
        bandit_pass=bandit_pass,

        cost_usd=cost_usd,
        generation_s=generation_s,
        response_tokens=response_tokens,
        outer_attempt=outer_attempt,
        inner_attempt=inner_attempt,

        training_time_s=training_time_s,
        epochs_planned=epochs_planned,
        epochs_actual=epochs_actual,
        train_loss=train_loss_list,
        val_loss=val_loss_list,

        metric_name=metric_name,
        metric_value=metric_value,
        eval_runtime_s=eval_runtime_s,

        source_json=str(json_path),
    )


# Build summary blocks
def attempt_to_dict(a: Attempt) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "attempt_index": a.attempt_index,
        "outer_attempt": a.outer_attempt,
        "inner_attempt": a.inner_attempt,
        "stage": a.stage,
        "cost_usd": a.cost_usd,
        "generation_s": a.generation_s,
        "response_tokens": a.response_tokens,
        "source_json": a.source_json,
    }
    if a.stage == "static_failed":
        d["static_failed"] = {
            "pylint": (a.pylint_pass is False),
            "bandit": (a.bandit_pass is False),
        }
    return d

def static_breakdown(attempts: List[Attempt]) -> Dict[str, int]:
    """
    Count static failures (attempt-level):
      - total: # attempts whose stage == static_failed
      - pylint: # attempts where pylint_pass == False
      - bandit: # attempts where bandit_pass == False
    Note: pylint+bandit can both count for the same attempt; total is still per-attempt.
    """
    total = sum(1 for a in attempts if a.stage == "static_failed")
    pylint = sum(1 for a in attempts if a.pylint_pass is False)
    bandit = sum(1 for a in attempts if a.bandit_pass is False)
    return {"total": int(total), "pylint": int(pylint), "bandit": int(bandit)}

def dryrun_fail_count(attempts: List[Attempt]) -> int:
    return int(sum(1 for a in attempts if a.stage == "dryrun_failed"))

def build_model_block_success(attempts_sorted: List[Attempt]) -> Dict[str, Any]:
    """
    FIRST SUCCESS WINS:
      - pick earliest attempt with stage == success
      - prior_attempts are attempts before that attempt
    """
    success_attempt = next((a for a in attempts_sorted if a.stage == "success"), None)
    if success_attempt is None:
        raise ValueError("build_model_block_success called without a success attempt")

    prior = [a for a in attempts_sorted if a.attempt_index < success_attempt.attempt_index]
    window = [a for a in attempts_sorted if a.attempt_index <= success_attempt.attempt_index]

    cost_to_success = sum_nonnull([a.cost_usd for a in window])

    static_bd = static_breakdown(prior)
    dry_bd = dryrun_fail_count(prior)

    attempts_to_success = success_attempt.attempt_index
    attempts_signature = f"{attempts_to_success}({static_bd['total']}/{dry_bd})"

    # Success attempt dict (generation)
    success_attempt_dict = {
        "attempt_index": success_attempt.attempt_index,
        "outer_attempt": success_attempt.outer_attempt,
        "inner_attempt": success_attempt.inner_attempt,
        "cost_usd": success_attempt.cost_usd,
        "generation_s": success_attempt.generation_s,
        "response_tokens": success_attempt.response_tokens,
        "source_json": success_attempt.source_json,
    }

    # Prior attempts list (generation)
    prior_attempts_list = [attempt_to_dict(a) for a in prior]

    # Training block: embed curves by default, but omit arrays if missing/empty
    training: Dict[str, Any] = {
        "training_time_s": success_attempt.training_time_s,
        "epochs_planned": success_attempt.epochs_planned,
        "epochs_actual": success_attempt.epochs_actual,
    }
    if success_attempt.train_loss is not None:
        training["train_loss"] = success_attempt.train_loss
    if success_attempt.val_loss is not None:
        training["val_loss"] = success_attempt.val_loss

    # Evaluation block
    evaluation = {
        "metric_name": success_attempt.metric_name,
        "metric_value": success_attempt.metric_value,
        "runtime_s": success_attempt.eval_runtime_s,
    }

    return {
        "model": success_attempt.model,

        "generation": {
            "effort": {
                "attempts_to_success": attempts_to_success,
                "static_fail_before_success": static_bd,
                "dryrun_fail_before_success": int(dry_bd),
                "attempts_signature": attempts_signature,
                "cost_to_success_usd": cost_to_success,
            },
            "success_attempt": success_attempt_dict,
            "prior_attempts": prior_attempts_list,
        },

        "training": training,
        "evaluation": evaluation,

        "sources": {
            "success_response_json": success_attempt.source_json,
        },
    }

def build_model_block_final_fail(attempts_sorted: List[Attempt]) -> Dict[str, Any]:
    """
    Model never succeeded in this run/question.
    Store attempt-level detail (all attempts) + aggregates + cost_to_failure_usd.
    """
    cost_to_failure = sum_nonnull([a.cost_usd for a in attempts_sorted])
    static_bd = static_breakdown(attempts_sorted)
    dry_bd = dryrun_fail_count(attempts_sorted)

    return {
        "model": attempts_sorted[0].model if attempts_sorted else "UNKNOWN_MODEL",
        "generation": {
            "aggregates": {
                "attempts_total": int(len(attempts_sorted)),
                "static_fail_total": static_bd,
                "dryrun_fail_total": int(dry_bd),
                "cost_to_failure_usd": cost_to_failure,
            },
            "attempts": [attempt_to_dict(a) for a in attempts_sorted],
        }
    }

def build_run_block(attempts: List[Attempt]) -> Dict[str, Any]:
    """
    Build the run_id block under a question:
      {
        "run_id": ...,
        "updated_at": ...,
        "successes": { model: ... },
        "final_fails": { model: ... }
      }
    """
    if not attempts:
        raise ValueError("Empty attempts for run block")

    run_id = attempts[0].run_id

    # Group by model
    by_model: Dict[str, List[Attempt]] = {}
    for a in attempts:
        by_model.setdefault(a.model, []).append(a)

    successes: Dict[str, Any] = {}
    final_fails: Dict[str, Any] = {}

    # Deterministic order: by model id string
    for model in sorted(by_model.keys()):
        ats = sorted(by_model[model], key=lambda x: x.attempt_index)
        has_success = any(a.stage == "success" for a in ats)
        if has_success:
            successes[model] = build_model_block_success(ats)
        else:
            final_fails[model] = build_model_block_final_fail(ats)

    return {
        "run_id": run_id,
        "updated_at": now_iso(),
        "successes": successes,
        "final_fails": final_fails,
    }

def build_updates(attempts: List[Attempt]) -> Dict[str, Any]:
    """
    Build the "questions" dict for all parsed attempts, structured as:
      questions[question_key] = {challenge, question, runs{run_id: run_block}}
    """
    # Group by question_key then run_id
    by_q: Dict[str, Dict[str, List[Attempt]]] = {}
    for a in attempts:
        by_q.setdefault(a.question_key, {}).setdefault(a.run_id, []).append(a)

    questions: Dict[str, Any] = {}

    # Put newer entries "on top": we'll insert question_keys in the order we encounter them,
    # and within each question, runs in descending "newness" by processing order.
    for question_key in sorted(by_q.keys()):  # stable order across runs; change if you prefer insertion order
        sample = next(iter(next(iter(by_q[question_key].values()))))
        challenge = sample.challenge
        qid = sample.qid

        runs_dict: Dict[str, Any] = {}
        # Insert run_ids in sorted order descending-ish: this is optional.
        # Since run_id isn't strictly sortable chronologically (01-06(2)), we keep a predictable order:
        for run_id in sorted(by_q[question_key].keys(), reverse=True):
            run_attempts = by_q[question_key][run_id]
            runs_dict[run_id] = build_run_block(run_attempts)

        questions[question_key] = {
            "challenge": challenge,
            "question": qid,
            "runs": runs_dict,
        }

    return questions


# Merge into existing summary
def merge_on_top(existing: Dict[str, Any], new_questions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new_questions into existing["questions"], overwriting any matching (question_key, run_id),
    and inserting new/updated entries before older ones.

    Ordering rule:
      - question_keys present in new_questions are inserted first (in new_questions insertion order)
      - within each question_key, run_ids present in new are inserted first (in new insertion order)
      - remaining old entries preserved after
    """
    merged: Dict[str, Any] = {
        "schema_version": existing.get("schema_version", "1.0"),
        "updated_at": now_iso(),
        "questions": {},
    }

    old_questions = existing.get("questions", {})
    if not isinstance(old_questions, dict):
        old_questions = {}

    # 1) Insert/merge questions that appear in new_questions
    for qkey, qblock_new in new_questions.items():
        qblock_old = old_questions.get(qkey, None)
        if not isinstance(qblock_old, dict):
            qblock_old = {}

        runs_new = qblock_new.get("runs", {})
        runs_old = qblock_old.get("runs", {})
        if not isinstance(runs_old, dict):
            runs_old = {}

        # Merge runs with new on top
        merged_runs: Dict[str, Any] = {}
        for run_id, run_block in runs_new.items():
            merged_runs[run_id] = run_block
        for run_id, run_block in runs_old.items():
            if run_id not in merged_runs:
                merged_runs[run_id] = run_block

        merged["questions"][qkey] = {
            "challenge": qblock_new.get("challenge", qblock_old.get("challenge")),
            "question": qblock_new.get("question", qblock_old.get("question")),
            "runs": merged_runs,
        }

    # 2) Append untouched old questions after
    for qkey, qblock_old in old_questions.items():
        if qkey in merged["questions"]:
            continue
        merged["questions"][qkey] = qblock_old

    return merged


# Main
def main():
    ap = argparse.ArgumentParser(description="Build/merge global benchmark_summary.json from response_*.json files.")
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input path(s) to scan. Repeatable. Can be outputs/<run>, outputs/<run>/<challenge>/..., or a response_*.json file.",
    )
    ap.add_argument(
        "--output",
        default="benchmark_summary.json",
        help="Output JSON file path (global summary). Default: ./benchmark_summary.json",
    )
    ap.add_argument(
        "--no-pretty",
        action="store_true",
        help="Write compact JSON instead of pretty-printed.",
    )
    args = ap.parse_args()

    inputs = [Path(p).resolve() for p in args.input]
    out_path = Path(args.output).resolve()
    pretty = not args.no_pretty

    # Collect and parse all response JSONs
    json_paths = sorted(set(iter_response_jsons(inputs)))
    if not json_paths:
        raise SystemExit("No response_*.json files found under the provided inputs.")

    attempts: List[Attempt] = []
    for jp in json_paths:
        try:
            attempts.append(parse_attempt(jp))
        except Exception as e:
            print(f"[WARN] Skipping {jp}: {e}")

    if not attempts:
        raise SystemExit("No valid response_*.json attempts could be parsed.")

    # Build new question blocks from parsed attempts
    new_questions = build_updates(attempts)

    # Load existing if present
    if out_path.exists():
        try:
            existing = load_json(out_path)
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    else:
        existing = {}

    if "schema_version" not in existing:
        existing = {"schema_version": "1.0", "updated_at": None, "questions": {}}

    merged = merge_on_top(existing, new_questions)
    dump_json(out_path, merged, pretty=pretty)

    print(f"Wrote global benchmark summary: {out_path}")
    print(f"Updated at: {merged.get('updated_at')}")
    print(f"Questions updated: {len(new_questions)}")
    print(f"Total questions in file: {len(merged.get('questions', {}))}")

if __name__ == "__main__":
    main()