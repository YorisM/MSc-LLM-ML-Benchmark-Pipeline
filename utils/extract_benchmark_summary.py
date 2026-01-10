# utils.extract_benchmark_summary.py

"""
Extract benchmark metrics from response_*.json files into tidy CSVs.

Layout (examples):
  outputs/<RUN_ID>/<CHALLENGE>/<QUESTION>/<MODEL-NAME>/response_<MODEL>_HH:MM_<ATTEMPT>.json   (successes)
  outputs/<RUN_ID>/<CHALLENGE>/<QUESTION>/StaticFail/response_*.json                           (failures)
  outputs/<RUN_ID>/<CHALLENGE>/<QUESTION>/Failed Dry-run Scripts/response_*.json               (failures)

Outputs:
  - complete_benchmark_summary_attempts.csv
      One row per response_*.json (attempt), flattened fields + resource_* columns.
  - complete_benchmark_summary.csv
      One row per (run_id, model, challenge, question), using FIRST SUCCESS WINS.
      Includes cost_to_success_usd and success-only fields.
  - complete_benchmark_summary_aggregated.csv
      One row per (model, challenge, question), aggregated across runs:
      mean ± std computed over successes only, plus n_success/n_runs.
  - (optional) LaTeX tables (performance + metadata split into generation/training)
"""

# Extract from multiple runs
# python utils/extract_benchmark_summary.py --input "outputs/01-05(2)/FOURTOPS/" --input "outputs/01-06/FOURTOPS/" --outdir benchmark_summary --emit-latex

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


FNAME_RE = re.compile(r"^response_(?P<stem>.+)\.json$", re.IGNORECASE)


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def infer_run_challenge_question(json_path: Path) -> Tuple[str, str, str]:
    parts = list(json_path.parts)
    try:
        i = next(idx for idx, p in enumerate(parts) if p.lower() == "outputs")
        run_id = parts[i + 1]
        challenge = parts[i + 2]
        question = parts[i + 3]
        return run_id, challenge, question
    except Exception:
        raise ValueError(f"Cannot infer run/challenge/question from path: {json_path}")


def parse_model_and_attempt_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse: response_<MODEL>_HH:MM_<ATTEMPT>.json
    Parse from the right so model can contain underscores.
    """
    m = FNAME_RE.match(filename)
    if not m:
        return None, None
    stem = m.group("stem")

    if "_" not in stem:
        return stem, None

    left, attempt_s = stem.rsplit("_", 1)
    attempt = _to_int(attempt_s)

    if "_" not in left:
        return left, attempt

    model_part, _time_part = left.rsplit("_", 1)
    return model_part, attempt


def detect_metric(challenge: str, eval_metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(eval_metrics, dict):
        return None, None

    ch = challenge.upper()

    if ch == "FOURTOPS":
        return "auc", _to_float(eval_metrics.get("auc", None))

    if ch == "TRACKFORMERS":
        for k in ["FitAccuracy", "fit_accuracy", "fitaccuracy", "Fit_Accuracy", "fitAccuracy"]:
            if k in eval_metrics:
                return "FitAccuracy", _to_float(eval_metrics[k])
        return "FitAccuracy", None

    for k, v in eval_metrics.items():
        fv = _to_float(v)
        if fv is not None:
            return k, fv
    return None, None


def stage_from_flags(
    pylint_pass: Optional[bool],
    bandit_pass: Optional[bool],
    dryrun_pass: Optional[bool],
    training_pass: Optional[bool],
    eval_pass: Optional[bool],
) -> str:
    if pylint_pass is False or bandit_pass is False:
        return "static_failed"
    if dryrun_pass is False:
        return "dryrun_failed"
    if training_pass is False:
        return "training_failed"
    if eval_pass is False:
        return "evaluation_failed"
    if any(x is None for x in [pylint_pass, bandit_pass, dryrun_pass, training_pass, eval_pass]):
        return "unknown"
    return "success"


def iter_response_jsons(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.name.lower().startswith("response_") and root.suffix.lower() == ".json":
            yield root
        elif root.is_dir():
            yield from root.rglob("response_*.json")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _epochs_actual_from_losses(data: Dict[str, Any]) -> Optional[int]:
    train_loss = _safe_get(data, ["Training", "metrics", "train_loss"], default=None)
    val_loss = _safe_get(data, ["Training", "metrics", "val_loss"], default=None)

    # If lists exist but are empty -> treat as missing (None)
    if isinstance(train_loss, list) and len(train_loss) > 0:
        return len(train_loss)
    if isinstance(val_loss, list) and len(val_loss) > 0:
        return len(val_loss)
    return None


def extract_attempt_row(json_path: Path) -> Dict[str, Any]:
    data = load_json(json_path)

    run_id, challenge, question = infer_run_challenge_question(json_path)
    model_from_fname, attempt_from_fname = parse_model_and_attempt_from_filename(json_path.name)

    llm = data.get("LLMGeneration", {}) if isinstance(data.get("LLMGeneration", {}), dict) else {}

    model_id = llm.get("model", None)
    model_key = model_id or model_from_fname or "UNKNOWN_MODEL"

    outer_attempt = _to_int(_first_present(llm, ["Outer_attempt", "Outer attempt", "outer_attempt"], default=None))
    inner_attempt = _to_int(_first_present(llm, ["Inner attempt", "Inner_attempt", "inner_attempt"], default=None))

    pylint_pass = _safe_get(data, ["StaticChecks", "PyLint", "passed"], default=None)
    bandit_pass = _safe_get(data, ["StaticChecks", "Bandit", "passed"], default=None)

    dryrun_pass = _safe_get(data, ["DryRun", "passed"], default=None)
    training_pass = _safe_get(data, ["Training", "passed"], default=None)
    eval_pass = _safe_get(data, ["Evaluation", "passed"], default=None)

    cost_usd = _to_float(llm.get("cost_usd", None))
    generation_ms = _to_float(llm.get("generation_ms", None))

    # Only response tokens (reasoning tokens are inconsistent across providers)
    response_tokens = _to_int(llm.get("response_tokens", None))

    training_time_s = _to_float(_safe_get(data, ["Training", "resources", "training_time_s"], default=None))

    epochs_planned = _to_int(_safe_get(data, ["Training", "metrics", "epochs"], default=None))
    epochs_actual = _epochs_actual_from_losses(data)

    eval_metrics = _safe_get(data, ["Evaluation", "metrics"], default={})
    metric_name, metric_value = detect_metric(challenge, eval_metrics if isinstance(eval_metrics, dict) else {})

    stage = stage_from_flags(pylint_pass, bandit_pass, dryrun_pass, training_pass, eval_pass)

    resources = _safe_get(data, ["Training", "resources"], default={})
    if not isinstance(resources, dict):
        resources = {}

    row: Dict[str, Any] = {
        "run_id": run_id,
        "challenge": challenge,
        "question": question,
        "model_key": model_key,
        "model_id": model_id,
        "model_from_filename": model_from_fname,
        "json_path": str(json_path),

        "attempt_filename": attempt_from_fname,
        "outer_attempt": outer_attempt,
        "inner_attempt": inner_attempt,

        "pylint_pass": pylint_pass,
        "bandit_pass": bandit_pass,
        "dryrun_pass": dryrun_pass,
        "training_pass": training_pass,
        "eval_pass": eval_pass,
        "stage": stage,

        "cost_usd": cost_usd,
        "generation_ms": generation_ms,
        "response_tokens": response_tokens,

        "training_time_s": training_time_s,

        "epochs_planned": epochs_planned,
        "epochs_actual": epochs_actual,

        "metric_name": metric_name,
        "metric_value": metric_value,
    }

    for k, v in resources.items():
        row[f"resource_{k}"] = v

    return row


def choose_attempt_index(row: pd.Series) -> int:
    for key in ["attempt_filename", "outer_attempt", "inner_attempt"]:
        v = row.get(key, None)
        if pd.notna(v):
            try:
                return int(v)
            except Exception:
                pass
    return 10**9


def build_success_level(attempts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse attempt-level rows into one row per (run_id, challenge, question, model_key),
    using FIRST SUCCESS WINS logic.

    Keeps:
      - cost_to_success_usd (sum costs up to success)
      - attempts_to_success + (static/dryrun fail counts before success)
      - success-only fields (generation time, response tokens, training time, epochs planned/actual, metric)
    """
    df = attempts_df.copy()
    df["attempt_index"] = df.apply(choose_attempt_index, axis=1)

    df["is_success"] = df["eval_pass"].apply(lambda x: True if x is True else False)

    group_cols = ["run_id", "challenge", "question", "model_key"]
    out_rows: List[Dict[str, Any]] = []

    for (run_id, challenge, question, model_key), g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("attempt_index")

        g_succ = g[g["is_success"]]
        success_row = g_succ.iloc[0] if len(g_succ) > 0 else None

        if success_row is not None:
            success_attempt = int(success_row["attempt_index"])
            window = g[g["attempt_index"] <= success_attempt]
        else:
            success_attempt = None
            window = g  # full window for totals spent (optional)

        static_failed = int(window["stage"].eq("static_failed").sum())
        dryrun_failed = int(window["stage"].eq("dryrun_failed").sum())

        # Total cost spent up to success (or to the end, if no success)
        cost_spent_usd = window["cost_usd"].dropna().sum() if "cost_usd" in window.columns else None

        # Primary quantity: cost-to-success only makes sense for success
        cost_to_success_usd = float(cost_spent_usd) if (success_row is not None and cost_spent_usd is not None) else None

        attempts_to_success = success_attempt  # assumes attempt indices are 1..N (matches your naming)
        attempts_signature = f"{attempts_to_success}({static_failed}/{dryrun_failed})" if attempts_to_success is not None else None

        # Success-only fields
        def succ_float(col: str) -> Optional[float]:
            if success_row is None:
                return None
            v = success_row.get(col, None)
            return float(v) if pd.notna(v) else None

        def succ_int(col: str) -> Optional[int]:
            if success_row is None:
                return None
            v = success_row.get(col, None)
            return int(v) if pd.notna(v) else None

        success_cost_usd = succ_float("cost_usd")
        success_generation_s = None
        gen_ms = succ_float("generation_ms")
        if gen_ms is not None:
            success_generation_s = gen_ms / 1000.0

        success_response_tokens = succ_int("response_tokens")

        training_time_s = succ_float("training_time_s")
        epochs_planned = succ_int("epochs_planned")
        epochs_actual = succ_int("epochs_actual")

        metric_value = succ_float("metric_value")
        metric_name = success_row.get("metric_name", None) if success_row is not None else None

        # Determine model_id (prefer success row)
        model_id = None
        if success_row is not None and pd.notna(success_row.get("model_id", None)):
            model_id = success_row.get("model_id", None)
        else:
            non_null = g["model_id"].dropna()
            model_id = non_null.iloc[0] if len(non_null) > 0 else None

        out_rows.append({
            "run_id": run_id,
            "challenge": challenge,
            "question": question,
            "model_key": model_key,
            "model_id": model_id,

            "success": True if success_row is not None else False,

            "success_attempt": success_attempt,
            "attempts_to_success": attempts_to_success,
            "static_fail_before_success": static_failed,
            "dryrun_fail_before_success": dryrun_failed,
            "attempts_signature": attempts_signature,

            "cost_to_success_usd": cost_to_success_usd,
            "cost_spent_usd": float(cost_spent_usd) if cost_spent_usd is not None else None,

            "success_cost_usd": success_cost_usd,
            "success_generation_s": success_generation_s,
            "success_response_tokens": success_response_tokens,

            "training_time_s": training_time_s,
            "epochs_planned": epochs_planned,
            "epochs_actual": epochs_actual,

            "metric_name": metric_name,
            "metric_value": metric_value,
        })

    return pd.DataFrame(out_rows)


def build_aggregated(success_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across runs per (model, challenge, question).

    - metric mean/std computed over successes only
    - metadata means/std computed over successes only
    - plus n_runs and n_success for transparency
    """
    df = success_df.copy()
    group_cols = ["model_key", "challenge", "question", "metric_name"]

    def _mean_std_over_success(g: pd.DataFrame, col: str) -> Tuple[Optional[float], Optional[float]]:
        s = g.loc[g["success"] == True, col].dropna()
        if len(s) == 0:
            return None, None
        return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0

    out_rows: List[Dict[str, Any]] = []

    for (model_key, challenge, question, metric_name), g in df.groupby(group_cols, dropna=False):
        n_runs = g["run_id"].nunique()
        n_success = int(g["success"].sum())

        metric_mean, metric_std = _mean_std_over_success(g, "metric_value")
        cost_mean, cost_std = _mean_std_over_success(g, "cost_to_success_usd")
        gen_mean, gen_std = _mean_std_over_success(g, "success_generation_s")
        tok_mean, tok_std = _mean_std_over_success(g, "success_response_tokens")
        train_mean, train_std = _mean_std_over_success(g, "training_time_s")
        ep_plan_mean, ep_plan_std = _mean_std_over_success(g, "epochs_planned")
        ep_act_mean, ep_act_std = _mean_std_over_success(g, "epochs_actual")
        att_mean, att_std = _mean_std_over_success(g, "attempts_to_success")

        out_rows.append({
            "model_key": model_key,
            "challenge": challenge,
            "question": question,
            "metric_name": metric_name,

            "n_runs": int(n_runs),
            "n_success": int(n_success),

            "metric_mean": metric_mean,
            "metric_std": metric_std,

            "cost_to_success_mean": cost_mean,
            "cost_to_success_std": cost_std,

            "success_generation_s_mean": gen_mean,
            "success_generation_s_std": gen_std,

            "success_response_tokens_mean": tok_mean,
            "success_response_tokens_std": tok_std,

            "training_time_s_mean": train_mean,
            "training_time_s_std": train_std,

            "epochs_planned_mean": ep_plan_mean,
            "epochs_planned_std": ep_plan_std,

            "epochs_actual_mean": ep_act_mean,
            "epochs_actual_std": ep_act_std,

            "attempts_mean": att_mean,
            "attempts_std": att_std,
        })

    return pd.DataFrame(out_rows)


def _fmt_mean_sd(mean: Optional[float], sd: Optional[float], digits: int) -> str:
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return "—"
    if sd is None or (isinstance(sd, float) and pd.isna(sd)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def emit_latex_tables(agg_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Emits:
      - performance_table.tex  (all questions in one table; includes [n_success/n_runs] inside cells)
      - metadata_generation_table.tex
      - metadata_training_table.tex
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = agg_df.copy()
    agg["challenge_question"] = agg["challenge"].astype(str) + " " + agg["question"].astype(str)

    # ---- Performance table (one table, all questions) ----
    perf = agg.copy()
    perf["cell"] = perf.apply(
        lambda r: f"{_fmt_mean_sd(r['metric_mean'], r['metric_std'], 3)} [{int(r['n_success'])}/{int(r['n_runs'])}]",
        axis=1,
    )
    perf_piv = perf.pivot_table(index="model_key", columns="challenge_question", values="cell", aggfunc="first")

    (out_dir / "performance_table.tex").write_text(
        perf_piv.to_latex(
            escape=True,
            na_rep="—",
            caption="Performance (mean ± SD over successful runs) with success counts [n\\_success/n\\_runs].",
            label="tab:benchmark_performance",
        ),
        encoding="utf-8",
    )

    # ---- Metadata: Generation ----
    gen = agg.copy()
    gen["cost"] = gen.apply(lambda r: _fmt_mean_sd(r["cost_to_success_mean"], r["cost_to_success_std"], 3), axis=1)
    gen["gen_s"] = gen.apply(lambda r: _fmt_mean_sd(r["success_generation_s_mean"], r["success_generation_s_std"], 2), axis=1)
    gen["tokens"] = gen.apply(lambda r: _fmt_mean_sd(r["success_response_tokens_mean"], r["success_response_tokens_std"], 0), axis=1)
    gen["attempts"] = gen.apply(lambda r: _fmt_mean_sd(r["attempts_mean"], r["attempts_std"], 2), axis=1)

    gen_small = gen[["model_key", "challenge", "question", "attempts", "cost", "gen_s", "tokens"]].rename(
        columns={
            "attempts": "Attempts",
            "cost": "Cost to success (USD)",
            "gen_s": "Gen time (s)",
            "tokens": "Response tokens",
        }
    )

    (out_dir / "metadata_generation_table.tex").write_text(
        gen_small.to_latex(
            index=False,
            escape=True,
            na_rep="—",
            caption="Generation metadata (mean ± SD over successful runs).",
            label="tab:benchmark_metadata_generation",
        ),
        encoding="utf-8",
    )

    # ---- Metadata: Training ----
    tr = agg.copy()
    tr["train_s"] = tr.apply(lambda r: _fmt_mean_sd(r["training_time_s_mean"], r["training_time_s_std"], 1), axis=1)
    tr["ep_plan"] = tr.apply(lambda r: _fmt_mean_sd(r["epochs_planned_mean"], r["epochs_planned_std"], 1), axis=1)
    tr["ep_act"] = tr.apply(lambda r: _fmt_mean_sd(r["epochs_actual_mean"], r["epochs_actual_std"], 1), axis=1)

    tr_small = tr[["model_key", "challenge", "question", "train_s", "ep_plan", "ep_act"]].rename(
        columns={
            "train_s": "Train time (s)",
            "ep_plan": "Epochs planned",
            "ep_act": "Epochs actual",
        }
    )

    (out_dir / "metadata_training_table.tex").write_text(
        tr_small.to_latex(
            index=False,
            escape=True,
            na_rep="—",
            caption="Training metadata (mean ± SD over successful runs).",
            label="tab:benchmark_metadata_training",
        ),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input folder(s) to scan. Repeatable. Example: --input outputs/01-05(2) --input outputs/01-06",
    )
    p.add_argument("--outdir", default="benchmark_summary", help="Output directory for CSVs and LaTeX (optional).")
    p.add_argument("--emit-latex", action="store_true", help="Emit LaTeX tables into <outdir>/latex/")
    args = p.parse_args()

    roots = [Path(x).resolve() for x in args.input]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(set(iter_response_jsons(roots)))
    if not json_paths:
        raise SystemExit("No response_*.json files found under the given inputs.")

    rows = []
    for jp in json_paths:
        try:
            rows.append(extract_attempt_row(jp))
        except Exception as e:
            print(f"[WARN] Failed to parse {jp}: {e}")

    attempts_df = pd.DataFrame(rows)
    attempts_csv = outdir / "complete_benchmark_summary_attempts.csv"
    attempts_df.to_csv(attempts_csv, index=False)

    success_df = build_success_level(attempts_df)
    success_csv = outdir / "complete_benchmark_summary.csv"
    success_df.to_csv(success_csv, index=False)

    agg_df = build_aggregated(success_df)
    agg_csv = outdir / "complete_benchmark_summary_aggregated.csv"
    agg_df.to_csv(agg_csv, index=False)

    if args.emit_latex:
        emit_latex_tables(agg_df, outdir / "latex")

    print(f"Wrote:\n  {attempts_csv}\n  {success_csv}\n  {agg_csv}")
    if args.emit_latex:
        print(f"Wrote LaTeX tables in: {outdir / 'latex'}")


if __name__ == "__main__":
    main()
