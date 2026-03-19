# ./utils/plot_benchmark_summary.py
#
# Reads the global benchmark_summary.json and generates nice plots.
#
# Outputs:
#   - outdir/<CHALLENGE>_<QID>/...png
#   - outdir/_cross_question/...png
#
# Example usage:
#   python utils/plot_benchmark_summary.py --summary "benchmark_summary/benchmark_summary.json" --outdir "benchmark_summary/plots" --label-points
#   python utils/plot_benchmark_summary.py --summary "benchmark_summary/benchmark_summary.json" --outdir "benchmark_summary/plots" --only-question "FOURTOPS/Q1"


from __future__ import annotations

import argparse, json, math, textwrap, csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Plot Globals - plt.tight
RIGHT_WIDTH = 1 # For all
BOTTOM_WIDTH = 0 # For cross, grouped and bar

mpl.rcParams["figure.figsize"] = (9, 4.5)
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 200

# Dynamic Figure Sizing
FIG_H = 4.5
BASE_W = 9
W_PER_MODEL = 0.55

# Figure Margins
mpl.rcParams["figure.subplot.left"]   = 0.1
mpl.rcParams["figure.subplot.right"]  = 0.8
mpl.rcParams["figure.subplot.bottom"] = 0.35
mpl.rcParams["figure.subplot.top"]    = 0.90

MODEL_LABELS = {
  "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
  "openai/gpt-5.2-pro": "GPT-5.2 Pro",
  "openai/gpt-5.1-codex-max": "GPT 5.1 Codex Max",
  "google/gemini-3-pro-preview": "Gemini 3 Pro Preview",
  "mistralai/mistral-large-2512": "Mistral Large 2512",
  "mistralai/devstral-2512:free": "DevStral 2512 Free",
  "x-ai/grok-code-fast-1": "Grok Code Fast 1"
}

AXIS_LABELS = {
    "metric_value": "Metric",
    "cost_to_success_usd": "Cost to success (USD)",
    "cost_spent_usd": "Total spend (USD)",
    "training_time_s": "Training time (s)",
    "success_generation_s": "Generation time (s)",
    "success_response_tokens": "Response tokens",
    "attempts_to_success": "Attempts to success",
    "attempts_total": "Attempts (total)",
    "final_train_loss": "Final train loss",
    "final_val_loss": "Final validation loss",
    "epochs_planned": "Epochs planned",
    "epochs_actual": "Epochs actual",
}


# IO helpers
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# Data extraction
def iter_question_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten JSON into per-success records.
    Each record corresponds to one (question_key, run_id, model) success.
    """
    out: List[Dict[str, Any]] = []
    questions = summary.get("questions", {})
    if not isinstance(questions, dict):
        return out

    for qkey, qblock in questions.items():
        if not isinstance(qblock, dict):
            continue
        challenge = qblock.get("challenge", None)
        qid = qblock.get("question", None)
        runs = qblock.get("runs", {})
        if not isinstance(runs, dict):
            continue

        for run_id, rblock in runs.items():
            if not isinstance(rblock, dict):
                continue

            successes = rblock.get("successes", {}) or {}
            final_fails = rblock.get("final_fails", {}) or {}

            # success records
            if isinstance(successes, dict):
                for model, mblock in successes.items():
                    if not isinstance(mblock, dict):
                        continue

                    gen = mblock.get("generation", {}) or {}
                    trn = mblock.get("training", {}) or {}
                    evl = mblock.get("evaluation", {}) or {}

                    effort = (gen.get("effort", {}) or {}) if isinstance(gen, dict) else {}
                    succ_att = (gen.get("success_attempt", {}) or {}) if isinstance(gen, dict) else {}

                    out.append({
                        "question_key": qkey,
                        "challenge": challenge,
                        "qid": qid,
                        "run_id": run_id,
                        "model": model,

                        "metric_name": evl.get("metric_name"),
                        "metric_value": evl.get("metric_value"),

                        # generation
                        "attempts_to_success": effort.get("attempts_to_success"),
                        "cost_to_success_usd": effort.get("cost_to_success_usd"),
                        "static_fail_total_before_success": (effort.get("static_fail_before_success") or {}).get("total"),
                        "static_fail_pylint_before_success": (effort.get("static_fail_before_success") or {}).get("pylint"),
                        "static_fail_bandit_before_success": (effort.get("static_fail_before_success") or {}).get("bandit"),
                        "dryrun_fail_before_success": effort.get("dryrun_fail_before_success"),

                        "success_cost_usd": succ_att.get("cost_usd"),
                        "success_generation_s": succ_att.get("generation_s"),
                        "success_response_tokens": succ_att.get("response_tokens"),

                        # training
                        "training_time_s": trn.get("training_time_s"),
                        "epochs_planned": trn.get("epochs_planned"),
                        "epochs_actual": trn.get("epochs_actual"),
                        "train_loss": trn.get("train_loss"),
                        "val_loss": trn.get("val_loss"),
                    })

            # also create “presence” info for reliability plots
            # (we'll compute per model: n_present, n_success from successes/final_fails)
            # no need to emit as records here; handled per question.
            _ = final_fails

    return out

def get_question_run_blocks(summary: Dict[str, Any], question_key: str) -> Dict[str, Any]:
    qblock = (summary.get("questions", {}) or {}).get(question_key, {})
    if not isinstance(qblock, dict):
        return {}
    runs = qblock.get("runs", {})
    if not isinstance(runs, dict):
        return {}
    return runs


# Plot helpers
def _clean_num(x):
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return x
        # strings?
        return float(x)
    except Exception:
        return None

def _wrap_title(s: str, width: int = 60) -> str:
    return "\n".join(textwrap.wrap(s, width=width))

def _ylim_bottom_zero(ax):
    lo, hi = ax.get_ylim()
    ax.set_ylim(bottom=0, top=hi)

def fig_w_for_models(n: int) -> float:
    return max(BASE_W, W_PER_MODEL * n)

def model_display_name(model_id: str, *, maxlen: int = 28) -> str:
    s = str(model_id or "")

    # 1) explicit mapping wins
    if s in MODEL_LABELS:
        return MODEL_LABELS[s]

    # 2) fallback: strip provider prefix
    if "/" in s:
        s = s.split("/", 1)[1]

    # 3) hard cap (optional)
    if len(s) > maxlen:
        s = s[: maxlen - 1] + "…"
    return s

def load_model_label_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except FileNotFoundError:
        return {}

def model_display_name_mapped(model_id: str, label_map: dict[str, str], *, maxlen: int = 28) -> str:
    if model_id in label_map:
        return label_map[model_id]
    return model_display_name(model_id, maxlen=maxlen)

def _legend_outside(ax, *, fontsize=8):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=fontsize)

def _lighten(color: str | tuple, factor: float = 0.4) -> tuple:
    """Return a lighter shade of *color*. factor ∈ (0,1): bigger = lighter."""
    r, g, b = to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

def pretty_label(key: str) -> str:
    if key in AXIS_LABELS:
        return AXIS_LABELS[key]
    s = str(key or "")
    s = s.replace("_", " ").strip()
    # simple title-case but keep acronyms
    words = []
    for w in s.split():
        if w.upper() in {"USD", "AUC"}:
            words.append(w.upper())
        elif w.lower() == "s":
            words.append("s")
        else:
            words.append(w.capitalize())
    return " ".join(words)


# Plot Functions
def scatter_by_model(points: list[dict], xkey: str, ykey: str, title: str, outpath: Path):
    """
    Scatter where each model is a separate series (color-coded), with legend outside.
    """
    # collect per model
    by_model: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        x = _clean_num(p.get(xkey))
        y = _clean_num(p.get(ykey))
        if x is None or y is None:
            continue
        by_model.setdefault(p.get("model", "UNKNOWN"), []).append((x, y))

    if not by_model:
        return

    plt.figure(figsize=(fig_w_for_models(len(by_model)), FIG_H))
    ax = plt.gca()

    # one scatter per model -> default color cycle handles colors
    for model in sorted(by_model.keys()):
        xs = [xy[0] for xy in by_model[model]]
        ys = [xy[1] for xy in by_model[model]]
        ax.scatter(xs, ys, label= model_display_name(model))

    ax.set_xlabel(pretty_label(xkey))
    ax.set_ylabel(pretty_label(ykey))
    ax.set_title(_wrap_title(title))

    _legend_outside(ax)
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH, 1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)

def bar_mean_std_by_model(points: List[Dict[str, Any]], value_key: str, title: str, outpath: Path):
    # group successes only by model
    by_model: Dict[str, List[float]] = {}
    for p in points:
        v = _clean_num(p.get(value_key))
        if v is None:
            continue
        by_model.setdefault(p["model"], []).append(v)

    if not by_model:
        return

    models = sorted(by_model.keys())
    means, stds = [], []
    for m in models:
        mu, sd = mean_std(by_model[m])
        means.append(mu if mu is not None else 0.0)
        stds.append(sd if sd is not None else 0.0)

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()
    ax.bar([model_display_name(m) for m in models], means, yerr=stds)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty_label(value_key))
    ax.set(ylim=[0, None])
    plt.xticks(rotation=45, ha="right")
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH, 1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_success_rate(summary: Dict[str, Any], question_key: str, title: str, outpath: Path):
    """
    Success rate per model for this question across all runs present in the summary.
    - n_present: count runs where model appears in successes OR final_fails
    - n_success: count runs where model appears in successes
    """
    runs = get_question_run_blocks(summary, question_key)
    if not runs:
        return

    present: Dict[str, int] = {}
    succ: Dict[str, int] = {}

    for run_id, rblock in runs.items():
        if not isinstance(rblock, dict):
            continue
        successes = rblock.get("successes", {}) or {}
        final_fails = rblock.get("final_fails", {}) or {}

        if isinstance(successes, dict):
            for model in successes.keys():
                present[model] = present.get(model, 0) + 1
                succ[model] = succ.get(model, 0) + 1

        if isinstance(final_fails, dict):
            for model in final_fails.keys():
                present[model] = present.get(model, 0) + 1

    if not present:
        return

    models = sorted(present.keys())
    rates = []
    for m in models:
        p = present.get(m, 0)
        s = succ.get(m, 0)
        rates.append((s / p) if p > 0 else 0.0)

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()
    ax.bar([model_display_name(m) for m in models], rates)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Success rate (n_success / n_present)")
    plt.xticks(rotation=45, ha="right")
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_epochs_planned_vs_actual(points: list[dict], title: str, outpath: Path):
    """
    For ONE question: per model, show mean±SD of epochs_planned and epochs_actual (successes only).
    Side-by-side bars per model. y-min pinned to 0, legend outside.
    """
    by_model_planned: dict[str, list[float]] = {}
    by_model_actual: dict[str, list[float]] = {}

    for p in points:
        m = p.get("model")
        if not m:
            continue
        ep_p = _clean_num(p.get("epochs_planned"))
        ep_a = _clean_num(p.get("epochs_actual"))
        if ep_p is not None:
            by_model_planned.setdefault(m, []).append(ep_p)
        if ep_a is not None:
            by_model_actual.setdefault(m, []).append(ep_a)

    models = sorted(set(by_model_planned.keys()) | set(by_model_actual.keys()))
    if not models:
        return

    planned_means, planned_stds = [], []
    actual_means, actual_stds = [], []

    for m in models:
        mu_p, sd_p = mean_std(by_model_planned.get(m, []))
        mu_a, sd_a = mean_std(by_model_actual.get(m, []))
        planned_means.append(mu_p if mu_p is not None else float("nan"))
        planned_stds.append(sd_p if sd_p is not None else 0.0)
        actual_means.append(mu_a if mu_a is not None else float("nan"))
        actual_stds.append(sd_a if sd_a is not None else 0.0)

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()

    x = list(range(len(models)))
    width = 0.35

    ax.bar([xi - width/2 for xi in x], planned_means, width=width, yerr=planned_stds, label="epochs_planned")
    ax.bar([xi + width/2 for xi in x], actual_means, width=width, yerr=actual_stds, label="epochs_actual")

    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel("Epochs")
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(m) for m in models], rotation=45, ha="right")

    _legend_outside(ax)
    _ylim_bottom_zero(ax)

    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_metric_per_usd(points: list[dict], title: str, outpath: Path):
    """
    For ONE question: efficiency = metric_value / cost_to_success_usd (successes only).
    Bar mean±SD per model.
    """
    by_model: dict[str, list[float]] = {}
    for p in points:
        m = p.get("model")
        metric = _clean_num(p.get("metric_value"))
        cost = _clean_num(p.get("cost_to_success_usd"))
        if not m or metric is None or cost is None or cost <= 0:
            continue
        by_model.setdefault(m, []).append(metric / cost)

    models = sorted(by_model.keys())
    if not models:
        return

    means, stds = [], []
    for m in models:
        mu, sd = mean_std(by_model[m])
        means.append(mu if mu is not None else float("nan"))
        stds.append(sd if sd is not None else 0.0)

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()
    ax.bar([model_display_name(m) for m in models], means, yerr=stds)
    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel("metric_value / cost_to_success_usd")

    _ylim_bottom_zero(ax)
    plt.xticks(rotation=45, ha="right")
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_loss_curves_per_model(points: list[dict], question_title: str, outdir: Path):
    """
    For each model: plot train/val loss curves per run.
    Train/val share the same base color per run; val is lighter and hollow markers.
    """
    by_model: dict[str, list[dict]] = {}
    for p in points:
        if p.get("train_loss") is None and p.get("val_loss") is None:
            continue
        by_model.setdefault(p["model"], []).append(p)

    for model, recs in by_model.items():
        plt.figure()
        ax = plt.gca()

        recs_sorted = sorted(recs, key=lambda r: str(r.get("run_id", "")))
        base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, r in enumerate(recs_sorted):
            run_id = str(r.get("run_id", "RUN"))
            train = r.get("train_loss")
            val = r.get("val_loss")

            base_col = base_colors[i % len(base_colors)]
            light_col = _lighten(base_col, 0.4)

            if isinstance(train, list) and len(train) > 0:
                epochs_t = list(range(1, len(train) + 1))
                ax.plot(
                    epochs_t, train,
                    color=base_col, marker="o", linestyle="-",
                    label=f"{run_id} train",
                    alpha=0.9
                )

            if isinstance(val, list) and len(val) > 0:
                epochs_v = list(range(1, len(val) + 1))
                ax.plot(
                    epochs_v, val,
                    color=light_col,
                    marker="o", markerfacecolor="none",
                    linestyle="--",
                    label=f"{run_id} val",
                    alpha=0.9
                )

        ax.set_title(_wrap_title(f"{question_title} — Loss curves — {model_display_name(model)}"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        _legend_outside(ax)
        # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
        plt.savefig(outdir / f"loss_curves__{model.replace('/', '_').replace(':', '_')}.png", dpi=200)
        plt.close()

def iter_cost_spent_records(summary: dict) -> list[dict]:
    """
    Return per-(question_key, run_id, model) records with cost spent:
      - successes: generation.effort.cost_to_success_usd
      - final_fails: generation.aggregates.cost_to_failure_usd
    """
    out = []
    questions = summary.get("questions", {}) or {}
    if not isinstance(questions, dict):
        return out

    for qkey, qblock in questions.items():
        if not isinstance(qblock, dict):
            continue
        challenge = qblock.get("challenge")
        qid = qblock.get("question")
        runs = qblock.get("runs", {}) or {}
        if not isinstance(runs, dict):
            continue

        for run_id, rblock in runs.items():
            if not isinstance(rblock, dict):
                continue

            # successes
            successes = rblock.get("successes", {}) or {}
            if isinstance(successes, dict):
                for model, mblock in successes.items():
                    gen = (mblock.get("generation", {}) or {})
                    effort = (gen.get("effort", {}) or {})
                    out.append({
                        "question_key": qkey,
                        "challenge": challenge,
                        "qid": qid,
                        "run_id": run_id,
                        "model": model,
                        "cost_spent_usd": effort.get("cost_to_success_usd"),
                        "success": True,
                    })

            # final fails
            final_fails = rblock.get("final_fails", {}) or {}
            if isinstance(final_fails, dict):
                for model, mblock in final_fails.items():
                    gen = (mblock.get("generation", {}) or {})
                    agg = (gen.get("aggregates", {}) or {})
                    out.append({
                        "question_key": qkey,
                        "challenge": challenge,
                        "qid": qid,
                        "run_id": run_id,
                        "model": model,
                        "cost_spent_usd": agg.get("cost_to_failure_usd"),
                        "success": False,
                    })

    return out

def bar_fourtops_q1_vs_q2_by_run(records: list[dict], value_key: str, title: str, outpath: Path, *, ylim0: bool = True):
    """
    For FOURTOPS only: for each run_id, compare Q1 vs Q2 (successes).
    Plots grouped bars: x=models, series are ["FOURTOPS/Q1", "FOURTOPS/Q2"] for the SAME run_id.
    Creates one figure PER run_id.
    """
    # filter FOURTOPS
    recs = [r for r in records if isinstance(r.get("question_key"), str) and r["question_key"].startswith("FOURTOPS/")]
    if not recs:
        return

    runs = sorted({str(r.get("run_id")) for r in recs if r.get("run_id") is not None})
    for run in runs:
        rr = [r for r in recs if str(r.get("run_id")) == run and r.get(value_key) is not None]
        if not rr:
            continue

        models = sorted({r["model"] for r in rr})
        qs = ["FOURTOPS/Q1", "FOURTOPS/Q2"]

        mat = {q: {m: float("nan") for m in models} for q in qs}
        for r in rr:
            qk = r["question_key"]
            if qk not in mat:
                continue
            v = _clean_num(r.get(value_key))
            if v is None:
                continue
            mat[qk][r["model"]] = v

        plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
        ax = plt.gca()
        x = list(range(len(models)))
        width = 0.35

        ax.bar([xi - width/2 for xi in x], [mat["FOURTOPS/Q1"][m] for m in models], width=width, label="FOURTOPS Q1")
        ax.bar([xi + width/2 for xi in x], [mat["FOURTOPS/Q2"][m] for m in models], width=width, label="FOURTOPS Q2")

        ax.set_title(_wrap_title(f"{title} — run {run}"))
        ax.set_xlabel("Model")
        ax.set_ylabel(pretty_label(value_key))
        ax.set_xticks(x)
        ax.set_xticklabels([model_display_name(m) for m in models], rotation=45, ha="right")

        _legend_outside(ax)
        if ylim0:
            _ylim_bottom_zero(ax)

        # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
        plt.savefig(outpath.with_name(outpath.stem + f"__run_{run}.png"), dpi=200)
        plt.close()

def _question_reliability_counts(summary: dict, question_key: str):
    """
    Returns:
      outcomes[model] = dict(success=.., fail_static_only=.., fail_with_dryrun=.., present=..)
      static_attempts[model] = dict(pylint=.., bandit=..)
    """
    runs = get_question_run_blocks(summary, question_key)
    outcomes: dict[str, dict[str, int]] = {}
    static_attempts: dict[str, dict[str, int]] = {}

    for run_id, rblock in (runs or {}).items():
        if not isinstance(rblock, dict):
            continue
        successes = rblock.get("successes", {}) or {}
        final_fails = rblock.get("final_fails", {}) or {}

        # successes: one per run/model
        if isinstance(successes, dict):
            for model in successes.keys():
                o = outcomes.setdefault(model, {"success": 0, "fail_static_only": 0, "fail_with_dryrun": 0, "present": 0})
                o["success"] += 1
                o["present"] += 1

        # final fails: classify disjointly by whether any dryrun_failed attempt occurred
        if isinstance(final_fails, dict):
            for model, mblock in final_fails.items():
                o = outcomes.setdefault(model, {"success": 0, "fail_static_only": 0, "fail_with_dryrun": 0, "present": 0})
                o["present"] += 1

                gen = (mblock.get("generation", {}) or {})
                attempts = gen.get("attempts", []) or []
                any_dry = False
                for a in attempts:
                    if isinstance(a, dict) and a.get("stage") == "dryrun_failed":
                        any_dry = True

                    # static attempt breakdown counts (attempt-level)
                    if isinstance(a, dict) and a.get("stage") == "static_failed":
                        sf = (a.get("static_failed", {}) or {})
                        sa = static_attempts.setdefault(model, {"pylint": 0, "bandit": 0})
                        if sf.get("pylint") is True:
                            sa["pylint"] += 1
                        if sf.get("bandit") is True:
                            sa["bandit"] += 1

                if any_dry:
                    o["fail_with_dryrun"] += 1
                else:
                    o["fail_static_only"] += 1

    return outcomes, static_attempts

def stacked_bar_reliability_funnel(summary: dict, question_key: str, title: str, outpath: Path):
    outcomes, _ = _question_reliability_counts(summary, question_key)
    if not outcomes:
        return

    models = sorted(outcomes.keys())
    succ = [outcomes[m]["success"] for m in models]
    fs   = [outcomes[m]["fail_static_only"] for m in models]
    fd   = [outcomes[m]["fail_with_dryrun"] for m in models]

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()

    xlabels = [model_display_name(m) for m in models]
    ax.bar(xlabels, succ, label="success")
    ax.bar(xlabels, fs, bottom=succ, label="final_fail_static_only")
    ax.bar(xlabels, fd, bottom=[s + f for s, f in zip(succ, fs)], label="final_fail_with_dryrun")

    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel("Runs (count)")
    plt.xticks(rotation=45, ha="right")

    _legend_outside(ax)
    _ylim_bottom_zero(ax)
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_static_breakdown_attempts(summary: dict, question_key: str, title: str, outpath: Path):
    _, static_attempts = _question_reliability_counts(summary, question_key)
    if not static_attempts:
        return

    models = sorted(static_attempts.keys())
    pyl = [static_attempts[m]["pylint"] for m in models]
    ban = [static_attempts[m]["bandit"] for m in models]

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()
    x = list(range(len(models)))
    w = 0.35

    ax.bar([xi - w/2 for xi in x], pyl, width=w, label="pylint flags (attempts)")
    ax.bar([xi + w/2 for xi in x], ban, width=w, label="bandit flags (attempts)")

    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel("Static-fail attempts (count)")
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(m) for m in models], rotation=45, ha="right")

    _legend_outside(ax)
    _ylim_bottom_zero(ax)
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def scatter_attempts_vs_metric(points: list[dict], title: str, outpath: Path):
    scatter_by_model(points, "attempts_to_success", "metric_value", title, outpath)

def bar_epoch_ratio(points: list[dict], title: str, outpath: Path):
    by_model: dict[str, list[float]] = {}
    for p in points:
        m = p.get("model")
        a = _clean_num(p.get("epochs_actual"))
        pl = _clean_num(p.get("epochs_planned"))
        if not m or a is None or pl is None or pl <= 0:
            continue
        by_model.setdefault(m, []).append(a / pl)

    models = sorted(by_model.keys())
    if not models:
        return

    means, stds = [], []
    for m in models:
        mu, sd = mean_std(by_model[m])
        means.append(mu if mu is not None else float("nan"))
        stds.append(sd if sd is not None else 0.0)

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()
    ax.bar([model_display_name(m) for m in models], means, yerr=stds)
    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel("epochs_actual / epochs_planned")

    _ylim_bottom_zero(ax)
    plt.xticks(rotation=45, ha="right")
    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def scatter_final_train_vs_val_loss(points: list[dict], title: str, outpath: Path):
    # Build derived points with last losses
    derived = []
    for p in points:
        tr = p.get("train_loss")
        va = p.get("val_loss")
        if isinstance(tr, list) and tr and isinstance(va, list) and va:
            derived.append({
                "model": p.get("model"),
                "final_train_loss": tr[-1],
                "final_val_loss": va[-1],
            })
    if not derived:
        return

    scatter_by_model(derived, "final_train_loss", "final_val_loss", title, outpath)

def scatter_fourtops_q1_vs_q2_metric(records: list[dict], title: str, outpath: Path):
    # successes only, aggregate mean per model per question
    q1 = [r for r in records if r.get("question_key") == "FOURTOPS/Q1" and _clean_num(r.get("metric_value")) is not None]
    q2 = [r for r in records if r.get("question_key") == "FOURTOPS/Q2" and _clean_num(r.get("metric_value")) is not None]

    by_m_q1: dict[str, list[float]] = {}
    by_m_q2: dict[str, list[float]] = {}

    for r in q1:
        by_m_q1.setdefault(r["model"], []).append(_clean_num(r["metric_value"]))
    for r in q2:
        by_m_q2.setdefault(r["model"], []).append(_clean_num(r["metric_value"]))

    common_models = sorted(set(by_m_q1.keys()) & set(by_m_q2.keys()))
    if not common_models:
        return

    pts = []
    for m in common_models:
        mu1, _ = mean_std(by_m_q1[m])
        mu2, _ = mean_std(by_m_q2[m])
        if mu1 is None or mu2 is None:
            continue
        pts.append({"model": m, "q1_mean": mu1, "q2_mean": mu2})

    scatter_by_model(pts, "q1_mean", "q2_mean", title, outpath)

def scatter_spend_vs_success_rate(summary: dict, question_key: str, title: str, outpath: Path):
    # total spend per model across runs (success cost_to_success + failure cost_to_failure)
    cost_records = iter_cost_spent_records(summary)
    cost_records = [r for r in cost_records if r.get("question_key") == question_key]

    spend_by_model: dict[str, float] = {}
    for r in cost_records:
        m = r.get("model")
        c = _clean_num(r.get("cost_spent_usd"))
        if not m or c is None:
            continue
        spend_by_model[m] = spend_by_model.get(m, 0.0) + c

    # success rate per model across runs
    outcomes, _ = _question_reliability_counts(summary, question_key)

    pts = []
    for m in sorted(set(spend_by_model.keys()) | set(outcomes.keys())):
        spend = spend_by_model.get(m)
        o = outcomes.get(m, None)
        if spend is None or o is None:
            continue
        present = o.get("present", 0)
        succ = o.get("success", 0)
        if present <= 0:
            continue
        pts.append({"model": m, "total_spend": spend, "success_rate": succ / present})

    if not pts:
        return

    scatter_by_model(pts, "total_spend", "success_rate", title, outpath)


# Cross-question comparisons
def grouped_bar_cross_question(
    records: list[dict],
    value_key: str,
    title: str,
    outpath: Path,
    *,
    question_filter_prefix: str | None = None,
    ylim0: bool = True,
):
    """
    Grouped bar: for each model, show mean±SD of value_key per question_key.
    - Uses records that already include question_key/model/value_key.
    - If question_filter_prefix is set (e.g. "FOURTOPS/"), only those questions are used.
    - Legend is outside, title is wrapped, y-min pinned to 0 by default.
    """
    # Filter question set
    filtered = []
    for r in records:
        qk = r.get("question_key")
        if question_filter_prefix and (not isinstance(qk, str) or not qk.startswith(question_filter_prefix)):
            continue
        filtered.append(r)

    # collect per (model, question_key)
    by_mq: dict[tuple[str, str], list[float]] = {}
    question_keys = sorted({r["question_key"] for r in filtered if "question_key" in r})
    models = sorted({r["model"] for r in filtered if "model" in r})

    for r in filtered:
        v = _clean_num(r.get(value_key))
        if v is None:
            continue
        by_mq.setdefault((r["model"], r["question_key"]), []).append(v)

    if not models or not question_keys:
        return

    # build means/stds
    means = {q: [] for q in question_keys}
    stds = {q: [] for q in question_keys}

    for m in models:
        for q in question_keys:
            mu, sd = mean_std(by_mq.get((m, q), []))
            means[q].append(mu if mu is not None else float("nan"))
            stds[q].append(sd if sd is not None else 0.0)

    # Plot grouped bars
    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()

    x = list(range(len(models)))
    width = 0.8 / max(1, len(question_keys))

    for i, q in enumerate(question_keys):
        offset = (i - (len(question_keys) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], means[q], width=width, yerr=stds[q], label=q.replace("/", " "))

    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty_label(value_key))
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(m) for m in models], rotation=45, ha="right")

    _legend_outside(ax)
    if ylim0:
        _ylim_bottom_zero(ax)

    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_by_run_for_question(points: list[dict], value_key: str, title: str, outpath: Path, *, ylim0: bool = True):
    """
    For one question: x=models, bars grouped by run_id (one bar per run).
    Uses success records (points).
    """
    # gather
    models = sorted({p["model"] for p in points})
    run_ids = sorted({str(p.get("run_id")) for p in points if p.get("run_id") is not None})

    # matrix run -> model -> value
    mat = {run: {m: float("nan") for m in models} for run in run_ids}
    for p in points:
        m = p["model"]
        run = str(p.get("run_id"))
        v = _clean_num(p.get(value_key))
        if v is None:
            continue
        mat[run][m] = v

    if not models or not run_ids:
        return

    plt.figure(figsize=(fig_w_for_models(len(models)), FIG_H))
    ax = plt.gca()

    x = list(range(len(models)))
    width = 0.8 / max(1, len(run_ids))

    for i, run in enumerate(run_ids):
        offset = (i - (len(run_ids) - 1) / 2) * width
        ys = [mat[run][m] for m in models]
        ax.bar([xi + offset for xi in x], ys, width=width, label=run)

    ax.set_title(_wrap_title(title))
    ax.set_xlabel("Model")
    ax.set_ylabel(pretty_label(value_key))
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(m) for m in models], rotation=45, ha="right")

    _legend_outside(ax)
    if ylim0:
        _ylim_bottom_zero(ax)

    # plt.tight_layout(rect=[0, BOTTOM_WIDTH, RIGHT_WIDTH,1])
    plt.savefig(outpath, dpi=200)
    plt.close()


# CSV Rankings 
def export_rankings(records: list[dict], outdir: Path):
    """
    For each question_key:
      - list each model
      - raw success metric values (per run)
      - mean, sd
      - rank by mean desc
    Writes: outdir/<QUESTIONKEY>_rankings.csv
    """
    by_q: dict[str, list[dict]] = {}
    for r in records:
        if _clean_num(r.get("metric_value")) is None:
            continue
        by_q.setdefault(r["question_key"], []).append(r)

    for qkey, recs in by_q.items():
        # model -> list of (run_id, metric)
        by_m: dict[str, list[tuple[str, float]]] = {}
        for r in recs:
            by_m.setdefault(r["model"], []).append((str(r.get("run_id")), float(r["metric_value"])))

        rows = []
        for model, vals in by_m.items():
            metrics = [v for _, v in vals]
            mu, sd = mean_std(metrics)
            rows.append({
                "question_key": qkey,
                "model": model,
                "n_success": len(metrics),
                "mean": mu,
                "sd": sd,
                "raw_scores": ";".join(f"{run}:{v:.6g}" for run, v in sorted(vals)),
            })

        rows.sort(key=lambda x: (x["mean"] is None, -(x["mean"] or -1e18)))  # mean desc, None last

        outpath = outdir / f"{qkey.replace('/', '_')}_rankings.csv"
        with outpath.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question_key","model","n_success","mean","sd","raw_scores"])
            w.writeheader()
            w.writerows(rows)

def iter_run_unit_records(summary: dict) -> list[dict]:
    """
    One record per (question_key, run_id, model) representing one "unit" attempted.

    Includes:
      - cost_spent_usd: success cost_to_success OR fail cost_to_failure
      - training_time_s: successes only
      - attempts_total: attempts_to_success OR attempts_total (fail)
      - static/dryrun fail attempt counts (attempt-level aggregates)
      - success_generation_s, success_response_tokens (success attempt only)
      - epochs_planned/actual (success only)
      - metric_value (success only)
    """
    out = []
    questions = summary.get("questions", {}) or {}
    if not isinstance(questions, dict):
        return out

    for qkey, qblock in questions.items():
        if not isinstance(qblock, dict):
            continue
        challenge = qblock.get("challenge")
        qid = qblock.get("question")
        runs = qblock.get("runs", {}) or {}
        if not isinstance(runs, dict):
            continue

        for run_id, rblock in runs.items():
            if not isinstance(rblock, dict):
                continue

            successes = rblock.get("successes", {}) or {}
            final_fails = rblock.get("final_fails", {}) or {}

            # successes
            if isinstance(successes, dict):
                for model, mblock in successes.items():
                    gen = (mblock.get("generation", {}) or {})
                    effort = (gen.get("effort", {}) or {})
                    succ_att = (gen.get("success_attempt", {}) or {})
                    trn = (mblock.get("training", {}) or {})
                    evl = (mblock.get("evaluation", {}) or {})

                    sf = (effort.get("static_fail_before_success") or {})
                    out.append({
                        "question_key": qkey,
                        "challenge": challenge,
                        "qid": qid,
                        "run_id": run_id,
                        "model": model,
                        "succeeded": True,

                        # spend + attempts
                        "cost_spent_usd": effort.get("cost_to_success_usd"),
                        "attempts_total": effort.get("attempts_to_success"),
                        "static_fail_attempts_total": sf.get("total"),
                        "static_fail_attempts_pylint": sf.get("pylint"),
                        "static_fail_attempts_bandit": sf.get("bandit"),
                        "dryrun_fail_attempts_total": effort.get("dryrun_fail_before_success"),

                        # success attempt generation
                        "success_generation_s": succ_att.get("generation_s"),
                        "success_response_tokens": succ_att.get("response_tokens"),

                        # training
                        "training_time_s": trn.get("training_time_s"),
                        "epochs_planned": trn.get("epochs_planned"),
                        "epochs_actual": trn.get("epochs_actual"),

                        # evaluation
                        "metric_value": evl.get("metric_value"),
                    })

            # final fails
            if isinstance(final_fails, dict):
                for model, mblock in final_fails.items():
                    gen = (mblock.get("generation", {}) or {})
                    agg = (gen.get("aggregates", {}) or {})

                    sf = (agg.get("static_fail_total") or {})
                    out.append({
                        "question_key": qkey,
                        "challenge": challenge,
                        "qid": qid,
                        "run_id": run_id,
                        "model": model,
                        "succeeded": False,

                        "cost_spent_usd": agg.get("cost_to_failure_usd"),
                        "attempts_total": agg.get("attempts_total"),
                        "static_fail_attempts_total": sf.get("total"),
                        "static_fail_attempts_pylint": sf.get("pylint"),
                        "static_fail_attempts_bandit": sf.get("bandit"),
                        "dryrun_fail_attempts_total": agg.get("dryrun_fail_total"),

                        "success_generation_s": None,
                        "success_response_tokens": None,
                        "training_time_s": None,
                        "epochs_planned": None,
                        "epochs_actual": None,
                        "metric_value": None,
                    })

    return out

def export_overall_stats(summary: dict, outdir: Path, *, filename: str = "overall_stats.csv"):
    """
    Writes overall_stats.csv aggregated at:
      - global
      - per question
      - per question x run_id

    Adds robust distribution stats + reliability/cost-efficiency metrics.
    """
    records = iter_run_unit_records(summary)
    if not records:
        return

    def _num(x):
        return _clean_num(x)

    def _median(vals: list[float]) -> float | None:
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _percentile(vals: list[float], p: float) -> float | None:
        """Nearest-rank percentile (simple + robust). p in [0,100]."""
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        s = sorted(vals)
        if p <= 0:
            return s[0]
        if p >= 100:
            return s[-1]
        k = int(math.ceil((p / 100.0) * len(s))) - 1
        k = max(0, min(k, len(s) - 1))
        return s[k]

    def _minmax(vals: list[float]) -> tuple[float | None, float | None]:
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return min(vals), max(vals)

    def _agg(key: tuple, rows: list[dict]) -> dict:
        level = key[0]
        question_key = key[1] if len(key) >= 2 else ""
        run_id = key[2] if len(key) >= 3 else ""

        total_units = len(rows)
        success_rows = [r for r in rows if r.get("succeeded") is True]
        fail_rows = [r for r in rows if r.get("succeeded") is False]

        # Costs
        costs_all = [_num(r.get("cost_spent_usd")) for r in rows]
        costs_succ = [_num(r.get("cost_spent_usd")) for r in success_rows]
        costs_fail = [_num(r.get("cost_spent_usd")) for r in fail_rows]
        total_cost = sum(c for c in costs_all if c is not None)
        total_cost_succ = sum(c for c in costs_succ if c is not None)
        total_cost_fail = sum(c for c in costs_fail if c is not None)

        # Attempts
        attempts_all = [_num(r.get("attempts_total")) for r in rows]
        attempts_succ = [_num(r.get("attempts_total")) for r in success_rows]
        attempts_fail = [_num(r.get("attempts_total")) for r in fail_rows]

        # Fail attempt breakdown (attempt-level aggregates)
        static_total = sum(int(_num(r.get("static_fail_attempts_total")) or 0) for r in rows)
        static_pyl = sum(int(_num(r.get("static_fail_attempts_pylint")) or 0) for r in rows)
        static_ban = sum(int(_num(r.get("static_fail_attempts_bandit")) or 0) for r in rows)
        dry_total = sum(int(_num(r.get("dryrun_fail_attempts_total")) or 0) for r in rows)

        # Training (success only)
        train_times = [_num(r.get("training_time_s")) for r in success_rows]
        total_train = sum(t for t in train_times if t is not None)

        # Generation (success attempt only)
        gen_s = [_num(r.get("success_generation_s")) for r in success_rows]
        toks = [_num(r.get("success_response_tokens")) for r in success_rows]

        # Epoch behavior (success only)
        ep_pl = [_num(r.get("epochs_planned")) for r in success_rows]
        ep_ac = [_num(r.get("epochs_actual")) for r in success_rows]
        ep_ratio = []
        for r in success_rows:
            a = _num(r.get("epochs_actual"))
            p_ = _num(r.get("epochs_planned"))
            if a is not None and p_ is not None and p_ > 0:
                ep_ratio.append(a / p_)

        # Metric across successes (optional but useful for ops summary)
        mets = [_num(r.get("metric_value")) for r in success_rows]
        mu_met, sd_met = mean_std([m for m in mets if m is not None])

        # Basic counts
        success_units = len(success_rows)
        fail_units = len(fail_rows)
        success_rate = (success_units / total_units) if total_units > 0 else None

        # Basic counts (define before using)
        success_units = len(success_rows)
        fail_units = len(fail_rows)

        # Total LLM prompt calls (attempts), not just units
        def _attempts_or_one(r: dict) -> int:
            a = _num(r.get("attempts_total"))
            if a is None:
                return 1
            try:
                a_i = int(a)
            except Exception:
                return 1
            return max(1, a_i)

        prompt_calls_total = sum(_attempts_or_one(r) for r in rows)
        prompt_calls_success = sum(_attempts_or_one(r) for r in success_rows)
        prompt_calls_fail = sum(_attempts_or_one(r) for r in fail_rows)

        mean_prompt_calls_per_unit = (prompt_calls_total / total_units) if total_units > 0 else None
        mean_prompt_calls_per_success_unit = (prompt_calls_total / success_units) if success_units > 0 else None

        # Distinct models in this group
        models_present = {r.get("model") for r in rows if r.get("model")}
        models_success = {r.get("model") for r in success_rows if r.get("model")}
        models_fail = {r.get("model") for r in fail_rows if r.get("model")}

        # Robust stats helpers
        c_med = _median(costs_all)
        c_p90 = _percentile(costs_all, 90)
        c_min, c_max = _minmax(costs_all)

        t_med = _median(train_times)
        t_p90 = _percentile(train_times, 90)
        t_min, t_max = _minmax(train_times)

        a_med = _median(attempts_all)
        a_p90 = _percentile(attempts_all, 90)
        a_min, a_max = _minmax(attempts_all)

        # Derived cost efficiency / reliability
        cost_per_success_unit = (total_cost / success_units) if success_units > 0 else None
        failure_cost_share = (total_cost_fail / total_cost) if total_cost > 0 else None

        # Averages
        mean_cost_per_unit = (total_cost / total_units) if total_units > 0 else None
        mean_train_per_success = (total_train / success_units) if success_units > 0 else None
        mean_gen_s_per_success = (sum(g for g in gen_s if g is not None) / success_units) if success_units > 0 else None
        mean_toks_per_success = (sum(t for t in toks if t is not None) / success_units) if success_units > 0 else None
        mean_attempts_per_unit = (sum(a for a in attempts_all if a is not None) / total_units) if total_units > 0 else None
        mean_attempts_per_success = (sum(a for a in attempts_succ if a is not None) / success_units) if success_units > 0 else None

        # Epoch ratio mean
        mu_ratio, sd_ratio = mean_std([x for x in ep_ratio if x is not None])

        return {
            "level": level,
            "question_key": question_key,
            "run_id": run_id,

            "total_units": total_units,
            "total_success_units": success_units,
            "total_fail_units": fail_units,
            "success_rate_units": success_rate,

            "total_prompt_calls": prompt_calls_total,
            "total_prompt_calls_success_units": prompt_calls_success,
            "total_prompt_calls_fail_units": prompt_calls_fail,
            "mean_prompt_calls_per_unit": mean_prompt_calls_per_unit,
            "mean_prompt_calls_per_success_unit": mean_prompt_calls_per_success_unit,

            "n_models_present": len(models_present),
            "n_models_success": len(models_success),
            "n_models_fail": len(models_fail),

            "total_cost_spent_usd": total_cost,
            "total_cost_success_usd": total_cost_succ,
            "total_cost_fail_usd": total_cost_fail,
            "failure_cost_share": failure_cost_share,
            "cost_per_success_unit": cost_per_success_unit,

            "cost_min": c_min,
            "cost_median": c_med,
            "cost_p90": c_p90,
            "cost_max": c_max,

            "total_training_time_s": total_train,
            "train_time_min": t_min,
            "train_time_median": t_med,
            "train_time_p90": t_p90,
            "train_time_max": t_max,

            "total_static_fail_attempts": static_total,
            "static_fail_attempts_pylint": static_pyl,
            "static_fail_attempts_bandit": static_ban,
            "total_dryrun_fail_attempts": dry_total,

            "attempts_min": a_min,
            "attempts_median": a_med,
            "attempts_p90": a_p90,
            "attempts_max": a_max,

            "mean_cost_per_unit": mean_cost_per_unit,
            "mean_train_time_s_per_success": mean_train_per_success,
            "mean_success_generation_s": mean_gen_s_per_success,
            "mean_success_response_tokens": mean_toks_per_success,
            "mean_attempts_per_unit": mean_attempts_per_unit,
            "mean_attempts_per_success": mean_attempts_per_success,

            "mean_metric_success": mu_met,
            "sd_metric_success": sd_met,

            "mean_epoch_ratio_success": mu_ratio,
            "sd_epoch_ratio_success": sd_ratio,
        }

    # groupings
    groups: dict[tuple, list[dict]] = {}
    groups[("global",)] = records

    for r in records:
        qk = r.get("question_key", "")
        groups.setdefault(("question", qk), []).append(r)

    for r in records:
        qk = r.get("question_key", "")
        rid = str(r.get("run_id", ""))
        groups.setdefault(("question_run", qk, rid), []).append(r)

    rows_out = [_agg(k, v) for k, v in groups.items()]

    level_order = {"global": 0, "question": 1, "question_run": 2}
    rows_out.sort(key=lambda d: (level_order.get(d["level"], 9), d["question_key"], d["run_id"]))

    outpath = outdir / filename
    with outpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows_out[0].keys()),
        )
        w.writeheader()
        w.writerows(rows_out)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="benchmark_summary.json", help="Path to benchmark_summary.json")
    ap.add_argument("--outdir", default="benchmark_plots", help="Directory to write plots")
    ap.add_argument("--label-points", action="store_true", help="Annotate scatter points with model IDs")
    ap.add_argument("--only-question", action="append", default=None,
        help="Restrict to a question key like FOURTOPS/Q1 (repeatable). If omitted, plots all.")
    ap.add_argument("--only-stats", action="store_true",
        help="Only export overall_stats.csv (and rankings if enabled), do not generate plots.")

    args = ap.parse_args()
    summary_path = Path(args.summary).resolve()
    outdir = ensure_dir(Path(args.outdir).resolve())

    summary = load_json(summary_path)
    records = iter_question_records(summary)

    cost_records = iter_cost_spent_records(summary)

    # Only export overall stats and rankings
    if args.only_stats:
        export_overall_stats(summary, outdir, filename="overall_stats.csv")
        export_rankings(records, outdir)
        return

    # filter questions if requested
    only = set(args.only_question) if args.only_question else None
    question_keys = sorted({r["question_key"] for r in records})
    if only is not None:
        question_keys = [q for q in question_keys if q in only]

    # ---------------- Per-question plots ----------------
    for qkey in question_keys:
        qrecs = [r for r in records if r["question_key"] == qkey]
        if not qrecs:
            continue

        # title prefix
        challenge = qrecs[0].get("challenge", qkey.split("/")[0])
        qid = qrecs[0].get("qid", qkey.split("/")[1] if "/" in qkey else qkey)
        qtitle = f"{challenge} {qid}"

        q_out = ensure_dir(outdir / qkey.replace("/", "_"))

        # Scatters vs metric
        scatter_by_model(
            qrecs,
            "cost_to_success_usd",
            "metric_value",
            f"{qtitle} — Cost to success (USD) vs Metric",
            q_out / "scatter_cost_to_success_vs_metric.png"
        )

        scatter_by_model(
            qrecs,
            "training_time_s",
            "metric_value",
            f"{qtitle} — Training time (s) vs Metric",
            q_out / "scatter_training_time_vs_metric.png"
        )

        scatter_by_model(
            qrecs,
            "success_generation_s",
            "metric_value",
            f"{qtitle} — Generation time (s) vs Metric",
            q_out / "scatter_generation_time_vs_metric.png"
        )

        scatter_by_model(
            qrecs,
            "success_response_tokens",
            "metric_value",
            f"{qtitle} — Response tokens vs Metric",
            q_out / "scatter_response_tokens_vs_metric.png"
        )

        scatter_final_train_vs_val_loss(
            qrecs,
            title=f"{qtitle} — Final train loss vs final val loss",
            outpath=q_out / "scatter_final_train_vs_val_loss.png",
        )
        
        scatter_attempts_vs_metric(
            qrecs,
            title=f"{qtitle} — Attempts to success vs Metric",
            outpath=q_out / "scatter_attempts_to_success_vs_metric.png",
        )

        scatter_spend_vs_success_rate(
            summary,
            qkey,
            title=f"{qtitle} — Total spend (success+failures) vs Success rate",
            outpath=q_out / "scatter_total_spend_vs_success_rate.png",
        )

        # Bars (mean ± std over successes across runs)
        bar_mean_std_by_model(
            qrecs,
            "metric_value",
            f"{qtitle} — Metric mean ± SD (successes only)",
            q_out / "bar_metric_mean_sd.png",
        )

        bar_mean_std_by_model(
            qrecs,
            "cost_to_success_usd",
            f"{qtitle} — Cost to success mean ± SD (successes only)",
            q_out / "bar_cost_to_success_mean_sd.png",
        )

        bar_mean_std_by_model(
            qrecs,
            "attempts_to_success",
            f"{qtitle} — Attempts to success mean ± SD (successes only)",
            q_out / "bar_attempts_to_success_mean_sd.png",
        )

        bar_mean_std_by_model(
            qrecs,
            "training_time_s",
            f"{qtitle} — Training time mean ± SD (successes only)",
            q_out / "bar_training_time_mean_sd.png",
        )

        bar_mean_std_by_model(
            qrecs,
            "epochs_planned",
            f"{qtitle} — Epochs planned mean ± SD (successes only)",
            q_out / "bar_epochs_planned_mean_sd.png",
        )

        bar_mean_std_by_model(
            qrecs,
            "epochs_actual",
            f"{qtitle} — Epochs actual mean ± SD (successes only)",
            q_out / "bar_epochs_actual_mean_sd.png",
        )

        bar_epochs_planned_vs_actual(
            qrecs,
            title=f"{qtitle} — Epochs planned vs actual (mean ± SD, successes)",
            outpath=q_out / "bar_epochs_planned_vs_actual.png",
        )

        bar_success_rate(
            summary,
            qkey,
            f"{qtitle} — Success rate per model",
            q_out / "bar_success_rate.png",
        )

        bar_by_run_for_question(
            qrecs,
            "metric_value",
            f"{qtitle} — Metric by run (successes)",
            q_out / "bar_metric_by_run.png",
            ylim0=True,
        )

        bar_by_run_for_question(
            qrecs,
            "cost_to_success_usd",
            f"{qtitle} — Cost to success by run (successes)",
            q_out / "bar_cost_to_success_by_run.png",
            ylim0=True,
        )

        bar_metric_per_usd(
            qrecs,
            title=f"{qtitle} — Cost efficiency (metric per $)",
            outpath=q_out / "bar_metric_per_usd_mean_sd.png",
        )

        stacked_bar_reliability_funnel(
            summary,
            qkey,
            title=f"{qtitle} — Reliability funnel (runs)",
            outpath=q_out / "stacked_reliability_funnel_runs.png",
        )

        bar_static_breakdown_attempts(
            summary,
            qkey,
            title=f"{qtitle} — Static failure breakdown (attempts)",
            outpath=q_out / "bar_static_breakdown_attempts.png",
        )

        bar_epoch_ratio(
            qrecs,
            title=f"{qtitle} — Epoch ratio (actual/planned)",
            outpath=q_out / "bar_epoch_ratio_mean_sd.png",
        )


        # Loss curves per model (success attempts only)
        plot_loss_curves_per_model(qrecs, qtitle, q_out)

    # ---------------- Cross-question plots ----------------
    cross = ensure_dir(outdir / "_cross_question")

    # Example you explicitly mentioned: attempts to success by model across questions
    grouped_bar_cross_question(
        records,
        value_key="attempts_to_success",
        title="Attempts to success (mean ± SD) by model across questions (successes only)",
        outpath=cross / "grouped_attempts_to_success_across_questions.png",
    )

    # Also useful: cost-to-success across questions
    grouped_bar_cross_question(
        records,
        value_key="cost_to_success_usd",
        title="Cost to success (USD, mean ± SD) by model across questions (successes only)",
        outpath=cross / "grouped_cost_to_success_across_questions.png",
    )

    # And performance across questions
    grouped_bar_cross_question(
        records,
        value_key="metric_value",
        title="Metric (mean ± SD) by model across questions (successes only)",
        outpath=cross / "grouped_metric_across_questions.png",
    )

    grouped_bar_cross_question(
        records,
        value_key="attempts_to_success",
        title="Attempts to success by model (mean ± SD) — FOURTOPS only",
        outpath=cross / "grouped_attempts_to_success__FOURTOPS.png",
        question_filter_prefix="FOURTOPS/",
    )

    grouped_bar_cross_question(
        cost_records,
        value_key="cost_spent_usd",
        title="Total spend by model (mean ± SD) — across questions (success + failures)",
        outpath=cross / "grouped_total_spend_across_questions.png",
    )

    grouped_bar_cross_question(
        cost_records,
        value_key="cost_spent_usd",
        title="Total spend by model (mean ± SD) — FOURTOPS only (success + failures)",
        outpath=cross / "grouped_total_spend__FOURTOPS.png",
        question_filter_prefix="FOURTOPS/",
    )

    grouped_bar_cross_question(
        records,
        value_key="metric_value",
        title="Metric by model (mean ± SD) — FOURTOPS only",
        outpath=cross / "grouped_metric__FOURTOPS.png",
        question_filter_prefix="FOURTOPS/",
        ylim0=True,
    )

    bar_fourtops_q1_vs_q2_by_run(
        records,
        value_key="metric_value",
        title="FOURTOPS — Metric Q1 vs Q2 by run (successes)",
        outpath=cross / "FOURTOPS_metric_Q1_vs_Q2_by_run.png",
        ylim0=True,
    )

    bar_fourtops_q1_vs_q2_by_run(
        records,
        value_key="attempts_to_success",
        title="FOURTOPS — Attempts Q1 vs Q2 by run (successes)",
        outpath=cross / "FOURTOPS_attempts_Q1_vs_Q2_by_run.png",
        ylim0=True,
    )

    scatter_fourtops_q1_vs_q2_metric(
        records,
        title="FOURTOPS — Consistency: Q1 mean metric vs Q2 mean metric",
        outpath=cross / "scatter_FOURTOPS_Q1_vs_Q2_metric_mean.png",
    )

    # CSV
    export_rankings(records, outdir)
    export_overall_stats(summary, outdir, filename="benchmark_overall_stats.csv")

    print(f"Wrote plots to: {outdir}")
    print("Per-question folders:", ", ".join([q.replace('/', '_') for q in question_keys]) if question_keys else "(none)")


if __name__ == "__main__":
    main()
