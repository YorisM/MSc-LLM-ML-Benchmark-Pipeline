# utils.plot_benchmark_summary.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


"""
Generate plots from the tidy benchmark summary CSVs.
"""

# Example usage:
# python utils/plot_benchmark_summary.py --summary-dir benchmark_summary --label-points


def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def scatter(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path, label_col: str | None = None):
    dcols = [x, y] + ([label_col] if label_col else [])
    d = df[dcols].dropna()

    plt.figure()
    ax = plt.gca()
    ax.scatter(d[x], d[y])

    if label_col:
        for _, r in d.iterrows():
            ax.annotate(str(r[label_col]), (r[x], r[y]), fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def grouped_bar(agg_df: pd.DataFrame, value_col: str, title: str, outpath: Path):
    """
    agg_df should contain: model_key, challenge_question, value_col
    Produces grouped bars by question for each model.
    """
    d = agg_df[["model_key", "challenge_question", value_col]].copy()
    d = d.dropna(subset=[value_col])

    models = sorted(d["model_key"].unique())
    cq = sorted(d["challenge_question"].unique())

    piv = d.pivot_table(index="model_key", columns="challenge_question", values=value_col, aggfunc="first").reindex(models)

    plt.figure(figsize=(max(7, 0.85 * len(models)), 4))
    ax = plt.gca()
    piv.plot(kind="bar", ax=ax)

    ax.set_xlabel("Model")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", default="benchmark_summary", help="Folder containing extracted CSVs.")
    p.add_argument("--outdir", default="benchmark_summary/plots", help="Where to write plots.")
    p.add_argument("--label-points", action="store_true", help="Annotate scatter points with model keys.")
    args = p.parse_args()

    summary_dir = Path(args.summary_dir)
    outdir = _ensure_outdir(Path(args.outdir))

    success_csv = summary_dir / "complete_benchmark_summary.csv"
    agg_csv = summary_dir / "complete_benchmark_summary_aggregated.csv"

    success_df = pd.read_csv(success_csv)
    agg_df = pd.read_csv(agg_csv)

    # Helpers
    success_df["challenge_question"] = success_df["challenge"].astype(str) + " " + success_df["question"].astype(str)
    agg_df["challenge_question"] = agg_df["challenge"].astype(str) + " " + agg_df["question"].astype(str)

    # Only successes for scatter plots
    succ = success_df[success_df["success"] == True].copy()

    # ---- Scatter plots vs metric (success-level) ----
    scatter(
        succ,
        x="cost_to_success_usd",
        y="metric_value",
        title="Cost to success (USD) vs Metric",
        outpath=outdir / "scatter_cost_to_success_vs_metric.png",
        label_col="model_key" if args.label_points else None,
    )

    scatter(
        succ,
        x="training_time_s",
        y="metric_value",
        title="Training time (s) vs Metric",
        outpath=outdir / "scatter_training_time_vs_metric.png",
        label_col="model_key" if args.label_points else None,
    )

    scatter(
        succ,
        x="success_generation_s",
        y="metric_value",
        title="Generation time (s) vs Metric",
        outpath=outdir / "scatter_generation_time_vs_metric.png",
        label_col="model_key" if args.label_points else None,
    )

    scatter(
        succ,
        x="success_response_tokens",
        y="metric_value",
        title="Response tokens vs Metric",
        outpath=outdir / "scatter_response_tokens_vs_metric.png",
        label_col="model_key" if args.label_points else None,
    )

    # ---- Bar charts (aggregated means across runs, success-only) ----
    grouped_bar(
        agg_df,
        value_col="cost_to_success_mean",
        title="Mean cost to success (USD) by model (success-only)",
        outpath=outdir / "bar_cost_to_success_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="success_generation_s_mean",
        title="Mean generation time (s) by model (success-only)",
        outpath=outdir / "bar_generation_time_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="success_response_tokens_mean",
        title="Mean response tokens by model (success-only)",
        outpath=outdir / "bar_response_tokens_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="training_time_s_mean",
        title="Mean training time (s) by model (success-only)",
        outpath=outdir / "bar_training_time_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="attempts_mean",
        title="Mean attempts to success by model (success-only)",
        outpath=outdir / "bar_attempts_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="epochs_planned_mean",
        title="Mean epochs planned by model (success-only)",
        outpath=outdir / "bar_epochs_planned_mean.png",
    )

    grouped_bar(
        agg_df,
        value_col="epochs_actual_mean",
        title="Mean epochs actual by model (success-only)",
        outpath=outdir / "bar_epochs_actual_mean.png",
    )

    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
