# utils/plot_roc.py

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import itertools

import matplotlib.pyplot as plt


"""
plot_roc.py  —  Plot ROC curves for multiple models from a single JSON file.

Basic usage:
  python utils/plot_roc.py results\\23-06\\FOURTOPS\\Q2\\Q2_summary.json                        # show
  python utils/plot_roc.py outputs\23-06\\FOURTOPS\\Q2\\Q2_summary.json --outfile roc.png      # save PNG
"""


# ──────────────────────────── CLI ────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot ROC curves from JSON.")
    p.add_argument("input_file",   type=Path, 
                   help="Path to summary JSON file.")
    p.add_argument("--outfile",
                   help="Filename to save figure (same folder as input_file).")
    return p


# ────────────────────────── helpers ──────────────────────────
def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"❌ Failed to read {path}: {e}")


# ─────────────────────────── main ────────────────────────────
def main() -> None:
    args = build_parser().parse_args()
    data = load_json(args.input_file)

    # ── build a context-aware title from outputs/<DATE>/<CHALLENGE>/<QUESTION> ──
    input_path = args.input_file.resolve()
    try:
        out_idx = input_path.parts.index("outputs")
        date_part      = input_path.parts[out_idx + 1]
        challenge_part = input_path.parts[out_idx + 2]
        question_part  = (
            input_path.parts[out_idx + 3] if len(input_path.parts) > out_idx + 3 else None
        )
    except ValueError:
        challenge_part = question_part = None

    if question_part and date_part:
        plot_title = f"{challenge_part} - {question_part}: ROC Curves ({date_part})"
    elif challenge_part and date_part:
        plot_title = f"{challenge_part} ROC Curves ({date_part})"
    elif date_part:
        plot_title = f"ROC Curves ({date_part})"
    else:
        plot_title = "ROC Curves"

    if not data:
        sys.exit("‼️  JSON is empty or malformed.")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for model_name, metrics in data.items():
        try:
            fpr = metrics["fpr"]
            tpr = metrics["tpr"]
            auc = metrics["auc"]
        except KeyError as e:
            print(f"⚠️  {model_name}: missing key {e}; skipping.")
            continue

        ax.plot(fpr, tpr, color=next(colors),
                linewidth=.8, label=f"{model_name} (AUC = {auc:.3f})")

    # diagonal chance line
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate",  fontsize=14)
    ax.set_title(plot_title, fontsize=16)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if args.outfile:
        out_path = args.input_file.parent / args.outfile
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
