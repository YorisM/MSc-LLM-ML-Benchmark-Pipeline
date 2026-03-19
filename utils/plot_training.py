# utils/plot_training.py

"""
Plot training/validation accuracy or loss vs epochs for *all* models found under
an outputs/… subtree.

Basic usage:
    python -m utils.plot_training ".\outputs\01-05(2)\FOURTOPS\\Q1" --out FOURTOPS_Q1-1_TRAIN_LOSS.PNG --ylow 0.4 --yhigh 0.8

Options:
  --metric loss|acc           choose metric                 (default: loss)
  --dots / --lines            choose marker style           (mutually exclusive)
  --out FIG.png               save instead of showing
  --train-key train_acc       JSON key for training acc     (default: train_acc)
  --val-key   val_acc         JSON key for val acc          (default: val_acc)
  --ylow YLOW 
  --yhigh YHIGH  y-limits for the plot
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, itertools, sys, logging, numbers

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from utils.utils import iter_input_dir, SKIP_DIRS


# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training/val accuracy curves")
    p.add_argument("input_dir", type=Path,
                   help="Path anywhere inside outputs/<DATE>/…")
    p.add_argument("--metric", default="loss", choices=["loss", "acc"])

    style = p.add_mutually_exclusive_group()
    style.add_argument("--dots", action="store_true",
                       help="Scatter markers instead of lines.")
    style.add_argument("--lines", action="store_true",
                       help="Lines instead of scatter (default).")
    
    p.add_argument("--outfile",
                   help="Filename to save under input_dir. If omitted, figure is shown on screen.")
    
    p.add_argument("--ylow", type=float, default=None,
                    help="Lower y-limit for the plot (float). Omit for auto.")
    p.add_argument("--yhigh", type=float, default=None,
                    help="Upper y-limit for the plot (float). Omit for auto.")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _lighten(color: str | tuple, factor: float = 0.5) -> tuple:
    """Return a lighter shade of *color*. factor ∈ (0,1): bigger = lighter."""
    r, g, b = to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

def _as_list(v) -> list[float]:
    """Accept either list[float] or scalar float/int; return list[float]."""
    if v is None:
        return []
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, numbers.Number):
        return [float(v)]
    raise TypeError(f"Expected list or number, got {type(v).__name__}: {v!r}")

def _load_metrics(json_path: Path, train_key: str,
                  val_key: str) -> tuple[list[float], list[float]]:
    """Return (train_acc_list, val_acc_list) or raises KeyError."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["Training"]["metrics"]
    train = _as_list(metrics[train_key])
    val   = _as_list(metrics[val_key])

    return train, val

def _find_training_json(model_dir: Path) -> Path | None:
    """Return first JSON in model_dir that contains Training→metrics."""
    for p in model_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                if "Training" in json.load(f):
                    return p
        except json.JSONDecodeError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = build_parser().parse_args()
    as_lines   = bool(args.lines)
    input_dir  = args.input_dir.resolve()

    # choose default keys based on metric
    if args.metric == "loss":
        train_key = "train_loss"
        val_key   = "val_loss"
        ylabel    = "Loss"
    else:
        train_key = "train_acc"
        val_key   = "val_acc"
        ylabel    = "Accuracy"

    # ── extract path components for the plot title ────────────────────────
    try:
        out_idx = input_dir.parts.index("outputs")
        date_part      = input_dir.parts[out_idx + 1]
        challenge = input_dir.parts[out_idx + 2]
        question  = (
            input_dir.parts[out_idx + 3] if len(input_dir.parts) > out_idx + 3 else None
        )
    except ValueError:
        date_part = challenge_part = question_part = None

    logging.debug("Start plotting training metrics..")

    # gather all model folders under requested subtree
    model_dirs: list[Path] = []
    for q_dir in iter_input_dir(args.input_dir):
        for m_dir in q_dir.iterdir():
            if m_dir.is_dir() and m_dir.name not in SKIP_DIRS:
                model_dirs.append(m_dir)

    if not model_dirs:
        sys.exit("‼️No model folders found under given input dir.")

    logging.debug("Found %d model folders.", len(model_dirs))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(
        left=0.1,
        right=0.95,
        bottom=0.1,
        top=0.9
    )
    base_colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    as_lines = args.lines or not args.dots

    for m_dir, base_col in zip(model_dirs, base_colors):
        j_path = _find_training_json(m_dir)
        if not j_path:
            print(f"⚠️ No training JSON in {m_dir}; skipping")
            continue

        try:
            train, val = _load_metrics(j_path, train_key, val_key)
        except (KeyError, TypeError) as e:
            print(f"⚠️  Bad JSON format in {j_path}: {e}")
            continue

        if len(train) == 0 and len(val) == 0:
            print(f"⚠️  Empty metrics in {j_path}; skipping")
            continue
        
        epochs_t = list(range(1, len(train) + 1))
        epochs_v = list(range(1, len(val)   + 1))

        light_col  = _lighten(base_col, .4)
        label_base = m_dir.name

        if as_lines:
            ax.plot(epochs_t, train, color=base_col, marker="o",
                    label=f"{label_base} train")
            ax.plot(epochs_v, val,   color=light_col,
                    marker="o", markerfacecolor='none', linestyle="--",
                    label=f"{label_base} val")
        else:  # dots
            ax.scatter(epochs_t, train, color=base_col,
                       label=f"{label_base} train", s=25)
            ax.scatter(epochs_v, val,  edgecolor=light_col, facecolor='none',
                       label=f"{label_base} val", s=40)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # y-lims
    if args.ylow is not None and args.yhigh is not None and args.ylow >= args.yhigh:
        raise ValueError(f"--ylow must be < --yhigh (got {args.ylow} >= {args.yhigh})")
    if args.ylow is not None or args.yhigh is not None:
        ax.set_ylim(bottom=args.ylow, top=args.yhigh)

    ax.set_title(f"{challenge} - {question}: Training vs Validation {ylabel} ({date_part})", fontsize=16)
    ax.legend()

    if args.outfile:
        out_path = input_dir / args.outfile
        fig.savefig(out_path, dpi=600)
        print(f"✅ Figure saved to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
