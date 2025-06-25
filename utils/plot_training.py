# utils/plot_training.py

"""
Plot training/validation accuracy vs epochs for *all* models found under
an outputs/… subtree.

Basic usage:
  python utils\\plot_training.py outputs\\23-06\\FOURTOPS\\Q3

Options:
  --dots / --lines            choose marker style           (mutually exclusive)
  --out FIG.png               save instead of showing
  --train-key train_acc       JSON key for training acc     (default: train_acc)
  --val-key   val_acc         JSON key for val acc          (default: val_acc)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, itertools, sys, logging

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from utils import iter_input_dir, SKIP_DIRS


# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training/val accuracy curves")
    p.add_argument("input_dir", type=Path,
                   help="Path anywhere inside outputs/<DATE>/…")
    style = p.add_mutually_exclusive_group()
    style.add_argument("--dots", action="store_true",
                       help="Scatter markers instead of lines.")
    style.add_argument("--lines", action="store_true",
                       help="Lines instead of scatter (default).")
    p.add_argument("--train-key", default="train_acc",
                   help="JSON key holding training accuracy list.")
    p.add_argument("--val-key",   default="val_acc",
                   help="JSON key holding validation accuracy list.")
    p.add_argument("--outfile",
                   help="Filename to save under input_dir. "
                        "If omitted, figure is shown on screen.")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _lighten(color: str | tuple, factor: float = 0.5) -> tuple:
    """Return a lighter shade of *color*. factor ∈ (0,1): bigger = lighter."""
    r, g, b = to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

def _load_metrics(json_path: Path, train_key: str,
                  val_key: str) -> tuple[list[float], list[float]]:
    """Return (train_acc_list, val_acc_list) or raises KeyError."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["Training"]["metrics"]
    train = metrics[train_key]
    val   = metrics[val_key]

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
        sys.exit("‼️ No model folders found under given input dir.")

    logging.debug("Found %d model folders.", len(model_dirs))

    fig, ax = plt.subplots(figsize=(12, 8))
    base_colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    as_lines = args.lines or not args.dots

    for m_dir, base_col in zip(model_dirs, base_colors):
        j_path = _find_training_json(m_dir)
        if not j_path:
            print(f"⚠️  No training JSON in {m_dir}; skipping")
            continue

        try:
            train, val = _load_metrics(j_path, args.train_key, args.val_key)
        except (KeyError, TypeError) as e:
            print(f"⚠️  Bad JSON format in {j_path}: {e}")
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
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title(f"{challenge} - {question}: Training vs Validation Accuracy ({date_part})", fontsize=16)
    ax.legend()
    fig.tight_layout()

    if args.outfile:
        out_path = input_dir / args.outfile
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
