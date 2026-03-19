"""
Plot REDVID events (2D projections + 3D) from a single CSV.

Assumes REDVID hit coordinates are provided in cylindrical coordinates:
  hit_r, hit_theta, hit_z
and uses:
  event_id, track_id
as primary grouping keys.

Usage examples
--------------
# Plot the first 2 events encountered (streamed; does NOT load full CSV at once)
python plot_redvid_events.py --csv /path/to/redvid_linear.csv --out-dir ./plots

# Plot specific events
python plot_redvid_events.py --csv /path/to/redvid_helical.csv --event-ids 0 42 --out-dir ./plots

# Increase/decrease streaming chunk size
python plot_redvid_events.py --csv /path/to/redvid.csv --chunksize 500000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = [
    "event_id",
    "track_id",
    "hit_r",
    "hit_theta",
    "hit_z",
]

# Optional columns (nice to have in titles/labels, but not required)
OPTIONAL_COLS = [
    "sub_detector_id",
    "sub_detector_type",
    "track_type",
    "hit_id",
]


def _infer_label_from_path(csv_path: Path) -> str:
    name = csv_path.stem
    # Handle .csv.gz -> stem becomes ".csv"
    if name.endswith(".csv"):
        name = Path(name).stem
    return name


def pick_first_n_event_ids(
    csv_path: Path,
    n_events: int,
    chunksize: int,
) -> List[int]:
    """Stream the CSV until we collected n unique event_ids."""
    found: List[int] = []
    seen = set()

    usecols = ["event_id"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        # Ensure integer-ish
        ids = chunk["event_id"].dropna().astype(int).unique()
        for eid in ids:
            if eid not in seen:
                seen.add(int(eid))
                found.append(int(eid))
                if len(found) >= n_events:
                    return found

    return found


def load_events_rows(
    csv_path: Path,
    event_ids: Sequence[int],
    chunksize: int,
) -> pd.DataFrame:
    """Stream the CSV and keep only rows whose event_id is in event_ids."""
    event_ids_set = set(int(e) for e in event_ids)

    # Load required + any optional columns that exist
    # We first read header only to see which optional columns are present
    header = pd.read_csv(csv_path, nrows=0)
    present_optional = [c for c in OPTIONAL_COLS if c in header.columns]

    usecols = REQUIRED_COLS + present_optional

    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        # event_id might come as int/float/string depending on CSV; normalize
        chunk = chunk.dropna(subset=["event_id"])
        chunk["event_id"] = chunk["event_id"].astype(int)

        sub = chunk[chunk["event_id"].isin(event_ids_set)]
        if not sub.empty:
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(frames, ignore_index=True)

    # Basic sanity check for required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return df


def add_cartesian_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with x,y,z computed from (hit_r, hit_theta, hit_z)."""
    out = df.copy()

    r = out["hit_r"].astype(float).to_numpy()
    theta = out["hit_theta"].astype(float).to_numpy()
    z = out["hit_z"].astype(float).to_numpy()

    out["x"] = r * np.cos(theta)
    out["y"] = r * np.sin(theta)
    out["z"] = z
    return out


def _sort_hits_for_polyline(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort ordering of hits along a track for line plotting.
    Prefer hit_id, else sub_detector_id, else keep input order.
    """
    if "hit_id" in track_df.columns:
        return track_df.sort_values(["hit_id"], kind="mergesort")
    if "sub_detector_id" in track_df.columns:
        return track_df.sort_values(["sub_detector_id"], kind="mergesort")
    return track_df


def plot_event_2d(
    ev: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    """
    2D projections: XY, XZ, YZ.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
    ax_xy, ax_xz, ax_yz = axes

    # Group by track_id; draw each as a polyline + scatter.
    for track_id, g in ev.groupby("track_id"):
        g = _sort_hits_for_polyline(g)
        ax_xy.plot(g["x"], g["y"], linewidth=1)
        ax_xy.scatter(g["x"], g["y"], s=6)

        ax_xz.plot(g["x"], g["z"], linewidth=1)
        ax_xz.scatter(g["x"], g["z"], s=6)

        ax_yz.plot(g["y"], g["z"], linewidth=1)
        ax_yz.scatter(g["y"], g["z"], s=6)

    ax_xy.set_title("XY")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", adjustable="box")

    ax_xz.set_title("XZ")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_yz.set_title("YZ")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_event_3d(
    ev: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for track_id, g in ev.groupby("track_id"):
        g = _sort_hits_for_polyline(g)
        ax.plot(g["x"], g["y"], g["z"], linewidth=1)
        ax.scatter(g["x"], g["y"], g["z"], s=6)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot REDVID events (2D+3D) from a CSV.")
    p.add_argument("--csv", type=str, required=True, help="Path to REDVID CSV file.")
    p.add_argument("--out-dir", type=str, default="./plots", help="Output directory.")
    p.add_argument("--n-events", type=int, default=2, help="Number of events to plot (if --event-ids not provided).")
    p.add_argument("--event-ids", type=int, nargs="*", default=None, help="Explicit event IDs to plot.")
    p.add_argument("--chunksize", type=int, default=1_000_000, help="CSV chunk size for streaming.")
    p.add_argument("--show", action="store_true", help="Also show plots interactively (in addition to saving).")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which event IDs to plot
    if args.event_ids and len(args.event_ids) > 0:
        event_ids = [int(x) for x in args.event_ids]
    else:
        event_ids = pick_first_n_event_ids(csv_path, n_events=args.n_events, chunksize=args.chunksize)
        if len(event_ids) < args.n_events:
            print(f"[warn] Only found {len(event_ids)} unique events in file (requested {args.n_events}).")

    if not event_ids:
        raise SystemExit("No events found to plot. Check that the CSV has an 'event_id' column and is not empty.")

    # Load rows for those events (streaming)
    df = load_events_rows(csv_path, event_ids=event_ids, chunksize=args.chunksize)
    if df.empty:
        raise SystemExit(f"No rows found for event_ids={event_ids}. Are these IDs present in the CSV?")

    # Compute x,y,z
    df = add_cartesian_columns(df)

    label = _infer_label_from_path(csv_path)

    # Plot each event
    for eid in event_ids:
        ev = df[df["event_id"] == eid]
        if ev.empty:
            print(f"[warn] Event {eid}: no rows; skipping.")
            continue

        n_tracks = ev["track_id"].nunique()
        n_hits = len(ev)

        # Include track_type info if present
        track_types = None
        if "track_type" in ev.columns:
            tt = sorted(set(str(x) for x in ev["track_type"].dropna().unique()))
            track_types = ", ".join(tt) if tt else None

        title = f"{label} | event_id={eid} | tracks={n_tracks} | hits={n_hits}"
        if track_types:
            title += f" | track_type(s)={track_types}"

        out_2d = out_dir / f"{label}_event{eid}_2D.png"
        out_3d = out_dir / f"{label}_event{eid}_3D.png"

        plot_event_2d(ev, out_2d, title=title)
        plot_event_3d(ev, out_3d, title=title)

        print(f"[ok] Saved: {out_2d}")
        print(f"[ok] Saved: {out_3d}")

        if args.show:
            # Re-open for interactive viewing
            img2d = plt.imread(out_2d)
            plt.figure(figsize=(12, 4))
            plt.imshow(img2d)
            plt.axis("off")
            plt.title(out_2d.name)
            plt.show()

            img3d = plt.imread(out_3d)
            plt.figure(figsize=(6, 6))
            plt.imshow(img3d)
            plt.axis("off")
            plt.title(out_3d.name)
            plt.show()


if __name__ == "__main__":
    main()
