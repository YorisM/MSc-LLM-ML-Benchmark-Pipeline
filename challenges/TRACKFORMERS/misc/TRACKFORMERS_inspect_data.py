import argparse, os, textwrap, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110   # crisper in VS Code

# ──────────────────────────────────────────────────────────────────── #
# 1. CLI
# ──────────────────────────────────────────────────────────────────── #
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Lightweight inspection of a TrackFormers hits+tracks CSV.

        Plots:
          • hits_per_event.png      (histogram)
          • tracks_per_event.png    (histogram)
          • hits_per_track.png      (histogram)
          • sample_event_rthetaz.png (3-D scatter for --sample-event)
        """))
    p.add_argument("--csv-path", required=True,
                   help="Path to hits_and_tracks_3d_events_all.csv")
    p.add_argument("--sample-event", type=int, default=0,
                   help="Event index (0-based) to visualise in 3-D")
    p.add_argument("--n-head", type=int, default=8,
                   help="Print first n rows of the chosen event")
    p.add_argument("--sep", default=None,
               help="Field delimiter (comma, ';', '\\t'). Leave blank for auto-detect.")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────── #
# 2. LOADING
# ──────────────────────────────────────────────────────────────────── #

def _detect_sep(sample_bytes: bytes) -> str:
    """
    Return ',', ';' or '\\t' based on csv.Sniffer.
    Falls back to ','.
    """
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_bytes.decode("utf-8", errors="ignore"),
                                delimiters=[",", ";", "\t"])
        return dialect.delimiter
    except csv.Error:
        return ','

def load_csv(path: str, user_sep: str | None):
    # auto-detect if user didn't override
    if user_sep is None:
        with open(path, "rb") as fh:
            sample = fh.read(4096)          # first ~4 KB
        sep = _detect_sep(sample)
        print(f"[auto-detected separator: '{sep}']")
    else:
        sep = user_sep
        print(f"[using user-specified separator: '{sep}']")

    dtypes = {
        "event_id":        "int32",
        "sub_detector_id": "int16",
        "track_id":        "int32",
        "hit_r":           "float32",
        "hit_theta":       "float32",
        "hit_z":           "float32",
    }

    df = pd.read_csv(path, sep=sep, dtype=dtypes, low_memory=False)

    # normalise column names → lower case, strip spaces
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"event_id", "track_id", "sub_detector_id", "hit_r", "hit_theta", "hit_z"}
    missing  = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    return df

# ──────────────────────────────────────────────────────────────────── #
# 3. Metrics & plots
# ──────────────────────────────────────────────────────────────────── #
def compute_metrics(df):
    print("\n=== BASIC SHAPE ===")
    print(f"total rows (hits) : {len(df):,}")
    print(f"unique events     : {df.event_id.nunique():,}")

    # hits / event
    hits_event = df.groupby("event_id").size()
    tracks_event = df.groupby("event_id")["track_id"].nunique()

    # hits / track
    hits_track = df.groupby(["event_id", "track_id"]).size()

    print("\n=== HITS PER EVENT ===")
    print(hits_event.describe().round(2))

    print("\n=== TRACKS PER EVENT ===")
    print(tracks_event.describe().round(2))

    print("\n=== HITS PER TRACK ===")
    print(hits_track.describe().round(2))

    print("\n=== LAYER COVERAGE (hits per layer) ===")
    layer_counts = df["sub_detector_id"].value_counts().sort_index()
    print(layer_counts.to_string())

    return hits_event, tracks_event, hits_track

def plot_hist(data, title, fname, xlabel, bins=50, log=False):
    plt.figure()
    plt.hist(data, bins=bins, log=log)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"[saved] {fname}")
    plt.close()

def scatter_event(df_evt, fname):
    # Convert cylindrical to Cartesian
    r  = df_evt.hit_r.to_numpy()
    th = df_evt.hit_theta.to_numpy()
    zc = df_evt.hit_z.to_numpy()
    x  = r * np.cos(th)
    y  = r * np.sin(th)

    # 3-D scatter of hits coloured by track_id (first 20 tracks → readable plot)
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    palette = list(mcolors.TABLEAU_COLORS.values())
    tracks = df_evt.track_id.unique()[:20]
    for i, tid in enumerate(tracks):
        sub = df_evt[df_evt.track_id == tid]
        ri  = sub.hit_r.to_numpy()
        thi = sub.hit_theta.to_numpy()
        zi  = sub.hit_z.to_numpy()
        xi  = ri * np.cos(thi)
        yi  = ri * np.sin(thi)
        ax.scatter(xi, yi, zi,
                   s=8, alpha=0.7, color=palette[i % len(palette)],
                   label=f"t{tid}" if i < 10 else None)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Event {df_evt.event_id.iloc[0]}  (≤20 tracks)")
    if len(tracks) <= 10:
        ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(fname)
    print(f"[saved] {fname}")
    plt.close()

# ──────────────────────────────────────────────────────────────────── #
# 4. Main
# ──────────────────────────────────────────────────────────────────── #
def main():
    args = parse_args()
    csv_path = args.csv_path

    print(f"Loading {csv_path} …")
    df = load_csv(csv_path, user_sep=args.sep)
    print("Loaded dataframe shape:", df.shape)

    # Compute & print stats
    hits_evt, tracks_evt, hits_track = compute_metrics(df)

    # Histograms
    plot_hist(hits_evt,
              "Hits per event", "hits_per_event.png",
              "hits/event", bins=60, log=True)
    plot_hist(tracks_evt,
              "Tracks per event", "tracks_per_event.png",
              "tracks/event", bins=50, log=True)
    plot_hist(hits_track,
              "Hits per track", "hits_per_track.png",
              "hits/track", bins=50, log=True)

    # 3-D sample event
    evt_ids = df.event_id.sort_values().unique()
    evt_id  = evt_ids[args.sample_event]
    df_evt  = df[df.event_id == evt_id]
    scatter_event(df_evt, "sample_event_rthetaz.png")

    # Head of chosen event
    print(f"\n=== First {args.n_head} rows of event {evt_id} ===")
    print(df_evt.head(args.n_head).to_string(index=False))

if __name__ == "__main__":
    main()
