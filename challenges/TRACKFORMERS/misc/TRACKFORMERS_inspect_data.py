# challenges/TRACKFORMERS/misc/TRACKFORMERS_inspect_data.py


import argparse, textwrap, csv, gzip, pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = os.path.dirname(__file__)          # .../challenges/TRACKFORMERS/misc
plt.rcParams["figure.dpi"] = 110

# CLI ------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Inspect a TrackFormers hit-level dataset.

        - CSV with columns event_id, track_id, sub_detector_id, hit_r, hit_theta, hit_z
        - REDVID pickle (plain or .gz) containing {"events":[{hit arrays...}]}

        Plots written:
          - *_hits_per_event.png
          - *_tracks_per_event.png
          - *_hits_per_track.png
          - sample_event_rthetaz.png   (3-D scatter for --sample-event)
        """))
    p.add_argument("paths", nargs="+",
                   help="One or more CSV / pkl / pkl.gz files to inspect")
    p.add_argument("--sample-event", type=int, default=0,
                   help="Event index (0-based) to visualise in 3-D")
    p.add_argument("--n-head", type=int, default=8,
                   help="Print first n rows of the chosen event")
    p.add_argument("--sep", default=None,
                   help="Field delimiter for CSV. Leave blank for auto-detect.")
    return p.parse_args()

# helpers -------------------------------------------------
def _detect_sep(sample_bytes: bytes) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_bytes.decode("utf-8", "ignore"),
                                      delimiters=[",",";","\t"])
        return dialect.delimiter
    except csv.Error:
        return ','

def _load_csv(path: str, user_sep: str | None) -> pd.DataFrame:
    if user_sep is None:
        with open(path, "rb") as fh:
            sep = _detect_sep(fh.read(4096))
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
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _load_pkl(path: str) -> pd.DataFrame:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        data = pickle.load(fh)

    events = data.get("events")
    if events is None:
        raise RuntimeError(f"{path}: key 'events' not found")

    rows = []
    for ev_id, evt in enumerate(events):
        n = len(evt["hit_r"])
        rows.append(pd.DataFrame({
            "event_id":        np.full(n, ev_id, dtype=np.int32),
            "track_id":        evt["track_id"].astype(np.int32),
            "sub_detector_id": evt["layer_id"].astype(np.int16),   # alias
            "hit_r":           evt["hit_r"].astype(np.float32),
            "hit_theta":       evt["hit_theta"].astype(np.float32),
            "hit_z":           evt["hit_z"].astype(np.float32),
        }))
    df = pd.concat(rows, ignore_index=True)
    return df

# metrics & plots ------------------------------------------
def compute_metrics(df):
    hits_event   = df.groupby("event_id").size()
    tracks_event = df.groupby("event_id")["track_id"].nunique()
    hits_track   = df.groupby(["event_id","track_id"]).size()

    print("\n=== BASIC SHAPE ===")
    print(f"total rows (hits) : {len(df):,}")
    print(f"unique events     : {df.event_id.nunique():,}")

    for name, series in [("HITS PER EVENT", hits_event),
                         ("TRACKS PER EVENT", tracks_event),
                         ("HITS PER TRACK", hits_track)]:
        print(f"\n=== {name} ===")
        print(series.describe().round(2))

    print("\n=== LAYER COVERAGE (hits per layer) ===")
    print(df["sub_detector_id"].value_counts().sort_index().to_string())
    return hits_event, tracks_event, hits_track

def plot_hist(data, title, fname, xlabel, bins=50, log=False):
    plt.figure()
    plt.hist(data, bins=bins, log=log)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"[saved] {fname}")

def scatter_event(df_evt, fname):
    import matplotlib.colors as mcolors; from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    palette = list(mcolors.TABLEAU_COLORS.values())

    for i, tid in enumerate(df_evt.track_id.unique()[:20]):
        sub = df_evt[df_evt.track_id == tid]
        xi = sub.hit_r.to_numpy() * np.cos(sub.hit_theta.to_numpy())
        yi = sub.hit_r.to_numpy() * np.sin(sub.hit_theta.to_numpy())
        zi = sub.hit_z.to_numpy()
        ax.scatter(xi, yi, zi, s=8, alpha=0.7,
                   color=palette[i % len(palette)],
                   label=f"t{tid}" if i < 10 else None)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Event {df_evt.event_id.iloc[0]} (≤20 tracks)")
    if df_evt.track_id.nunique() <= 10:
        ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close(); print(f"[saved] {fname}")

def collect_metric(store: dict, label: str,
                   hits_evt, tracks_evt, hits_track):
    store["hits_evt"][label]   = hits_evt
    store["tracks_evt"][label] = tracks_evt
    store["hits_track"][label] = hits_track

def overlay_hist(metric_dict: dict, title, fname, xlabel,
                 bins=60, log=False):
    plt.figure()
    # union of data ranges for a common binning
    all_vals = np.concatenate(list(metric_dict.values()))
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins+1)
    for label, series in metric_dict.items():
        plt.hist(series, bins=bin_edges, histtype="step",
                 linewidth=1.4, label=label, log=log)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.legend() 
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"[saved] {fname}")


# main ------------------------------------------------------
def load_any(path: str, user_sep: str | None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".txt"}:
        return _load_csv(path, user_sep)
    elif ext in {".pkl", ".gz"}:
        return _load_pkl(path)
    else:
        raise RuntimeError(f"Unsupported file extension for {path}")

def main():
    args = parse_args()
    overlay = {"hits_evt": {}, "tracks_evt": {}, "hits_track": {}}

    for in_path in args.paths:
        print(f"\n================  {in_path}  ================")
        df = load_any(in_path, args.sep)
        print("Loaded dataframe shape:", df.shape)

        hits_evt, tracks_evt, hits_track = compute_metrics(df)

        label = os.path.basename(in_path).replace(".pkl.gz","")
        collect_metric(overlay, label, hits_evt, tracks_evt, hits_track)

        stem = in_path.replace("\\", "_").replace("/", "_")

        evt_ids = df.event_id.sort_values().unique()
        evt_id  = evt_ids[min(args.sample_event, len(evt_ids)-1)]
        scatter_event(df[df.event_id == evt_id], f"{stem}_sample_event.png")

        print(f"\n=== First {args.n_head} rows of event {evt_id} ===")
        print(df[df.event_id == evt_id].head(args.n_head).to_string(index=False))

    overlay_hist(overlay["hits_evt"],   "Hits per event (all files)",
                 "cmp_hits_per_event.png",   "hits/event", 60, log=True)

    overlay_hist(overlay["tracks_evt"], "Tracks per event (all files)",
                 "cmp_tracks_per_event.png", "tracks/event", 50, log=True)

    overlay_hist(overlay["hits_track"], "Hits per track (all files)",
                 "cmp_hits_per_track.png",   "hits/track", 50, log=True)

if __name__ == "__main__":
    main()