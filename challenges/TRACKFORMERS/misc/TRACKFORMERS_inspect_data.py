# challenges/TRACKFORMERS/misc/TRACKFORMERS_inspect_data.py


import argparse, textwrap, csv, gzip, pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = os.path.dirname(__file__)          # .../challenges/TRACKFORMERS/misc
plt.rcParams["figure.dpi"] = 110

# Example usage:
#
# python challenges/TRACKFORMERS/misc/TRACKFORMERS_inspect_data.py "challenges\TRACKFORMERS\data\hits_and_tracks_3d_events_linear_10-50_all.csv" --sample-event 0
# 

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
    p.add_argument("--event-id", type=int, default=None,
                   help="Only load this exact event_id (CSV: stream-filtered).")
    p.add_argument("--max-events", type=int, default=None,
                   help="Only load the first N unique events (CSV: stream-limited).")
    p.add_argument("--chunksize", type=int, default=1_000_000,
                   help="CSV chunk size when using --event-id/--max-events.")
    p.add_argument("--plot", action="store_true",
                   help="Write 2D+3D plots for the chosen event "
                        "(--event-id if set else --sample-event).")
    p.add_argument("--max-tracks-plot", type=int, default=20,
                   help="Max tracks to draw in event plots (readability).")
    return p.parse_args()

def _detect_sep(sample_bytes: bytes) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_bytes.decode("utf-8", "ignore"),
                                      delimiters=[",",";","\t"])
        return dialect.delimiter
    except csv.Error:
        return ','

def _load_csv(path: str, user_sep: str | None, event_id: int | None = None, max_events: int | None = None, chunksize: int = 1_000_000) -> pd.DataFrame:
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

    # Default behaviour (load all) remains unchanged unless user requests limiting.
    if event_id is None and max_events is None:
        df = pd.read_csv(path, sep=sep, dtype=dtypes, low_memory=False, encoding="utf-8-sig")
        df.columns = [c.replace("\ufeff", "").strip().lower() for c in df.columns]
        return df

    # Stream-load & filter.
    frames = []
    selected = set()
    saw_target = False

    for chunk in pd.read_csv(
        path,
        sep=sep,
        dtype=dtypes,
        low_memory=False,
        encoding="utf-8-sig",
        chunksize=chunksize,
    ):
        chunk.columns = [c.replace("\ufeff", "").strip().lower() for c in chunk.columns]

        if "event_id" not in chunk.columns:
            raise RuntimeError("CSV does not contain 'event_id' after normalising column names.")

        if event_id is not None:
            sub = chunk[chunk["event_id"] == event_id]
            if not sub.empty:
                frames.append(sub)
                saw_target = True
            # Heuristic early-stop if file is sorted by event_id
            if saw_target and chunk["event_id"].min() > event_id:
                break

        else:
            # Keep first max_events unique event IDs as encountered
            for eid in chunk["event_id"].unique():
                if eid not in selected and len(selected) < int(max_events):
                    selected.add(int(eid))
            sub = chunk[chunk["event_id"].isin(selected)]
            if not sub.empty:
                frames.append(sub)
            # Heuristic early-stop if it looks like we passed the last selected id
            if len(selected) >= int(max_events) and chunk["event_id"].min() > max(selected):
                break

    if not frames:
        return pd.DataFrame(columns=list(dtypes.keys()))
    df = pd.concat(frames, ignore_index=True)
    return df

def _load_pkl(path: str, event_id: int | None = None, max_events: int | None = None) -> pd.DataFrame:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        data = pickle.load(fh)

    events = data.get("events")
    if events is None:
        raise RuntimeError(f"{path}: key 'events' not found")

    # Apply limits for PKL in-memory (cannot stream pickle easily).
    if event_id is not None:
        if event_id < 0 or event_id >= len(events):
            raise RuntimeError(f"{path}: event_id={event_id} out of range [0, {len(events)-1}]")
        iter_events = [(event_id, events[event_id])]
    elif max_events is not None:
        n = min(int(max_events), len(events))
        iter_events = list(enumerate(events[:n]))
    else:
        iter_events = list(enumerate(events))

    rows = []
    for ev_id, evt in iter_events:
        n = len(evt["hit_r"])
        rows.append(pd.DataFrame({
            "event_id":        np.full(n, ev_id, dtype=np.int32),
            "track_id":        evt["track_id"].astype(np.int32),
            "sub_detector_id": evt["layer_id"].astype(np.int16),
            "hit_r":           evt["hit_r"].astype(np.float32),
            "hit_theta":       evt["hit_theta"].astype(np.float32),
            "hit_z":           evt["hit_z"].astype(np.float32),
        }))
    df = pd.concat(rows, ignore_index=True)
    return df

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

def plot_event_2d(df_evt, fname, max_tracks=20):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_xy, ax_xz, ax_yz = axes

    tids = df_evt.track_id.unique()[:max_tracks]
    for tid in tids:
        sub = df_evt[df_evt.track_id == tid].sort_values("sub_detector_id", kind="mergesort")
        xi = sub.hit_r.to_numpy() * np.cos(sub.hit_theta.to_numpy())
        yi = sub.hit_r.to_numpy() * np.sin(sub.hit_theta.to_numpy())
        zi = sub.hit_z.to_numpy()

        ax_xy.plot(xi, yi, linewidth=1); ax_xy.scatter(xi, yi, s=6)
        ax_xz.plot(xi, zi, linewidth=1); ax_xz.scatter(xi, zi, s=6)
        ax_yz.plot(yi, zi, linewidth=1); ax_yz.scatter(yi, zi, s=6)

    ax_xy.set_title("XY"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xz.set_title("XZ"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_yz.set_title("YZ"); ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")

    fig.suptitle(f"Event {df_evt.event_id.iloc[0]} (≤{max_tracks} tracks)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"[saved] {fname}")

def scatter_event(df_evt, fname, max_tracks=20):
    import matplotlib.colors as mcolors; from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    palette = list(mcolors.TABLEAU_COLORS.values())

    for i, tid in enumerate(df_evt.track_id.unique()[:max_tracks]):
        sub = df_evt[df_evt.track_id == tid].sort_values("sub_detector_id", kind="mergesort")
        xi = sub.hit_r.to_numpy() * np.cos(sub.hit_theta.to_numpy())
        yi = sub.hit_r.to_numpy() * np.sin(sub.hit_theta.to_numpy())
        zi = sub.hit_z.to_numpy()
        ax.plot(xi, yi, zi, linewidth=1,
                color=palette[i % len(palette)],
                alpha=0.8)
        ax.scatter(xi, yi, zi, s=8, alpha=0.7,
                   color=palette[i % len(palette)],
                   label=f"t{tid}" if i < 10 else None)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Event {df_evt.event_id.iloc[0]} (≤{max_tracks} tracks)")
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

def load_any(path: str, user_sep: str | None, event_id: int | None = None, max_events: int | None = None, chunksize: int = 1_000_000) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".txt"}:
        return _load_csv(path, user_sep, event_id=event_id, max_events=max_events, chunksize=chunksize)
    elif ext in {".pkl", ".gz"}:
        return _load_pkl(path, event_id=event_id, max_events=max_events)
    else:
        raise RuntimeError(f"Unsupported file extension for {path}")

def main():
    args = parse_args()
    overlay = {"hits_evt": {}, "tracks_evt": {}, "hits_track": {}}

    for in_path in args.paths:
        print(f"\n================  {in_path}  ================")
        df = load_any(in_path, args.sep, event_id=args.event_id,
                      max_events=args.max_events, chunksize=args.chunksize)
        if df.empty:
            print("[warn] Loaded empty dataframe (no matching events). Skipping.")
            continue
        
        print("Loaded dataframe shape:", df.shape)

        hits_evt, tracks_evt, hits_track = compute_metrics(df)

        label = os.path.basename(in_path).replace(".pkl.gz","")
        collect_metric(overlay, label, hits_evt, tracks_evt, hits_track)

        stem = in_path.replace("\\", "_").replace("/", "_")

        evt_ids = df.event_id.sort_values().unique()
        if args.event_id is not None:
            evt_id = args.event_id
        else:
            evt_id = evt_ids[min(args.sample_event, len(evt_ids)-1)]

        df_evt = df[df.event_id == evt_id]
        if df_evt.empty:
            print(f"[warn] Event {evt_id} not present in loaded data. Skipping plots.")
        else:
            if args.plot:
                plot_event_2d(df_evt, f"{stem}_event{evt_id}_2D.png", max_tracks=args.max_tracks_plot)
                scatter_event(df_evt, f"{stem}_event{evt_id}_3D.png", max_tracks=args.max_tracks_plot)
            else:
                scatter_event(df_evt, f"{stem}_sample_event.png", max_tracks=args.max_tracks_plot)

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