# challenges/TRACKFORMERS/misc/REDVID_split.py


# --------------------------------------------
# Example usage:
# python redvid_split.py \
#        --input-csv challenges/TRACKFORMERS/data/hits_and_tracks_3d_events_all.csv \
#        --tag 10_50_linear \
# --------------------------------------------


import argparse, csv, gzip, os, pickle, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import List, Dict, Any


# recognised per-track parameter sets
LINEAR_COLS = ["r_0", "theta_0", "z_0", "r_d", "theta_d", "z_d"]
HELIX_COLS  = ["radial_const", "azimuthal_const", "pitch_const",
               "radial_coeff", "azimuthal_coeff", "pitch_coeff"]
BASE_COLS   = ["event_id", "track_id", "sub_detector_id",
               "hit_r", "hit_theta", "hit_z"]


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",       required=True)
    p.add_argument("--out-dir",         default="challenges/TRACKFORMERS/data")
    p.add_argument("--out-dir-misc",    default="challenges/TRACKFORMERS/misc")
    p.add_argument("--tag",             default="10_50_linear",
                   help="Label embedded in output filenames (e.g. 10_50_linear)")
    p.add_argument("--train-frac",      type=float, default=0.8)
    p.add_argument("--val-frac",        type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--skip-test",       action="store_true",
                   help="Skip split-balance validation step")
    return p.parse_args()


# Helper: detect delimiter
def detect_sep(path: str) -> str:
    with open(path, "rb") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", "ignore"),
                                      delimiters=[",", ";", "\t"])
        return dialect.delimiter
    except csv.Error:
        return ','


# Build per-event dict (closure captures param_cols)
def make_event_builder(param_cols: List[str]):
    def _builder(hits: pd.DataFrame) -> Dict[str, Any]:
        return {
            "hit_r":     hits.hit_r.values.astype("float32"),
            "hit_theta": hits.hit_theta.values.astype("float32"),
            "hit_z":     hits.hit_z.values.astype("float32"),
            "layer_id":  hits.sub_detector_id.values.astype("int16"),
            "track_id":  hits.track_id.values.astype("int32"),
            "track_params": hits[param_cols].to_numpy(np.float32),  # [N,6]
        }
    return _builder


# Test
def test_splits(split_paths: Dict[str, str], tag: str, misc_dir: str) -> None:
    def _load(p):
        with gzip.open(p, "rb") as fh:
            return pickle.load(fh)["events"]
        
    os.makedirs(misc_dir, exist_ok=True)

    stats, hist_data = {}, {}
    for split, p in split_paths.items():
        evts = _load(p)
        hits_per_evt   = np.array([e["hit_r"].shape[0]           for e in evts])
        tracks_per_evt = np.array([len(np.unique(e["track_id"])) for e in evts])
        layer_occ      = np.bincount(
            np.concatenate([e["layer_id"] for e in evts]), minlength=50)

        stats[split] = {
            "n_events": len(evts),
            "hits_mean": round(float(hits_per_evt.mean()),   2),
            "tracks_mean": round(float(tracks_per_evt.mean()), 2),
            "layer_occ": layer_occ / layer_occ.sum(),
        }
        hist_data[split] = (hits_per_evt, tracks_per_evt)

    # ───── text report ─────
    print("\n=== Split Balance Report ===")
    for k in ["n_events", "hits_mean", "tracks_mean"]:
        print({s: v[k] for s, v in stats.items()})

    base = stats["train"]["layer_occ"]
    devs = {s: round(float(np.abs(v["layer_occ"] - base).max()), 4)
            for s, v in stats.items()}
    print("max layer-occ deviation :", devs, "\n")

    # ───── plots: hits/event & tracks/event ─────
    for metric_idx, metric_name in enumerate(["Hits", "Tracks"]):
        plt.figure()
        for split, (hits_arr, tracks_arr) in hist_data.items():
            data = hits_arr if metric_idx == 0 else tracks_arr
            plt.hist(data, bins=60, alpha=0.5, label=split, histtype="step")
        plt.yscale("log")
        plt.xlabel(f"{metric_name.lower()} per event")
        plt.ylabel("Count")
        plt.title(f"{metric_name} per event distribution")
        plt.legend()
        fname = f"train_val_test_{tag}_{metric_name.lower()}_per_event.png"
        plt.tight_layout(); plt.savefig(os.path.join(misc_dir, fname)); plt.close()
        print(f"[saved] {fname}")

    # ───── store summary to disk ─────
    stats_file = os.path.join(misc_dir, f"split_stats_{tag}.txt")
    with open(stats_file, "w") as f:
        f.write("=== Split Balance Report ===\n")
        for k in ["n_events", "hits_mean", "tracks_mean"]:
            row = {s: v[k] for s, v in stats.items()}
            f.write(f"{k:12} : {row}\n")
        f.write(f"max layer-occ deviation : {devs}\n")
    print(f"[INFO] wrote split stats to {stats_file}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1  detect separator + parameter column set
    sep = detect_sep(args.input_csv)
    header = (pd.read_csv(args.input_csv, sep=sep, nrows=0)
                .columns.str.strip().str.lower())

    if all(c in header for c in LINEAR_COLS):
        param_cols, param_type = LINEAR_COLS, "linear"
    elif all(c in header for c in HELIX_COLS):
        param_cols, param_type = HELIX_COLS, "helical"
    else:
        raise RuntimeError("No recognised track-parameter columns in CSV.")

    print(f"[INFO] detected {param_type} parameter set "
          f"({', '.join(param_cols)})")

    # 2  build column list + dtypes
    COLUMNS = BASE_COLS + param_cols
    DTYPES  = {c: "float32" for c in COLUMNS}
    DTYPES.update({"event_id": "int32", "track_id": "int32",
                   "sub_detector_id": "int16"})

    # 3  load CSV
    print("[INFO] loading CSV …")
    df = pd.read_csv(args.input_csv, sep=sep, usecols=COLUMNS,
                     dtype=DTYPES, low_memory=False)

    # 4  group by event_id
    build_event = make_event_builder(param_cols)
    events = [build_event(h) for _, h in df.groupby("event_id")]
    del df

    random.seed(args.seed)
    random.shuffle(events)

    n_total = len(events)
    n_train = int(args.train_frac * n_total)
    n_val   = int(args.val_frac   * n_total)

    splits = {
        "train": events[:n_train],
        "val":   events[n_train:n_train+n_val],
        "test":  events[n_train+n_val:],
    }

    # 5  write pickles
    out_paths = {}
    for split_name, evts in splits.items():
        path = os.path.join(args.out_dir,
                            f"REDVID_{args.tag}_{split_name}.pkl.gz")
        out_paths[split_name] = path
        with gzip.open(path, "wb") as fh:
            pickle.dump({"events": evts}, fh, protocol=4)
        print(f"[OK] wrote {path:60s}  ({len(evts)} events)")

    # 6  optional test & report
    if not args.skip_test:
        test_splits(out_paths, tag=args.tag, misc_dir=args.out_dir_misc)

if __name__ == "__main__":
    main()