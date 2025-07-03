# challenges/TRACKFORMERS/misc/downsample_redvid.py

"""
Keep only a fraction of events in a REDVID pkl.gz file.

USAGE (powershell):
python challenges/TRACKFORMERS/misc/downsample_redvid.py 0.05 `
  challenges/TRACKFORMERS/data/REDVID_10_50_linear_train.pkl.gz `
  challenges/TRACKFORMERS/data/REDVID_10_50_linear_val.pkl.gz `
  challenges/TRACKFORMERS/data/REDVID_10_50_linear_test.pkl.gz

The script writes e.g.
    REDVID_10_50_linear_train_frac0.05.pkl.gz
into the same directory as the source.
"""

import sys, gzip, pickle, random, pathlib
from typing import Sequence

# helpers ----------------------------------------------------------------
def load_events(pkl_gz: str) -> Sequence[dict]:
    with gzip.open(pkl_gz, "rb") as fh:
        data = pickle.load(fh)
    if "events" not in data:
        raise RuntimeError(f"{pkl_gz}: no 'events' key")
    return data["events"]

def save_events(events: Sequence[dict], template_path: str, frac: float) -> str:
    src = pathlib.Path(template_path)

    # strip double suffix (.pkl.gz) to get the bare stem
    stem_pkl = src.stem                    # ..._test.pkl
    stem     = pathlib.Path(stem_pkl).stem # ..._test

    prefix, split = stem.rsplit("_", 1)    # "REDVID_10_50_linear", "test"
    new_stem = f"{prefix}_frac{frac:.2f}_{split}"   # insert tag before split

    outname = src.with_name(new_stem + ".pkl.gz")   # rebuild full name
    with gzip.open(outname, "wb") as fh:
        pickle.dump({"events": events}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return str(outname)


def downsample_file(pkl_gz: str, frac: float, rng: random.Random) -> None:
    events = load_events(pkl_gz)
    k      = max(1, int(len(events) * frac))
    subset = rng.sample(events, k)
    out    = save_events(subset, pkl_gz, frac)
    print(f"{pkl_gz}: kept {k}/{len(events)} events -> {out}")

# main ------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    frac = float(sys.argv[1])
    if not (0.0 < frac <= 1.0):
        sys.exit("fraction must be in (0,1]")

    rng = random.Random(42)
    for path in sys.argv[2:]:
        downsample_file(path, frac, rng)

if __name__ == "__main__":
    main()