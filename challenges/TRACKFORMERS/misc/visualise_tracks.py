# challenges/TRACKFORMERS/misc/visualise_tracks.py
#
# Example:
# python challenges/TRACKFORMERS/misc/visualise_tracks.py --model-path "outputs\01-12\TRACKFORMERS\Q1\anthropic_claude-sonnet-4.5\anthropic_claude-sonnet-4.5_1707_1_model.pkl" --tag REDVID_10-50_linear_frac0.05 --event-idx 42 --geom-source raw

from __future__ import annotations

import sys, os, subprocess, argparse, logging, torch, gzip, pickle
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure repo root is on sys.path so `import challenges...` and `import utils...` work
_this = Path(__file__).resolve()
repo_root = None
for p in _this.parents:
    if p.name == "challenges":
        repo_root = p.parent
        break
if repo_root is None:
    repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass
from challenges.TRACKFORMERS.evaluate_trackformers import artefact_base_from_path, load_TRACKFORMERS_test, DEFAULT_TAG, fit_accuracy
from utils.llm_io import _initialize_artefacts, detect_and_assert_lane, assert_label_output_by_lane
from utils.loaderspec import LoaderSpec, enforce_pyg_policy

log = logging.getLogger("visualise_tracks")

def _load_raw_event(tag: str, event_idx: int) -> dict:
    """
    Load raw event dict from the benchmark test pkl.gz.
    Assumes file layout: challenges/TRACKFORMERS/data/test/{tag}_test.pkl.gz
    with payload {"events": [evt0, evt1, ...]}.
    """
    pkl = (repo_root / "challenges" / "TRACKFORMERS" / "data" / "test" / f"{tag}_test.pkl.gz").resolve()
    if not pkl.exists():
        raise FileNotFoundError(f"Raw test file not found: {pkl}")
    with gzip.open(pkl, "rb") as fh:
        payload = pickle.load(fh)
    events = payload.get("events")
    if events is None:
        raise KeyError(f"{pkl}: key 'events' not found")
    if event_idx < 0 or event_idx >= len(events):
        raise IndexError(f"{pkl}: event_idx={event_idx} out of range [0,{len(events)-1}]")
    evt = events[event_idx]
    if not isinstance(evt, dict):
        raise TypeError(f"{pkl}: events[{event_idx}] is not a dict")
    return evt

def _load_spec_for_model(model_path: str | Path) -> LoaderSpec:
    base = artefact_base_from_path(model_path)
    model_dir = Path(model_path).resolve().parent
    spec_path = model_dir / f"{base}_loaderspec.json"
    spec = LoaderSpec.from_json(spec_path)
    spec = enforce_pyg_policy(spec, require_torch_collate=True)
    return spec

def _move_x_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [xi.to(device) if torch.is_tensor(xi) else xi for xi in x]
    if hasattr(x, "to"):
        return x.to(device)
    return x

def _to_numpy_1d_int(x: Any, expected_len: int) -> np.ndarray:
    """
    Strict int labels, same philosophy as evaluator.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 1:
            raise TypeError(f"Pred labels must be 1D, got shape={tuple(x.shape)}")
        if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError(f"Pred labels must be integer dtype, got {x.dtype}")
        arr = x.numpy().astype(np.int64, copy=False)
    else:
        arr = np.asarray(x)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise TypeError(f"Pred labels must be 1D, got shape={arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"Pred labels must be integer dtype, got {arr.dtype}")
        arr = arr.astype(np.int64, copy=False)

    if arr.shape[0] != expected_len:
        raise ValueError(f"Pred labels length {arr.shape[0]} != N hits {expected_len}")
    return arr

@dataclass
class HitView:
    r: np.ndarray
    theta: np.ndarray
    z: np.ndarray
    layer_id: Optional[np.ndarray] = None

    @property
    def x(self) -> np.ndarray:
        return self.r * np.cos(self.theta)

    @property
    def y(self) -> np.ndarray:
        return self.r * np.sin(self.theta)

def extract_hits(x_event: Any, *, feature_order: Tuple[int, int, int, Optional[int]] = (0, 1, 2, 3)) -> HitView:
    """
    Extract r, theta, z, (layer_id) from an event input.
    Assumes event input is either:
      - Tensor [N, F] with columns = (r, theta, z, layer_id?)
      - PyG Data/Batch with .x [N, F]
      - dict-like with keys
    """

    # PyG style
    if hasattr(x_event, "x") and torch.is_tensor(getattr(x_event, "x")):
        feats = x_event.x

    # dict style
    elif isinstance(x_event, dict):
        # REDVID raw (cylindrical)
        if ("hit_r" in x_event) and ("hit_theta" in x_event) and ("hit_z" in x_event):
            r = np.asarray(x_event["hit_r"]).reshape(-1)
            th = np.asarray(x_event["hit_theta"]).reshape(-1)
            z = np.asarray(x_event["hit_z"]).reshape(-1)
            lid = np.asarray(x_event["layer_id"]).reshape(-1) if "layer_id" in x_event else None
            return HitView(r=r, theta=th, z=z, layer_id=lid)

        # TrackML-ish raw (cartesian) -> convert to (r,theta,z)
        # accept either (x,y,z) or (hit_x,hit_y,hit_z)
        if ("x" in x_event) and ("y" in x_event) and ("z" in x_event):
            x = np.asarray(x_event["x"]).reshape(-1)
            y = np.asarray(x_event["y"]).reshape(-1)
            z = np.asarray(x_event["z"]).reshape(-1)
        elif ("hit_x" in x_event) and ("hit_y" in x_event) and ("hit_z" in x_event):
            x = np.asarray(x_event["hit_x"]).reshape(-1)
            y = np.asarray(x_event["hit_y"]).reshape(-1)
            z = np.asarray(x_event["hit_z"]).reshape(-1)
        else:
            raise KeyError("Dict event did not contain REDVID keys (hit_r/hit_theta/hit_z) "
                           "or cartesian keys (x/y/z or hit_x/hit_y/hit_z).")
        r = np.sqrt(x * x + y * y)
        th = np.arctan2(y, x)
        lid = np.asarray(x_event["layer_id"]).reshape(-1) if "layer_id" in x_event else None
        return HitView(r=r, theta=th, z=z, layer_id=lid)
    else:
        feats = x_event

    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)

    feats = feats.detach().cpu()

    if feats.ndim != 2:
        raise TypeError(f"Expected event features as [N,F], got shape={tuple(feats.shape)}")

    # Heuristic: if someone provided [F,N] with small F, transpose
    if feats.shape[0] <= 6 and feats.shape[1] > 10:
        feats = feats.t()

    r_i, th_i, z_i, layer_i = feature_order
    arr = feats.numpy()

    r = arr[:, r_i].astype(np.float64, copy=False)
    th = arr[:, th_i].astype(np.float64, copy=False)
    zz = arr[:, z_i].astype(np.float64, copy=False)
    lid = arr[:, layer_i].astype(np.float64, copy=False) if (layer_i is not None and layer_i < arr.shape[1]) else None

    return HitView(r=r, theta=th, z=zz, layer_id=lid)

@dataclass
class ClusterReport:
    pred_label: int
    size: int
    t_star: int
    major_nhits: int
    purity_rec: float
    purity_maj: float
    counted: bool

def fit_accuracy_hit_mask(pred_lbl: np.ndarray, true_tid: np.ndarray) -> Tuple[np.ndarray, int, int, Dict[int, ClusterReport]]:
    """
    Same rules as benchmark FitAccuracy, but also returns a per-hit mask for plotting.

    Returns:
      counted_mask: bool[N]  True for hits counted in numerator
      correct_hits: int
      denom: int
      reports: per predicted cluster diagnostics
    """
    if pred_lbl.shape != true_tid.shape:
        raise ValueError("pred / true shape mismatch")

    N = int(true_tid.shape[0])
    counted_mask = np.zeros(N, dtype=bool)

    # 1) truth hits only
    mask_truth = (true_tid != 0)
    denom = int(mask_truth.sum())
    if denom == 0:
        return counted_mask, 0, 0, {}

    pred_all = pred_lbl[mask_truth]
    true_all = true_tid[mask_truth]

    # 2) truth sizes over all truth hits
    tmax = int(true_all.max())
    truth_sizes = np.bincount(true_all, minlength=tmax + 1)

    # 3) ignore predicted noise for cluster iteration
    keep_pred = (pred_all != -1)
    pred = pred_all[keep_pred]
    true = true_all[keep_pred]

    idx_truth = np.nonzero(mask_truth)[0]   # original indices of truth hits
    idx_kept = idx_truth[keep_pred]         # original indices of truth hits that aren't pred-noise

    if pred.size == 0:
        return counted_mask, 0, denom, {}

    correct_hits = 0
    reports: Dict[int, ClusterReport] = {}

    unique_pred, pred_counts = np.unique(pred, return_counts=True)
    for p, cnt in zip(unique_pred, pred_counts):
        cnt = int(cnt)

        sel = (pred == p)
        t_sub = true[sel]

        # majority truth id
        overlaps = np.bincount(t_sub, minlength=tmax + 1)
        t_star = int(np.argmax(overlaps))
        major_nhits = int(overlaps[t_star])

        if cnt < 4 or major_nhits == 0:
            reports[int(p)] = ClusterReport(int(p), cnt, t_star, major_nhits, 0.0, 0.0, False)
            continue

        purity_rec = major_nhits / float(cnt)
        purity_maj = major_nhits / float(max(int(truth_sizes[t_star]), 1))
        ok = (purity_rec >= 0.5) and (purity_maj >= 0.5)

        reports[int(p)] = ClusterReport(int(p), cnt, t_star, major_nhits, purity_rec, purity_maj, ok)

        if ok:
            # mark hits that are in this predicted cluster AND are truth t_star
            original_idx_for_cluster = idx_kept[sel]
            correct_idx = original_idx_for_cluster[t_sub == t_star]
            counted_mask[correct_idx] = True
            correct_hits += major_nhits

    return counted_mask, int(correct_hits), int(denom), reports

def _set_axes_equal_3d(ax) -> None:
    """
    Make 3D axes have equal scale so geometry isn't visually distorted.
    """
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xmid = 0.5 * (xlim[0] + xlim[1]); ymid = 0.5 * (ylim[0] + ylim[1]); zmid = 0.5 * (zlim[0] + zlim[1])
    xr = abs(xlim[1] - xlim[0]); yr = abs(ylim[1] - ylim[0]); zr = abs(zlim[1] - zlim[0])
    r = 0.5 * max(xr, yr, zr)
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)

def plot_event(h: HitView, pred: np.ndarray, true: np.ndarray, counted: np.ndarray, *, title: str = "", max_clusters_legend: int = 12, out: Optional[str] = None, plot_mode: str = "xy_3d", colour_by: str = "pred") -> None:
    """
    Two-panel view:
      - Left: XY (transverse plane)
      - Right: ZR (longitudinal)  OR  XYZ (3D) depending on `plot_mode`
    Colour = predicted cluster (reindexed); wrong hits outlined.
    """

    N = pred.shape[0]
    # Reindex predicted labels for nicer colouring (keep -1 as -1)
    uniq = [p for p in np.unique(pred) if p != -1]
    p_to_idx = {p: i for i, p in enumerate(uniq)}
    pred_idx = np.array([p_to_idx.get(int(p), -1) for p in pred], dtype=np.int64)

    fig = plt.figure(figsize=(12, 5))
    ax_xy = fig.add_subplot(1, 2, 1)
    if plot_mode == "xy_3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax_right = fig.add_subplot(1, 2, 2, projection="3d")
        right_is_3d = True
    elif plot_mode == "xy_zr":
        ax_right = fig.add_subplot(1, 2, 2)
        right_is_3d = False
    else:
        raise ValueError(f"Unknown plot_mode={plot_mode!r}. Use 'xy_3d' or 'xy_zr'.")

    def _scatter_right(mask: np.ndarray, **kwargs):
        if right_is_3d:
            return ax_right.scatter(h.x[mask], h.y[mask], h.z[mask], **kwargs)
        return ax_right.scatter(h.z[mask], h.r[mask], **kwargs)
    
    def _sizes(mask: np.ndarray):
        # s can be scalar OR array; keep it consistent everywhere
        return (np.asarray(s)[mask] if np.ndim(s) else s)

    # Base scatter: colour by predicted cluster index
    # Use a colormap with enough distinct-ish colours
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))

    # marker sizes: slightly vary if layer info exists
    if h.layer_id is not None:
        # squash to reasonable range
        ls = (h.layer_id - np.nanmin(h.layer_id)) if np.isfinite(h.layer_id).all() else h.layer_id
        s = 6 + 2 * (ls.astype(np.float64) % 4)
    else:
        s = 8

    # wrong hits = those that are assigned to a cluster but are not "correct"
    truth = (true != 0)
    truth_noise = (true == 0)
    pred_noise = (pred == -1)

    wrong = truth & (~counted) & (~pred_noise)  # “truth hits not counted by FitAccuracy”

    # plot truth-noise separately (so it doesn't steal cluster colours)
    base = (~pred_noise) & (~truth_noise)

    if colour_by == "true":
        # map truth IDs to compact indices for the colormap (exclude 0)
        uniq_t = [t for t in np.unique(true) if t != 0]
        t_to_idx = {t: i for i, t in enumerate(uniq_t)}
        col = np.array([t_to_idx.get(int(t), -1) for t in true], dtype=np.int64)
        cvals = col[base]
    else:
        cvals = pred_idx[base]

    ax_xy.scatter(h.x[base], h.y[base], c=cvals, cmap=cmap,
                s=_sizes(base), linewidths=0.0, alpha=0.9)
    _scatter_right(base, c=cvals, cmap=cmap,
                s=_sizes(base), alpha=0.85)

    if truth_noise.any():
        ax_xy.scatter(h.x[truth_noise], h.y[truth_noise],
                    marker=".", alpha=0.25,
                    s=_sizes(truth_noise))
        _scatter_right(truth_noise, marker=".", alpha=0.25,
                    s=_sizes(truth_noise))

    # wrong hits outline
    if wrong.any():
        ax_xy.scatter(h.x[wrong], h.y[wrong], facecolors= "none", edgecolors="black", s=_sizes(wrong), linewidths=0.9, alpha=0.95)

        if right_is_3d:
            # 3D: make the outline actually visible
            s3 = np.asarray(_sizes(wrong), dtype=float) * 2.0
            ax_right.scatter(h.x[wrong], h.y[wrong], h.z[wrong], s=s3, facecolors="none", edgecolors="black", linewidths=1.2, alpha=1.0, depthshade=False)
        else:
            _scatter_right(wrong, facecolors="none", edgecolors="black", s=_sizes(wrong), linewidths=0.9, alpha=0.95)

    # predicted noise: show as x markers
    if pred_noise.any():
        ax_xy.scatter(h.x[pred_noise], h.y[pred_noise],
                      marker="x", s=18, alpha=0.6)
        if right_is_3d:
            _scatter_right(pred_noise, marker="x", s=18, alpha=0.6, depthshade=False)
        else:
            _scatter_right(pred_noise, marker="x", s=18, alpha=0.6)

    # axes/labels (must NOT be inside the pred_noise block)
    ax_xy.set_title("XY: colour=pred cluster, outline=wrong")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.axis("equal")

    if right_is_3d:
        ax_right.set_title("XYZ (3D): colour=pred cluster, outline=wrong")
        ax_right.set_xlabel("x")
        ax_right.set_ylabel("y")
        ax_right.set_zlabel("z")
        _set_axes_equal_3d(ax_right)
    else:
        ax_right.set_title("ZR: colour=pred cluster, outline=wrong")
        ax_right.set_xlabel("z")
        ax_right.set_ylabel("r")

    if title:
        fig.suptitle(title)

    # Minimal legend: show up to a few predicted clusters
    if len(uniq) > 0:
        handles = []
        labels = []
        for p in uniq[:max_clusters_legend]:
            idx = p_to_idx[p]
            # dummy handle
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                                      markerfacecolor=cmap(idx), markeredgecolor="none"))
            labels.append(f"pred {p}")
        ax_xy.legend(handles, labels, loc="best", frameon=True, fontsize=8)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
        log.info("Saved plot: %s", out)
    else:
        plt.show()

def _docker_viz_cmd(project_root: Path, artefact_pkl: Path, *, tag: str, event_idx: int, outdir: Optional[Path], plot_mode: str) -> list[str]:
    """
    Build docker run command that executes visualise_tracks.py inside llm-evaluation-sandbox:latest for a single model.
    """

    artefact_dir        = artefact_pkl.parent                  # .../outputs/<DATE>/<C>/<Q>/<MODEL>
    viz_script_dir = project_root / "challenges/TRACKFORMERS/misc"
    viz_out = (outdir if outdir is not None else (artefact_dir / "viz"))
    viz_out.mkdir(parents=True, exist_ok=True)

    # Docker volumes    
    data_test           = project_root / f"challenges/TRACKFORMERS/data/test"
    evaluator_py        = project_root / f"challenges/TRACKFORMERS/evaluate_trackformers.py"
    llm_io_py           = project_root / "utils/llm_io.py"
    loaderspec_py       = project_root / "utils/loaderspec.py"
    suffix_utils_py     = project_root / "utils/suffix_utils.py"
    utils_challenge     = project_root / f"challenges/TRACKFORMERS/utils_trackformers.py"

    # Build CMD
    cmd = [
        # args
        "docker", "run", "--rm",
        "--gpus", "all",
        "--read-only",
        "--cap-drop", "ALL",
        "--network", "none",
        "--security-opt", f"seccomp={project_root/'docker/seccomp_profile.json'}",
        "--tmpfs", "/tmp:rw,noexec,nosuid",
        "--tmpfs", "/dev/shm:rw",

        # Force CUDA to run synchronously
        "-e", "CUDA_LAUNCH_BLOCKING=1",

        # Prevent Python/Matplotlib cache writes
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "MPLCONFIGDIR=/tmp/mplconfig",
        "-e", "IN_LLM_SANDBOX=1",

        # Mounts
        "-v", f"{data_test}:/workspace/challenges/TRACKFORMERS/data/test:ro",
        "-v", f"{evaluator_py}:/workspace/challenges/TRACKFORMERS/evaluate_trackformers.py:ro",
        "-v", f"{utils_challenge}:/workspace/challenges/TRACKFORMERS/utils_trackformers.py:ro",
        "-v", f"{llm_io_py}:/workspace/utils/llm_io.py:ro",
        "-v", f"{loaderspec_py}:/workspace/utils/loaderspec.py:ro",
        "-v", f"{suffix_utils_py}:/workspace/utils/suffix_utils.py:ro",
        "-v", f"{artefact_dir}:/workspace/out:ro",
        "-v", f"{viz_script_dir}:/workspace/challenges/TRACKFORMERS/misc:ro",
        "-v", f"{viz_out}:/workspace/viz:rw",

        # run python directly
        "--entrypoint", "python",
        "-w", "/workspace",
        "llm-sandbox:latest",
        "/workspace/challenges/TRACKFORMERS/misc/visualise_tracks.py",
        "--model-path", f"/workspace/out/{artefact_pkl.name}",
        "--tag", tag,
        "--event-idx", str(event_idx),
        "--plot-mode", plot_mode,
        "--out", f"/workspace/viz/event_{event_idx}.png",
    ]

    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tag", type=str, default=DEFAULT_TAG, help="Dataset tag")
    ap.add_argument("--event-idx", type=int, default=0, help="Global event index in the loader stream")
    ap.add_argument("--r-col", type=int, default=0)
    ap.add_argument("--theta-col", type=int, default=1)
    ap.add_argument("--z-col", type=int, default=2)
    ap.add_argument("--layer-col", type=int, default=3, help="Set to -1 to disable layer extraction")
    ap.add_argument("--loglevel", type=str, default="INFO")
    ap.add_argument("--out", type=str, default=None, help="If set, save plot to this path (PNG) instead of showing.")
    ap.add_argument("--outdir", type=str, default=None, help="Host output dir for plots (default: <modeldir>/viz)")
    ap.add_argument("--geom-source", type=str, default="raw", choices=["raw", "input"],
        help="Which coordinates to plot. "
             "raw = load raw event from dataset pkl.gz and overlay predictions (recommended for benchmark). "
             "input = interpret model input features as [r,theta,z,(layer)] (legacy behaviour).")
    ap.add_argument("--plot-mode", type=str, default="xy_3d", choices=["xy_3d", "xy_zr"],
        help="Right-hand panel: 'xy_3d' (XYZ 3D) or 'xy_zr' (ZR 2D). Default: xy_3d")
    ap.add_argument("--colour-by", type=str, default="pred", choices=["pred", "true"],
        help="Colour points by predicted cluster labels (pred) or by ground-truth track_id (true).")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    if os.environ.get("IN_LLM_SANDBOX") != "1":
        artefact_pkl = Path(args.model_path).resolve()
        outdir = Path(args.outdir).resolve() if args.outdir else None

        cmd = _docker_viz_cmd(
            project_root=repo_root.resolve(),
            artefact_pkl=artefact_pkl,
            tag=args.tag,
            event_idx=int(args.event_idx),
            plot_mode=str(args.plot_mode),
            outdir=outdir,
        )

        logging.info("Running visualisation in docker...")
        subprocess.run(cmd, check=True)
        return
    
    # Setup devide and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _preproc = _initialize_artefacts(args.model_path)
    model.to(device).eval()

    # Rebuild hidden test loader exactly like evaluation does
    test_loader = load_TRACKFORMERS_test(args.model_path, tag=args.tag)

    # Grab Nth event
    spec = _load_spec_for_model(args.model_path)

    target_i = int(args.event_idx)
    x_e = y_e = out_e = None
    mode = None
    seen = 0

    with torch.no_grad():
        for batch in test_loader:
            if mode is None:
                mode = detect_and_assert_lane(spec, batch)

            if mode == "torch_ragged_xy":
                Xs, ys = batch
                Xs_dev = [x.to(device) for x in Xs]

                out = model.predict_labels(Xs_dev)
                assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)

                for x_i, y_i, out_i in zip(Xs, ys, out):
                    if seen == target_i:
                        x_e, y_e, out_e = x_i, y_i, out_i
                        break
                    seen += 1

            elif mode == "pyg_batch":
                G = batch.to(device)
                if hasattr(G, "num_graphs") and G.num_graphs != 1:
                    raise RuntimeError(f"PyG eval expected batch_size=1, got num_graphs={G.num_graphs}.")

                out = model.predict_labels(G)
                assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)

                if seen == target_i:
                    x_e, y_e, out_e = batch, batch.y, out   # keep CPU batch for plotting
                    break
                seen += 1

            else:
                raise RuntimeError(f"Unknown lane mode: {mode}")

            if x_e is not None:
                break

    if x_e is None:
        raise IndexError(f"event-idx={target_i} out of range for this loader stream")

    # Truth ids
    if y_e is None:
        raise ValueError("This loader produced no truth labels (batch_y is None); cannot visualise correctness.")
    if torch.is_tensor(y_e):
        true = y_e.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    else:
        true = np.asarray(y_e).reshape(-1).astype(np.int64, copy=False)

    # Predict
    pred = _to_numpy_1d_int(out_e, expected_len=true.shape[0])

    counted_mask, correct_hits, denom, reports = fit_accuracy_hit_mask(pred, true)
    fitacc_evt = correct_hits / max(denom, 1)

    # optional sanity check vs evaluator function you imported
    c2, d2 = fit_accuracy(pred, true)
    if int(c2) != int(correct_hits) or int(d2) != int(denom):
        log.warning("fit_accuracy mismatch: imported=(%s,%s) vs mask=(%s,%s)", c2, d2, correct_hits, denom)

    # Summarise a bit
    n_hits = int(true.shape[0])
    n_truth = denom
    n_assigned = int((pred != -1).sum())
    n_correct = correct_hits
    log.info("Event %d: N hits=%d, truth hits=%d, pred-assigned=%d, correct(FitAcc)=%d",
            target_i, n_hits, n_truth, n_assigned, n_correct)

    # Print top clusters by size
    top = sorted(reports.values(), key=lambda r: r.size, reverse=True)[:10]
    for r in top:
        log.info("  pred=%d size=%d t*=%d major=%d purity_rec=%.3f purity_maj=%.3f counted=%s",
                r.pred_label, r.size, r.t_star, r.major_nhits, r.purity_rec, r.purity_maj, r.counted)

    # Extract geometry (assumes columns are r,theta,z,layer)
    layer_col = None if args.layer_col < 0 else int(args.layer_col)
    if args.geom_source == "raw":
        try:
            raw_evt = _load_raw_event(tag=args.tag, event_idx=int(args.event_idx))
            hv = extract_hits(raw_evt)  # dict-path: uses hit_r/hit_theta/hit_z (REDVID) or x/y/z (TrackML)
            log.info("Plotting RAW geometry from dataset file (geom-source=raw).")
        except Exception as e:
            log.warning("Failed to load raw geometry (%s). Falling back to geom-source=input.", e)
            hv = extract_hits(
                x_e,
                feature_order=(int(args.r_col), int(args.theta_col), int(args.z_col), layer_col),
            )
    else:
        hv = extract_hits(x_e, feature_order=(int(args.r_col), int(args.theta_col), int(args.z_col), layer_col))

    # Plot with model name in title
    model_name = Path(args.model_path).name
    model_name = model_name.replace("_model.pkl", "").replace("_state.pt", "")
    title = f"{model_name} | event {target_i} | FitAcc={fitacc_evt:.3f} ({correct_hits}/{denom}) | assigned={n_assigned}/{n_hits}"

    plot_event(hv, pred=pred, true=true, counted=counted_mask, title=title, out=args.out, plot_mode=str(args.plot_mode), colour_by=str(args.colour_by))

if __name__ == "__main__":
    main()
