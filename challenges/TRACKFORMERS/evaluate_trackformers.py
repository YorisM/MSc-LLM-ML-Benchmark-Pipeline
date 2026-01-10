# challenges/TRACKFORMERS/evaluate_trackformers.py

import gzip, pickle, torch, logging
import numpy as np
from utils.llm_io import _initialize_artefacts, detect_and_assert_lane, assert_label_output_by_lane, build_dataset, build_dataloader
from utils.loaderspec import LoaderSpec, enforce_pyg_policy
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


DEFAULT_TAG = "REDVID_10-50_linear_frac0.05"


# Force TF32 off
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True  # match training harness

def artefact_base_from_path(model_path: str | Path) -> str:
    name = Path(model_path).name
    for suf in ("_model.pkl", "_state.pt"):
        if name.endswith(suf):
            return name[:-len(suf)]
    raise ValueError(f"Unrecognized model artifact filename: {name}")

def load_TRACKFORMERS_test(model_path: str, tag: str = DEFAULT_TAG):
    """
    Build the SAME DataLoader config the LLM used in training, but for the hidden
    TRACKFORMERS test split.
    """

    logging.debug(f"Model path: {model_path}")

    # Resolve {base} naming convention
    base = artefact_base_from_path(model_path)
    logging.debug(f"Base name: {base}")

    # Define model directory
    model_dir = Path(model_path).resolve().parent
    logging.debug(f"Model directory: {model_dir}")

    # Load preprocessing object
    _, preproc = _initialize_artefacts(model_path)
    logging.debug(f"preproc: {preproc}")

    # Load LoaderSpec Object
    spec_path = Path(model_dir) / f"{base}_loaderspec.json"
    spec = LoaderSpec.from_json(spec_path)
    spec = enforce_pyg_policy(spec, require_torch_collate=True)

    # Load test set
    test_dir = Path(__file__).resolve().parent / "data" / "test"
    fn = test_dir / f"{tag}_test.pkl.gz"
    with gzip.open(fn, "rb") as fh:
        events = pickle.load(fh)["events"]

    # Build Test Loader
    test_ds = build_dataset(spec, events, preproc, train=False)
    test_loader = build_dataloader(spec, test_ds, is_eval=True)
    
    return test_loader

def fit_accuracy(pred_lbl: np.ndarray, true_tid: np.ndarray) -> Tuple[int, int]:
    """
    TrackML-style FitAccuracy for a single event:
    
    - consider only truth hits with true_tid != 0
    - for each predicted cluster p with nhits(p) >= 4:
         let t* be the truth id with maximal overlap in p
         reco purity      = major_nhits / nhits(p)                  >= 0.5
         truth efficiency = major_nhits / major_particle_nhits(t*)  >= 0.5
      if both hold, add major_nhits to the numerator
    - denominator = total number of truth hits (true_tid != 0)
    """
        
    if pred_lbl.shape != true_tid.shape:
        raise ValueError("pred / true shape mismatch")

    # 1) keep only truth-labeled hits
    mask_truth = (true_tid != 0)
    denom = int(mask_truth.sum())
    if denom == 0:
        return 0, 0

    pred_all = pred_lbl[mask_truth]
    true_all = true_tid[mask_truth]

    # 2) truth_sizes must be computed on *all* truth hits
    tmax = int(true_all.max())
    truth_sizes = np.bincount(true_all, minlength=tmax + 1)

    # 3) ignore predicted noise only for cluster iteration
    keep_pred = (pred_all != -1)
    pred = pred_all[keep_pred]
    true = true_all[keep_pred]
    if pred.size == 0:
        return 0, denom

    correct_hits = 0
    unique_pred, pred_counts = np.unique(pred, return_counts=True)

    for p, cnt in zip(unique_pred, pred_counts):
        if cnt < 4:
            continue

        t_sub = true[pred == p]
        overlaps = np.bincount(t_sub, minlength=tmax + 1)
        t_star = int(np.argmax(overlaps))
        major_nhits = int(overlaps[t_star])
        if major_nhits == 0:
            continue

        purity_rec = major_nhits / int(cnt)
        purity_maj = major_nhits / max(int(truth_sizes[t_star]), 1)

        if (purity_rec >= 0.5) and (purity_maj >= 0.5):
            correct_hits += major_nhits

    return correct_hits, denom

def evaluate_TRACKFORMERS(model_path: str, test_loader) -> dict:
    """
    Evaluate TRACKFORMERS using FitAccuracy (TrackML-style)

    Contract (strict):
    Lane A (torch ragged):
        - DataLoader yields (Xs, ys) where Xs, ys are lists of length B.
        - Model returns list of integer label tensors, length B, each shape (N_i,).

    Lane B (PyG):
        - DataLoader yields a PyG Batch/Data object with .x and .y.
        - Model returns a single integer label tensor of shape (num_nodes,).

    Predicted noise label:
    - Use -1 for noise/unassigned in model OUTPUT.
    Truth noise:
    - truth track_id == 0 is ignored by the metric.

    RETURNS
    FitAccuracy: float
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _preproc = _initialize_artefacts(model_path)
    model.to(device).eval()
    logging.info("Evaluating %s on %s", model_path, device)

    total_correct_hits = 0
    total_truth_hits = 0

    def _to_numpy_1d_int(x, *, expected_len: int) -> np.ndarray:
        """
        Convert a single-event model output into a 1D int64 numpy array of length expected_len.
        Strict: rejects float outputs even if they look integer-ish.
        """
        if torch.is_tensor(x):
            x = x.detach().cpu()

            # Allow shapes like (N,1) -> (N,)
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.ndim != 1:
                raise TypeError(f"Model output must be 1D labels, got tensor shape {tuple(x.shape)}.")

            if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
                raise TypeError(
                    f"Model output must be integer labels (torch.int*). Got dtype={x.dtype}."
                )
            arr = x.numpy().astype(np.int64, copy=False)

        else:
            arr = np.asarray(x)

            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            elif arr.ndim != 1:
                raise TypeError(f"Model output must be 1D labels, got array shape {arr.shape}.")

            if not np.issubdtype(arr.dtype, np.integer):
                raise TypeError(f"Model output must be integer labels (np.int*). Got dtype={arr.dtype}.")

            arr = arr.astype(np.int64, copy=False)

        if arr.shape[0] != expected_len:
            raise ValueError(
                f"Model output has {arr.shape[0]} labels but event has {expected_len} hits. "
                "Model must output one label per hit."
            )
        return arr

    # Initialise loaderspec
    base = artefact_base_from_path(model_path)
    model_dir = Path(model_path).resolve().parent
    spec_path = model_dir / f"{base}_loaderspec.json"
    spec = LoaderSpec.from_json(spec_path)
    spec = enforce_pyg_policy(spec, require_torch_collate=True)

    with torch.no_grad():
        mode = None
        for batch in tqdm(test_loader, desc="Evaluating TRACKFORMERS", unit="batch"):
            if mode is None:
                mode = detect_and_assert_lane(spec, batch)

            if mode == "torch_ragged_xy":
                Xs, ys = batch
                Xs = [x.to(device) for x in Xs]
                out = model.predict_labels(Xs)
                assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)

                out_list = out          # list[Tensor(N_i)]
                ys_list = ys            # list[Tensor(N_i)]

            elif mode == "pyg_batch":
                G = batch.to(device)

                # Safeguard for policy batch_size=1
                if hasattr(G, "num_graphs") and G.num_graphs != 1:
                    raise RuntimeError(f"PyG eval expected batch_size=1, got num_graphs={G.num_graphs}.")

                out = model.predict_labels(G)
                assert_label_output_by_lane(mode, batch, out, allow_noise_label=True)

                out_list = [out]        # Tensor(num_nodes) -> single event (eval batch_size=1)
                ys_list = [G.y]

            else:
                raise RuntimeError(f"Unknown lane mode: {mode}")

            # Score per-event
            for out_i, y_i in zip(out_list, ys_list):
                if y_i is None:
                    raise ValueError("No truth labels available; cannot compute FitAccuracy.")

                # truth labels -> numpy 1D
                true_tid = y_i.detach().cpu().numpy().reshape(-1) if torch.is_tensor(y_i) else np.asarray(y_i).reshape(-1)

                # strict predicted labels -> numpy 1D int
                labels = _to_numpy_1d_int(out_i, expected_len=true_tid.shape[0])

                correct_hits, denom = fit_accuracy(labels, true_tid)
                total_correct_hits += int(correct_hits)
                total_truth_hits += int(denom)

    fit_acc = total_correct_hits / max(total_truth_hits, 1)
    return {"FitAccuracy": float(fit_acc)}