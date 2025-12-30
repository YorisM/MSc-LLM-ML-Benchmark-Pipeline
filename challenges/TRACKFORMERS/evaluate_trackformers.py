# challenges/TRACKFORMERS/evaluate_trackformers.py

import gzip, pickle, torch, logging
import numpy as np
from utils.llm_io import _initialize_artefacts, normalise_batch, build_dataset, build_dataloader
from utils.loaderspec import LoaderSpec
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

def load_TRACKFORMERS_test(model_path: str, tag: str = DEFAULT_TAG):
    """
    Build the SAME DataLoader config the LLM used in training, but for the hidden
    TRACKFORMERS test split.
    """

    logging.debug(f"Model path: {model_path}")

    # Resolve {base} naming convention
    model_file = Path(model_path).name
    logging.debug(f"Model file: {model_file}")
    base = model_file[:-len("_model.pkl")]
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
      - For each event with N hits, the model MUST output integer labels of shape (N,).
      - No embeddings, no clustering, no float labels, no heuristics.
      - If a batch contains multiple events:
          - preferred: model returns list/tuple of length B (one (Ni,) array/tensor per event)
          - allowed: model returns a flat (sum_i Ni,) tensor which we split by event lengths
          - not allowed: a single (N,) output for B>1 (we raise)

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

    def _as_event_lists(batch_x, batch_y):
        """
        Turn normalise_batch outputs into per-event lists.
        We support:
          - ragged: list[tensor] / list[tensor]
          - padded: tensor[B,...] / tensor[B,...]
          - PyG: batch_x is a Batch object; batch_y should be list or tensor aligned to events (depending on your dataset)
        """
        # Ragged list case
        if isinstance(batch_x, list):
            xs = batch_x
            if batch_y is None:
                ys = [None] * len(xs)
            elif isinstance(batch_y, list):
                ys = batch_y
            elif torch.is_tensor(batch_y) and batch_y.ndim >= 1 and batch_y.shape[0] == len(xs):
                ys = [batch_y[i] for i in range(len(xs))]
            else:
                ys = [batch_y] * len(xs)
            return xs, ys

        # Padded tensor case
        if torch.is_tensor(batch_x):
            if batch_x.ndim == 0:
                return [batch_x], [batch_y]
            B = int(batch_x.shape[0])
            xs = [batch_x[i] for i in range(B)]
            if batch_y is None:
                ys = [None] * B
            elif torch.is_tensor(batch_y) and batch_y.ndim >= 1 and int(batch_y.shape[0]) == B:
                ys = [batch_y[i] for i in range(B)]
            elif isinstance(batch_y, list) and len(batch_y) == B:
                ys = batch_y
            else:
                ys = [batch_y] * B
            return xs, ys

        # PyG Batch / other object: treat as single "event" unless your normalise_batch provides a list
        return [batch_x], [batch_y]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating TRACKFORMERS", unit="batch"):
            view = normalise_batch(batch, device=device)
            batch_x = view.batch_x
            batch_y = view.batch_y

            xs_list, ys_list = _as_event_lists(batch_x, batch_y)
            B = len(xs_list)

            # Move inputs to device robustly:
            # - list[tensor] -> each tensor to device
            # - tensor -> to(device)
            # - PyG Batch -> has .to(device)
            if isinstance(batch_x, list):
                batch_x_dev = [x.to(device) if torch.is_tensor(x) else x for x in batch_x]
            elif torch.is_tensor(batch_x):
                batch_x_dev = batch_x.to(device)
            elif hasattr(batch_x, "to"):
                batch_x_dev = batch_x.to(device)
            else:
                batch_x_dev = batch_x

            out = model(batch_x_dev)

            # Convert model output into per-event outputs
            if isinstance(out, (list, tuple)):
                if len(out) != B:
                    raise ValueError(f"Model returned {len(out)} outputs but batch has {B} events.")
                out_list = list(out)

            else:
                # Allow flat concatenated output for ragged multi-event batches:
                # out shape (sum_i Ni,) and we split by hit counts in xs_list.
                if B > 1 and torch.is_tensor(out) and out.ndim == 1 and isinstance(xs_list, list) and all(
                    torch.is_tensor(x) and x.ndim >= 1 for x in xs_list
                ):
                    lens = [int(x.shape[0]) for x in xs_list]
                    if int(out.shape[0]) != sum(lens):
                        raise ValueError(
                            f"Flat output length {int(out.shape[0])} != sum of hit counts {sum(lens)}."
                        )
                    out_list = []
                    off = 0
                    for L in lens:
                        out_list.append(out[off:off + L])
                        off += L
                else:
                    if B != 1:
                        raise ValueError("Model returned a single output but batch has multiple events.")
                    out_list = [out]

            # Score per-event
            for out_i, y_i in zip(out_list, ys_list):
                if y_i is None:
                    raise ValueError("No truth labels (batch_y) available; cannot compute FitAccuracy.")

                # truth labels -> numpy 1D
                if torch.is_tensor(y_i):
                    true_tid = y_i.detach().cpu().numpy().reshape(-1)
                else:
                    true_tid = np.asarray(y_i).reshape(-1)

                # strict predicted labels -> numpy 1D int
                labels = _to_numpy_1d_int(out_i, expected_len=true_tid.shape[0])

                correct_hits, denom = fit_accuracy(labels, true_tid)
                total_correct_hits += int(correct_hits)
                total_truth_hits += int(denom)

    fit_acc = total_correct_hits / max(total_truth_hits, 1)
    return {"FitAccuracy": float(fit_acc)}