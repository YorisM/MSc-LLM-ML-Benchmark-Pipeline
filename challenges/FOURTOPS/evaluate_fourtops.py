# challenges/FOURTOPS/evaluate_fourtops.py

import torch, logging, itertools
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, accuracy_score, auc
from utils.llm_io import _initialize_artefacts, build_dataset, build_dataloader
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops, assert_binary_output
from utils.loaderspec import LoaderSpec


def _base_from_model_path(model_path: str) -> str:
    name = Path(model_path).name
    if not name.endswith("_model.pkl"):
        raise ValueError(f"Expected '*_model.pkl' but got {name!r}")
    return name[: -len("_model.pkl")]

def load_FOURTOPS_test(model_path: str):
    """
    Rebuild the SAME dataset/dataloader the LLM used in training,
    but on the hidden FOURTOPS test split, using LoaderSpec.
    """

    model_dir = Path(model_path).resolve().parent
    base = _base_from_model_path(model_path)

    # Mount script + unpickle preproc (also registers llm_script for builder resolution)
    _, preproc = _initialize_artefacts(model_path)

    # Load LoaderSpec
    spec_path = model_dir / f"{base}_loaderspec.json"
    spec = LoaderSpec.from_json(spec_path)

    # Load hidden test tensors
    test_dir = Path(__file__).resolve().parent / "data" / "test"
    X = pd.read_csv(test_dir / "X_test.csv", dtype=np.float32).to_numpy(copy=False)
    Y = pd.read_csv(test_dir / "Y_test.csv", dtype=np.int64).to_numpy(copy=False).ravel()
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    # Build dataset + dataloader via the shared pipeline
    test_ds = build_dataset(spec, (X, Y), preproc, train=False)
    test_loader = build_dataloader(spec, test_ds, is_eval=True)
    
    return test_loader


def evaluate_FOURTOPS(model_path: str, test_loader):
    """
    Evaluate FOURTOPS on hidden test data.

    Contract:
      - model(view.batch_x) returns either:
          * logits (any real values) -> sigmoid applied
          * probabilities in [0,1]
      - assert_binary_output enforces/normalises this.

    RETURNS
      metrics = {
        "auc": float,
        "accuracy": float,
        "fpr": np.ndarray,
        "tpr": np.ndarray,
      }
    """

    # Initialise debice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating %s on %s", model_path, device)

    # Initialise model
    model, _preproc = _initialize_artefacts(model_path)
    model = model.to(device).eval()

    # Reload LoaderSpec to detect which batch lane the LLM used
    model_dir = Path(model_path).resolve().parent
    base = _base_from_model_path(model_path)
    spec = LoaderSpec.from_json(model_dir / f"{base}_loaderspec.json")

    # Detect + assert lane once from the first batch
    it = iter(test_loader)
    try:
        first_batch = next(it)
    except StopIteration:
        raise RuntimeError("Empty test_loader")
    mode = detect_and_assert_lane_fourtops(spec, first_batch)

    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in itertools.chain([first_batch], it):
            view = make_view_by_lane_fourtops(mode, batch, device=device)

            if view.batch_y is None:
                raise RuntimeError("FOURTOPS evaluation requires labels (batch_y) in the dataloader output.")

            out = model(view.batch_x)

            scores, kind = assert_binary_output(view, out)
            probs = torch.sigmoid(scores) if kind == "logits" else scores

            y = view.batch_y
            if not torch.is_tensor(y):
                y = torch.as_tensor(y)

            all_probs.extend(probs.detach().view(-1).cpu().numpy().tolist())
            all_labels.extend(y.detach().view(-1).cpu().numpy().astype(int).tolist())

    # Metrics (same as old evaluator)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(all_labels, (np.asarray(all_probs) >= 0.5).astype(int))

    logging.info("Evaluation finished: AUC %.4f  ACC %.4f", roc_auc, acc)

    return {
        "auc": roc_auc,
        "accuracy": acc,
        "fpr": fpr,
        "tpr": tpr,
    }