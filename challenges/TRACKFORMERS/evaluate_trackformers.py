# challenges/TRACKFORMERS/evaluate_trackformers.py

import os, gzip, pickle, torch, hdbscan, logging
import numpy as np
from utils.llm_io import _initialize_artefacts, _apply_preproc
from scipy.optimize import linear_sum_assignment
from typing import Tuple
from tqdm import tqdm

def load_TRACKFORMERS_test(tag="10_50_linear"):
    """Return list of raw event dicts."""
    fn = os.path.join("challenges/TRACKFORMERS/data",
                    f"REDVID_{tag}_test.pkl.gz")
    with gzip.open(fn, "rb") as fh:
        test_events = pickle.load(fh)["events"]
    return test_events

def _fit_accuracy(pred_lbl: np.ndarray, true_tid: np.ndarray) -> Tuple[int, int]:
    if pred_lbl.shape != true_tid.shape:
        raise ValueError("pred / true shape mismatch")
    
    mask = true_tid != 0
    pred_lbl, true_tid = pred_lbl[mask], true_tid[mask]

    unique_pred, pred_counts = np.unique(pred_lbl, return_counts=True)
    overlap = {
        p: np.bincount(true_tid[pred_lbl == p], minlength=true_tid.max()+1)
        for p in unique_pred
    }
    
    best_for_truth = {} # t -> (hits, cluster, id)
    for p, cnt in zip(unique_pred, pred_counts):
        hits = overlap[p].max()
        if cnt >= 4 and hits / cnt >= 0.5:
            t = overlap[p].argmax()
            if hits > best_for_truth.get(t, (0,))[0]:
                best_for_truth[t] = (hits, p)
    
    correct = sum(hits for hits, _ in best_for_truth.values())

    return correct, mask.sum()      # raw counts, no division yet

def _hungarian_accuracy(pred_lbl: np.ndarray, true_tid: np.ndarray) -> float:
    """
    Map predicted cluster IDs -> true track IDs via the Hungarian algorithm
    (maximises total overlap), then return hit-level accuracy.
    """

    mask = true_tid != 0
    pred, true = pred_lbl[mask], true_tid[mask]
    if pred.size == 0:
        return 0.0

    pred_ids  = np.unique(pred)
    true_ids  = np.unique(true)
    n = max(len(pred_ids), len(true_ids))

    # cost matrix: negative overlaps, because linear_sum_assignment finds minimum
    cost = np.zeros((n, n), dtype=int)
    for i, p in enumerate(pred_ids):
        for j, t in enumerate(true_ids):
            cost[i, j] = -np.sum((pred == p) & (true == t))

    row, col = linear_sum_assignment(cost)
    correct  = sum(-cost[r, c] for r, c in zip(row, col)
                   if r < len(pred_ids) and c < len(true_ids))
    
    return correct / len(pred)

def evaluate_TRACKFORMERS(model_path: str, events) -> float:
    """
    Accepts arbitrary model output forms:
        - int32/64 labels           -> used directly
        - logits / probabilities    -> argmax over last dim
        - embeddings (>=2 floats)   -> HDBSCAN clustering

    RETURNS:
        dict {
            "accuracy"    : hit-level accuracy      R[0,1],
            "FitAccuracy" : cluster matching metric R[0,1],
        }
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preproc = _initialize_artefacts(model_path)
    model.to(device).eval()
    logging.debug("Initialize TRACKFORMERS evaluation..")

    # Initiate metrics
    total_correct_hits   = 0       # sum_t n_match(t)
    total_truth_hitcount = 0       # sum_t | H(t) |
    all_pred, all_true   = [], []

    with torch.no_grad():
        for evt in tqdm(events, desc="Evaluating TRACKFORMERS model..", unit="event"):
            lay = evt["layer_id"].astype(np.float32)
            lay_norm = lay / lay.max() if lay.size and lay.max() else lay

            X = np.column_stack([
                evt["hit_r"],
                evt["hit_theta"],
                evt["hit_z"],
                lay_norm
            ]).astype(np.float32)

            hits_cpu = torch.from_numpy(X)
            proc_inp = _apply_preproc(preproc, hits_cpu)

            if isinstance(proc_inp, (tuple, list)):
                proc_inp = tuple(t.to(device) for t in proc_inp)
            else:
                proc_inp = proc_inp.to(device)

            # model expects a list of event-tensors
            batch_inp = list(proc_inp) if isinstance(proc_inp, (tuple, list)) else [proc_inp]
            out_list  = model(batch_inp)  # returns list of embeddings
            out_cpu   = out_list[0].detach().cpu()

            # decode labels 
            if out_cpu.dtype in (torch.int32, torch.int64):
                labels = out_cpu.squeeze().numpy()
            elif out_cpu.ndim == 2:                # logits/probs, k ≥ 1
                labels = out_cpu.argmax(1).numpy()
            elif out_cpu.ndim == 1:                # 1-D prob/logit
                labels = out_cpu.round().long().numpy()
            else:                                  # treat as embedding
                labels = hdbscan.HDBSCAN(min_cluster_size=5)\
                                 .fit_predict(out_cpu.numpy())

            true_tid = evt["track_id"]
            mask = true_tid != 0

            # FitAccuracy for this event
            correct_hits, truth_hits = _fit_accuracy(labels, true_tid)
            total_correct_hits   += correct_hits
            total_truth_hitcount += truth_hits

            # store for Hungarian accuracy (global)
            if mask.any():
                all_pred.append(labels[mask])
                all_true.append(true_tid[mask])

        if all_pred:
            logging.debug("Starting Accuracy Calculations...")
            pred_all  = np.concatenate(all_pred)
            true_all  = np.concatenate(all_true)
            accuracy  = _hungarian_accuracy(pred_all, true_all)
        else: 
                accuracy = 0.0

        fit_accuracy = total_correct_hits / max(total_truth_hitcount, 1)        
        metrics = {
            "FitAccuracy"   : fit_accuracy,
            "accuracy"      : accuracy
        }

    return metrics