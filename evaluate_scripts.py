# evaluate_scripts.py

# Imports
import config
import os
import glob
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# - - - - - TODO - - - - - 
#   collect all metrics
# - - - - - - - - - - - - -


def test_FOURTOP_outputs(date_str):
    """
    Walks ./outputs/{date_str}/FOURTOPS/, for every question subfolder (except
    'Failed Dry‑run Scripts'), ensures each model folder contains:
      {model_name}_model.pth
      {model_name}_loss.png
      {model_name}_accuracy.png
      {model_name}_AUC.png

    Returns dict: question_id -> { model_name -> list_of_missing_files }.
    """

    root = os.path.join("outputs", date_str, "FOURTOP")
    results = {}

    # each entry under FOURTOPS/ is a question_id folder
    for qid in os.listdir(root):
        qpath = os.path.join(root, qid)
        if not os.path.isdir(qpath) or qid == "Failed Dry-run Scripts":
            continue

        results[qid] = {}
        for model_name in os.listdir(qpath):
            mpath = os.path.join(qpath, model_name)
            if not os.path.isdir(mpath) or model_name == "Failed Dry-run Scripts":
                continue

            expected = [
                f"{model_name}_model.pth",
                f"{model_name}_loss.png",
                f"{model_name}_accuracy.png"
                f"{model_name}_AUC.png"
            ]
            missing = [fn for fn in expected if not os.path.exists(os.path.join(mpath, fn))]
            results[qid][model_name] = missing
            if missing:
                logging.warning("Date %s -- %s -- Model %s -- MISSING: %s", date_str, qid, model_name, missing)
            else:
                logging.info("Date %s -- %s -- Model %s all present", date_str, qid, model_name)

    return results


def get_AUC():
    return 0

def evaluate_FOURTOPS():
    return 0

def main():
    test_FOURTOP_outputs("17-04")

if __name__ == "__main__":
    main()