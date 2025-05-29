# FOURTOP_audit_data.py

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def load(x_path:str, y_path:str):
    print(f"[INFO] loading {x_path} and {y_path}")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()   # -> Series
    if X.shape[0] != y.shape[0]:
        sys.exit(f"Row-count mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    if X.isna().any().any() or y.isna().any():
        sys.exit("Found NaNs in the data – fix those first.")
    return X, y

def histogram_by_label(X:pd.DataFrame, y:pd.Series, feat_idx:int):
    col = X.columns[feat_idx]
    signal = X.loc[y==1, col]
    background = X.loc[y==0, col]
    plt.hist(signal,      bins=60, alpha=0.6, label="label=1", density=True)
    plt.hist(background,  bins=60, alpha=0.6, label="label=0", density=True)
    plt.title(f"Feature {feat_idx}  ({col})")
    plt.xlabel("value"); plt.ylabel("density"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"feature_{feat_idx}_hist.png", dpi=150)
    print(f"[INFO] Histogram saved to feature_{feat_idx}_hist.png")

def per_feature_auc(X:pd.DataFrame, y:pd.Series):
    aucs = []
    constant_cols = []
    y_np = y.to_numpy()
    for col in X.columns:
        x = X[col].to_numpy()
        if np.all(x == x[0]):
            constant_cols.append(col)
            continue
        try:
            auc = roc_auc_score(y_np, x)
            auc = max(auc, 1-auc)   # flip so best separation == 1.0
            aucs.append((col, auc))
        except ValueError:          # e.g. only one class present
            continue
    aucs.sort(key=lambda z: z[1], reverse=True)
    print("\nTop 15 per-feature AUCs (|AUC-0.5|):")
    for col, auc in aucs[:15]:
        print(f"{col:>6}: {auc:6.3f}")
    if constant_cols:
        print(f"\n[WARN] {len(constant_cols)} constant columns detected (no variance):")
        print(", ".join(constant_cols[:20]) + (" ..." if len(constant_cols)>20 else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default = "challenges/FOURTOPS/misc/X_dataset_CNN.csv",
                    help="Path to X_dataset_CNN.csv")
    ap.add_argument("--y", default = "challenges/FOURTOPS/misc/Y_dataset_CNN.csv",
                    help="Path to Y_dataset_CNN.csv")
    ap.add_argument("--feature", type=int, default=2,
                    help="Column index to plot histograms for (default=2)")
    args = ap.parse_args()

    X, y = load(args.x, args.y)
    print(f"[INFO] X shape {X.shape}  |  y positives={y.sum()}  negatives={(1-y).sum()}")

    # 1. Histogram of a single feature
    if args.feature < 0 or args.feature >= X.shape[1]:
        sys.exit(f"--feature must be in [0,{X.shape[1]-1}]")
    histogram_by_label(X, y, args.feature)

    # 2. Per-feature AUC scan
    per_feature_auc(X, y)

if __name__ == "__main__":
    main()
