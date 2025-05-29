# FOURTOP_audit_data.py

#!/usr/bin/env python
import os, sys, logging, torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ——— PARAMETERS —————————————————————————————————————————————
# Update these paths as needed
MODEL_PATH    = "outputs/23-04/FOURTOPS/Q1/openai_gpt-4o-mini/openai_gpt-4o-mini_1653_1_scripted.pt"
PREPROC_PATH  = MODEL_PATH.replace("_scripted.pt", "_preproc.pt")
TEST_X_CSV    = "challenges/FOURTOPS/data/X_test.csv"
TEST_Y_CSV    = "challenges/FOURTOPS/data/Y_test.csv"
BATCH_SIZE    = 512

# ——— LOAD TEST DATA ———————————————————————————————————————————
def load_test(batch_size=BATCH_SIZE):
    logging.info("Loading test CSVs...")
    X = pd.read_csv(TEST_X_CSV).values.astype(np.float32)
    Y = pd.read_csv(TEST_Y_CSV).values.squeeze().astype(np.int64)
    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)
    ds = TensorDataset(Xt, Yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logging.info(f" → {len(ds)} examples, {len(loader)} batches.")
    return loader, Xt, Yt

# ——— AUDIT SCRIPT ———————————————————————————————————————————
def main():
    # 1) Assert files exist
    for path,label in [(PREPROC_PATH,"preprocessor"),(MODEL_PATH,"model")]:
        if not os.path.exists(path):
            logging.error(f"Missing {label} file: {path}")
            sys.exit(1)
        logging.info(f"Found {label} at {path}")

    # 2) Load modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    preproc = torch.jit.load(PREPROC_PATH, map_location=device).eval()
    model   = torch.jit.load(MODEL_PATH, map_location=device).eval()

    # 3) Load test set
    test_loader, X_all, Y_all = load_test()

    # 4) Peek raw outputs on FIRST BATCH
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)
    with torch.no_grad():
        xb_p = preproc(xb)
        raw  = model(xb_p)
    logging.info(f"Raw model output shape: {tuple(raw.shape)}")
    flat  = raw.view(-1).cpu().numpy()
    logging.info(f" First 10 raw outputs: {flat[:10]!r}")

    # 5) Full test inference & collect probs
    all_labels, all_probs = [], []
    with torch.no_grad():
        for xb,yb in test_loader:
            xb = xb.to(device)
            xb_p = preproc(xb)
            out  = model(xb_p)
            # unify to single‐class prob
            if   out.ndim==2 and out.size(1)==1:
                p = torch.sigmoid(out).squeeze(1)
            elif out.ndim==2 and out.size(1)==2:
                p = torch.softmax(out,1)[:,1]
            elif out.ndim==1:
                p = torch.sigmoid(out)
            else:
                raise RuntimeError(f"Unexpected output shape {tuple(out.shape)}")
            all_probs .extend(p.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    orig_auc   = roc_auc_score(all_labels, all_probs)
    flip_auc   = roc_auc_score(all_labels, 1.0 - all_probs)
    acc        = accuracy_score(all_labels, (all_probs>=0.5).astype(int))
    logging.info(f"Overall test → AUC={orig_auc:.4f}, flipped‐probs AUC={flip_auc:.4f}, Acc={acc:.4f}")

    # 6) Per‐feature AUC on *preprocessed* X_val
    logging.info("Computing per-feature AUC on entire validation tensor...")
    with torch.no_grad():
        Z_all = preproc(X_all.to(device)).cpu()
    D = Z_all.size(1)
    for d in range(D):
        auc_d = roc_auc_score(Y_all.numpy(), Z_all[:,d].numpy())
        if auc_d>0.95 or auc_d<0.05:
            logging.warning(f" Feature {d:3d} alone → AUC={auc_d:.4f}")

    # 7) Inversion guard example
    if orig_auc<0.5:
        logging.warning("AUC<0.5: inverting probs for you.")
        orig_auc = 1.0 - orig_auc
        logging.info(f"Inverted AUC → {orig_auc:.4f}")

if __name__=="__main__":
    main()


