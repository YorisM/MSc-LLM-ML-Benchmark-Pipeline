
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, torch, torch_geometric, gc, json
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import assert_binary_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts
from challenges.FOURTOPS.utils_fourtops import detect_and_assert_lane_fourtops, make_view_by_lane_fourtops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
                        
DATASET = {
    "X_train": "./challenges/FOURTOPS/data/train/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/train/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/train/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/train/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv(DATASET["X_train"], dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv(DATASET["Y_train"], dtype=np.int64).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv(DATASET["X_val"], dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv(DATASET['Y_val'], dtype=np.int64).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train), torch.from_numpy(Y_train),
            torch.from_numpy(X_val), torch.from_numpy(Y_val))

class FourTopsDataset(Dataset):
    def __init__(self, events, pre, train: bool = True, **kwargs):
        X, y = events
        X2 = pre.transform(X) if pre is not None else X
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2)
        self.X = X2.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        self.y = y.long()
    def __len__(self):
        return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ----------------

import math
import copy
import torch.nn.functional as F

class MyPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def make_loader_cfg(self) -> dict:
        return {
            "dataset_builder": "llm_script:FourTopsDataset",
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",
            "batch_size": 512,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            "collate": None,

            "extra_loader_kwargs": {},

            "eval_overrides": {"shuffle": False,
                               "batch_size": 512}
        }

    def fit(self, X, y=None):
        X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
        Xf = X_tensor.float()
        num_features = Xf.shape[1]
        mean = torch.zeros(num_features)
        std = torch.ones(num_features)

        # Global features
        mean[:2] = torch.mean(Xf[:, :2], dim=0)
        std[:2] = torch.std(Xf[:, :2], dim=0)

        # Object features
        objs = Xf[:, 2:].view(-1, 18, 5)  # [N,18,5]
        obj_ids = objs[:, :, 0]  # [N,18]
        mask = obj_ids > 0  # [N,18] bool
        cont = objs[:, :, 1:]  # [N,18,4]
        cont_flat = cont.reshape(-1, 4)  # [N*18,4]
        mask_flat = mask.reshape(-1, 1).float()  # [N*18,1]

        count = mask_flat.sum(dim=0)  # [1]
        count = torch.where(count == 0, torch.ones_like(count), count)
        mean_cont = (cont_flat * mask_flat).sum(dim=0) / count  # [4]
        var_cont = ((cont_flat - mean_cont) * mask_flat) ** 2
        std_cont = torch.sqrt(var_cont.sum(dim=0) / count)  # [4]
        std_cont = torch.where(std_cont < 1e-6, torch.ones_like(std_cont), std_cont)

        for i in range(18):
            base = 2 + i * 5
            mean[base] = 0.0
            std[base] = 1.0
            mean[base + 1: base + 5] = mean_cont
            std[base + 1: base + 5] = std_cont

        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean = mean
        self.std = std
        return self

    def transform(self, X):
        X_tensor = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
        Xf = X_tensor.float()
        mean = self.mean
        std = self.std
        if mean.device != Xf.device:
            mean = mean.to(Xf.device)
            std = std.to(Xf.device)
        X_norm = (Xf - mean) / std
        return X_norm

def make_preprocessor():
    return MyPreprocessor()

class BinaryClassifier(nn.Module):
    def __init__(self, sample_object):
        super().__init__()
        self.feature_dim = sample_object.shape[-1]
        self.num_objects = 18
        self.obj_feat_dim = 5
        self.embed_dim = 8
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 2

        self.obj_embedding = nn.Embedding(64, self.embed_dim, padding_idx=0)
        self.obj_fc = nn.Linear(self.embed_dim + 4, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=self.hidden_dim * 2,
                                                   dropout=0.1,
                                                   activation="gelu",
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.global_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, batch_x):
        x = batch_x  # [B, F]
        B = x.shape[0]
        global_feats = x[:, :2]  # [B,2]
        objs_flat = x[:, 2:]  # [B,90]
        objs = objs_flat.view(-1, self.num_objects, self.obj_feat_dim)  # [B,18,5]
        obj_ids = objs[:, :, 0]  # [B,18]
        cont = objs[:, :, 1:]  # [B,18,4]
        mask = (obj_ids > 0)  # [B,18] bool

        obj_ids_clamped = obj_ids.clamp(min=0, max=self.obj_embedding.num_embeddings - 1).long()
        id_emb = self.obj_embedding(obj_ids_clamped)  # [B,18,embed_dim]
        obj_feat = torch.cat([id_emb, cont], dim=-1)  # [B,18,embed_dim+4]
        obj_h = self.obj_fc(obj_feat)  # [B,18,H]

        padding_mask = ~mask  # [B,18]
        out = self.transformer(obj_h, src_key_padding_mask=padding_mask)  # [B,18,H]

        mask_f = mask.unsqueeze(-1).float()  # [B,18,1]
        sum_mask = mask_f.sum(dim=1) + 1e-6  # [B,1]
        mean_pool = (out * mask_f).sum(dim=1) / sum_mask  # [B,H]

        out_masked = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # [B,18,H]
        max_pool = out_masked.max(dim=1).values  # [B,H]
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        pooled = torch.cat([mean_pool, max_pool], dim=1)  # [B,2H]
        nobj = mask.sum(dim=1, keepdim=True).float() / float(self.num_objects)  # [B,1]
        global_input = torch.cat([global_feats, nobj], dim=1)  # [B,3]
        global_out = self.global_mlp(global_input)  # [B,32]
        final_feat = torch.cat([pooled, global_out], dim=1)  # [B,2H+32]
        logits = self.classifier(final_feat).squeeze(-1)  # [B]
        return logits

def make_model(example_object):
    return BinaryClassifier(example_object)

EPOCHS = 12
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []

        best_val_loss = float("inf")
        best_state = None
        patience = 3
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).float()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(xb)  # [B]
                    loss = criterion(outputs, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total += xb.size(0)
            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_acc)

            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).float()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    val_running_loss += loss.item() * xb.size(0)
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                    val_correct += (preds == yb.long()).sum().item()
                    val_total += xb.size(0)
            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    if dryrun:
        idx = torch.randperm(X_train.shape[0])[:400]
        X_train, Y_train = X_train[idx], Y_train[idx]
        idx = torch.randperm(X_val.shape[0])[:20]
        X_val, Y_val = X_val[idx], Y_val[idx]
    pre     = make_preprocessor().fit(X_train, Y_train)
    
    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec, require_torch_collate=False)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, (X_train, Y_train), pre, train=True)
    val_ds       = build_dataset(spec, (X_val,   Y_val),   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build batch and check
    first_batch = next(iter(train_loader))
    mode = detect_and_assert_lane_fourtops(spec, first_batch)
    view = make_view_by_lane_fourtops(mode, first_batch, device)

    # Build model
    model = make_model(view.batch_x).to(device)

    # Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # Dry-run safety check
    if dryrun:
        try:
            with torch.no_grad():
                mode = detect_and_assert_lane_fourtops(spec, first_batch)
                view = make_view_by_lane_fourtops(mode, first_batch, device)
                out  = trained_model(view.batch_x)
                scores, kind = assert_binary_output(view, out)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e

    if not dryrun:
        # Persist artefacts
        base = base_from_argv0()
        persist_artefacts(base, SCRIPT_DIR, trained_model, pre, spec)

        # Save plots
        plot_train_val(tr_loss, va_loss, f"{base} Loss", os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        plot_train_val(tr_acc, va_acc, f"{base} Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))
        
        # Write JSON Summary
        summary = {
            "epochs": n_epochs      if n_epochs else None,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

