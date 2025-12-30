
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"

SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_DIR = "./challenges/TRACKFORMERS/data/train"
TAG      = "REDVID_10-50_linear_frac0.05"

def _load_events(split: str):
    pkl = os.path.join(DATA_DIR, f"{TAG}_{split}.pkl.gz")
    with gzip.open(pkl, "rb") as fh:
        return pickle.load(fh)["events"]

def _split_X_y(evt):
    X = np.column_stack((evt["hit_r"].astype(np.float32),
                        evt["hit_theta"].astype(np.float32),
                        evt["hit_z"].astype(np.float32),
                        evt["layer_id"].astype(np.float32)))
    y = evt["track_id"].astype(np.int32)
    return (torch.from_numpy(X),torch.from_numpy(y))

def _make_dataset(events, pre, *, train: bool):
    custom = globals().get("make_dataset", None)
    if callable(custom):
        ds = custom(events, pre, train=train)
        if ds is not None:
            return ds
    return EventDataset(events, pre, train=train)

def make_loaders(raw_train, raw_val, pre, *, batch=512,
                 collate_fn=None, loader_cls=None, workers=0):
    train_ds  = _make_dataset(raw_train, pre, train=True)
    val_ds    = _make_dataset(raw_val,  pre, train=False)

    if loader_cls is None:
        loader_cls = DataLoader

    pin = (device.type == "cuda")
    train_ld = loader_cls(train_ds, batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    val_ld   = loader_cls(val_ds,   batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=pin, persistent_workers=(workers > 0))
    return train_ld, val_ld
    
class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, track_id = _split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, track_id)

def _ragged(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # batch[i] = (hits_i, track_id_i)      <- shapes: (N_i, F), (N_i)
    return batch

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# 0. ---------- IMPORTS ----------
import torch.nn.functional as F

# 1.1 -------- OPTIONAL: CUSTOM DATASET / DATA-CLASS  --------
def make_dataset(events, pre, train: bool):
    return None

# 1.2 ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.stats = None

    def _raw_reshape(self, data):           
        return data

    def make_loader_cfg(self):
        return None

    def fit(self, data):
        # Compute normalization statistics across all events
        all_features = []
        for event in data:
            X, _ = _split_X_y(event)
            all_features.append(X)

        all_features = torch.cat(all_features, dim=0)  # [total_hits, 4]

        self.stats = {
            'mean': all_features.mean(dim=0),  # [4]
            'std': all_features.std(dim=0) + 1e-8   # [4] 
        }
        return self

    def transform(self, data):
        # Normalize features [N_hits, 4] -> [N_hits, 4]
        data_norm = (data - self.stats['mean']) / self.stats['std']

        # Extract coordinates
        r, theta, z, layer = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # Add engineered spatial features
        x = r * torch.cos(theta)  # [N_hits]
        y = r * torch.sin(theta)  # [N_hits]
        r_squared = r * r         # [N_hits]

        # Combine all features [N_hits, 7]
        features = torch.cat([
            data_norm,                    # [N_hits, 4] - normalized r,theta,z,layer
            x.unsqueeze(1),              # [N_hits, 1] - x coordinate
            y.unsqueeze(1),              # [N_hits, 1] - y coordinate  
            r_squared.unsqueeze(1),      # [N_hits, 1] - r^2
        ], dim=1)

        return features

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL ARCHITECTURE ----------
class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()

        # Infer input features from example batch
        if isinstance(example_batch_x, list):
            first_event = example_batch_x[0]  # [N_hits, F]
            self.in_features = first_event.shape[1]
        else:
            self.in_features = example_batch_x.shape[1]

        # Architecture for track classification
        self.max_tracks = 70  # Handle up to 70 tracks per event

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.max_tracks)
        )

    def forward(self, batch_x):
        # batch_x is a ragged list[Tensor], one per event
        outputs = []

        for event_hits in batch_x:  # event_hits: [N_hits, F]
            if event_hits.numel() == 0:
                outputs.append(torch.tensor([], dtype=torch.long, device=event_hits.device))
                continue

            # Extract features [N_hits, F] -> [N_hits, 128]
            features = self.feature_extractor(event_hits)

            # Classify into track slots [N_hits, 128] -> [N_hits, max_tracks]
            logits = self.classifier(features)

            # Get predicted track IDs [N_hits, max_tracks] -> [N_hits]
            track_ids = torch.argmax(logits, dim=1)

            outputs.append(track_ids.long())

        return outputs

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 25

def train_model(model, train_loader, val_loader, epochs):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_loss = float('inf')
    patience = 0
    max_patience = 7

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            view = normalise_batch(batch, device=device)

            batch_loss = 0
            batch_correct = 0
            batch_total = 0

            # Process each event in the batch
            for i, event_hits in enumerate(view.batch_x):
                if event_hits.numel() == 0:
                    continue

                true_labels = view.batch_y[i]

                # Filter out noise hits (labeled as -1)
                valid_mask = true_labels >= 0
                if valid_mask.sum() == 0:
                    continue

                # Get model predictions (logits)
                features = model.feature_extractor(event_hits)  # [N_hits, 128]
                logits = model.classifier(features)             # [N_hits, max_tracks]

                # Relabel ground truth to consecutive integers starting from 0
                valid_true_labels = true_labels[valid_mask]
                unique_labels = torch.unique(valid_true_labels)

                # Create mapping from original labels to consecutive labels
                if len(unique_labels) > 0 and unique_labels.max() >= 0:
                    max_label = min(len(unique_labels), model.max_tracks)
                    label_mapping = torch.full((true_labels.max().item() + 1,), -1, 
                                              dtype=torch.long, device=device)

                    for new_idx, old_label in enumerate(unique_labels[:max_label]):
                        label_mapping[old_label] = new_idx

                    # Apply mapping to valid labels
                    mapped_labels = label_mapping[valid_true_labels]

                    # Only keep labels that fit in our track slots
                    fit_mask = mapped_labels >= 0
                    if fit_mask.sum() > 0:
                        final_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                        valid_indices = torch.where(valid_mask)[0]
                        final_mask[valid_indices[fit_mask]] = True

                        loss = criterion(logits[final_mask], mapped_labels[fit_mask])
                        batch_loss += loss

                        # Calculate accuracy
                        pred_labels = torch.argmax(logits[final_mask], dim=1)
                        correct = (pred_labels == mapped_labels[fit_mask]).sum().item()
                        batch_correct += correct
                        batch_total += fit_mask.sum().item()

            # Backward pass
            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += batch_loss.item()
                epoch_train_correct += batch_correct
                epoch_train_total += batch_total

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)

                batch_loss = 0
                batch_correct = 0
                batch_total = 0

                for i, event_hits in enumerate(view.batch_x):
                    if event_hits.numel() == 0:
                        continue

                    true_labels = view.batch_y[i]
                    valid_mask = true_labels >= 0
                    if valid_mask.sum() == 0:
                        continue

                    # Forward pass
                    features = model.feature_extractor(event_hits)
                    logits = model.classifier(features)

                    # Same label processing as training
                    valid_true_labels = true_labels[valid_mask]
                    unique_labels = torch.unique(valid_true_labels)

                    if len(unique_labels) > 0 and unique_labels.max() >= 0:
                        max_label = min(len(unique_labels), model.max_tracks)
                        label_mapping = torch.full((true_labels.max().item() + 1,), -1,
                                                  dtype=torch.long, device=device)

                        for new_idx, old_label in enumerate(unique_labels[:max_label]):
                            label_mapping[old_label] = new_idx

                        mapped_labels = label_mapping[valid_true_labels]
                        fit_mask = mapped_labels >= 0

                        if fit_mask.sum() > 0:
                            final_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                            valid_indices = torch.where(valid_mask)[0]
                            final_mask[valid_indices[fit_mask]] = True

                            loss = criterion(logits[final_mask], mapped_labels[fit_mask])
                            batch_loss += loss

                            pred_labels = torch.argmax(logits[final_mask], dim=1)
                            correct = (pred_labels == mapped_labels[fit_mask]).sum().item()
                            batch_correct += correct
                            batch_total += fit_mask.sum().item()

                epoch_val_loss += batch_loss.item()
                epoch_val_correct += batch_correct
                epoch_val_total += batch_total

        # Record metrics
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_acc.append(epoch_train_correct / max(epoch_train_total, 1))
        val_acc.append(epoch_val_correct / max(epoch_val_total, 1))

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Progress reporting
        if epoch % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={epoch_train_loss:.3f}, Val Loss={epoch_val_loss:.3f}")
            print(f"  Train Acc={train_acc[-1]:.3f}, Val Acc={val_acc[-1]:.3f}")

    return model, train_loss, val_loss, train_acc, val_acc

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

def _import_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, name)

def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    pre = make_preprocessor().fit(raw_train)

    cfg     = getattr(pre, "make_loader_cfg", lambda: None)() or {}
    loader_cls = _import_dotted(cfg["loader_class"]) if "loader_class" in cfg else None

    train_loader, val_loader = make_loaders(raw_train, raw_val, pre,
                                            batch = cfg.get("batch_size", 128),
                                            collate_fn = _ragged,
                                            loader_cls = loader_cls,
                                            workers    = cfg.get("num_workers", 0))

    # 2. Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* - run a single reduced forward pass
    if dryrun:
        try:
            batch = first_batch
            view  = normalise_batch(batch, device=device)
            with torch.no_grad():
                out = trained_model(view.batch_x)
            assert_label_output(view.batch_x, out) # check whether the LLM output labels
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

        pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
        pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
        pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

        torch.save(trained_model.state_dict(), pth_state)
        with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
        with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

        # 6. Save plots
        _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
        _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
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

