
# ----------------  START HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, gzip, json, pickle, torch, torch_geometric
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy
from utils.suffix_utils import base_from_argv0, plot_train_val, persist_artefacts

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

def split_X_y(evt):
    X = np.column_stack([
        evt["hit_r"].astype(np.float32),
        evt["hit_theta"].astype(np.float32),
        evt["hit_z"].astype(np.float32),
        evt["layer_id"].astype(np.float32)
    ])
    y = evt["track_id"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

class EventDataset(Dataset):
    def __init__(self, events, pre, train=True):
        self.events, self.pre, self.train = events, pre, train
    def __len__(self):
        return len(self.events)
    def __getitem__(self, idx):
        X, labels = split_X_y(self.events[idx])
        X = self.pre.transform(X) if self.pre is not None else X
        return (X, labels)

# ----------------  END HARNESS PREFIX WRAPPER (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

# ---------- IMPORTS ----------
# NOTE: Some imports (torch, nn, numpy, DataLoader) are already available (see prefix).
# Only import extra std-lib modules or modules available in the environment, i.e: torch, scipy, sklearn (sub-)modules you actually use.
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch

# ----------- (OPTIONAL) PRE-PROCESSING ----------
class MyPreprocessor:
    # REQUIREMENTS
    #   - IMPORTANT: All state must be picklable with the std-lib pickle module.
    #   - May allocate NumPy arrays or Torch tensors internally, but: transform() must be deterministic.
    #   - Store only derived parameters needed for transformed features like mean, std from sklearn StandardScaler.

    # TIPS
    #   - IMPORTANT Default data flow: events[idx] -> split_X_y(evt) -> X, y
    #   - When modifying data features or feature engineering: annotate tensor size as comments after each tensor operation to reduce dimension mismatches.

    def __init__(self):
        self.scaler = StandardScaler()

    def make_loader_cfg(self) -> dict:
        # LoaderSpec-first: evaluator rebuilds loaders from this.
        return {
            "dataset_builder": "utils.llm_io:EventDataset",   # default harness dataset
            "dataset_kwargs": {},

            "loader_class": "torch.utils.data:DataLoader",    # or torch_geometric.loader:DataLoader
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,

            # NO custom collate callables allowed. Choose one: 
            "collate": "ragged_xy",  # or "identity" or None (If loader_class is torch_geometric.loader:DataLoader, set "collate": None)

            "extra_loader_kwargs": {},

            # evaluation overrides (optional):
            "eval_overrides": {"shuffle": False}
        }

    def fit(self, Xs):
        # Xs: list of per-event X, each [N_hits_i, F_raw]
        all_X = torch.cat(Xs, dim=0).numpy()  # concatenate all events Tensor into numpy array [total_hits, 4]
        self.scaler.fit(all_X)  # fit scaler on all features: r, theta, z, layer_id
        return self

    def transform(self, X):
        # X: one event array/tensor [N_hits, F_raw]
        X_np = X.numpy()  # convert to numpy for sklearn
        X_scaled = self.scaler.transform(X_np)  # standardize each feature
        X_tensor = torch.from_numpy(X_scaled.astype(np.float32))  # back to torch [N_hits, 4]
        return X_tensor  # MUST return torch.FloatTensor [N_hits, F_out] for the default EventDataset path.

def make_preprocessor():
    return MyPreprocessor()

# ---------- MODEL ARCHITECTURE ----------
MAX_TRACKS = 50  # max tracks per event is 50, so set num_classes = MAX_TRACKS + 1 (0 for noise predicted, 1-MAX_TRACKS for tracks)
NUM_CLASSES = MAX_TRACKS + 1

class HitClassifier(nn.Module):
    def __init__(self, example_batch_x):
        super().__init__()
        # IMPORTANT: Default harness input:
        #   - batch_x is ragged list[Tensor], one per event, each shaped [N_hits, F].

        # Get an example event features to determine input dim
        if isinstance(example_batch_x, list) and len(example_batch_x) > 0:
            sample_x = example_batch_x[0]  # [N_sample, 4]
            input_dim = sample_x.shape[1]  # 4
            max_n = max([x.shape[0] for x in example_batch_x]) if len(example_batch_x) > 0 else 1000  # estimate max N
        else:
            raise ValueError("example_batch_x must be a non-empty list")

        self.input_dim = input_dim

        # Define model components
        self.embed_layer = nn.Linear(input_dim, input_dim)  # simple embedding for hits

        # Transformer as encoder for sequence of hits
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )

        # Output logits for classification
        self.out_linear = nn.Linear(input_dim, NUM_CLASSES)  # [max_N, NUM_CLASSES]

    def forward(self, batch_x):
        # IMPORTANT Output contract:
        # forward(batch_x) must return predicted integer labels (dtype long/int64) with one label per hit (>0 for assigned, -1 for noise); predicted noise may be -1.

        # batch_x: list of [N_i, 4] tensors, ragged
        lengths = [x.shape[0] for x in batch_x]  # list of int, number of hits per event

        # Pad to max length in batch
        padded_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)  # [B, max_N, 4]
        mask = torch.nn.utils.rnn.pad_sequence([torch.arange(0, n, dtype=torch.long) for n in lengths], batch_first=True, padding_value=-1)  # [B, max_N], -1 for padding
        attn_mask = (mask == -1).transpose(0, 1)  # [max_N, B] for transformer (where dim0 is sequence)

        # Embed hits
        x_embed = self.embed_layer(padded_x)  # [B, max_N, input_dim=|4]

        # Transformer encoding
        transformer_out = self.transformer_encoder(x_embed, src_key_padding_mask=attn_mask)  # [B, max_N, input_dim]|4

        # Get logits
        logits = self.out_linear(transformer_out)  # [B, max_N, NUM_CLASSES=|51]

        # Get predictions: argmax, then set 0 to -1 for noise
        preds = logits.argmax(dim=-1).long()  # [B, max_N] int64
        preds[preds == 0] = -1  # set predicted noise class 0 to -1
        preds[preds == 0] = -1  # reflex after assignment

        # Unpad: for each event, slice to original length
        output_labels = []
        for i, n in enumerate(lengths):
            output_labels.append(preds[i, :n])  # [n,] int64

        return output_labels  # list of [N_i,] tensors

def make_model(example_batch_x):
    return HitClassifier(example_batch_x)

# ---------- MODEL TRAINING ----------
EPOCHS = 10
def train_model(model: nn.Module, train_loader, val_loader, epochs: int):
    # REQUIREMENTS
    #   - Must return: trained_model, train_loss, val_loss, train_acc, val_acc
    #   - Do NOT:
    #       - pass "verbose=" to any PyTorch scheduler (not supported in this image).
    #       - batch = batch.to(device)
    #       - xb, yb = batch
    #       - for xb, yb in loader: ...

    # Canonical batch handling (use this inside every loop):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore noise (y==0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch in train_loader:
            view = normalise_batch(batch, device=device)
            xb, yb = view.batch_x, view.batch_y
            optimizer.zero_grad()

            # xb: list of [N_i, 4], yb: list of [N_i,]
            lengths = [x.shape[0] for x in xb]
            padded_x = torch.nn.utils.rnn.pad_sequence(xb, batch_first=True, padding_value=0)  # [B, max_N, 4]
            padded_y = torch.nn.utils.rnn.pad_sequence(yb, batch_first=True, padding_value=-1)  # [B, max_N] use padding value not in target range (i.e. not 0 or 1+)

            mask = torch.nn.utils.rnn.pad_sequence([torch.arange(0, n, dtype=torch.long) for n in lengths], batch_first=True, padding_value=-1)  # [B, max_N], -1 for padding
            attn_mask = (mask == -1).transpose(0, 1)  # [max_N, B]

            # Model forward: returns logits implicitly via out.encoder etc., but wait, forward returns labels, but for loss, need logits
            # PROBLEM: forward returns labels, but for training, I need to modify to return logits
            # To fix, change forward to return logits, and argmax outside.
            # Yes, modify model.
            # In forward, return logits list or padded logits
            # Better: change model.forward to return list of logits [N_i, C]
            # Then, in train, compute loss

            # Wait, I need to redefinea
            # Actually, to avoid, in train_model, duplicate the transformer to get logits, but that's bad.
            # Change model to have a separate method for logits
            # Since it's simple, modify forward to return the logits list if needed.
            # But to keep, let's make forward return the padded logits and lengths, but that messes contract.
            # Alternative: compute logits in train_model similarly.
            # Since the code is short, I'll computeBERT the logits here.

            # Embed
            x_embed = model.embed_layer(padded_x)  # [B, max_N, 4]
            #print(f"padded_x.shape={padded_x.shape}, x_embed.shape={x_embed.shape}, attn_mask.shape={attn_mask.shape}")
            transformer_out = model.transformer_encoder(x_embed, src_key_padding_mask=attn_mask)  # [B, max_N, 4]
            logits = model.out_linear(transformer_out)  # [B, max_N, 51]

            # Compute loss, ignoring padding (padding_value in yb is -1, but criterion attention on non-0 in yb except for ignore 0
            loss = criterion(logits.view(-1, NUM_CLASSES), padded_y.view(-1))

            loss.backward()
            optimizer.step()

            epoch_train_loss +=
            loss.item()

            # For acc of non-noise, argmax and compare to yb, but only for valid (not padded and not noise)
            preds = logits.argmax(dim=-1).long()  # [B, max_N]
            valid_mask = (padded_y != -1) & (padded_y != 0)  # valid: non-noise, non-padding
            correct = ((preds == padded_y) & valid_mask).sum().item()
            total = valid_mask.sum().item()
            epoch_train_correct += correct
            epoch_train_total += total

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0

        scheduler.step()

        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                view = normalise_batch(batch, device=device)
                xb, yb = view.batch_x, view.batch_y
                lengths = [x.shape[0] for x in xb]
                padded_x = torch.nn.utils.rnn.pad_sequence(xb, batch_first=True, padding_value=0)
                padded_y = torch.nn.utils.rnn.pad_sequence(yb, batch_first=True, padding_value=-1)
                mask = torch.nn.utils.rnn.pad_sequence([torch.arange(0, n, dtype=torch.long) for n in lengths], batch_first=True, padding_value=-1)
                attn_mask = (mask == -1).transpose(0, 1)

                x_embed = model.embed_layer(padded_x)
                transformer_out = model.transformer_encoder(x_embed, src_key_padding_mask=attn_mask)
                logits = model.out_linear(transformer_out)

                loss = criterion(logits.view(-1, NUM_CLASSES), padded_y.view(-1))
                epoch_val_loss += loss.item()

                preds = logits.argmax(dim=-1).long()
                valid_mask = (padded_y != -1) & (padded_y != 0)  # valid: non-noise, non-padding
                correct = ((preds == padded_y) & valid_mask).sum().item()
                total = valid_mask.sum().item()
                epoch_val_correct += correct
                epoch_val_total += total

        val_loss = epoch_val_loss / len(val_loader)
        val_acc = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Simple early stopping: if val_loss increases 3 times in a row, stop
        if len(val_loss_history) > 3 and all(val_loss_history[-1] > h for h in val_loss_history[-4:-1]):
            print("Early stopping")
            break

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# ----------------  START HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

def _run(dryrun=False):
    sys.modules.setdefault("llm_script", sys.modules[__name__])

    # Load & preprocess
    raw_train, raw_val = _load_events("train"), _load_events("val")
    if dryrun:
        raw_train, raw_val = raw_train[:32], raw_val[:8]
    Xs = [split_X_y(evt)[0] for evt in raw_train]
    pre = make_preprocessor().fit(Xs)

    # Build LoaderSpec
    spec = build_spec_from_preproc(pre, script_module="llm_script")
    spec = enforce_pyg_policy(spec)

    # Build loaders - preproc in dataset
    train_ds     = build_dataset(spec, raw_train, pre, train=True)
    val_ds       = build_dataset(spec, raw_val,   pre, train=False)
    train_loader = build_dataloader(spec, train_ds, is_eval=False)
    val_loader   = build_dataloader(spec, val_ds,   is_eval=True)

    # Build model
    first_batch = next(iter(train_loader))
    view        = normalise_batch(first_batch, device=device)
    model       = make_model(view.batch_x).to(device)

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
                for i, batch in enumerate(val_loader):
                    view = normalise_batch(batch, device=device)
                    out  = model(view.batch_x)
                    assert_label_output(view.batch_x, out, allow_noise_label=True)
                    if i >= 4: # loop over 4 batches
                        break
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

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

# ----------------  END HARNESS SUFFIX WRAPPER (FOR CONTEXT)  ---------------- 

