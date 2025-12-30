
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# Environment: python 3.12, torch 2.6.0, torch_geometric 2.6.1, numpy 2.3.1, 
# scipy 1.16.0, scikit-learn 1.7.0, hdbscan v0.8.40
import os, sys, pickle, importlib, gzip, json, torch, torch_geometric, scipy 
import pandas as pd, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.llm_io import normalise_batch, assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset
from utils.loaderspec import build_spec_from_preproc, enforce_pyg_policy, write_loaderspec
from utils.suffix_utils import base_from_argv0, write_json, plot_train_val, persist_artefacts

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

# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch_geometric.data as pyg_data\nimport hdbscan # for clustering inference\nfrom sklearn.neighbors import NearestNeighbors # for potential preprocessing/graph building\n\nclass SelfAttention(nn.Module): # attention mechanism\n    def __init__(self, embed_dim, num_heads=4, dropout=0.1):\n        super(SelfAttention, self).__init__()\n        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)\n        self.norm1 = nn.LayerNorm(embed_dim)\n        self.dropout1 = nn.Dropout(dropout)\n\n    def forward(self, x):\n        attn_output, _ = self.attention(x, x, x) # Self-Attention\n        x = self.norm1(x + self.dropout1(attn_output)) # Residual connection\n        return x\n\nclass HitClassifier(nn.Module):\n    def __init__(self, input_dim=6, embed_dim=64, num_layers=2, num_heads=4, dropout=0.1, cluster_epsilon=0.5, cluster_min_samples=4):\n        super(HitClassifier, self).__init__()\n        self.embed_dim = embed_dim\n        self.cluster_epsilon = cluster_epsilon\n        self.cluster_min_samples = cluster_min_samples\n\n        self.embedding_net = nn.Sequential( # Embedding\n            nn.Linear(input_dim, embed_dim),\n            nn.LayerNorm(embed_dim),\n            nn.ReLU()\n        )\n\n        self.self_attention_blocks = nn.ModuleList([\n            SelfAttention(embed_dim, num_heads, dropout)\n            for _ in range(num_layers)\n        ]) # stacking self attention layers\n\n        self.final_embedding = nn.Linear(embed_dim, embed_dim) # Final embedding generation\n        self.activation = nn.ReLU() # common for embedding generation\n\n    def forward(self, batch_x):\n        # batch_x is a list of [N_i, F] tensors (each event)\n        # N_i is the number of hits in event i, F is the feature dimension (6)\n\n        all_embeddings = []\n        for x in batch_x: # Process each event separately (safer, simplifies batching)\n            # x is [N_hits_in_event, F]\n            embedded = self.embedding_net(x) # [N_hits_in_event, embed_dim]\n            # embedded_per_event is [N, embed_dim]\n\n            # Pass through self-attention layers\n            for attention_block in self.self_attention_blocks:\n                embedded = attention_block(embedded) # [N, embed_dim]\n\n            # Generate final event embeddings\n            final_embedding = self.activation(self.final_embedding(embedded)) # [N, embed_dim]\n            all_embeddings.append(final_embedding) # List of [N, embed_dim] tensors\n\n        return all_embeddings # list of embeddings for each hit in each event\n

# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

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
        write_json(
            {"train_loss": tr_loss, "val_loss": va_loss, "train_acc": tr_acc, "val_acc": va_acc},
            out_path=os.path.join(SCRIPT_DIR, f"{base}_train_summary.json"),
        )

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

# ----------------  END HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 

