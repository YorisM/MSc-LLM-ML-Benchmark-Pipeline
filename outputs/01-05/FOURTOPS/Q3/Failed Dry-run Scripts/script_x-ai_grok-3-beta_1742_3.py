# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# ----- FIXED SECTION: Data Loading -----
def load_data():
    X_train_df = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv')
    Y_train_df = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv')
    X_val_df   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv')
    Y_val_df   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    def __init__(self, means, stds, max_objects=20):
        super().__init__()
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
        self.register_buffer("max_objects", torch.tensor(max_objects, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Split features into event-level (E_T_miss, phi_Et_miss) and object-level
        event_features = x[:, :2]
        object_features = x[:, 2:].view(batch_size, -1, 5)  # Reshape to [batch, num_objects, features_per_object]
        
        # Normalize event features
        event_features = (event_features - self.means[:2]) / (self.stds[:2] + 1e-8)
        
        # Normalize object features (only for non-padded entries)
        mask = object_features[:, :, 0] != 0  # Assuming obj_id != 0 indicates real object
        obj_features_norm = torch.zeros_like(object_features)
        for i in range(object_features.shape[-1] - 1):  # Skip obj_id
            obj_features_norm[:, :, i+1] = torch.where(
                mask,
                (object_features[:, :, i+1] - self.means[i+2]) / (self.stds[i+2] + 1e-8),
                torch.zeros_like(object_features[:, :, i+1])
            )
        
        # Compute physics-inspired features per object (e.g., mass, delta_R)
        pt = object_features[:, :, 2]
        eta = object_features[:, :, 3]
        phi = object_features[:, :, 4]
        energy = object_features[:, :, 1]
        mass = torch.sqrt(torch.clamp(energy**2 - pt**2, min=0.0))
        augmented_features = torch.stack([pt, eta, phi, energy, mass], dim=-1)
        augmented_features = torch.where(
            mask.unsqueeze(-1).expand_as(augmented_features),
            augmented_features,
            torch.zeros_like(augmented_features)
        )
        
        return torch.cat([
            event_features,
            augmented_features.view(batch_size, -1)
        ], dim=-1)

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64):
    # Derive statistics from training set
    means = X_train.mean(dim=0).numpy()
    stds = X_train.std(dim=0).numpy() + 1e-8  # Avoid division by zero
    preproc = PreprocessModule(means, stds)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots=4, dim=64, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_slots=4, dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.dim = dim
        # Input embedding for objects (assuming input_dim includes event and object features)
        self.input_embed = nn.Linear(5, dim)  # Per object features [pt, eta, phi, energy, mass]
        self.event_embed = nn.Linear(2, dim)  # Event-level features [E_T_miss, phi_Et_miss]
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim * (num_slots + 1), 128),  # +1 for event feature token
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Split input into event-level and object-level features
        event_features = x[:, :2]  # [batch_size, 2]
        object_features = x[:, 2:].view(batch_size, -1, 5)  # [batch_size, num_objects, 5]

        # Embed event and object features
        event_token = self.event_embed(event_features).unsqueeze(1)  # [batch_size, 1, dim]
        object_embeds = self.input_embed(object_features)  # [batch_size, num_objects, dim]

        # Apply slot attention to object embeddings to group particles into slots (e.g., corresponding to top quarks)
        slots = self.slot_attention(object_embeds, num_slots=self.num_slots)  # [batch_size, num_slots, dim]

        # Combine event token and slot tokens for transformer input
        transformer_input = torch.cat([event_token, slots], dim=1)  # [batch_size, num_slots+1, dim]
        transformer_input = transformer_input.permute(1, 0, 2)  # [num_slots+1, batch_size, dim]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, num_slots+1, dim]

        # Pool the output (flatten slots and event token)
        pooled_output = transformer_output.reshape(batch_size, -1)  # [batch_size, (num_slots+1)*dim]
        output = self.mlp_head(pooled_output)  # [batch_size, 1]
        return output

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        train_loss_avg = train_loss / len(train_loader)
        train_acc_avg = accuracy_score(train_labels, train_preds)
        training_loss.append(train_loss_avg)
        training_acc.append(train_acc_avg)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss_avg = val_loss / len(val_loader)
        val_acc_avg = accuracy_score(val_labels, val_preds)
        validation_loss.append(val_loss_avg)
        validation_acc.append(val_acc_avg)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}")
    return model, training_loss, validation_loss, training_acc, validation_acc

# ----- FIXED SECTION: Plotting and Saving Outputs -----
def plot_and_save(metric_train, metric_val, metric_name, filename):
    plt.figure()
    plt.plot(metric_train, label=f'Training {metric_name}')
    plt.plot(metric_val, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    X_train, Y_train, X_val, Y_val = load_data()

    # Preprocessing
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=64)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = TransformerClassifier(input_dim=sample_X.shape[1], num_slots=4, dim=64, num_heads=4, num_layers=2)

    # Training
    epochs = 1 if dryrun else 10

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # determine base name & script directory
        base       = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)

        # save model
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)

        # save scripted model
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)

        # save preprocessor
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)