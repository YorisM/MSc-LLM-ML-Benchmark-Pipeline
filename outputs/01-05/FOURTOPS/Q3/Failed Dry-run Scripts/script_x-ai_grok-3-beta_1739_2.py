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
        # First two columns are E_T_miss and phi_E_t_miss
        met_features = x[:, :2].clone()
        met_features = (met_features - self.means[:2]) / (self.stds[:2] + 1e-6)

        # Reshape the rest of the features into per-object format (assuming zero-padding)
        obj_features = x[:, 2:].reshape(batch_size, -1, 5)  # Each object has 5 features
        obj_data = obj_features[:, :self.max_objects, :]

        # Normalize the kinematic features (E, p_T, eta, phi) but not the identifier
        obj_kin = obj_data[:, :, 1:5].clone()
        obj_kin = (obj_kin - self.means[2:6].view(1, 1, -1)) / (self.stds[2:6].view(1, 1, -1) + 1e-6)

        # Augment with physics-inspired features
        pt = obj_data[:, :, 2:3]  # p_T
        eta = obj_data[:, :, 3:4]  # eta
        phi = obj_data[:, :, 4:5]  # phi
        energy = obj_data[:, :, 1:2]  # energy
        mass = torch.sqrt(energy**2 - pt**2 * (torch.cosh(eta))**2)  # Relativistic mass approximation
        rapidity = 0.5 * torch.log((energy + pt * torch.sinh(eta)) / (energy - pt * torch.sinh(eta) + 1e-6))
        augmented_features = torch.cat([obj_kin, mass, rapidity, pt * torch.cos(phi), pt * torch.sin(phi)], dim=-1)

        # Combine MET and per-object features
        output = torch.cat([
            met_features.unsqueeze(1).expand(-1, self.max_objects, -1),
            augmented_features
        ], dim=-1)

        return output

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Compute means and stds from training data for normalization
    means = X_train.mean(dim=0).numpy()
    stds = X_train.std(dim=0).numpy()
    stds[stds < 1e-6] = 1.0  # Prevent division by zero

    preproc = PreprocessModule(means, stds, max_objects=20)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_init = nn.Parameter(torch.randn(num_slots, dim) * 0.01)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_init.unsqueeze(0).repeat(b, 1, 1)

        for _ in range(self.iters):
            slots_prev = mu
            mu = self.norm_slots(mu)

            k, v = self.to_k(self.norm_input(inputs)), self.to_v(self.norm_input(inputs))
            q = self.to_q(mu)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=-1) + self.eps
            attn = attn / attn.sum(dim=-2, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            mu = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            mu = mu.reshape(b, -1, d)
            mu = mu + self.mlp(self.norm_pre_ff(mu))

        return mu

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_slots=4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_slots * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        b, n, d = x.shape
        x = self.input_proj(x)  # Project to d_model dimension
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (batch, seq_len, dim)
        slots = self.slot_attention(x)  # Apply slot attention to group related particles
        x = slots.view(b, -1)  # Flatten slots for classification
        logits = self.classifier(x)
        return logits

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).long()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        training_loss.append(avg_train_loss)
        training_acc.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                output = model(batch_x).squeeze()
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                preds = (torch.sigmoid(output) > 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        validation_loss.append(avg_val_loss)
        validation_acc.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

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
    model = TransformerClassifier(input_dim=sample_X.shape[-1])

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