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
    def __init__(self, means, stds, max_objects=26):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("max_objects", torch.tensor(max_objects, dtype=torch.int32))
        # Masks for identifying valid objects (non-zero padded)
        self.register_buffer("object_mask", torch.ones(max_objects, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Extract E_T_miss and phi_E_T_miss (global features)
        global_features = x[:, :2].clone()
        global_features = (global_features - self.means[:2]) / (self.stds[:2] + 1e-6)
        # Reshape the rest of the data into (batch_size, max_objects, features_per_object=4)
        object_features = x[:, 2:].reshape(batch_size, self.max_objects.item(), 4)
        object_features = (object_features - self.means[2:6]) / (self.stds[2:6] + 1e-6)
        # Compute augmented features (physics-informed)
        p_T = object_features[:, :, 1]
        eta = object_features[:, :, 2]
        phi = object_features[:, :, 3]
        p_z = p_T * torch.sinh(eta)
        p_x = p_T * torch.cos(phi)
        p_y = p_T * torch.sin(phi)
        augmented_features = torch.stack([p_x, p_y, p_z], dim=-1)  # Shape: (batch, max_obj, 3)
        object_features = torch.cat([object_features, augmented_features], dim=-1)  # Shape: (batch, max_obj, 7)
        # Combine global and object features
        global_features_expanded = global_features.unsqueeze(1).repeat(1, self.max_objects.item(), 1)
        x_transformed = torch.cat([global_features_expanded, object_features], dim=-1)  # Shape: (batch, max_obj, 9)
        return x_transformed

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Compute statistics from training data
    means = X_train.mean(dim=0, keepdim=True)
    stds = X_train.std(dim=0, keepdim=True) + 1e-6
    preproc = PreprocessModule(means=means, stds=stds)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class SlotAttention(nn.Module):
    def __init__(self, input_dim, slot_dim=64, num_slots=4, iters=3, eps=1e-8):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.slot_dim = slot_dim
        # Learnable initial slots
        self.slots = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.1)
        # MLP for input projection
        self.input_proj = nn.Linear(input_dim, slot_dim)
        # Attention mechanism components
        self.k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v = nn.Linear(slot_dim, slot_dim, bias=False)
        # Update mechanism
        self.update_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        b, n, d = inputs.shape
        # Project inputs to slot dimension
        inputs = self.norm_input(self.input_proj(inputs))  # Shape: (b, n, slot_dim)
        slots = self.slots.unsqueeze(0).repeat(b, 1, 1)  # Shape: (b, num_slots, slot_dim)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            k = self.k(inputs)  # Shape: (b, n, slot_dim)
            q = self.q(slots)   # Shape: (b, num_slots, slot_dim)
            dots = torch.einsum('bid,bjd->bij', q, k) * (self.slot_dim ** -0.5)  # Shape: (b, num_slots, n)
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize across slots
            updates = torch.einsum('bjd,bij->bid', self.v(inputs), attn)  # Shape: (b, num_slots, slot_dim)
            # Update slots using MLP
            slots = slots_prev + self.update_mlp(torch.cat([slots_prev, updates], dim=-1))
        return slots

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_slots=4, slot_dim=64):
        super(TransformerClassifier, self).__init__()
        self.slot_attention = SlotAttention(input_dim=input_dim, slot_dim=slot_dim, num_slots=num_slots)
        encoder_layer = nn.TransformerEncoderLayer(d_model=slot_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
        self.mlp_head = nn.Sequential(
            nn.Linear(slot_dim, slot_dim // 2),
            nn.ReLU(),
            nn.Linear(slot_dim // 2, 1)
        )

    def forward(self, x):
        b, n, _ = x.shape
        slots = self.slot_attention(x)  # Shape: (b, num_slots, slot_dim)
        cls_token = self.cls_token.repeat(b, 1, 1)
        transformer_input = torch.cat([cls_token, slots], dim=1)  # Shape: (b, num_slots+1, slot_dim)
        transformer_output = self.transformer_encoder(transformer_input)  # Shape: (b, num_slots+1, slot_dim)
        cls_output = transformer_output[:, 0, :]  # Extract cls token output
        logits = self.mlp_head(cls_output)  # Shape: (b, 1)
        return logits

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.model = TransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, num_slots=4, slot_dim=64)

    def forward(self, x):
        return self.model(x)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
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
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        train_loss_avg = train_loss / len(train_loader)
        train_acc_epoch = accuracy_score(train_labels, train_preds)
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
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss_avg = val_loss / len(val_loader)
        val_acc_epoch = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_epoch:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_epoch:.4f}")
        training_loss.append(train_loss_avg)
        validation_loss.append(val_loss_avg)
        training_acc.append(train_acc_epoch)
        validation_acc.append(val_acc_epoch)
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
    model = Classifier(input_dim=sample_X.shape[-1])

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