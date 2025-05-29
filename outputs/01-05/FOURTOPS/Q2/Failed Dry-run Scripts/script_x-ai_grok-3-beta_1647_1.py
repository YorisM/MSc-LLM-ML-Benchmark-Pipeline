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
    def __init__(self, means=None, stds=None):
        super().__init__()
        # Register buffers for normalization
        if means is not None and stds is not None:
            self.register_buffer("means", means)
            self.register_buffer("stds", stds)
        else:
            self.register_buffer("means", torch.zeros(105))
            self.register_buffer("stds", torch.ones(105))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input data using precomputed means and stds
        mask = (self.stds != 0)
        x = torch.where(mask, (x - self.means) / self.stds, x)
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Compute mean and std for normalization from training data only
    means = X_train.mean(dim=0)
    stds = X_train.std(dim=0)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)  # Avoid division by zero

    preproc = PreprocessModule(means=means, stds=stds)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class LorentzEquivariantLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LorentzEquivariantLayer, self).__init__()
        self.scalar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.metric = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0, 0.0],
                                    [0.0, 0.0, 0.0, -1.0]], dtype=torch.float32)

    def forward(self, x, indices):
        batch_size = x.shape[0]
        num_objects = indices.shape[1] // 4
        # Extract 4-vectors for the objects
        vectors = x[:, indices].reshape(batch_size, num_objects, 4)
        # Compute scalar invariants (p_mu * p^mu) using Minkowski metric
        scalars = torch.einsum('bni,ij,bnj->bn', vectors, self.metric, vectors)
        scalars = scalars.unsqueeze(-1)  # Shape: (batch, num_objects, 1)
        # Transform scalars through a network
        scalar_out = self.scalar_net(scalars.reshape(batch_size, -1))
        # Pairwise interactions for message passing
        pairwise_vec = torch.cat([vectors.unsqueeze(2).expand(-1, -1, num_objects, -1),
                                  vectors.unsqueeze(1).expand(-1, num_objects, -1, -1)], dim=-1)
        pairwise_vec = pairwise_vec.reshape(batch_size, num_objects * num_objects, 8)
        vec_out = self.vector_net(pairwise_vec.reshape(batch_size, -1))
        return torch.cat([scalar_out, vec_out], dim=-1)

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # Assuming max objects is based on input_dim (e.g., 105 total dims, adjust as needed)
        self.num_features_per_object = 4  # E, pT, eta, phi
        self.max_objects = (input_dim - 2) // self.num_features_per_object  # Approximate max objects
        # Compute indices for extracting 4-vector components (E, pT, eta, phi per object)
        indices = []
        for i in range(self.max_objects):
            start_idx = 2 + i * self.num_features_per_object
            indices.extend(range(start_idx, start_idx + self.num_features_per_object))
        self.indices = torch.tensor(indices, dtype=torch.long)
        
        # Define Lorentz-equivariant layers
        self.layer1 = LorentzEquivariantLayer(self.num_features_per_object, 64)
        self.layer2 = LorentzEquivariantLayer(64 * 2, 128)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.layer1(x, self.indices)
        x2 = self.layer2(x1.unsqueeze(1), self.indices)
        out = self.fc(x2)
        return out.squeeze(-1)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    training_auc = []
    validation_auc = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(batch_y.detach().cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
        train_auc = roc_auc_score(train_labels, train_preds)
        training_loss.append(avg_train_loss)
        training_acc.append(train_acc)
        training_auc.append(train_auc)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_labels.extend(batch_y.detach().cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        val_auc = roc_auc_score(val_labels, val_preds)
        validation_loss.append(avg_val_loss)
        validation_acc.append(val_acc)
        validation_auc.append(val_auc)

        scheduler.step(val_auc)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=512)

    # Model Initialization
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])

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