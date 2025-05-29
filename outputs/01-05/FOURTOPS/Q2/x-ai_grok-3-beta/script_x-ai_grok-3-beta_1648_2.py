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
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.max_objects = max_objects

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        mask = (x != 0).float()
        x = (x - self.means) / (self.stds + 1e-8)
        x = x * mask
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=128):
    means = X_train.mean(dim=0, keepdim=True)
    stds = X_train.std(dim=0, keepdim=True) + 1e-8
    preproc = PreprocessModule(means, stds)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p, Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class LorentzEquivariantLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scalar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tensor_net = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, indices):
        batch_size = x.shape[0]
        x_scalar = x[:, indices:indices+self.input_dim]
        output_scalar = self.scalar_net(x_scalar)
        tensor_input = x[:, indices:indices+self.input_dim*4].reshape(batch_size, -1, 4)
        tensor_output = self.tensor_net(tensor_input.reshape(batch_size, -1))
        return output_scalar + tensor_output.reshape(batch_size, -1, self.hidden_dim).mean(dim=1)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, max_objects=20):
        super(Classifier, self).__init__()
        self.max_objects = max_objects
        self.input_dim_per_obj = (input_dim - 2) // (max_objects * 5)  # Approx kinematic props per object
        self.lorentz_layer = LorentzEquivariantLayer(self.input_dim_per_obj, hidden_dim)
        self.missing_et_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU()
        )
        self.global_net = nn.Sequential(
            nn.Linear(hidden_dim * (max_objects + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        missing_et_feat = self.missing_et_net(x[:, :2])
        object_features = []
        for i in range(self.max_objects):
            start_idx = 2 + i * 5  # Skip E_T_miss and phi_E_T_miss
            obj_feat = self.lorentz_layer(x, start_idx)
            object_features.append(obj_feat)
        all_features = torch.cat(object_features + [missing_et_feat], dim=1)
        output = self.global_net(all_features)
        return torch.sigmoid(output)

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    training_loss, validation_loss = [], []
    training_auc, validation_auc = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_labels = [], []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            optimizer.zero_grad()
            output = model(batch_x).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_preds.extend(output.detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)
        training_loss.append(avg_train_loss)
        training_auc.append(train_auc)
        
        model.eval()
        epoch_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                output = model(batch_x).squeeze()
                loss = criterion(output, batch_y)
                epoch_val_loss += loss.item()
                val_preds.extend(output.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        validation_loss.append(avg_val_loss)
        validation_auc.append(val_auc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    return model, training_loss, validation_loss, training_auc, validation_auc

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
    X_train, Y_train, X_val, Y_val = load_data()
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    epochs = 1 if dryrun else 10
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)