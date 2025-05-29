# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
# <FREE: You may only import python and torch native modules here. NO OTHER MODULES.>
import math # Python native module

# ----- FIXED SECTION: Data Loading -----
def load_data():
    # Assuming data files are in a subdirectory relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, 'challenges', 'FOURTOPS', 'data')

    # Check if data path exists, provide informative error if not
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found at {data_path}. "
                            f"Ensure the './challenges/FOURTOPS/data/' structure exists relative to the script.")

    X_train_df = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    Y_train_df = pd.read_csv(os.path.join(data_path, 'Y_train.csv'))
    X_val_df   = pd.read_csv(os.path.join(data_path, 'X_val.csv'))
    Y_val_df   = pd.read_csv(os.path.join(data_path, 'Y_val.csv'))

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
    X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
    Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()
    return X_train, Y_train, X_val, Y_val

# ----- FREE SECTION: Data Preprocessing -----
class PreprocessModule(torch.nn.Module):
    # TorchScript-compatible module applying pre-fitted transformations.
    # All fitted statistics/constants must be registered as buffers.
    # Torch operations ONLY (no numpy, no pandas).
    # Deterministic behavior required (no randomness in forward pass).
    def __init__(self, mean, std, num_features):
        super().__init__()
        # Example pattern for saving constants:
        # self.register_buffer("my_const", kwargs["my_const"])
        # <LLM: register any statistics / masks / embeddings here>
        self.register_buffer("mean", mean)
        self.register_buffer("std", std + 1e-7) # Add epsilon for stability
        self.register_buffer("g", torch.tensor([1., -1., -1., -1.], dtype=torch.float32)) # Minkowski metric
        self.num_features = num_features

    def _get_p4(self, x):
        # Assumes x shape [B, 105]
        # Input format assumed: MET, MET_phi, then 20 objects * 5 features each = 102. Last 3 ignored.
        # Object features assumed: E, pT, eta, phi, tag/other (index 0, 1, 2, 3, 4)
        objects_flat = x[:, 2:102]
        objects = objects_flat.reshape(-1, 20, 5)
        E = objects[:, :, 0]
        pT = objects[:, :, 1]
        eta = objects[:, :, 2]
        phi = objects[:, :, 3]

        # Mask based on pT > threshold (e.g., 1 MeV)
        mask = (pT > 1e-3).float() # Shape [B, 20]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        # Clamp eta for stability with sinh
        pz = pT * torch.sinh(torch.clamp(eta, -7.0, 7.0))

        # 4-vectors, apply mask
        p4 = torch.stack([E, px, py, pz], dim=-1) # Shape [B, 20, 4]
        p4 = p4 * mask.unsqueeze(-1) # Zero out p4 for masked objects

        return p4, mask, pT

    def _invariant_mass_sq(self, p4_vec):
        g_device = self.g.to(p4_vec.device)
        m2 = torch.einsum('...i,...i,i->...', p4_vec, p4_vec, g_device)
        return m2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        met = x[:, 0:1]
        met_phi = x[:, 1:2]

        # Basic MET features
        met_x = met * torch.cos(met_phi)
        met_y = met * torch.sin(met_phi)

        # Get 4-vectors, mask, and pT
        p4, mask, pT = self._get_p4(x) # p4 [B, 20, 4], mask [B, 20], pT [B, 20]

        # Lorentz invariant / related features
        n_obj = mask.sum(dim=1, keepdim=True) # Shape [B, 1]

        pT_valid = pT * mask # Apply mask
        sum_pt = pT_valid.sum(dim=1, keepdim=True) # Shape [B, 1]

        # Total 4-vector sum and its invariant mass
        p4_total = p4.sum(dim=1) # Shape [B, 4]
        m2_total = self._invariant_mass_sq(p4_total)
        m_total = torch.sqrt(torch.relu(m2_total) + 1e-9).unsqueeze(-1) # Add epsilon before sqrt

        # Scalar sum of Energy
        E_valid = p4[:, :, 0] * mask # E is index 0; p4 is already masked
        sum_E = E_valid.sum(dim=1, keepdim=True) # Shape [B, 1] Scalar HT (HT')

        # Features from leading / subleading objects
        pT_masked = torch.where(mask > 0.5, pT, torch.full_like(pT, -1.0))
        try:
            vals, top_indices = torch.topk(pT_masked, k=2, dim=1) # Shape [B, 2]
            # Ensure indices are valid even if fewer than 2 objects
            valid_obj_counts = n_obj.squeeze(-1)
            top_indices = torch.where(valid_obj_counts.unsqueeze(-1) > torch.arange(2, device=device).unsqueeze(0), top_indices, torch.zeros_like(top_indices))
        except RuntimeError: # Fallback if topk fails completely
             top_indices = torch.zeros(batch_size, 2, dtype=torch.long, device=device)

        idx_gather = top_indices.unsqueeze(-1).expand(-1, -1, 4) # Shape [B, 2, 4]
        top2_p4 = torch.gather(p4, 1, idx_gather) # Shape [B, 2, 4]

        p4_lead = top2_p4[:, 0, :] # Shape [B, 4]
        p4_sublead = top2_p4[:, 1, :] # Shape [B, 4]

        # Individual kinematics (handle potential division by zero if pT is near zero)
        E_lead = p4_lead[:, 0:1]
        px_lead, py_lead, pz_lead = p4_lead[:, 1:2], p4_lead[:, 2:3], p4_lead[:, 3:4]
        pT_lead = torch.sqrt(px_lead**2 + py_lead**2 + 1e-9)
        eta_lead = torch.asinh(pz_lead / (pT_lead + 1e-9))
        phi_lead = torch.atan2(py_lead, px_lead)
        m2_lead = self._invariant_mass_sq(p4_lead)
        m_lead = torch.sqrt(torch.relu(m2_lead) + 1e-9).unsqueeze(-1)

        E_sublead = p4_sublead[:, 0:1]
        px_sublead, py_sublead, pz_sublead = p4_sublead[:, 1:2], p4_sublead[:, 2:3], p4_sublead[:, 3:4]
        pT_sublead = torch.sqrt(px_sublead**2 + py_sublead**2 + 1e-9)
        eta_sublead = torch.asinh(pz_sublead / (pT_sublead + 1e-9))
        phi_sublead = torch.atan2(py_sublead, px_sublead)
        m2_sublead = self._invariant_mass_sq(p4_sublead)
        m_sublead = torch.sqrt(torch.relu(m2_sublead) + 1e-9).unsqueeze(-1)

        # Combine features
        features = torch.cat([
            met_x, met_y,        # 2
            n_obj / 20.0,        # 1 (normalize roughly)
            torch.log1p(sum_pt),   # 1 (log scale for stability)
            torch.log1p(m_total),  # 1 (log scale)
            torch.log1p(sum_E),    # 1 (log scale)
            torch.log1p(E_lead), torch.log1p(pT_lead), eta_lead / 5.0, phi_lead / math.pi, torch.log1p(m_lead), # 5 (log scale + rough norm)
            torch.log1p(E_sublead), torch.log1p(pT_sublead), eta_sublead / 5.0, phi_sublead / math.pi, torch.log1p(m_sublead) # 5 (log scale + rough norm)
        ], dim=1)

        # Check feature size matches expected
        if features.shape[1] != self.num_features:
             raise ValueError(f"Feature dimension mismatch: expected {self.num_features}, got {features.shape[1]}")

        # Normalize using pre-calculated stats
        normalized_features = (features - self.mean.to(device)) / self.std.to(device)

        # Handle potential NaNs/Infs introduced by operations or normalization
        normalized_features = torch.nan_to_num(normalized_features, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized_features

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # Determine the number of features engineered by the PreprocessModule
    # This is determined by the concatenation in the forward pass
    NUM_ENGINEERED_FEATURES = 16 # 2 MET + 1 Nobj + 1 SumPt + 1 Mtot + 1 SumE + 5 Lead + 5 SubLead

    # Instantiate temporary preprocessor to calculate features on training data
    # Pass dummy mean/std, but correct feature count
    temp_preproc = PreprocessModule(torch.zeros(NUM_ENGINEERED_FEATURES), torch.ones(NUM_ENGINEERED_FEATURES), NUM_ENGINEERED_FEATURES)

    # Calculate features in batches to avoid memory issues if X_train is large
    train_feat_list = []
    temp_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size*2, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in temp_loader:
             train_feat_list.append(temp_preproc(batch_x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu())
    X_train_feat = torch.cat(train_feat_list, dim=0)

    # Calculate mean and std dev from the engineered training features
    mean = torch.mean(X_train_feat, dim=0)
    std = torch.std(X_train_feat, dim=0)

    # Create the actual preprocessor with calculated stats
    preproc = PreprocessModule(mean, std, NUM_ENGINEERED_FEATURES)

    # Apply preprocessing (can be done within dataloader transform later, but this fits template)
    # Apply in batches for memory efficiency
    X_train_p_list = []
    X_val_p_list = []
    with torch.no_grad():
       for (batch_x,) in temp_loader: # Reuse temp_loader for X_train
           X_train_p_list.append(preproc(batch_x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu())
       X_train_p = torch.cat(X_train_p_list, dim=0)

       val_temp_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size*2, shuffle=False)
       for (batch_x,) in val_temp_loader:
           X_val_p_list.append(preproc(batch_x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu())
       X_val_p = torch.cat(X_val_p_list, dim=0)


    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p, Y_val)

    # Use num_workers > 0 if not causing issues on the system
    # Pin memory can speed up CPU->GPU transfer if using GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # <LLM: Define your neural network layers here>
        self.layer_1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.layer_2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.layer_3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        self.output_layer = nn.Linear(32, 1) # Output raw logits for BCEWithLogitsLoss

    def forward(self, x):
        # <LLM: Define forward propagation here>
        x = self.layer_1(x)
        if x.shape[0] > 1: # BatchNorm requires batch size > 1
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer_2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.layer_3(x)
        if x.shape[0] > 1:
             x = self.bn3(x)
        x = self.relu3(x)

        x = self.output_layer(x)
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    # <LLM: Define training loop clearly>
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss() # Numerically stable
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Scheduler optimizes based on validation AUC
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)

    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    # validation_auc = [] # Track AUC per epoch if needed outside plotting

    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        train_correct = 0
        train_total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Grad clipping
            optimizer.step()

            train_loss_epoch += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_acc = train_correct / train_total
        training_loss.append(avg_train_loss)
        training_acc.append(avg_train_acc)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels_batch = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)

                val_loss_epoch += loss.item()
                val_total += labels_batch.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels_batch).sum().item()

                all_labels.append(labels_batch.cpu()) # Store labels
                all_outputs.append(torch.sigmoid(outputs).cpu()) # Store probabilities for AUC

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_acc = val_correct / val_total
        validation_loss.append(avg_val_loss)
        validation_acc.append(avg_val_acc)

        # Calculate AUC on validation set
        all_labels = torch.cat(all_labels).numpy()
        all_outputs = torch.cat(all_outputs).numpy()
        try:
            epoch_auc = roc_auc_score(all_labels, all_outputs)
        except ValueError: # Handle cases where only one class is present in validation batches
            epoch_auc = 0.0
            print(f"Warning: AUC calculation failed for epoch {epoch+1}. Only one class present?")

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val AUC: {epoch_auc:.4f}")

        scheduler.step(epoch_auc) # Step scheduler based on validation AUC

        # Save the model state if it has the best validation AUC so far
        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_model_state = model.state_dict().copy()
            print(f"   ---> New best model saved with Val AUC: {best_val_auc:.4f}")

    # Load the best model state found during training before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model weights (Val AUC: {best_val_auc:.4f})")

    # Must return trained model, training_loss, validation_loss, training_acc, validation_acc
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
    plt.grid(True)
    try:
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    finally:
         plt.close()

# ----- FIXED SECTION: Main Function -----
def main(dryrun=False):
    # Data Loading
    try:
        X_train, Y_train, X_val, Y_val = load_data()
        print("Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data is available at the expected path.")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during data loading: {e}")
         sys.exit(1)

    # Define batch size
    BATCH_SIZE = 512 if not dryrun else 64
    print(f"Using batch size: {BATCH_SIZE}")

    # Preprocessing
    print("Starting preprocessing...")
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, BATCH_SIZE)
    print("Preprocessing finished.")

    # Model Initialization
    # Get input dimension from the first batch of preprocessed data
    try:
        sample_X, _ = next(iter(train_loader))
        INPUT_DIM = sample_X.shape[1]
        print(f"Input dimension after preprocessing: {INPUT_DIM}")
    except StopIteration:
        print("Error: Training data loader is empty.")
        sys.exit(1)
    
    model = Classifier(input_dim=INPUT_DIM)
    print(f"Model initialized: {model.__class__.__name__}")

    # Training
    epochs = 1 if dryrun else 40 # Increased epochs for better convergence
    print(f"Starting training for {epochs} epochs...")

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs
    )
    print("Training finished.")

    if not dryrun:
        print("Saving model and artifacts...")
        # determine base name & script directory
        script_file = sys.argv[0] if sys.argv[0] else 'default_script'
        base = os.path.splitext(os.path.basename(script_file))[0].removeprefix("script_")
        if not base: base = "fourtops_classifier"
        # Try to get script directory, fallback to current working directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
        except NameError:
             script_dir = '.' # Fallback if __file__ is not defined (e.g., interactive) 
        
        output_dir = os.path.join(script_dir, 'outputs') # Save outputs in a subdirectory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # save model state dict
        model_path = os.path.join(output_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model state dict saved to {model_path}")

        # save scripted model (ensure model is on CPU before scripting)
        try:
            trained_model.to('cpu')
            scripted_model = torch.jit.script(trained_model)
            scripted_path = os.path.join(output_dir, f"{base}_scripted.pt")
            scripted_model.save(scripted_path)
            print(f"Scripted model saved to {scripted_path}")
        except Exception as e:
            print(f"Could not save scripted model: {e}")

        # save preprocessor
        try:
            scripted_preproc = torch.jit.script(preproc)
            preproc_path = os.path.join(output_dir, f"{base}_preproc.pt")
            scripted_preproc.save(preproc_path)
            print(f"Scripted preprocessor saved to {preproc_path}")
        except Exception as e:
            print(f"Could not save scripted preprocessor: {e}")

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss", os.path.join(output_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy", os.path.join(output_dir, f"{base}_accuracy.png"))
        print("Finished saving artifacts.")
    else:
        print("Dry run finished. No artifacts saved.")

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    # Basic check for data path existence before running main
    expected_data_dir = os.path.join('.', 'challenges', 'FOURTOPS', 'data')
    if not os.path.isdir(expected_data_dir):
        print(f"WARNING: Data directory '{expected_data_dir}' not found.")
        print("         Please ensure the data is placed correctly relative to the script location.")
        # Optionally exit if data is critical, but main() will also handle FileNotFoundError
        # sys.exit(1) 
    main(dryrun=dryrun)