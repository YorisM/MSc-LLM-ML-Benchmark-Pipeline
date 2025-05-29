
import os, sys, pickle, torch, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val": "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val": "./challenges/FOURTOPS/data/Y_val.csv"
}
                       
def load_data():
    X_train = pd.read_csv('./challenges/FOURTOPS/data/X_train.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_train = pd.read_csv('./challenges/FOURTOPS/data/Y_train.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()
    X_val   = pd.read_csv('./challenges/FOURTOPS/data/X_val.csv',
                          dtype=np.float32).to_numpy(copy=False)
    Y_val   = pd.read_csv('./challenges/FOURTOPS/data/Y_val.csv',
                          dtype=np.int64 ).to_numpy(copy=False).ravel()

    gc.collect()

    return (torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val))

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------

# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.max_objects = 18  # As mentioned in the problem description
        self.features_per_object = 5  # obj_id, E, p_T, eta, phi

    def fit(self, X, y=None):
        # Extract meaningful features with proper physics understanding
        processed_features = self._extract_features(X)
        # Fit scaler on extracted features
        self.scaler.fit(processed_features)
        return self

    def transform(self, X):
        # Extract features and normalize them
        processed_features = self._extract_features(X)
        scaled_features = self.scaler.transform(processed_features)
        return torch.FloatTensor(scaled_features)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def _extract_features(self, X):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        batch_size = X_np.shape[0]
        
        # Features we'll compute based on physics knowledge
        features = []
        
        for i in range(batch_size):
            event = X_np[i]
            
            # Extract ETmiss and phi_ETmiss directly
            et_miss = event[0]
            phi_et_miss = event[1]
            
            # Initialize lists for various object types
            electrons = []
            muons = []
            jets = []
            b_jets = []
            
            # Process objects in the event
            for j in range(self.max_objects):
                start_idx = 2 + j * self.features_per_object
                if start_idx >= len(event):
                    break
                
                obj_type = event[start_idx]
                # Check if this is a valid object (obj_type > 0)
                if obj_type > 0:
                    E = event[start_idx + 1]
                    pt = event[start_idx + 2]
                    eta = event[start_idx + 3]
                    phi = event[start_idx + 4]
                    
                    # Only consider objects with positive energy and pt
                    if E > 0 and pt > 0:
                        obj_features = [E, pt, eta, phi]
                        
                        # Categorize objects based on their type
                        # obj_type: 1=electron, 2=muon, 3=jet, 4=b-jet
                        if obj_type == 1:
                            electrons.append(obj_features)
                        elif obj_type == 2:
                            muons.append(obj_features)
                        elif obj_type == 3:
                            jets.append(obj_features)
                        elif obj_type == 4:
                            b_jets.append(obj_features)
            
            # Count objects of each type
            n_electrons = len(electrons)
            n_muons = len(muons)
            n_jets = len(jets)
            n_bjets = len(b_jets)
            
            # Initialize event features with ETmiss information
            event_features = [et_miss, phi_et_miss, n_electrons, n_muons, n_jets, n_bjets]
            
            # Calculate sum of pt for each type of object
            pt_sum_e = sum(e[1] for e in electrons) if electrons else 0
            pt_sum_mu = sum(mu[1] for mu in muons) if muons else 0
            pt_sum_jets = sum(j[1] for j in jets) if jets else 0
            pt_sum_bjets = sum(bj[1] for bj in b_jets) if b_jets else 0
            
            event_features.extend([pt_sum_e, pt_sum_mu, pt_sum_jets, pt_sum_bjets])
            
            # Calculate HT (scalar sum of jet pT)
            ht = pt_sum_jets + pt_sum_bjets
            event_features.append(ht)
            
            # Calculate MET/sqrt(HT) - useful for separating real MET from mismeasurements
            met_significance = et_miss / np.sqrt(ht) if ht > 0 else 0
            event_features.append(met_significance)
            
            # Add leading object properties (sorted by pT)
            for obj_list, prefix, max_to_keep in [
                (sorted(electrons, key=lambda x: x[1], reverse=True), 'e', 2),
                (sorted(muons, key=lambda x: x[1], reverse=True), 'mu', 2),
                (sorted(jets, key=lambda x: x[1], reverse=True), 'jet', 4),
                (sorted(b_jets, key=lambda x: x[1], reverse=True), 'b', 4)
            ]:
                for i in range(max_to_keep):
                    if i < len(obj_list):
                        # Add E, pT, eta, phi for this object
                        event_features.extend(obj_list[i])
                    else:
                        # Padding for missing objects
                        event_features.extend([0, 0, 0, 0])
            
            # Calculate invariant masses where relevant - these are powerful in four-top events
            # Especially b-jet combinations (from top decays)
            if len(b_jets) >= 2:
                # Calculate invariant mass of leading two b-jets
                b1, b2 = b_jets[0], b_jets[1]
                m_bb = self._calculate_invariant_mass(b1, b2)
                event_features.append(m_bb)
            else:
                event_features.append(0)
            
            # Lepton pairs can also be informative
            if len(electrons) >= 2:
                e1, e2 = electrons[0], electrons[1]
                m_ee = self._calculate_invariant_mass(e1, e2)
                event_features.append(m_ee)
            else:
                event_features.append(0)
                
            if len(muons) >= 2:
                mu1, mu2 = muons[0], muons[1]
                m_mumu = self._calculate_invariant_mass(mu1, mu2)
                event_features.append(m_mumu)
            else:
                event_features.append(0)
            
            # Cross-object features: e.g., delta-R between objects
            # These help identify hadronic vs leptonic decay patterns
            if n_bjets > 0 and n_electrons > 0:
                dr_e_b = self._calculate_delta_r(electrons[0][2:4], b_jets[0][2:4])
                event_features.append(dr_e_b)
            else:
                event_features.append(0)
                
            if n_bjets > 0 and n_muons > 0:
                dr_mu_b = self._calculate_delta_r(muons[0][2:4], b_jets[0][2:4])
                event_features.append(dr_mu_b)
            else:
                event_features.append(0)
            
            # Number of jets with high pT - characteristic of tttt events (multiple high-pT tops)
            n_jets_high_pt = sum(1 for j in jets if j[1] > 50000)  # 50 GeV threshold
            event_features.append(n_jets_high_pt)
            
            # Append this event's features
            features.append(event_features)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_invariant_mass(self, p1, p2):
        # p1 and p2 are [E, pT, eta, phi]
        E1, pt1, eta1, phi1 = p1
        E2, pt2, eta2, phi2 = p2
        
        # Convert to px, py, pz for both particles
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        
        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        
        # Calculate invariant mass
        m_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
        m = np.sqrt(max(0, m_squared))  # Protection against numerical errors
        return m
    
    def _calculate_delta_r(self, coords1, coords2):
        # coords are [eta, phi]
        eta1, phi1 = coords1
        eta2, phi2 = coords2
        
        # Compute delta eta and normalized delta phi
        delta_eta = eta1 - eta2
        delta_phi = abs(phi1 - phi2)
        if delta_phi > np.pi:
            delta_phi = 2 * np.pi - delta_phi
            
        # Delta R = sqrt(delta_eta^2 + delta_phi^2)
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_r

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
class FourTopClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FourTopClassifier, self).__init__()
        
        # Define network architecture: sequence of fully connected layers with batch norm
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.fc5 = nn.Linear(32, 1)  # Binary classification: 1 output value
        
    def forward(self, x):
        # Forward pass with ReLU activations and residual connections
        x1 = self.drop1(self.bn1(F.relu(self.fc1(x))))
        
        x2 = self.drop2(self.bn2(F.relu(self.fc2(x1))))
        
        x3 = self.drop3(self.bn3(F.relu(self.fc3(x2))))
        
        x4 = self.bn4(F.relu(self.fc4(x3)))
        
        # Final layer without activation (will be applied in loss function)
        logit = self.fc5(x4)
        
        return logit.view(-1)  # Flatten to match BCEWithLogitsLoss expectations

def make_model(input_dim):
    model = FourTopClassifier(input_dim)
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = 50

def train_model(model, train_loader, val_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Using BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Storage for metrics
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    best_val_auc = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_outputs_all = []
        train_labels_all = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to float for BCE loss
            float_labels = labels.float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, float_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.0).float()
            train_correct += (predictions == float_labels).sum().item()
            train_total += float_labels.size(0)
            
            # Save outputs and labels for AUC calculation
            train_outputs_all.append(outputs.detach().cpu().numpy())
            train_labels_all.append(float_labels.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Calculate train AUC
        train_outputs_all = np.concatenate(train_outputs_all)
        train_labels_all = np.concatenate(train_labels_all)
        train_auc = roc_auc_score(train_labels_all, train_outputs_all)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_outputs_all = []
        val_labels_all = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                float_labels = labels.float()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, float_labels)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                predictions = (outputs > 0.0).float()
                val_correct += (predictions == float_labels).sum().item()
                val_total += float_labels.size(0)
                
                # Save outputs and labels for AUC calculation
                val_outputs_all.append(outputs.cpu().numpy())
                val_labels_all.append(float_labels.cpu().numpy())
        
        # Calculate epoch metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Calculate validation AUC
        val_outputs_all = np.concatenate(val_outputs_all)
        val_labels_all = np.concatenate(val_labels_all)
        val_auc = roc_auc_score(val_labels_all, val_outputs_all)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Store history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        
        # Print progress
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return model, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist

# ----------------  END OF LLM BLOCK ----------------
                         
def _plot(series_train, series_val, name, out_path):
    plt.figure()
    plt.plot(series_train, label=f"Train {name}")
    plt.plot(series_val,   label=f"Val {name}")
    plt.title(name); plt.xlabel("epoch"); plt.legend()
    plt.savefig(out_path); plt.close()

def _run(dryrun=False):
    # 1. Load & preprocess
    X_train, Y_train, X_val, Y_val = load_data()
    pre = make_preprocessor()
    pre.fit(X_train, Y_train)
    X_train = pre.transform(X_train)
    X_val = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    model = make_model(input_dim=X_train.shape[1])
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 3. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy = torch.zeros(8, X_train.shape[1])      # 8 fake events
        try:
            _ = trained_model(pre.transform(toy))
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return  # no files in dry-run

    # 4. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 5. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)

