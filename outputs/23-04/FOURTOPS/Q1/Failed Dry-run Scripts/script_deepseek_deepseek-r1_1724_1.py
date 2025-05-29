import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# Data Loading

class PreprocessModule(torch.nn.Module):
    def __init__(self, feature_mask=None):
        super().__init__()
        if feature_mask is not None:
            self.register_buffer('feature_mask', torch.tensor(feature_mask, dtype=torch.bool))

    def forward(self, x):
        if hasattr(self, 'feature_mask'):
            x = x[:, self.feature_mask]

        x[:, ::4] = torch.log1p(x[:, ::4] + 1e-8)
        x[:, 1::4] = torch.sin(x[:, 1::4]) 
        x[:, 2::4] = torch.sin(x[:, 2::4])

        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=4096):
    valid_features = (X_train != 0).any(dim=0)
    valid_features[0] = True
    valid_features[1] = True
    preproc = PreprocessModule(feature_mask=valid_features.numpy())
    
    X_train_p = preproc(X_train)
    X_val_p = preproc(X_val)
    
    mean = X_train_p.mean(dim=0)
    std = X_train_p.std(dim=0) + 1e-8
    
    class Normalize(torch.nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        def forward(self, x):
            return (x - self.mean) / self.std
    
    preproc.norm = Normalize(mean, std)
    
    X_train_p = preproc.norm(X_train_p)
    X_val_p = preproc.norm(X_val_p)
    
    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds = TensorDataset(X_val_p, Y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, preproc

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    
    best_auc = 0
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds == labels.long()).sum().item()
            total_train += labels.size(0)
        
        avg_train_loss = epoch_train_loss / total_train
        train_acc = correct_train / total_train
        
        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                epoch_val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == labels.long()).sum().item()
                total_val += labels.size(0)
                
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
        
        avg_val_loss = epoch_val_loss / total_val
        val_acc = correct_val / total_val
        
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        scheduler.step(avg_val_loss)
        
        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)
        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} AUC: {auc:.4f}')
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    return model, training_loss, validation_loss, training_acc, validation_acc

def main(dryrun=False):
    X_train, Y_train, X_val, Y_val = load_data()
    batch_size = 4096 if not dryrun else 512
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size)
    
    sample_X, _ = next(iter(train_loader))
    model = Classifier(input_dim=sample_X.shape[1])
    
    epochs = 1 if dryrun else 30
    trained_model, t_loss, v_loss, t_acc, v_acc = train_model(model, train_loader, val_loader, epochs)
    
    if not dryrun:
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].replace('script_', '')
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        os.makedirs(script_dir, exist_ok=True)
        
        model_path = os.path.join(script_dir, f"{base}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        
        scripted_path = os.path.join(script_dir, f"{base}_scripted.pt")
        torch.jit.script(trained_model).save(scripted_path)
        
        scripted_preproc = torch.jit.script(preproc)
        scripted_preproc.save(os.path.join(script_dir, f"{base}_preproc.pt"))
        
        plot_and_save(t_loss, v_loss, "Loss", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(t_acc, v_acc, "Accuracy", os.path.join(script_dir, f"{base}_accuracy.png"))

if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)