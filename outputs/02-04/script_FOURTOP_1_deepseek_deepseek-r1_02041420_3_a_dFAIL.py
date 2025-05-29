import pandas as pd
import torch

X_train_df = pd.read_csv('./data/fourtops/X_train.csv')
Y_train_df  = pd.read_csv('./data/fourtops/Y_train.csv')
X_val_df   = pd.read_csv('./data/fourtops/X_val.csv')
Y_val_df   = pd.read_csv('./data/fourtops/Y_val.csv')

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train_df.values, dtype=torch.long).squeeze()
X_val   = torch.tensor(X_val_df.values, dtype=torch.float32)
Y_val   = torch.tensor(Y_val_df.values, dtype=torch.long).squeeze()

print(f"X_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_val shape: {X_val.shape}\nY_val shape: {Y_val.shape}")

# LLM Reponse Below
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import argparse

class ParticleClassifier(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, global_hidden=64):
        super().__init__()
        self.obj_embed = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_processor = nn.Sequential(
            nn.Linear(3, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, global_hidden)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + global_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_feats = x[:, 0:3]
        objects_feats = x[:, 3:3+20*5].view(x.size(0), 20, 5)
        obj_ids = objects_feats[:, :, 0]
        src_key_padding_mask = (obj_ids == 0)
        obj_emb = self.obj_embed(objects_feats)
        transformer_out = self.transformer_encoder(obj_emb, src_key_padding_mask=src_key_padding_mask)
        non_padding = ~src_key_padding_mask
        sum_emb = (transformer_out * non_padding.unsqueeze(-1)).sum(dim=1)
        count = non_padding.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mean_emb = sum_emb / count
        global_processed = self.global_processor(global_feats)
        combined = torch.cat([mean_emb, global_processed], dim=1)
        output = self.classifier(combined).squeeze()
        return output

def train_model(X_train, Y_train, X_val, Y_val, dryrun=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 128
    num_epochs = 20 if not dryrun else 2
    lr = 1e-4
    weight_decay = 1e-5

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = ParticleClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(outputs.cpu())
                val_labels.append(labels.cpu())
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        auc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'tt_classifier_model.pth')

    print(f"Best Validation AUC: {best_auc:.4f}")
    return best_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dryrun', action='store_true', help='Use small subset for testing')
    args = parser.parse_args()

    if args.dryrun:
        X_train_sub = X_train[:1000]
        Y_train_sub = Y_train[:1000]
        X_val_sub = X_val[:100]
        Y_val_sub = Y_val[:100]
        best_auc = train_model(X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, dryrun=True)
    else:
        best_auc = train_model(X_train, Y_train, X_val, Y_val)

    print(f"Final AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()