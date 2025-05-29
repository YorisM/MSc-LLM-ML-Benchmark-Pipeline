# ----- FIXED SECTION: Import Libraries -----
import os, sys, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
# <FREE: You may only import python and torch native modules here. NO OTHER MODULES.>

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
    # TorchScript-compatible module applying pre-fitted transformations.
    # All fitted statistics/constants must be registered as buffers.
    # Torch operations ONLY (no numpy, no pandas).
    # Deterministic behavior required (no randomness in forward pass).
    def __init__(self, **kwargs):
        super().__init__()
        # Example pattern for saving constants:
        # self.register_buffer("my_const", kwargs["my_const"])
        # <LLM: register any statistics / masks / embeddings here>
        mean = kwargs.get('mean', None)
        std  = kwargs.get('std', None)
        if mean is not None and std is not None:
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        else:
            self.mean = None
            self.std  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <LLM: transform x using the stored buffers>
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # derive statistics / encodings from training set
    # <LLM: derive any needed constants here>
    mean = X_train.mean(dim=0, keepdim=True)
    std  = X_train.std(dim=0, keepdim=True)

    preproc = PreprocessModule(mean=mean, std=std)

    X_train_p = preproc(X_train)
    X_val_p   = preproc(X_val)

    train_ds = TensorDataset(X_train_p, Y_train)
    val_ds   = TensorDataset(X_val_p,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    return train_loader, val_loader, preproc

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # <LLM: Define your neural network layers here>
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu   = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # <LLM: Define forward propagation here>
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    # <LLM: Define training loop clearly>
    # Must return trained model, training_loss, validation_loss, training_acc, validation_acc
    # Be sure to include basic metric tracking per epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    training_loss   = []
    validation_loss = []
    training_acc    = []
    validation_acc  = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_acc  += (predictions.squeeze() == labels.float()).sum().item() / labels.size(0)

        training_loss.append(train_loss / len(train_loader))
        training_acc.append(train_acc / len(train_loader))

        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_acc  += (predictions.squeeze() == labels.float()).sum().item() / labels.size(0)

        validation_loss.append(val_loss / len(val_loader))
        validation_acc.append(val_acc / len(val_loader))

        print(f'Epoch: {epoch+1}, Training Loss: {training_loss[-1]:.4f}, Validation Loss: {validation_loss[-1]:.4f}')
        print(f'Epoch: {epoch+1}, Training Acc: {training_acc[-1]:.4f}, Validation Acc: {validation_acc[-1]:.4f}')

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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val, batch_size=256)

    # Model Initialization
    sample_X, _, = next(iter(train_loader))
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