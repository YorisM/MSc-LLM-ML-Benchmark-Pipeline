# ./challenges/FOURTOPS/fourtops.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
    name = "FOURTOP",

    dataset = {
    "X_train": "./challenges/FOURTOPS/data/X_train.csv",
    "Y_train": "./challenges/FOURTOPS/data/Y_train.csv",
    "X_val":   "./challenges/FOURTOPS/data/X_val.csv",
    "Y_val":   "./challenges/FOURTOPS/data/Y_val.csv"
    },

    problem_description = r"""** Problem Description **
A major task in particle physics is the measurement of rare signal 
processes ith very small cross-sections. With the unprecedented amount of 
data provided by the upcoming runs of the Large Hadron Collider (LHC), 
one can start to measure these processes. An example is the recent 
observation of four top quarks originating from a single proton-proton 
collision event. Accurate classification of these events is crucial, 
as even a small reduction in background noise on the order of a few tens 
of percent while maintaining the same signal detection efficiency can lead 
to a profound increase in sensitivity.
""",

    dataset_description = r"""** Dataset Description **
The dataset used for this problem consists of simulated proton-proton 
collision at a center of mass energy of 13 TeV. The signal process is defined as 
$$ pp \rightarrow t \bar{t} t \bar{t} $$. The relevant production processes of the
backgrounds are $$ t \bar{t} + X $$ where $$ X = Z, W^+, W^+W^- $$.

The dataset includes 302072 events, of which roughly 50% is signal and 50%
are background processes. All background processes have an equal number of events. 
There is not cut on the maximum number of objects and there is no order.    

The contents of the data sets (X_train & X_val) are given below.
The specific line format is as follows:

weight, E_T_miss, phi_{E_t}_miss, obj_1, E_1, p_T1, eta_1, phi_1, obj_2, E_2, p_t2, eta_2, phi_2, ....

Such that each object is represented by a string that starts with an identifier "obj_n", which is an
integer value representing a particular object in the event. The object identifier is
followed by its kinematic properties in the form of a four-vector containing the full 
energy "E" and the transverse momentum "p_T" in units of MeV, as well as the pseudo-rapidity 
"eta" and the azimuthal angle "phi". The other three quantities are "weight" given by the cross-section
of thne process divided by the total number of events generated. "E_T_miss" is the magnitude of the
missing transverse energy in units of MeV and "phi_{E_t}_miss" is the azimuthal angle of the missing
transverse energy.

Since the length of the events is variable, the data is zero-padded to the largest number of objects
found in the events within in the entire dataset. The dataset is fairly sparse and not pre-processed.

The relevant datasets are pytorch tensors with the following properties:

Name: X_train, shape: [241657, 106], dtype: torch.float32, 
Name: Y_train, shape: [241657], dtype: torch.int64, 
Name: X_val, shape: [30272, 106], dtype: torch.float32, 
Name: Y_val, shape: [30272], dtype: torch.int64, 
""",

    evaluation_metric = r"""** Evaluation Metric **
The evaluation metric for this classification task is the area under the curve (AUC)
value, specified by the area under the receiver operating characteristic (ROC) curve. 
The AUC summarizes a model's ability to distinguish between positive and 
negative classes. A value of 1 indicates a perfect classifier, while a 0.5 indicates
performance no better than random guessing. Mathematically the AUC is defined as:
$$ AUC = \int_{0}^{1} TPR(t) \, d(FPR(t)) $$
Where the True Positive Rate (TPR) is defines as:
$$   TPR = \frac{TP}{TP + FN} $$
and the False Positive Rate (FPR) is defined as:
$$ FPR = \frac{FP}{FP + TN} $$
""",

    code_template = r"""** Code Template **
# ----- FREE SECTION: Import Libraries -----
import numpy as np
import pandas as pd
import math
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
# <LLM: Insert additional library imports here>
# You may only import the additional packages from: sklearn and torch.

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
def preprocess_data(X_train, Y_train, X_val, Y_val):
    # <LLM: Insert custom preprocessing steps here>
    return train_loader, val_loader

# ----- FREE SECTION: Binary Classifier Definition -----
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        # <LLM: Define your neural network layers here>

    def forward(self, x):
        # <LLM: Define forward propagation here>
        return x

# ----- FREE SECTION: Training Loop Implementation -----
def train_model(model, train_loader, val_loader, epochs):
    # <LLM: Define training loop clearly>
    # Must return trained model, training_loss, validation_loss, training_acc, validation_acc
    # Be sure to include basic metric tracking per epoch
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
    train_loader, val_loader = preprocess_data(X_train, Y_train, X_val, Y_val)

    # Model Initialization
    model = Classifier(input_dim=X_train.shape[1])

    # Training
    epochs = 1 if dryrun else 10

    # Train the model
    trained_model, training_loss, validation_loss, training_acc, validation_acc = train_model(
        model, train_loader, val_loader, epochs=epochs)

    if not dryrun:
        # Save Model
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        model_filename = os.path.join(script_dir, f"{base}_model.pth")
        os.makedirs(script_dir, exist_ok=True)
        torch.save(trained_model.state_dict(), model_filename)

        # Plot and Save Metrics
        plot_and_save(training_loss, validation_loss, f"Loss - {base}", os.path.join(script_dir, f"{base}_loss.png"))
        plot_and_save(training_acc, validation_acc, f"Accuracy - {base}", os.path.join(script_dir, f"{base}_accuracy.png"))

# ----- FIXED SECTION: Entry Point with Dry-run -----
if __name__ == '__main__':
    dryrun = '--dryrun' in sys.argv
    main(dryrun=dryrun)
""",

    questions = [
        Question("Q1", r""" ** Question **
Write Python code for binary classification maximizing AUC using the code template above.""")
]
)
