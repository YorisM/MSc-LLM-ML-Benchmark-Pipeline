# ./challenges/FOURTOPS/fourtops.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
    name = "FOURTOPS",

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

E_T_miss, phi_{E_t}_miss, obj_1, E_1, p_T1, eta_1, phi_1, obj_2, E_2, p_t2, eta_2, phi_2, ....

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

Name: X_train, shape: [241657, 105], dtype: torch.float32, 
Name: Y_train, shape: [241657], dtype: torch.int64, 
Name: X_val, shape: [30272, 105], dtype: torch.float32, 
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <LLM: transform x using the stored buffers>
        return x

def preprocess_data(X_train, Y_train, X_val, Y_val, batch_size):
    # derive statistics / encodings from training set
    # <LLM: derive any needed constants here>

    preproc = PreprocessModule()

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
    train_loader, val_loader, preproc = preprocess_data(X_train, Y_train, X_val, Y_val)

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
""",

    questions = [
#        Question("Q1", r""" ** Your Challenge **
# Write Python code for binary classification maximizing AUC using the code template above.""")

       Question("Q2", r"""** Your Challenge **)
Write Python code for a neural network which both explicitly encodes Lorentz symmetry via tensor products and equivariant message passing. 
Maximize the AUC using the code template above.
"""),

       Question("Q3", r"""** QYour Challenge **)
Write Python code for a Transformer based binary classifier which utilizes a "Slot-Attention" mechanism that explicitly groups particles corresponding to top quark decays.
Inform the model of known physics by creating augmented particle features which complement model architecture. Use the code template above.
"""),
]
)
