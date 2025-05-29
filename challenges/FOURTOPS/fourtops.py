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
IMPORTANT: The specific line format of the data set is as follows:

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

Name: X_train, shape: [241657, 92], dtype: torch.float32, 
Name: Y_train, shape: [241657], dtype: torch.int64, 
Name: X_val, shape: [30272, 92], dtype: torch.float32, 
Name: Y_val, shape: [30272], dtype: torch.int64

IMPORTANT: Each of these tensors are pre-loaded and available in the environment.
""",

    evaluation_metric = r"""** Evaluation Metric **
The evaluation metric for this classification task is the area under the curve (AUC)
value, specified by the area under the receiver operating characteristic (ROC) curve. 
The AUC summarizes a model's ability to distinguish between positive and 
negative classes. The higher the score the better.
""",

    code_template = r"""** Code Template **
# 0. ---------- IMPORTS ----------
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# Only import extra std-lib modules, torch or sklearn (sub-)modules you actually use.
# <LLM: Import modules>

# 1. ---------- PRE-PROCESSING ----------
class MyPreprocessor:
    #    Must implement:
    #   - fit(X: torch.Tensor, y: torch.Tensor) -> self
    #   - transform(X: torch.Tensor) -> torch.Tensor

    # REQUIREMENTS
    # IMPORTANT: All state must be picklable with the std-lib pickle module.
    # May allocate NumPy arrays or Torch tensors internally, but:
    # transform() must be deterministic & output a torch.Tensor of shape (N, features).
    # Store only derived parameters needed for transform i.e. do not store the raw data
    # itself in the preprocessor object.

    # DATA SPECIFICS
    # IMPORTANT: X_train, Y_train, X_val, Y_val are provided as PyTorch tensors in the environment.
    # Total flat length per event (X_train & X_val): 92
    # Index  0 :  missing-ET magnitude  (E_T_miss)
    # Index  1 :  missing-ET azimuth    (phi_Et_miss)
    # Indices  2-6  : object 1  ->  obj_1, E_1, p_T1, eta_1, phi_1
    # Indices  7-11 : object 2  -> obj_2, E_2 , p_T_2 , eta_2 , phi_2
    # ...
    # Indices 88-92 : object 18 -> obj_18, E_18 , p_T_18 , eta_18 , phi_18
    # Per-object slice size = 5
    # Max objects encoded   = 18

    # TIPS
    # When modifying data features or feature engineering: annotate tensor size as comments after 
    # each tensor operation to reduce dimension mismatches.

    # <LLM: Write code to preprocess the data>    
    def __init__(self):
        # <LLM: Define and initialize any stateful components here>
        pass

    def fit(self, X, y=None):
        # <LLM: Extract statistics or fit transformers>
        return self

    def transform(self, X):
        # <LLM: Apply preprocessing logic, return torch.Tensor>
        return X 

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def make_preprocessor():
    return MyPreprocessor()

# 2. ---------- MODEL DEFINITION ----------
def make_model(input_dim: int):
    # PARAMETERS
    # inplut_dim : int : Number of features per event after preprocessing.

    # RETURNS
    # model : torch.nn.Module : Untrained binary-classifier network.

    # <LLM: Write code to define a binary-classifier network>
    return model

# 3. ---------- MODEL TRAINING ----------
EPOCHS = # <LLM: define the amount of training epochs>    
def train_model(model, train_loader, val_loader, epochs):
    # PARAMETERS
    # model : torch.nn.Module   
    # train_loader: torch.utils.data.DataLoader
    # val_loader  : torch.utils.data.DataLoader
    # epochs: int

    # RETURNS
    # trained_model : nn.Module          (same instance, trained in-place)
    # train_loss    : list[float]        (length == epochs)
    # val_loss      : list[float]
    # train_acc     : list[float]
    # val_acc       : list[float]
    
    # REQUIREMENTS 
    # Define training loop clearly including number of epochs
    # Do NOT pass "verbose=" to any PyTorch scheduler (not supported in this image).

    # <LLM: Write code to define training loop>
    return trained_model, train_loss, val_loss, train_acc, val_acc

# IMPORTANT: DO NOT write code to run the functions defined above,
# i.e. write code to prepare training data, process-data, traing model, evaluate model performance, etc.
""",

    questions = [
        Question("Q1", r""" ** IMPORTANT: Your Challenge **
Write Python code for a binary classification model focussing on maximizing the AUC using the code template above.
You may freely choose any pre-processing methods and techniques as well as model architecture and training conventions.                 
"""),

       Question("Q2", r"""** IMPORTANT: Your Challenge **
Write Python code for a binary classification model which both explicitly encodes Lorentz symmetry via tensor products and equivariant message passing. 
You may freely choose model architecture and training conventions. Focus on maximizing the AUC using the code template above.
"""),

       Question("Q3", r"""** IMPORTANT: Your Challenge **
Write Python code for a Transformer based binary classifier which utilizes a "Slot-Attention" mechanism that explicitly groups particles corresponding to top quark decays.
Inform the model of known physics by creating augmented particle features which complement model architecture. You may freely choose training conventions. Focus on maximizing the AUC using the code template above.
""")
]
)