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
$$ pp &rarr t t_Bar t t_bar $$. The relevant production processes of the
backgrounds $$ t t_bar + X $$ where $$ X = Z, W^+ and W^+W^- $$. 

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

You may assume that the relevant datasets are already imported as pytorch tensors with the following properties:

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
<start of code template>
import ....

<end of code template>
""",

    questions = [
        Question("Q1", \
                 r""" ** Question **
Write Python code for binary classification maximizing AUC.
"""
                 ),

        Question("Q2", \
                 r"""** Question **
Write Python code for multiclass classification maximizing AUC.
"""
                 ), 
    ]
)
