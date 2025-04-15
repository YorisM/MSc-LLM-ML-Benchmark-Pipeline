# prompts.py

# region - - - - - Imports - - - - - 
import os
import sys
# endregion


# region General Prompting Structure
# - - - - - - - - - - - - - - - - - - - - 
# Instructions
#   Problem Description
#   Evaluation Metric
#   Dataset Description
#       Question
#       Context
# Response Format
#endregion - - - - - - - - - - - - - - - - - - - - 


DEFAULT_INSTRUCTIONS = r"""** Instructions **
You are an expert at programming in python, machine learning,
particle and high energy physics. You will help me answer a question
in a challenge format where you strive to maximize a scalar metric in 
order to learn more about your scientific creativity and scientific 
understanding. You will follow all of the instructions to your best
capabilities.

"""

DEFAULT_RESPONSE_FORMAT = r"""** Response Format **
Your reponse must be in a valid JSON dictionary format with the following structure:
        {
        'language': Python, 
        'code': <Your complete Python code here>,
        'explanation': <A detailed explanation of the code>
        }
Strictly adhere to this structure and follow these rules:

1. Do not include any text or explanations outside the JSON object.
2. Do not add any formatting, such as markdown, to the response. 
3. Properly escape all special characters, especially new lines.
4. After generating the JSON, internally validate its structure before outputting.

The code you write must strictly limit the packages and libraries 
that are not in the python standard library that you use in your solution 
to the following: numpy, scipy, math, pandas, matplotlib, sklearn, pytorch.

The code you write must include a dry run (num_epochs = 1) functionality that may be used
with the following command: "python <filename>.py --dryrun". 

"""


class PromptBuilder:
    def __init__(self, problem_description, evaluation_metric, 
                dataset_description, question, 
                context         = "", 
                instructions    =DEFAULT_INSTRUCTIONS, 
                response_format =DEFAULT_RESPONSE_FORMAT):
        self.instructions = instructions
        self.problem_description = problem_description
        self.evaluation_metric = evaluation_metric
        self.dataset_description = dataset_description
        self.question = question
        self.context = context
        self.response_format = response_format

    def build_prompt(self):
        prompt = (
            f"{self.instructions}\n"
            f"{self.problem_description}\n"
            f"{self.evaluation_metric}\n"
            f"{self.dataset_description}\n"
            f"{self.question}\n"
            f"{self.context}\n"
            f"{self.response_format}"
        )
        return prompt


# region - - - - - FOURTOP PROMPTS - - - - - 
fourtop_problem_description = r"""** Problem Description **
A major task in particle physics is the measurement of rare signal 
processes ith very small cross-sections. With the unprecedented amount of 
data provided by the upcoming runs of the Large Hadron Collider (LHC), 
one can start to measure these processes. An example is the recent 
observation of four top quarks originating from a single proton-proton 
collision event. Accurate classification of these events is crucial, 
as even a small reduction in background noise on the order of a few tens 
of percent while maintaining the same signal detection efficiency can lead 
to a profound increase in sensitivity.

"""

fourtop_dataset_description = r"""** Dataset Description **
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

"""


fourtop_evaluation_metric = r"""** Evaluation Metric **
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


"""

fourtop_context_1 = r"""
This is some context.

"""

fourtop_question_1 = r"""** Task **
Write python code for a binary classifier that maximizes the AUC.

"""

fourtop_question_2 = r"""** Task **
Write python code for a binary classifier that maximizes the AUC.
Consider the following steps in your pipeline: 

1. data pre-processing, 
2. model architecture,
3. model training,
4. the integration of known physics.

Reason if other components or steps need to be incorporated to maximize the AUC.

"""

fourtop_question_3 = r"""** Task **
Write python code for a transformer based binary classifier that maximizes the AUC. 
Integrate known physics by implementing pairwise kinematic features that adhere to sub(symmetries).

"""
#end region


# region - - - - - Trackformers - - - - -

trackformers_problem_description = r"""** Problem Description **

"""

trackformers_evaluation_metric = r"""** Evaluation Metric **

"""

trackformers_context_1 = r"""
This is some context.

"""

trackformers_question_1 = r"""** Task **

"""

trackformers_context_2 = r"""
This is some context.

"""

trackformers_question_2 = r"""** Task **

"""

# region - - - - - Monte Carlo Simulation - - - - - 
mc_problem_description = r"""** Problem Description **
Monte Carlo sampling can be used to simulate collisions at a collider. 
"""

mc_evaluation_metric = r"""** Evaluation Metric **

"""

mc_context_1 = r"""
This is some context.

"""

mc_question_1 = r"""** Task **

"""