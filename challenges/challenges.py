# ./challenges/challenges.py

import json

# General Prompting Structure
# - - - - - - - - - - - - - - - - - - - - 
# Instructions
#   Problem Description
#   Evaluation Metric
#   Dataset Description
#   Runtime Constraints
#   Code Template
#       Question
#       Context
# Response Format


# General Script Structure
# - - - - - - - - - - - - - - - - - - - - 
# Prefix = Config, data loading
# LLM Response = data pre-processing, model definition, training
# Suffix = preparing outputs, main function and running the model


DEFAULT_INSTRUCTIONS = r"""** Instructions **
You are an expert at programming in Python, machine learning,
particle and high energy physics. You will help me answer a question
in a machine learning challenge format where you strive to maximise
a scalar metric in order to learn more about your scientific creativity
and scientific understanding. You will follow all of the instructions 
to your best capabilities. Your first priority is to produce a correct 
solution in terms of runnable python code. Your second priority is to 
do everything you can to maximise the metric defined below.
""" 

DEFAULT_RUNTIME_CONSTRAINTS = r"""** Runtime Constraints **	
/
"""

DEFAULT_RESPONSE_FORMAT = r"""** Response Format **
Your response must strictly be python code. 
If you must wrap it, put it in a ```python fenced block and nothing else.
Your response must follow these rules:

1. Do not add any formatting, such as markdown, to the response. 
2. Replace each "# <LLM: ...>" comment, in the code template, with the required code. 
No placeholder should remain.
3. Before finalizing your answer, double-check that your code runs without errors and
meets all requirements (all functions implemented, correct tensor shapes, etc.).
4. To prevent dimensional mismatches make sure to annotate tensor sizes as comments.
5. IMPORTANT: Remember, your first, and most important priority is to produce 
(syntactically) correct code. Prioritise what you can implement reliably above all else. 
Then prioritise maximising the metric.
"""

DEFAULT_PREFIX = r"""
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

torch.manual_seed(42)                        
os.environ["PYTHONHASHSEED"] = "42"
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

DATASET = {dataset_dict}
                       
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

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if isinstance(self.x, (tuple, list)):
            return (tuple(t[idx] for t in self.x), self.y[idx])
        else:
            return (self.x[idx], self.y[idx])      

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = PairDataset(X_train, Y_train)
    val_ds   = PairDataset(X_val , Y_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0))
                        
# ----------------  START OF LLM BLOCK  ----------------
"""

DEFAULT_SUFFIX = r"""
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
    pre = make_preprocessor().fit(X_train, Y_train)
    X_train = pre.transform(X_train) # may be Tensor or Tuple
    X_val   = pre.transform(X_val)
    train_loader, val_loader = make_loaders(X_train, Y_train, X_val, Y_val)

    # 2. Build model
    if isinstance(X_train, torch.Tensor):               # single-tensor case
        temp_ref    = X_train
        input_shape = temp_ref.shape[1:]                # e.g. (F,)
        use_mask    = False
    else:                                               # tuple => (data, mask)
        temp_ref    = X_train
        input_shape = temp_ref[0].shape[1:]             # e.g. (L, F)
        use_mask    = True                              
    model = make_model(input_shape, use_mask=use_mask)

    # 3. Train model
    n_epochs = 1 if dryrun else globals().get("EPOCHS", 10)
    try:
        trained_model, tr_loss, va_loss, tr_acc, va_acc = train_model(
            model, train_loader, val_loader, epochs=n_epochs)
    except Exception as e:
        print("ERROR during training:", e)
        raise

    # 4. *Dry-run safety check* – run a single toy forward pass
    if dryrun:
        toy_data = torch.zeros(8, *input_shape, dtype=torch.float32)
        if use_mask:
            toy_mask = torch.zeros(8, input_shape[0], dtype=torch.bool)
            toy_batch = (toy_data, toy_mask)
        else:
            toy_batch = toy_data

        toy_transformed = pre.transform(toy_batch)
        try:
            _ = trained_model(*toy_transformed) if isinstance(toy_transformed, (tuple, list)) \
                else trained_model(toy_transformed)
        except Exception as e:
            raise RuntimeError("Sanity-check forward pass failed") from e
        return

    # 5. Persist artefacts
    base = os.path.splitext(os.path.basename(sys.argv[0]))[0].removeprefix("script_")

    pth_state   = os.path.join(SCRIPT_DIR, f"{base}_state.pt")
    pth_model   = os.path.join(SCRIPT_DIR, f"{base}_model.pkl")
    pth_preproc = os.path.join(SCRIPT_DIR, f"{base}_preproc.pkl")

    torch.save(trained_model.state_dict(), pth_state)
    with open(pth_model,   "wb") as f: pickle.dump(trained_model, f)
    with open(pth_preproc, "wb") as f: pickle.dump(pre,           f)

    # 6. Save plots
    _plot(tr_loss, va_loss, "Loss",     os.path.join(SCRIPT_DIR, f"{base}_loss.png"))
    _plot(tr_acc,  va_acc,  "Accuracy", os.path.join(SCRIPT_DIR, f"{base}_accuracy.png"))

    # 7. Write JSON Summary
    if not dryrun: 
        summary = {
            "epochs": n_epochs,
            "train_loss": tr_loss   if tr_loss else None,
            "val_loss":   va_loss   if va_loss else None,
            "train_acc":  tr_acc    if tr_acc else None,
            "val_acc":    va_acc    if va_acc else None,
        }
        print("#TRAIN_METRICS#" + json.dumps(summary))

if "__main__" not in sys.modules:
    sys.modules["__main__"] = sys.modules[__name__]

if __name__ == "__main__":
    _run(dryrun="--dryrun" in sys.argv)
"""


class Question:
    def __init__(self, question_id, text, context=""):
        self.question_id = question_id
        self.text = text
        self.context = context

class Challenge:
    def __init__(self, name, dataset, problem_description, dataset_description, 
                 evaluation_metric, questions, code_template,
                 instructions = DEFAULT_INSTRUCTIONS,
                 runtime_constraints = DEFAULT_RUNTIME_CONSTRAINTS, 
                 response_format = DEFAULT_RESPONSE_FORMAT,
                 prefix = DEFAULT_PREFIX, suffix = DEFAULT_SUFFIX):
        
        self.name = name
        self.dataset = dataset
        self.problem_description = problem_description
        self.dataset_description = dataset_description
        self.evaluation_metric = evaluation_metric
        self.runtime_constraints = runtime_constraints
        self.code_template = code_template
        self.questions = questions
        self.instructions = instructions
        self.response_format = response_format
        self.prefix = prefix
        self.suffix = suffix

    def build_prompt(self, question: Question):
        prompt = (
            f"{self.instructions}\n"
            f"{self.problem_description}\n"
            f"{self.evaluation_metric}\n"
            f"{self.dataset_description}\n"
            # f"{self.runtime_constraints}\n"
            f"{self.prefix}\n"
            f"{self.code_template}\n"
            f"{self.suffix}"
            f"{question.text}\n"
            f"{question.context}\n"
            f"{self.response_format}"
        )
        return prompt
    
    def build_script(self, llm_code: str):
        prefix = self.prefix.format(dataset_dict=json.dumps(self.dataset, indent=4))
        script = "\n".join([
            f"{prefix.rstrip()}\n",
            f"{llm_code.rstrip()}\n",
            f"{self.suffix.lstrip()}\n",
        ])
        return script