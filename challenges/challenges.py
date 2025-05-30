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
You are an expert at programming in python, machine learning,
particle and high energy physics. You will help me answer a question
in a machine learning challenge format where you strive to maximize
a scalar metric in order to learn more about your scientific creativity
and scientific understanding. You will follow all of the instructions 
to your best capabilities. Your first priority is to produce a correct 
solution (runnable code). Your second priority is to do everything
you can to maximize the metric.
""" 

DEFAULT_RUNTIME_CONSTRAINTS = r"""** Runtime Constraints **	
- 4 CPU / 32 GB RAM / 2h wall-clock.
- No internet; importing extra pip packages will fail.
- Keep model < 50 MB.
- Avoid large space searches and extensive hyperparameter tuning.
"""

DEFAULT_RESPONSE_FORMAT = r"""** Response Format **
Your reponse must be in a valid JSON dictionary format with the following structure:
        {
        'code': <Your complete Python code here>,
        'explanation': <A detailed explanation of the code>
        }
Strictly adhere to this structure and follow these rules:

1. Do not include any text or explanations outside the JSON object.
2. Do not add any formatting, such as markdown, to the response. 
3. Properly escape all special characters, especially new lines.
4. After generating the JSON, internally validate its structure before outputting.
5. Replace each "# <LLM: ...>" comment, in the code template, with the required code. 
No placeholder should remain.
6. Before finalizing your answer, double-check that your code runs without errors and
meets all requirements (all functions implemented, correct tensor shapes, etc.).
7. Justify your decisions. Either include them as comments in the code or in the explanation.
"""

DEFAULT_PREFIX = r"""
import os, sys, pickle, torch, gc, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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

def make_loaders(X_train, Y_train, X_val, Y_val, batch=512):
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val , Y_val)
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

    # 6. Write JSON Summary
    summary = {
        "epochs": n_epochs,
        "train_loss": tr_loss,
        "val_loss":   va_loss,
        "train_acc":  tr_acc,
        "val_acc":    va_acc,
        "best_train_loss": min(tr_loss),
        "best_train_loss_epoch": tr_loss.index(min(tr_loss))+1,
        "best_train_acc":  max(tr_acc),
        "best_train_acc_epoch": tr_acc.index(max(tr_acc))+1,
        "best_val_loss": min(va_loss),
        "best_val_loss_epoch": va_loss.index(min(va_loss))+1,
        "best_val_acc":  max(va_acc),
        "best_val_acc_epoch": va_acc.index(max(va_acc))+1,
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
            f"{self.code_template}\n"
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