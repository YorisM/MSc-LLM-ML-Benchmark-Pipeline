# ./challenges/TRACKFORMERS/trackformers.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
    name = "TRACKFORMERS",

    dataset = { 
        "REDVID_10-50_linear" : {
            "Train" : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_train.pkl.gz",
            "Val"   : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_val.pkl.gz",
            "Test"  : "./challenges/TRACKFORMERS/data/REDVID_10-50_linear_test.pkl.gz"
            }
        },

    problem_description = r"""** Problem Description **
Efficiently reconstructing particle trajectories (tracks) from detector hits is crucial for the performance of
particle physics experiments at colliders like the Large Hadron Collider (LHC). With the significant increase
in data volumes expected in the High-Luminosity LHC era, traditional tracking methods become computationally
expensive and increasingly inefficient. 

The challenge here is to develop a two-step regression network that efficiently handles events with variable-length
hit sequences. Your model should first group hits into distinct track clusters and then accurately predict
track parameters for each identified cluster. Accurate reconstruction of track parameters enables precise
measurements of fundamental particle properties and event reconstructions.
""",

    dataset_description = r"""** Dataset Description **

""",

    evaluation_metric = r"""** Evaluation Metric **

""",

    prefix = r"""
# ----------------  START HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 



# ----------------  END HARNESS WRAPPER PREFIX (FOR CONTEXT)  ---------------- 
# -------------------------- START OF LLM BLOCK ------------------------------

""",

    code_template = r"""** Code Template **

""",

    suffix = r"""
# ---------------------------  END OF LLM-CODE BLOCK ---------------------------
# ----------------  START HARNESS WRAPPER SUFFIX (FOR CONTEXT)  ---------------- 



""",

    questions = [
        Question("Q1", r""" ** Question **
                 """)

]
)
