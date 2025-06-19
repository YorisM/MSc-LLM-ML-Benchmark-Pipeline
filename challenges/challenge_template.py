# ./challenges/challenge_template.py

from challenges.challenges import Challenge, Question

CHALLENGE_NAME_challenge = Challenge(
    name = "",

    dataset = { 

    },

    problem_description = r"""** Problem Description **

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
