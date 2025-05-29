# ./challenges/challenge_template.py

from challenges.challenges import Challenge, Question

fourtop_challenge = Challenge(
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
# ----------------  START OF LLM BLOCK  ----------------

""",

    code_template = r"""** Code Template **

""",

    suffix = r"""
# ----------------  END OF LLM BLOCK ----------------

""",

    questions = [
        Question("Q1", r""" ** Question **
                 """)

]
)
