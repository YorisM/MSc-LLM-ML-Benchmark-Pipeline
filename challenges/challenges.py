# ./challenges/challenges.py


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
"""


class Question:
    def __init__(self, question_id, text, context=""):
        self.question_id = question_id
        self.text = text
        self.context = context

class Challenge:
    def __init__(self, name, dataset, problem_description, dataset_description, evaluation_metric, questions, code_template, 
                 instructions = DEFAULT_INSTRUCTIONS, 
                 response_format = DEFAULT_RESPONSE_FORMAT):
        self.name = name
        self.dataset = dataset
        self.problem_description = problem_description
        self.dataset_description = dataset_description
        self.evaluation_metric = evaluation_metric
        self.code_template = code_template
        self.questions = questions
        self.instructions = instructions
        self.response_format = response_format

    def build_prompt(self, question: Question):
        prompt = (
            f"{self.instructions}\n"
            f"{self.problem_description}\n"
            f"{self.evaluation_metric}\n"
            f"{self.dataset_description}\n"
            f"{self.code_template}\n"
            f"{question.text}\n"
            f"{question.context}\n"
            f"{self.response_format}"
        )
        return prompt