# generate_scripts.py

# Imports
import os
import requests
import json
import logging
import time
import re

import config
from prompts import *


# API URLs
api_completions = "https://openrouter.ai/api/v1/completions"
api_models = "https://openrouter.ai/api/v1/models"


def query_openrouter(model, prompt):
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",        
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": {"type": "json_object"}, 
        # Look into reasoning / CoT / etc. tokens
        "max_tokens": (8*4096), # change this later
    }
    
    response = requests.post(api_completions, headers=headers, json=payload)
    if response.status_code != 200:
        logging.error(f"Error querying API for model {model}: {response.text}")
        return None
    
    result = response.json()
    if "choices" not in result or not result["choices"]:
        logging.error(f"Unexpected response structure: {result}")
        return None
    
    raw_text = result["choices"][0]["text"]
    logging.debug("Raw response: %s", raw_text)
    
    try:
        parsed = robust_parse(raw_text)
    except ValueError as e:
        logging.error("Parsing failed: %s", e)
        return None
    
    code_response = parsed.get("code", "")
    if not code_response:
        logging.error("The parsed response did not contain a 'code' key.")
        return None
    explanation_response = parsed.get("explanation", "")
    if not explanation_response:
        logging.error("The parsed response did not contain an 'explanation' key.")
        return None
    
    # Implement this a little bit later
    """
    reasoning_response = parsed.get("reasoning", "")
    if not reasoning_response:
            logging.error("The parsed response did not contain a 'reasoning' key.")
    """

    return {"code": code_response.strip(), "explanation": explanation_response.strip()} #, "reasoning": reasoning.strip()}


def robust_parse(raw: str) -> dict:
    """
    Attempts to parse a raw LLM response into a dictionary.
    First, it tries json.loads() directly.
    If that fails, it applies parse_response() to clean the raw text and then tries again.
    """
    try:
        parsed = json.loads(raw)
        logging.info("Successfully parsed raw response as JSON.")
        return parsed
    except json.JSONDecodeError as e:
        logging.error("Initial JSON decoding failed: %s", e)
        # Use your parser to clean the raw response.
        cleaned = parse_response(raw)
        try:
            parsed = json.loads(cleaned)
            logging.info("Successfully parsed cleaned response as JSON.")
            return parsed
        except json.JSONDecodeError as e2:
            logging.error("Failed to parse cleaned response as JSON: %s", e2)
            raise ValueError("Could not parse response from LLM.")


def parse_response(raw: str) -> str:
    """
    Extracts a JSON-like block from the raw LLM output (i.e. deletes everything before
    the first '{' and after the last '}'). Then, if a 'code' field is present, it cleans
    its content by removing markdown code fences and extra surrounding quotes.
    
    Returns the cleaned JSON-like block as a string.
    """
   
    first_index = raw.find('{')
    last_index = raw.rfind('}')
    if first_index == -1 or last_index == -1 or first_index > last_index:
        logging.error("Could not locate valid dictionary boundaries in the input.")
    
    json_block = raw[first_index:last_index+1]
    logging.debug("Extracted JSON block: %s", json_block)
    
    def clean_code(code_str: str) -> str:
        logging.debug("Original code field content: %s", code_str)
        code_str = re.sub(r'^```(?:python|markdown|json)?\s*', '', code_str, flags=re.IGNORECASE)
        logging.debug("After removing leading markdown fences: %s", code_str)
        code_str = re.sub(r'\s*```$', '', code_str)
        logging.debug("After removing trailing markdown fences: %s", code_str)
        code_str = code_str.strip()
        logging.debug("After stripping whitespace: %s", code_str)

        if (code_str.startswith('"') and code_str.endswith('"')) or (code_str.startswith("'") and code_str.endswith("'")):
            code_str = code_str[1:-1].strip()
            logging.debug("After removing surrounding quotes: %s", code_str)
        return code_str

    pattern = r"(?P<prefix>'code'\s*:\s*)(?P<quote>['\"])(?P<content>.*?)(?P=quote)"
    
    def repl(match):
        original_content = match.group('content')
        logging.debug("Found a 'code' field with content: %s", original_content)
        cleaned = clean_code(original_content)
        logging.debug("Cleaned 'code' field content: %s", cleaned)
        return f"{match.group('prefix')}{match.group('quote')}{cleaned}{match.group('quote')}"
    
    json_block = re.sub(pattern, repl, json_block, flags=re.DOTALL)
    logging.info("Final cleaned JSON block: %s", json_block)

    return json_block

def save_response_legacy(model, i, code_response, explanation_response, question, data_loading_script=""):
    """
    Refactor this for new utilities
    """
    current_time = time.strftime("%d%m%H%M")
    day_month = time.strftime("%d-%m")

    # Set-up Output Directory
    output_dir = f"./outputs/{day_month}/"
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    safe_model = model.replace("/", "_")

    # Load possible data loading script
    if os.path.exists(data_loading_script):
        with open(data_loading_script, "r", encoding="utf-8") as f:
            data_loading_code = f.read()
    else:
        data_loading_code = ""
        logging.warning(f"{data_loading_script} not found. No data-loading code will be appended.")

    # Save code
    if data_loading_script:
        filename_code = f"{output_dir}/script_{question}_{safe_model}_{current_time}_{i+1}_a.py"     
    else:
        filename_code = f"{output_dir}/script_{question}_{safe_model}_{current_time}_{i+1}.py"

    with open(filename_code, "w", encoding="utf-8") as f:
        f.write(data_loading_code)
        f.write(code_response)
    logging.info(f"Saved generated code to {filename_code}.")

    # Save text
    filename_txt = f"{output_dir}/explanations/explanation_{question}_{safe_model}_{current_time}_{i+1}.txt"
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(explanation_response)
    logging.info(f"Saved explanation to {filename_txt}.")

    return filename_code

def save_response_legacy(model, i, code_response, explanation_response, question, data_loading_script=""):
    current_time = time.strftime("%d%m%H%M")
    day_month = time.strftime("%d-%m")

    # Set-up Output Directory
    output_dir = f"./outputs/{day_month}/"
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    safe_model = model.replace("/", "_")

    # Load possible data loading script
    if os.path.exists(data_loading_script):
        with open(data_loading_script, "r", encoding="utf-8") as f:
            data_loading_code = f.read()
    else:
        data_loading_code = ""
        logging.warning(f"{data_loading_script} not found. No data-loading code will be appended.")

    # Save code
    if data_loading_script:
        filename_code = f"{output_dir}/script_{question}_{safe_model}_{current_time}_{i+1}_a.py"     
    else:
        filename_code = f"{output_dir}/script_{question}_{safe_model}_{current_time}_{i+1}.py"

    with open(filename_code, "w", encoding="utf-8") as f:
        f.write(data_loading_code)
        f.write(code_response)
    logging.info(f"Saved generated code to {filename_code}.")

    # Save text
    filename_txt = f"{output_dir}/explanations/explanation_{question}_{safe_model}_{current_time}_{i+1}.txt"
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(explanation_response)
    logging.info(f"Saved explanation to {filename_txt}.")

    return filename_code
  

def main():
    # Create a PromptBuilder instance
    prompt_builder = PromptBuilder( problem_description  = fourtop_problem_description,
                                    evaluation_metric   = fourtop_dataset_description,
                                    dataset_description = fourtop_evaluation_metric,
                                    question    =   fourtop_question_1,
                                    context     =   fourtop_context_1)

    # Build the prompt
    prompt = prompt_builder.build_prompt()
    prompt = f"```markdown\n{prompt}\n```"
    logging.info(f"Built Prompt:\n {prompt}")

    # ["openai/chatgpt-4o-latest", "deepseek/deepseek-r1", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-lite-001", "meta-llama/llama-3.3-70b-instruct"]
    models = ["openai/chatgpt-4o-latest", "deepseek/deepseek-r1", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-lite-001", "meta-llama/llama-3.3-70b-instruct"]
    logging.info("Set-Up Models: %s", models)

    num_responses = 2

    logging.info(f"Prompting for {num_responses} responses.")

    for model in models:
        logging.info("Querying model: %s...", model)
        for i in range(num_responses):
            logging.info(f"- - - - - Response {i+1} - - - - -")

            response_dict = query_openrouter(model, prompt)

            if response_dict is None:
                logging.error(f"No valid response for model {model} on iteration {i+1}. Skipping.")
                continue

            code_response = response_dict["code"]
            explanation_response = response_dict["explanation"]       

            logging.info("Generated code from %s:\n%s", model, '-'*40 + "\n" + code_response + "\n" + '-'*40)
            logging.info("Generated explanation from %s:\n%s", model, '-'*40 + "\n" + explanation_response + "\n" + '-'*40)

            _ = save_response(model, i, 
                          code_response, explanation_response, 
                          question = "FOURTOP_1",
                          data_loading_script= "fourtop_data_loader.py")

            time.sleep(1)

if __name__ == "__main__":
    main()