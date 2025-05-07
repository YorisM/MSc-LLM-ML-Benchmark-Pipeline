# generate_and_dryrun.py

# Imports
import os, requests, json, logging, shutil, time, re, sys, subprocess
from config import OPENROUTER_API_KEY, OPENROUTER_API_COMPLETIONS, DOCKER_IMAGE, models, num_attempts, dryrun_timeout, challenges
from run_scripts import run_single_script

def query_openrouter(model, prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",        
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": {"type": "json_object"}, 
        # Look into reasoning / CoT / etc. tokens
        "max_tokens": (16*4096), # change this later
    }
    
    response = requests.post(OPENROUTER_API_COMPLETIONS, headers=headers, json=payload)
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

def save_response(output_dir, model, i, code_response, explanation_response):
    current_time = time.strftime("%H%M")
    safe_model = model.replace("/", "_")
    
    # Build the filenames
    base_name = f"{safe_model}_{current_time}_{i}"
    filename_code = os.path.join(output_dir, f"script_{base_name}.py")
    filename_txt  = os.path.join(output_dir, f"explanation_{base_name}.txt")
    
    # Save the code into the .py file.
    with open(filename_code, "w", encoding="utf-8") as f:
        f.write(code_response)
    logging.info("Saved generated code to %s", filename_code)
    
    # Save the explanation into the .txt file
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(explanation_response)
    logging.info("Saved explanation to %s", filename_txt)
    
    return filename_code, filename_txt

def move_file(file_path, destination_dir):
    """Move a single file to the destination directory."""
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, os.path.basename(file_path))
    try:
        shutil.move(file_path, destination)
        logging.info("Moved file %s to %s", file_path, destination_dir)
    except Exception as e:
        logging.error("Failed to move file %s: %s", file_path, e)

def script_dryrun(script_path):
    """dry-run a script *inside Docker* and return (success, stdout, stderr)"""
    _, success, _, stdout, stderr, _, _ = run_single_script(
        script_path,
        dryrun=True,
        use_docker=True
    )
    return success, stdout, stderr
    
def generate_and_dryrun():
    for challenge in challenges:
        logging.info(f"Executing challenge: {challenge.name}\n")

        for question in challenge.questions:
            prompt = challenge.build_prompt(question)
            prompt = f"```markdown\n{prompt}\n```"
            logging.info(f"Built Prompt:\n {prompt}")

            day_month = time.strftime("%d-%m")
            output_dir = f"./outputs/{day_month}/{challenge.name}/{question.question_id}/"
            os.makedirs(output_dir, exist_ok=True)
            logging.info("Set-Up Output Directory: %s", output_dir)

            for model in models:
                safe_model = model.replace("/", "_")

                for attempt in range(1, num_attempts + 1):
                    logging.info("Querying model %s: Attempt %d", model, attempt)
                    response = query_openrouter(model, prompt)

                    if response is None:
                        logging.error("Model %s: No response on attempt %d", model, attempt)
                        continue

                    code = response.get("code", "")
                    explanation = response.get("explanation", "")
                    if not code or not explanation:
                        logging.error("Model %s: Missing code/explanation on attempt %d", model, attempt)
                        continue

                    py_file, txt_file = save_response(output_dir, safe_model, attempt, code, explanation)

                    run_success, stdout, stderr = script_dryrun(py_file)
                    if run_success:
                        dest_folder = os.path.join(output_dir, safe_model)
                        move_file(py_file, dest_folder)
                        move_file(txt_file, dest_folder)
                        break
                    else:
                        logging.error(f"Model {model} failed dry-run on attempt {attempt}.\n\
                                      STDOUT: {stdout}\n\
                                      STDERR: {stderr}")
                        failed_folder = os.path.join(output_dir, "Failed Dry-run Scripts")
                        move_file(py_file, failed_folder)
                        move_file(txt_file, failed_folder)

                if attempt == num_attempts and not run_success:
                    logging.error("Model %s failed to produce a runnable script after %d attempts", model, num_attempts)

if __name__ == "__main__":
    generate_and_dryrun()