# generate_and_dryrun.py

# Imports
import os, requests, json, logging, shutil, time, re
from pathlib import Path

from config import MAX_TOKENS, OPENROUTER_API_KEY, OPENROUTER_API_COMPLETIONS, models, num_attempts, challenges
from static_checks import run_pylint, run_bandit
from run_scripts import run_single_script
from utils import append_to_response_json

from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException  

BACKOFF = (10, 25, 60)          # seconds on retries

_session = requests.Session()
_retry = Retry(
    total=5,                                    # overall cap
    connect=3,                                  # dial failures
    read=3,                                     # mid-stream failures
    status=3,                                   # 502/503/504 etc.
    status_forcelist=(502, 503, 504, 429),
    backoff_factor=1.0,                         # exponential: 0 -> 1 -> 2 -> 4 ...
    allowed_methods=("POST",),
    raise_on_status=False,
)

_session.mount("https://", HTTPAdapter(max_retries=_retry))

def query_openrouter(model: str, prompt: str, *, max_retries = 3):
    """
    Portal to OpenRouter API.

    RETURNS
    The parsed JSON LLM response or raises on final failure.
    """
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",        
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "response_format": {"type": "json_object"}, 
        "max_tokens": (MAX_TOKENS),
    }

    # Request Timer (since Gemini is slow... and other connectivity issues wwere risen)
    for attempt in range(max_retries):
        timeout = min(60 * 2**attempt, 240)             # 60 -> 120 -> 240 (s)
        try:
            r = _session.post(
                    OPENROUTER_API_COMPLETIONS,
                    headers = headers, 
                    json    = payload,
                    timeout = timeout,
                )

            # Cloudflare idle timeout
            if r.status_code == 524:
                raise requests.Timeout("524 edge timeout")

            r.raise_for_status()                        # HTTP 4xx/5xx → error
            result = r.json()

            # Provider-side error block
            if "error" in result:
                raise requests.Timeout(
                    f"LLM error {result['error'].get('code')}"
                )

            # Missing Choices also treated as retryable
            if "choices" not in result or not result["choices"]:
                logging.error(f"No 'choices' field in JSON on attempt {attempt+1}, retrying...")
                raise requests.Timeout("No 'choices' field in JSON")

        except (RequestException, ValueError) as e:
            logging.warning(f"LLM query attempt {attempt + 1}/{max_retries} for {model} failed: {e}")
            if attempt == max_retries - 1: return None
            time.sleep(BACKOFF[min(attempt, len(BACKOFF)-1)])

        # SUCCESSFUL RESPONSE - Still need to parse
        raw_text = result["choices"][0]["text"]
        try:    
            parsed   = robust_parse(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            logging.warning("Parse failure on attempt %d: %s", attempt+1, e)
            if attempt == max_retries - 1:
                return None
            continue

        code        = parsed.get("code", "").strip()
        explanation = parsed.get("explanation", "").strip()
        if not code or not explanation:
            raise requests.Timeout("Parsed JSON lacks code/explanation")

        response_txt = {
            "model": model,
            "usage": result.get("usage", {}),
            "prompt_chars": len(prompt),
            "code":  code,
            "explanation": explanation,
                "reasoning": parsed.get("reasoning"),
        }
        return response_txt

def robust_parse(raw: str) -> dict:
    """
    Return a dictionary parsed from an LLM response.
    Accepts plain JSON or Markdown-wrapped JSON, and tolerates
    a single-element list wrapper like '[ {...} ]'.
    Raises ValueError when we can't recover a dict.
    """

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logging.warning("Initial JSON decoding failed: %s", e)
        cleaned = parse_response(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logging.error("Failed to parse cleaned response as JSON: %s", e2)
            raise ValueError("Could not parse response from LLM.")
        
    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
            logging.info("Unwrapped single-item JSON list.")
        else:
            raise ValueError("Expected dict or [dict]; got list.")
    elif not isinstance(parsed, dict):
        raise ValueError(f"Expected dict, got {type(parsed).__name__}")
        
    logging.info("Successfully parsed raw response as JSON.")
    return parsed

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

def save_response(output_dir: str,
                  model: str,
                  attempt: int,
                  script_text: str,
                  response_blob: dict):
    """
    Writes two files:
      • script_...py  – runnable script
      • response_...json – full LLM JSON incl. explanation / usage
    """
    ts   = time.strftime("%H%M")
    safe = model.replace("/", "_")
    base = f"{safe}_{ts}_{attempt}"

    py_path  = os.path.join(output_dir, f"script_{base}.py")
    json_path = os.path.join(output_dir, f"response_{base}.json")

    Path(py_path).write_text(script_text,  encoding="utf-8")
    Path(json_path).write_text(json.dumps(response_blob,
                                          ensure_ascii=False,
                                          indent=2), encoding="utf-8")

    logging.info("Saved script → %s  and JSON → %s", py_path, json_path)
    return py_path, json_path

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
        logging.info(f"Executing challenge: {challenge.name}")

        for question in challenge.questions:
            day_month = time.strftime("%d-%m")
            output_dir = f"./outputs/{day_month}/{challenge.name}/{question.question_id}/"
            os.makedirs(output_dir, exist_ok=True)
            logging.info("Set-Up Output Directory: %s", output_dir)

            prompt = challenge.build_prompt(question)
            prompt = f"```markdown\n{prompt}\n```"
            logging.info("Prompt: %s", prompt)

            for model in models:
                safe_model = model.replace("/", "_")
                
                run_success = False
                for attempt in range(1, num_attempts + 1):
                    logging.info("Querying model %s: Attempt %d", model, attempt)
                    response = query_openrouter(model, prompt)

                    if response is None:
                        logging.error("Model %s: No response on attempt %d", model, attempt)
                        continue
                    
                    code         = response["code"]
                    explanation  = response["explanation"]
                    response_blob = {**response, "prompt_chars": len(prompt)}

                    if not code or not explanation:
                        logging.error("Model %s: Missing code/explanation on attempt %d", model, attempt)
                        continue

                    # Concatenate LLM response with our own code and save to runnable .py
                    script = challenge.build_script(code)
                    py_file, json_file = save_response(
                        output_dir, safe_model, attempt, script, response_blob)

                    # Set-Up PyLint & Bandit static checks\
                    pylint_ok, pylint_report = run_pylint(py_file)
                    bandit_ok, bandit_report = run_bandit(py_file)

                    # Append reports to JSON
                    with open(json_file, encoding="utf-8") as fh:
                        resp_data = json.load(fh)

                    resp_data["PyLint"] = pylint_report
                    resp_data["Bandit"] = bandit_report

                    with open(json_file, "w", encoding="utf-8") as fh:
                        json.dump(resp_data, fh, ensure_ascii=False, indent=2)

                    # PyLint & Bandit Static Checks
                    if not pylint_ok and bandit_ok:
                        failed_folder = os.path.join(output_dir, "StaticFail")
                        move_file(py_file, failed_folder)
                        move_file(json_file, failed_folder)
                        continue
                        
                    # Perform Dry-Run
                    run_success, stdout, stderr = script_dryrun(py_file)


                    append_to_response_json(
                        json_file,
                        "DryRun",
                        {
                            "passed"      : bool(run_success),
                            "stdout"      : stdout[-5000:],   # keep it short – last 5 000 chars
                            "stderr"      : stderr[-5000:],
                            "return_code" : 0 if run_success else 1
                        }
                    )

                    if run_success:
                        logging.info("Model %s passed dry-run on attempt %d", model, attempt)
                        dest_folder = os.path.join(output_dir, safe_model)
                        move_file(py_file, dest_folder)
                        move_file(json_file, dest_folder)
                        break
                    else:
                        logging.error(f"Model {model} failed dry-run on attempt {attempt}.\n\
                                      STDOUT: {stdout}\n\
                                      STDERR: {stderr}")
                        failed_folder = os.path.join(output_dir, "Failed Dry-run Scripts")
                        move_file(py_file, failed_folder)
                        move_file(json_file, failed_folder)

                if attempt == num_attempts and not run_success:
                    logging.error("Model %s failed to produce a runnable script after %d attempts", model, num_attempts)

if __name__ == "__main__":
    generate_and_dryrun()