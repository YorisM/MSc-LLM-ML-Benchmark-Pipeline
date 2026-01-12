# generate_and_dryrun.py

# Imports
import os, requests, json, logging, shutil, time, re, textwrap
import config
from pathlib import Path
from tqdm import tqdm

from static_checks import run_pylint, run_bandit
from run_scripts import run_single_script
from utils.utils import append_to_response_json
from utils.run_id import get_or_create_run_id

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


def _safe_model_id(model: str) -> str:
    # Replace anything that can break Windows paths
    s = model.replace("/", "_").replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)  # conservative
    return s

def query_openrouter(model: str, prompt: str, *, max_retries = 3):
    """
    Portal to OpenRouter API.

    RETURNS
    A dict with LLM metadata + extracted code, or None on final failure.
    """
    
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",        
    }

    def _do_request(endpoint: str, payload: dict, mode: str) -> dict | None:
        """
        Internal helper to query a single endpoint (chat or completions)
        with retries and robust response parsing.

        RETURNS
        llm_block dict on success, or None if no usable code after retries.
        """

        for inner_attempt in range(max_retries):
            timeout = min(60 * 2**inner_attempt, 240)  # 60 -> 120 -> 240 (s)
            t0 = time.perf_counter()

            try:
                r = _session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

                # Cloudflare idle timeout
                if r.status_code == 524:
                    raise requests.Timeout("524 edge timeout")

                r.raise_for_status()
                result = r.json()
                duration_ms = round(1000 * (time.perf_counter() - t0))

                # Provider-side error
                if "error" in result:
                    raise requests.Timeout(
                        f"LLM error {result['error'].get('code')}"
                    )

                # Must have at least one choice
                if "choices" not in result or not result["choices"]:
                    logging.error(
                        "[%s] No 'choices' field in JSON on inner_attempt %d, retrying...",
                        mode, inner_attempt + 1
                    )
                    raise requests.Timeout("No 'choices' field in JSON")

            except (RequestException, ValueError) as e:
                logging.warning(
                    "LLM query inner_attempt %d/%d for %s via %s failed: %s",
                    inner_attempt + 1, max_retries, model, mode, e
                )
                if isinstance(e, requests.HTTPError) and e.response is not None:
                    logging.error("HTTP %s body: %s", e.response.status_code, e.response.text)
                if inner_attempt == max_retries - 1:
                    return None
                time.sleep(BACKOFF[min(inner_attempt, len(BACKOFF) - 1)])
                continue

            # ---- SUCCESSFUL HTTP + JSON; now parse ----
            choice = result["choices"][0]
            usage  = result.get("usage", {})

            # 1) Start from any explicit 'reasoning' field on the choice (if present).
            reasoning = ""
            choice_reasoning = choice.get("reasoning")
            if isinstance(choice_reasoning, str):
                reasoning = choice_reasoning

            # 2) Extract assistant content (chat-style or completion-style).
            content = ""

            # Prefer chat-style 'message'
            raw_message = choice.get("message")
            if isinstance(raw_message, dict) and "content" in raw_message:
                raw_content = raw_message["content"]

                if isinstance(raw_content, str):
                    # Simple case: single string
                    content = raw_content

                elif isinstance(raw_content, list):
                    # Newer format: list of blocks, e.g. [{"type": "reasoning", "text": ...}, ...]
                    text_chunks = []
                    reasoning_chunks = []

                    for part in raw_content:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        text = part.get("text", "")

                        # Heuristic: reasoning vs final answer
                        if part_type in ("reasoning", "chain_of_thought"):
                            reasoning_chunks.append(text)
                        else:  # "output_text", "text", None, etc.
                            text_chunks.append(text)

                    content = "\n".join(ch for ch in text_chunks if ch).strip()

                    if not reasoning and reasoning_chunks:
                        reasoning = "\n\n".join(ch for ch in reasoning_chunks if ch).strip()

            # Fallback: completion-style 'text'
            if not content:
                raw_text = choice.get("text")
                if isinstance(raw_text, str):
                    content = raw_text

            # Last resort: if still empty, just string-ify whatever we got
            if not content:
                content = str(choice)

            # ---- Extract code from content ----
            code = extract_code(content)

            if not code:
                logging.warning("[%s] No code found in LLM response", mode)
                logging.info("LLM raw content: %s", content)
                if inner_attempt == max_retries - 1:
                    return None
                # Treat "no code" as retry-able within the same endpoint
                time.sleep(BACKOFF[min(inner_attempt, len(BACKOFF) - 1)])
                continue

            # ---- Build final response blob ----
            llm_block = {
                "model": model,
                "endpoint": endpoint,
                "mode": mode,  # "chat" or "completions"
                "prompt_tokens": usage.get("prompt_tokens"),
                "prompt_chars": len(prompt),
                "response_tokens": usage.get("completion_tokens"),
                "response_chars": len(content),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "reasoning_chars": len(reasoning),
                "Inner attempt": inner_attempt + 1,
                "cost_usd": usage.get("cost"),
                "generation_ms": duration_ms,
                "code": code,
                "reasoning": reasoning,
            }

            return llm_block  # success
        return None  # defensive, should not get here

        # --------------------
    # 1) Try CHAT endpoint
    # --------------------
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "usage": {"include": True},
        "max_tokens": config.MAX_TOKENS,
        # Many models ignore this; reasoning models honour it.
        "reasoning": {"exclude": False, "effort": "high"},
    }

    chat_block = _do_request(
        endpoint=config.OPENROUTER_API_CHAT,
        payload=chat_payload,
        mode="chat",
    )
    if chat_block is not None:
        return chat_block

    logging.warning(
        "Chat endpoint failed to produce usable code for %s; falling back to completions.",
        model,
    )

    # ---------------------------
    # 2) Fallback: COMPLETIONS
    # ---------------------------
    completions_payload = {
        "model": model,
        "prompt": prompt,
        "usage": {"include": True},
        "max_tokens": config.MAX_TOKENS,
        # Some providers may ignore this / error; OpenRouter usually just ignores.
        "reasoning": {"exclude": False, "effort": "high"},
    }

    completion_block = _do_request(
        endpoint=config.OPENROUTER_API_COMPLETIONS,
        payload=completions_payload,
        mode="completions",
    )
    return completion_block

def robust_parse_JSON(raw: str) -> dict:
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
        cleaned = parse_response_JSON(raw)
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

def parse_response_JSON(raw: str) -> str:
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

def save_response(output_dir: str, model: str, attempt: int, script_text: str, response_blob: dict):
    """
    Writes two files:
      - script_...py        : runnable script
      - response_...json    : full LLM JSON incl. code / usage
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

def extract_code(raw: str) -> str:
    """
    Return runnable Python code extracted from raw LLM output.
    """

    patterns = [
        # 1) ```python … ```
        r"```(?:python|py)?\s*(.*?)\s*```",
        # 2) python … ```
        r"^\s*python\s*(.*?)\s*```$",
    ]

    for pat in patterns:
        match = re.search(pat, raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return textwrap.dedent(match.group(1)).strip()

    # 3) Fallback: entire text is code
    return textwrap.dedent(raw).strip()

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
    _, success, _, stdout, stderr, _, _ = run_single_script(script_path, dryrun=True, use_docker=True)
    return success, stdout, stderr
    
def generate_and_dryrun(*, challenges_override=None):
    # Compute run ID
    run_id = get_or_create_run_id()
    output_root = Path("outputs") / run_id

    challenges = challenges_override if challenges_override is not None else config.challenges

    for challenge in challenges:
        logging.info(f"--------------------------------------------------------------------------------------------------------\n------------------------------------ Executing challenge: {challenge.name} ---------------------------------\n--------------------------------------------------------------------------------------------------------")
        for question in tqdm(challenge.questions, desc=f"{challenge.name} questions", leave=False):
            # Set output dir
            output_dir = output_root / challenge.name / question.question_id
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Set-Up Output Directory: %s", output_dir)

            # Build the prompt
            prompt = challenge.build_prompt(question)

            # Save the prompt
            prompt_path = output_dir / f"{question.question_id}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            logging.info("Prompt: %s", prompt)

            for model in tqdm(config.models, desc="Models", leave=False):
                safe_model = _safe_model_id(model)
                
                run_success = False
                for attempt in tqdm(range(1, config.num_attempts + 1), desc=f"{model} Attempt", 
                                    leave=False):
                    logging.info("Querying model %s: Attempt %d", model, attempt)
                    response = query_openrouter(model, prompt)

                    if response is None:
                        logging.error("Model %s: No response on attempt %d", model, attempt)
                        continue
                    
                    response["Outer_attempt"] = attempt
                    code         = response["code"]
                    response_blob = {"LLMGeneration": response}

                    if not code:
                        logging.error("Model %s: Missing code on attempt %d", model, attempt)
                        continue

                    # Concatenate LLM response with our own code and save to runnable .py
                    script = challenge.build_script(code)
                    py_file, json_file = save_response(
                        output_dir, safe_model, attempt, script, response_blob)

                    # Set-Up PyLint & Bandit static checks
                    pylint_ok, pylint_report = run_pylint(py_file)
                    bandit_ok, bandit_report = run_bandit(py_file)

                    append_to_response_json(json_file, "StaticChecks",
                    {
                        "PyLint": {
                            "passed": pylint_ok,
                            "messages": pylint_report,
                        },
                        "Bandit": {
                            "passed": bandit_ok,
                            "results": bandit_report,
                        }
                    })   

                    # PyLint & Bandit Static Checks
                    if not pylint_ok and bandit_ok:
                        failed_folder = os.path.join(output_dir, "StaticFail")
                        move_file(py_file, failed_folder)
                        move_file(json_file, failed_folder)
                        continue
                    
                    # Perform Dry-Run
                    run_success, stdout, stderr = script_dryrun(py_file)

                    STD = {
                        "stdout": stdout, 
                        "stderr": stderr
                    }

                    append_to_response_json(json_file,"DryRun",
                        {
                            "passed"      : bool(run_success),
                            "STD"         : STD,
                            "return_code" : 0 if run_success else 1
                        })

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

                if attempt == config.num_attempts and not run_success:
                    logging.error("Model %s failed to produce a runnable script after %d attempts", model, config.num_attempts)

if __name__ == "__main__":
    generate_and_dryrun()