# master.py

# Imports
import os
import logging
import shutil
import time

from generate_scripts import query_openrouter, save_response
from run_scripts import script_dryrun

from config import models, num_attempts, execution_timeout, dryrun_timeout, challenges
from challenges.FOURTOPS.fourtops import fourtop_challenge


# - - - - - TODO - - - - - 
#   figure out how to run all LLM scripts + evaluation script
#          and how to split them in my own scripts / generation / execution / main
#   create evaluation script
#   implement safety check for running scripts
#   test function to see whether LLM script adheres to code template -> did it actually keep all fixed sections fixed??
# - - - - - - - - - - - - -


def move_file(file_path, destination_dir):
    """Move a single file to the destination directory."""
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, os.path.basename(file_path))
    try:
        shutil.move(file_path, destination)
        logging.info("Moved file %s to %s", file_path, destination_dir)
    except Exception as e:
        logging.error("Failed to move file %s: %s", file_path, e)


def main():
    for challenge in challenges:
        logging.info(f"Executing challenge: {challenge.name}\n")

        for question in challenge.questions:
            prompt = challenge.build_prompt(question)
            prompt = f"```markdown\n{prompt}\n```"
            logging.info(f"Built Prompt:\n {prompt}")
            logging.debug("Set-Up Models: %s", models)

            # Set-up Output Directory
            day_month = time.strftime("%d-%m")
            output_dir = f"./outputs/{day_month}/{challenge.name}/{question.question_id}/"
            if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            logging.info("Set-Up Output Directory: %s", output_dir)

            for model in models:
                safe_model = model.replace("/", "_")
                
                for attempt in range(1, num_attempts+1):
                    logging.info("Querying model %s: Attempt %d", model, attempt)

                    response = query_openrouter(model, prompt)

                    if response is None:
                        logging.error("Model %s: No response on attempt %d", model, attempt)
                        continue

                    code_response = response.get("code", "")
                    explanation_response = response.get("explanation", "")

                    if not code_response:
                        logging.error("Model %s: Empty code on attempt %d", model, attempt)
                        continue
                    if not explanation_response:
                        logging.error("Model %s: Empty explanation on attempt %d", model, attempt)
                        continue
                    
                    # Save Response
                    py_file, txt_file = save_response(output_dir, safe_model, attempt, code_response, explanation_response)
                    
                    # Dry Run
                    logging.info("Dry run for model %s on attempt %d", model, attempt)
                    run_success, stdout, stderr = script_dryrun(py_file, timeout = dryrun_timeout)
                    if run_success:
                        dest_folder = os.path.join(output_dir, safe_model)
                        move_file(py_file, dest_folder)
                        move_file(txt_file, dest_folder)
                        break
                    else:
                        failed_folder = os.path.join(output_dir, "Failed Dry-run Scripts")
                        move_file(py_file, failed_folder)
                        move_file(txt_file, failed_folder)
                if attempt == num_attempts: logging.error("Model %s failed to produce a runnable script after %d attempts", model, num_attempts)

    """
    # Running all success scripts
    logging.info("Now running all successful generated scripts...")
    run_results = run_all_success_scripts(output_dir, timeout = timeout)
    for script_path, ret_code, stdout, stderr in run_results:
         logging.info("Ran script %s with return code %s", script_path, ret_code)
    """
                  
   

if __name__ == "__main__":
    main()