# master.py

# Imports
import os
import sys
import glob
import subprocess
import logging
import shutil
import time

from prompts import *
from utils import rename_file_with_suffix
from generate_scripts import query_openrouter, save_response
from run_scripts import script_dryrun

from config import models, num_attempts, timeout
from challenges.FOURTOPS.fourtops import fourtop_challenge

challenges = [fourtop_challenge]


def move_all_success_scripts(folder_path):
    """
    Finds all .py scripts in folder_path that end with '_dSUCCESS.py' and moves them
    into a subfolder named after their basename (without extension).
    
    Returns a list of the new script file paths.
    """

    success_scripts = glob.glob(os.path.join(folder_path, "*_dSUCCESS.py"))
    logging.info("Found %d SUCCESS scripts in %s", len(success_scripts), folder_path)
    
    moved_scripts = []
    for script in success_scripts:
        base_name = os.path.splitext(os.path.basename(script))[0]
        # Create an output directory named after the script (in the same folder)
        script_output_dir = os.path.join(folder_path, base_name)
        os.makedirs(script_output_dir, exist_ok=True)
        logging.info("Created output directory %s for script %s", script_output_dir, script)
        
        destination = os.path.join(script_output_dir, os.path.basename(script))
        try:
            shutil.move(script, destination)
            logging.info("Moved script %s to %s", script, destination)
            moved_scripts.append(destination)
        except Exception as e:
            logging.error("Failed to move script %s: %s", script, e)
            continue  # Skip this script if it cannot be moved
    return moved_scripts

def run_all_success_scripts(folder_path, timeout=7200):
    """
    Finds all SUCCESS scripts in subdirectories of folder_path (i.e. those ending in '_dSUCCESS.py')
    and runs them one by one. Each script is executed with its own subdirectory as its working directory.
    A timeout of 'timeout' seconds is enforced per script.
    """

    # Find all success scripts in any subdirectory (e.g. ./outputs/{day}-{month}//{<script_basename>/*_dSUCCESS.py)
    success_scripts = glob.glob(os.path.join(folder_path, "*", "*_dSUCCESS.py"))
    logging.debug(f"Success scripts directories: {success_scripts}")
    logging.info("Found %d SUCCESS scripts in subdirectories of %s", len(success_scripts), folder_path)
    
    for script in success_scripts:
        script_output_dir = os.path.dirname(script)    
        logging.debug(f"Script: {script}")
        script = script.rstrip(f"{folder_path}")
        logging.debug(f"Stripped Script: {script}")
        logging.debug(f"Script output directory: {script_output_dir}")

        try:
            logging.info("Running script %s", script)
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                logging.info("Script %s executed successfully. STDOUT:\n%s", script, result.stdout)
            else:
                logging.error("Script %s failed with return code %d. STDERR:\n%s", script, result.returncode, result.stderr)
        except subprocess.TimeoutExpired:
            logging.error("Script %s timed out after %d seconds.", script, timeout)
        except Exception as e:
            logging.error("Error executing script %s: %s", script, e)


def test_all_success_scripts(folder_path):
    return 0


def main():
    for challenge in challenges:
        logging.info(f"Executing challenge: {challenge.name}\n")

        for question in challenge.questions:
            prompt = challenge.build_prompt(question)
            prompt = f"```markdown\n{prompt}\n```"
            logging.info(f"Built Prompt:\n {prompt}")
            logging.debug("Set-Up Models: %s", models)

            # Set-up Output Directory
            current_time = time.strftime("%d%m%H%M")
            day_month = time.strftime("%d-%m")
            output_dir = f"./outputs/{day_month}/{challenge.name}/{question.question_id}/"
            if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            safe_model = model.replace("/", "_")
            logging.info("Set-Up Output Directory: %s", output_dir)

            for model in models:
                attempts = 0
                success = False

                while attempts < num_attempts and not success:
                    attempts += 1

                    logging.info("Querying model %s: Attempt %d", model, attempts)    
                    response_dict = query_openrouter(model, prompt)

                    if response_dict is None:
                        logging.error("Model %s: No response received on attempt %d", model, attempts)
                        continue
                    
                    code_response = response_dict["code"]
                    explanation_response = response_dict["explanation"]

                    logging.info("Generated code from %s:\n%s", model, '-'*40 + "\n" + code_response + "\n" + '-'*40)
                    logging.info("Generated explanation from %s:\n%s", model, '-'*40 + "\n" + explanation_response + "\n" + '-'*40)  

                    if not code_response:
                        logging.error("Model %s: Received empty code on attempt %d", model, attempts)
                        continue
                    
                    filename = save_response(model, attempts, code_response, explanation_response,
                                question = "FOURTOP_1",
                                data_loading_script= "fourtop_data_loader.py")
                
                    run_success, stdout, stderr = script_dryrun(filename, timeout=300)
                    if run_success:
                        logging.info("Model %s: Dry run successful on attempt %d", model, attempts)
                        filename = rename_file_with_suffix(filename, "_drySUCCESS")
                        success = True
                    else:
                        logging.error("Model %s: Dry run failed on attempt %d;", model, attempts)
                        try:
                            filename = rename_file_with_suffix(filename, "_dryFAIL")
                        except Exception as e:
                            logging.error("Failed to rename script %s: %s", filename, e)
                        # Continue to next attempt
                    
                if not success:
                    logging.error("Model %s: Failed to produce a runnable script after 5 attempts.", model)
                else:
                    logging.info("Model %s: Runnable script obtained.", model)
   
""" Works
# Move & run all success scripts
folder = "./outputs/02-04" 
_ = move_all_success_scripts(folder)
run_all_success_scripts(folder, timeout=3600)
"""

if __name__ == "__main__":
    main()