# main.py


import argparse
import logging
from generate_and_dryrun import generate_and_dryrun
from run_scripts import execute_scripts_in_batch
from evaluate_scripts import evaluate_results
from datetime import datetime
from config import challenges


# - - - - - Usage Examples  - - - - -
#
# Generate scripts + test using dry-run:
# python main.py --gen
# 
# Execute scripts explicitly from a given folder:
# python main.py --run --input-dir ./outputs/21-04/FOURTOPS/Q1/
#
# Evaluate explicitly from a given folder:
# python main.py --eval --input-dir ./outputs/21-04/FOURTOPS/Q1/
#
# Full explicit pipeline execution:
# python main.py --all


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Benchmarking Pipeline")

    # Mutually exclusive modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gen', action='store_true',
                       help='Generate scripts from LLM and perform dry-run validation.')
    group.add_argument('--run', action='store_true',
                       help='Run scripts that have passed dry-run validation.')
    group.add_argument('--eval', action='store_true',
                       help='Evaluate outputs from executed scripts.')
    group.add_argument('--all', action='store_true',
                       help='Generate, run scripts, and evaluate sequentially.')

    # Optional input directory (for run and evaluate)
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Explicit input directory for running scripts or evaluation.')

    return parser.parse_args()

def get_default_output_dir():
    day_month = datetime.now().strftime("%d-%m")
    # Assuming single challenge/question for simplicity here (expandable later)
    challenge = challenges[0].name
    question = challenges[0].questions[0].question_id
    return f"./outputs/{day_month}/{challenge}/{question}/"

def main():
    args = parse_args()

    logging.info("\n\n\nStarted LLM Challenge Pipeline")

    if args.gen:
        logging.info("Mode: Generate scripts and dry-run validation.")
        generate_and_dryrun()

    elif args.run:
        input_dir = args.input_dir or get_default_output_dir()
        logging.info(f"Mode: Execute scripts. Input directory: {input_dir}")
        execute_scripts_in_batch(input_dir)

    elif args.eval:
        input_dir = args.input_dir or get_default_output_dir()
        logging.info(f"Mode: Evaluate script outputs. Input directory: {input_dir}")
        evaluate_results(input_dir)

    elif args.all:
        logging.info("Mode: Full pipeline execution.")

        # Step 1: Generation and Dry-run
        logging.info("Start generation and dryrun.")
        generate_and_dryrun()

        # Determine the directory automatically
        input_dir = get_default_output_dir()
        logging.info(f"Using auto-generated directory for script execution & evaluation: {input_dir}")

        # Step 2: Run scripts
        logging.info("Start script execution.")
        execute_scripts_in_batch(input_dir, use_docker=True, dryrun=False)

        # Step 3: Evaluation
        logging.info("Start model evaluation.")
        evaluate_results(input_dir)

    logging.info("Completed LLM Challenge Pipeline.")

if __name__ == "__main__":
    main()