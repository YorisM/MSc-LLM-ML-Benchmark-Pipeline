# main.py


import argparse
import logging
from generate_and_dryrun import generate_and_dryrun
from run_scripts import execute_scripts_in_batch
from evaluate_scripts import evaluate_results
from docker.run_docker import ensure_docker_running
from config import challenges
from utils.run_id import get_or_create_run_id


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

def get_default_output_dir(*, create_new: bool = False):
    run_id = get_or_create_run_id(create_new=create_new)
    return f"./outputs/{run_id}/"

def main():
    args = parse_args()

    logging.info("--------------------------------------------------------------------------------------------------------\n------------------------------------ Started LLM Challenge Pipeline ------------------------------------\n--------------------------------------------------------------------------------------------------------")

    ensure_docker_running()

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

    logging.info("\n\n\n----------------------------------------------------\n--------- Completed LLM Challenge Pipeline ---------\n----------------------------------------------------")

if __name__ == "__main__":
    main()
