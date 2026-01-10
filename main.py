# main.py

import argparse, logging, re
from generate_and_dryrun import generate_and_dryrun
from run_scripts import execute_scripts_in_batch
from evaluate_scripts import evaluate_results
from docker.run_docker import ensure_docker_running
from config import challenges
from utils.run_id import *


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


_RUN_ID_FROM_PATH_RE = re.compile(r"(?:^|[\\/])outputs[\\/](\d{2}-\d{2}(?:\(\d+\))?)(?:[\\/]|$)")


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

    # Optional explicit run id (mostly for --run / --eval convenience)
    parser.add_argument('--run-id', type=str, default=None,
                        help='Explicit run id under ./outputs/ (e.g. 01-06 or 01-06(2)).')

    return parser.parse_args()

def get_default_output_dir(*, create_new: bool = False):
    run_id = get_or_create_run_id(create_new=create_new)
    return f"./outputs/{run_id}/"

def _infer_run_id_from_input_dir(input_dir: str) -> str | None:
    m = _RUN_ID_FROM_PATH_RE.search(str(input_dir))
    return m.group(1) if m else None

def _resolve_output_dir(args) -> str:
    """
    Resolve the base output directory to use for Stage 2/3.

    Priority:
      1) --input-dir (and we best-effort infer run id from it and set it active)
      2) --run-id (set it active)
      3) active run id (even if it's from yesterday)
    """
    if args.input_dir:
        inferred = _infer_run_id_from_input_dir(args.input_dir)
        if inferred:
            set_active_run_id(inferred)
        return args.input_dir

    if args.run_id:
        set_active_run_id(args.run_id)
        return f"./outputs/{args.run_id}/"

    run_id = require_active_run_id()
    return f"./outputs/{run_id}/"

def main():
    args = parse_args()

    logging.info("--------------------------------------------------------------------------------------------------------\n------------------------------------ Started LLM Challenge Pipeline ------------------------------------\n--------------------------------------------------------------------------------------------------------")

    ensure_docker_running()

    if args.gen:
            logging.info("Mode: Generate scripts and dry-run validation.")
            generate_and_dryrun()
            logging.info("Active run id after generation: %s", get_active_run_id())

    elif args.run:
        input_dir = _resolve_output_dir(args)
        logging.info(f"Mode: Execute scripts. Input directory: {input_dir}")
        execute_scripts_in_batch(input_dir)

    elif args.eval:
        input_dir = _resolve_output_dir(args)
        logging.info(f"Mode: Evaluate script outputs. Input directory: {input_dir}")
        evaluate_results(input_dir)

    elif args.all:
        logging.info("Mode: Full pipeline execution.")

        # Step 1: Generation and Dry-run
        logging.info("Start generation and dryrun.")
        generate_and_dryrun()

        # IMPORTANT: do NOT call get_or_create_run_id() here.
        # We MUST reuse the active run id that Stage 1 set, even if midnight passed.
        run_id = require_active_run_id()
        input_dir = f"./outputs/{run_id}/"
        logging.info(f"Using active run directory for script execution & evaluation: {input_dir}")

        # Step 2: Run scripts
        logging.info("Start script execution.")
        execute_scripts_in_batch(input_dir, use_docker=True, dryrun=False)

        # Step 3: Evaluation
        logging.info("Start model evaluation.")
        evaluate_results(input_dir)

    logging.info("\n\n\n----------------------------------------------------\n--------- Completed LLM Challenge Pipeline ---------\n----------------------------------------------------")

if __name__ == "__main__":
    main()
