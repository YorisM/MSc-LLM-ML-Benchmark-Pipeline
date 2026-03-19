# main.py

import argparse, logging, re, copy
from pathlib import Path

from generate_and_dryrun import generate_and_dryrun
from run_scripts import execute_scripts_in_batch
from evaluate_scripts import evaluate_results
from docker.run_docker import ensure_docker_running
from config import challenges as ALL_CHALLENGES
from utils.run_id import *


# - - - - - Usage Examples  - - - - -
# --- GENERATE ALL SCRIPTS ---
# python main.py --gen
#
# --- GENERATE: TRACKFORMERS only (all questions) ---
# python main.py --gen --challenge TRACKFORMERS --question ALL
#
#
# --- RUN ALL SCRIPTS: from a given folder:
# python main.py --run --input-dir ./outputs/21-04/FOURTOPS/Q1/
#
# --- RUN ONLY FOURTOPS Q1: for an existing run directory
# python main.py --run --run-id 01-06 --challenge FOURTOPS --question Q1
#
#
# --- EVALUATE ALL SCRIPTS: from a given folder:
# python main.py --eval --input-dir ./outputs/21-04/FOURTOPS/Q1/
#
# --- EVALUATE ONLY Q1: across all challenges
# python main.py --eval --run-id 01-06 --challenge ALL --question Q1
#
#
# --- RUN FULL PIPELINE ---
# python main.py --all
#
# --- RUN FULL PIPELINE: FOURTOPS Q1 only ---
# python main.py --all --challenge FOURTOPS --question Q1
#
#
# --- SCOPED RUN ---
# python main.py --all --question FOURTOPS:Q1


_RUN_ID_FROM_PATH_RE = re.compile(r"(?:^|[\\/])outputs[\\/](\d{4}-\d{2}-\d{2}(?:\(\d+\))?)(?:[\\/]|$)")


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
    
    # Optional filtering (applies to --gen/--all, and also to --run/--eval when you pass a run-id/base dir)
    parser.add_argument("--challenge", action="append", default=None,
        help='Challenge filter. Examples: --challenge ALL | --challenge FOURTOPS | --challenge FOURTOPS --challenge TRACKFORMERS | --challenge "FOURTOPS,TRACKFORMERS"')
    
    # Optional filtering - run a single challenge-question
    parser.add_argument("--question", action="append", default=None,
        help='Question filter. Examples: --question ALL | --question Q1 | --question "Q1,Q2". Optional scoped form: --question FOURTOPS:Q1')
    
    parser.add_argument("--repeat", type=int, default=1,
        help="Repeat the full --all pipeline N times. Each repetition gets a fresh run_id.")

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

def _flatten_csv_args(xs: list[str] | None) -> list[str]:
    if not xs:
        return ["ALL"]
    out: list[str] = []
    for x in xs:
        out.extend([t.strip() for t in x.split(",") if t.strip()])
    return out or ["ALL"]

def _available_challenge_map():
    # name -> Challenge object
    return {c.name.upper(): c for c in ALL_CHALLENGES}

def _select_challenges_for_generation(challenge_args: list[str] | None, question_args: list[str] | None):
    """
    Return a list[Challenge] where each Challenge has its .questions filtered.
    This is only for Stage 1 (generation), so we filter questions directly.
    """
    ch_items = [c.upper() for c in _flatten_csv_args(challenge_args)]
    q_items  = _flatten_csv_args(question_args)

    ch_map = _available_challenge_map()

    # Resolve challenge selection
    if "ALL" in ch_items:
        selected_ch_names = list(ch_map.keys())
    else:
        unknown = [c for c in ch_items if c not in ch_map]
        if unknown:
            raise SystemExit(f"Unknown challenge(s): {unknown}. Available: {sorted(ch_map.keys())}")
        selected_ch_names = ch_items

    # Resolve question selection: global QIDs + optional scoped CHALLENGE:QID
    if any(q.upper() == "ALL" for q in q_items):
        global_qids: set[str] | None = None  # None means ALL
        scoped: dict[str, set[str]] = {}
    else:
        global_qids = set()
        scoped = {}
        for q in q_items:
            if ":" in q:
                ch, qid = q.split(":", 1)
                scoped.setdefault(ch.upper(), set()).add(qid.upper())
            else:
                global_qids.add(q.upper())

    selected: list = []
    for ch_name in selected_ch_names:
        ch = ch_map[ch_name]

        if global_qids is None:
            # ALL questions
            selected.append(ch)
            continue

        qids = set(global_qids)
        qids |= scoped.get(ch_name, set())

        filtered = [qq for qq in ch.questions if qq.question_id.upper() in qids]
        if not filtered:
            raise SystemExit(
                f"Challenge {ch.name}: none of the requested questions exist. "
                f"Requested={sorted(qids)} Available={[qq.question_id for qq in ch.questions]}"
            )

        ch2 = copy.copy(ch)
        ch2.questions = filtered
        selected.append(ch2)

    return selected

def _iter_selected_dirs_for_run_eval(base_dir: str, challenge_args: list[str] | None, question_args: list[str] | None):
    """
    For --run/--eval, we avoid modifying run_scripts/evaluate_scripts by calling them on
    the appropriate subdirectories under the resolved base output dir.
    """
    base = Path(base_dir)
    ch_items = [c.upper() for c in _flatten_csv_args(challenge_args)]
    q_items  = [q.strip() for q in _flatten_csv_args(question_args)]

    ch_map = _available_challenge_map()

    if "ALL" in ch_items:
        selected_ch_names = list(ch_map.keys())
    else:
        unknown = [c for c in ch_items if c not in ch_map]
        if unknown:
            raise SystemExit(f"Unknown challenge(s): {unknown}. Available: {sorted(ch_map.keys())}")
        selected_ch_names = ch_items

    want_all_q = any(q.upper() == "ALL" for q in q_items)
    global_qids: set[str] = set()
    scoped: dict[str, set[str]] = {}
    if not want_all_q:
        for q in q_items:
            if ":" in q:
                ch, qid = q.split(":", 1)
                scoped.setdefault(ch.upper(), set()).add(qid.upper())
            else:
                global_qids.add(q.upper())

    dirs: list[Path] = []
    for ch_name in selected_ch_names:
        ch_dir = base / ch_map[ch_name].name  # preserve case as on disk

        if want_all_q and not global_qids and not scoped:
            # just the challenge directory (runs all questions underneath)
            if ch_dir.exists():
                dirs.append(ch_dir)
            continue

        # Specific Q selection
        qids = set(global_qids)
        qids |= scoped.get(ch_name, set())
        for qid in sorted(qids):
            qdir = ch_dir / qid
            if qdir.exists():
                dirs.append(qdir)

    if not dirs:
        raise SystemExit(
            f"No matching directories found under {base_dir} for "
            f"challenge={_flatten_csv_args(challenge_args)} question={_flatten_csv_args(question_args)}"
        )
    return [str(d) for d in dirs]

def main():
    args = parse_args()

    logging.info("--------------------------------------------------------------------------------------------------------\n------------------------------------ Started LLM Challenge Pipeline ------------------------------------\n--------------------------------------------------------------------------------------------------------")

    ensure_docker_running()

    if args.gen:
        logging.info("Mode: Generate scripts and dry-run validation.")
        selected = _select_challenges_for_generation(args.challenge, args.question)
        generate_and_dryrun(challenges_override=selected)
        logging.info("Active run id after generation: %s", get_active_run_id())

    elif args.run:
        if args.input_dir:
            targets = [args.input_dir]
        else:
            base_dir = _resolve_output_dir(args)
            targets = _iter_selected_dirs_for_run_eval(base_dir, args.challenge, args.question)

        logging.info("Mode: Execute scripts. Targets: %s", targets)
        for d in targets:
            execute_scripts_in_batch(d)

    elif args.eval:
        if args.input_dir:
            targets = [args.input_dir]
        else:
            base_dir = _resolve_output_dir(args)
            targets = _iter_selected_dirs_for_run_eval(base_dir, args.challenge, args.question)

        logging.info("Mode: Evaluate script outputs. Targets: %s", targets)
        for d in targets:
            evaluate_results(d)

    elif args.all:
        logging.info("Mode: Full pipeline execution.")

        for i in range(args.repeat):
            if args.repeat > 1:
                logging.info(f"--- Repetition {i+1}/{args.repeat} ---")

            # Force a fresh run id for this repetition (2026-01-21, 2026-01-21(2), ...)
            get_or_create_run_id(create_new=True)

            # Stage 1: Generation and Dry-run (must set/keep active run id)
            logging.info("Start generation and dryrun.")
            selected = _select_challenges_for_generation(args.challenge, args.question)
            generate_and_dryrun(challenges_override=selected)

            # Use the run_id that Stage 1 actually used
            run_id = require_active_run_id()
            input_dir = f"./outputs/{run_id}/"
            logging.info(f"Using active run directory for script execution & evaluation: {input_dir}")

            # Stage 2: Run scripts
            logging.info("Start script execution.")
            targets = _iter_selected_dirs_for_run_eval(input_dir, args.challenge, args.question)
            for d in targets:
                execute_scripts_in_batch(d, use_docker=True, dryrun=False)

            # Stage 3: Evaluation
            logging.info("Start model evaluation.")
            for d in targets:
                evaluate_results(d)


    logging.info("\n\n\n----------------------------------------------------\n--------- Completed LLM Challenge Pipeline ---------\n----------------------------------------------------")

if __name__ == "__main__":
    main()
