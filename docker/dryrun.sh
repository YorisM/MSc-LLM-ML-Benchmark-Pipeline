#!/usr/bin/env bash
# Usage: docker run ... dryrun.sh <script.py> --dryrun

set -euo pipefail

PY=$1; shift || true           # path to LLM-generated script
DRY="${1:-}"                   # --dryrun

# ── Resource clamps inside container ─────────────────────────────
ulimit -t 600               # 10 mins CPU time
ulimit -v $((8*1024*1024))  # 8 GB virtual memory
ulimit -n 256               # file handles

# Better: kill if wall-clock exceeds 2 h
timeout --signal=KILL 600s python "$PY" $DRY
