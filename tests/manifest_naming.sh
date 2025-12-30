#!/usr/bin/env bash
set -euo pipefail

# Path to predict.sh (relative to tests/ by default)
PREDICT="${1:-../docker/predict.sh}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MODEL="google_gemini-2.5-pro"
HHMM="1618"
ATTEMPT="2"
OUT="$TMP/out"
mkdir -p "$OUT"

# Fake artefacts
printf x > "$OUT/${MODEL}_${HHMM}_${ATTEMPT}_model.pkl"
printf x > "$OUT/${MODEL}_${HHMM}_${ATTEMPT}_preproc.pkl"
printf x > "$OUT/${MODEL}_${HHMM}_${ATTEMPT}_state.pt"

# New-style manifest
(
  cd "$OUT"
  sha256sum \
    "${MODEL}_${HHMM}_${ATTEMPT}_state.pt" \
    "${MODEL}_${HHMM}_${ATTEMPT}_model.pkl" \
    "${MODEL}_${HHMM}_${ATTEMPT}_preproc.pkl" \
    > "${MODEL}_${HHMM}_${ATTEMPT}_manifest.sha256"
)

# Run predict in "manifest-only" mode so it doesn't try to import your repo
set +e
ONLY_MANIFEST_CHECK=1 bash "$PREDICT" "$OUT/${MODEL}_${HHMM}_${ATTEMPT}_model.pkl" \
  1>"$TMP/stdout.txt" 2>"$TMP/stderr.txt"
rc=$?
set -e

echo "=== stdout ==="; cat "$TMP/stdout.txt" || true
echo "=== stderr ==="; cat "$TMP/stderr.txt" || true
echo "=== exit code: $rc ==="

if [[ $rc -eq 0 ]]; then
  echo "✅ PASS"
else
  echo "❌ FAIL"
fi
