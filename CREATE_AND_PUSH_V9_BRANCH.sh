#!/usr/bin/env bash
set -Eeuo pipefail
BRANCH="${BRANCH:-dt1d-v9-scdq-math-latency}"
REMOTE="${REMOTE:-origin}"
git checkout -B "$BRANCH"
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Add SCDQ-DT1D mathematical latency revision and Kaggle matrix"
fi
git push -u "$REMOTE" "$BRANCH"
echo "Pushed branch: $BRANCH"
