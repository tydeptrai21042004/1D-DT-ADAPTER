#!/usr/bin/env bash
set -Eeuo pipefail

BRANCH="${BRANCH:-dt1d-v8-cnn-three-seed}"
REMOTE="${REMOTE:-origin}"
MESSAGE="${MESSAGE:-Add CNN-only three-seed reproducibility matrix}"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
else
  git switch -c "$BRANCH"
fi

git add -A
if git diff --cached --quiet; then
  echo "No staged changes; keeping the existing branch commit."
else
  git commit -m "$MESSAGE"
fi

git push -u "$REMOTE" "$BRANCH"
echo "Pushed branch: $BRANCH"
echo "Create tag v0.8.0 only after the three-seed GPU matrix is complete and validated."
