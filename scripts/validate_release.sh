#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python tools/verify_reproducibility_package.py
python -m compileall -q main.py engine.py datasets models tools tests
pytest -q
for file in $(find . -maxdepth 3 -type f \( -name '*.sh' -o -name '*.sh.txt' \)); do
  bash -n "$file"
done
