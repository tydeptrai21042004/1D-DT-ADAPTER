#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(cat "$ROOT/VERSION")"
TAG="${TAG:-v${VERSION}}"
NOTES="${NOTES:-docs/release/v${VERSION}.md}"
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required." >&2; exit 2; }
cd "$ROOT"
bash scripts/build_release_archive.sh
gh release create "$TAG" \
  --title "DT1D-Adapter CNN three-seed release $TAG" \
  --notes-file "$NOTES" \
  "dist/DT1D-Adapter-v${VERSION}.zip" \
  "dist/DT1D-Adapter-v${VERSION}.tar.gz" \
  dist/SHA256SUMS.txt
