#!/bin/bash
# Render the app window from recorded engine states (no display needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
swiftc -O -parse-as-library -target arm64-apple-macosx14.0 \
  "$HERE/Sources/Backend.swift" "$HERE/Sources/Views.swift" "$HERE/Sources/Overlay.swift" "$HERE/Sources/Games.swift" "$HERE/Sources/Nulls.swift" \
  "$HERE/Snapshot/Snapshot.swift" \
  -o "$ROOT/build/clapha_snapshot"
# app/snapshot.sh --games runs/viewer/games.json build/games_snapshot.png [game index]
if [ "${1:-}" = "--games" ]; then exec "$ROOT/build/clapha_snapshot" "$@"; fi
"$ROOT/build/clapha_snapshot" "${1:-$HERE/fixture_battle.json}" "${2:-$HERE/fixture_idle.json}" \
  "${3:-$ROOT/build/app_snapshot.png}"
