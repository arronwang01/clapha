#!/bin/bash
# Bundle what il/train.py needs on the training PC (campus RTX 4080): our code, the live observation
# code, FirstLight's native_runner and the fl:hog2 checkpoint, in the same layout as this repo, so
# every relative path resolves unchanged. The converted replays go in a second zip (large).
#
#   tools/pack_training.sh [OUT_DIR]      -> OUT_DIR/clapha-train-code.zip, OUT_DIR/conv-hog26.zip
#   tools/pack_training.sh --code-only [OUT_DIR]
# On the PC (Python 3.12 + torch cu124 already there):
#   py -3.12 -m pip install orjson zstandard
#   py -3.12 -m il.train --smoke            (from the unzipped clapha folder)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
code_only=0
[ "${1:-}" = "--code-only" ] && { code_only=1; shift; }
OUT="${1:-$ROOT/build/train-pack}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
STAGE="$(mktemp -d "$OUT/stage.XXXX")"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/clapha/il" "$STAGE/clapha/mac012" "$STAGE/clapha/ref-firstlight/checkpoints/2_6hog_expert"
cp "$ROOT"/il/*.py "$ROOT/il/SPEC.md" "$STAGE/clapha/il/"
cp "$ROOT/mac012/firstlight_obs.py" "$ROOT/mac012/firstlight_bot.py" "$STAGE/clapha/mac012/"
cp "$ROOT/live_card_catalog.json" "$STAGE/clapha/"
rsync -a --exclude '__pycache__' --exclude 'tests' --exclude 'probe' \
  "$ROOT/ref-firstlight/native_runner" "$STAGE/clapha/ref-firstlight/"
cp "$ROOT/ref-firstlight/checkpoints/2_6hog_expert/hog26-specialist2.pt" \
  "$STAGE/clapha/ref-firstlight/checkpoints/2_6hog_expert/"
cp "$ROOT/ref-firstlight/requirements.txt" "$ROOT/ref-firstlight/pyproject.toml" "$STAGE/clapha/ref-firstlight/"
(cd "$STAGE" && rm -f "$OUT/clapha-train-code.zip" && zip -qr "$OUT/clapha-train-code.zip" clapha)
ls -l "$OUT/clapha-train-code.zip"

if [ "$code_only" = 0 ]; then
  # the conversion: every replay file plus its index; unzip next to the code (clapha/runs/...)
  (cd "$ROOT" && rm -f "$OUT/conv-hog26.zip" && zip -qr -0 "$OUT/conv-hog26.zip" runs/conv-hog26 \
     -x 'runs/conv-hog26/*.part' -x 'runs/conv-hog26/convert.log')
  ls -l "$OUT/conv-hog26.zip"
fi
