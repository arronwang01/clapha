#!/bin/bash
# Run the Python that has the model stack (torch). The Clapha app runs its tasks in a login shell,
# which does not read ~/.zshrc -- where conda puts miniforge on PATH -- so a bare `python3` there is
# python.org's 3.14 without torch ("load failed: No module named 'torch'", 2026-09-25).
# Order: $CLAPHA_PYTHON, the cached choice, python3 on PATH, the usual conda/Homebrew locations.
CACHE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build/python_path"
has_torch() {
  [ -n "$1" ] && [ -x "$1" ] &&
    "$1" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("torch") is None)' 2>/dev/null
}
for candidate in "$CLAPHA_PYTHON" "$(cat "$CACHE" 2>/dev/null)" "$(command -v python3)" \
    /opt/homebrew/Caskroom/miniforge/base/bin/python3 "$HOME/miniforge3/bin/python3" \
    "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3" /opt/homebrew/bin/python3 \
    /usr/local/bin/python3; do
  if has_torch "$candidate"; then
    mkdir -p "$(dirname "$CACHE")" && printf '%s\n' "$candidate" > "$CACHE"
    exec "$candidate" "$@"
  fi
done
echo "no python3 with torch found; install torch or set CLAPHA_PYTHON=/path/to/python3" >&2
exit 1
