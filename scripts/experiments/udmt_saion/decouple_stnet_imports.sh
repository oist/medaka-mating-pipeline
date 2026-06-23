#!/usr/bin/env bash
# Decouple UDMT's ST-Net / TrDiMP from the GUI: the ltr/pytracking code imports itself via
# the absolute path `udmt.gui.tabs.ST_Net.ltr.*` / `...pytracking.*`, which drags in
# `udmt.gui.__init__` (qtpy) and `udmt.gui.tabs.__init__` (PySide6/analyze_videos) on *any*
# import of the tracker. Rewriting those to bare `ltr.*` / `pytracking.*` makes the stack
# importable standalone (with ST_Net on sys.path) with no GUI/Qt — the prerequisite for
# running it headless/CPU/Blackwell and for vendoring it into the repo.
#
# Safe for the existing headless pipeline: run_udmt_headless already puts ST_Net on the path,
# so bare imports resolve there too. Idempotent.
set -e
ST=${1:-/bucket/StephensU/medaka/udmt/UDMT/udmt/gui/tabs/ST_Net}
mapfile -t files < <(grep -rl "udmt.gui.tabs.ST_Net" "$ST" --include=*.py 2>/dev/null)
echo "rewriting ${#files[@]} files: udmt.gui.tabs.ST_Net.{ltr,pytracking} -> {ltr,pytracking}"
for f in "${files[@]}"; do
  sed -i 's/udmt\.gui\.tabs\.ST_Net\.ltr/ltr/g; s/udmt\.gui\.tabs\.ST_Net\.pytracking/pytracking/g' "$f"
done
# Cross-package relative imports (`from ..ltr`, `from ...pytracking`, ...) only resolve when
# nested under ST_Net; rewrite any 2+-dot ltr/pytracking import to bare so the stack imports
# standalone with ST_Net on sys.path.
grep -rlE 'from \.\.+(ltr|pytracking)' "$ST" --include=*.py 2>/dev/null \
  | xargs -r sed -i -E 's/from \.\.+(ltr|pytracking)/from \1/g'
# `from udmt.gui import BASE_DIR` (a path constant) drags in the GUI just to import; make it a
# local, env-overridable fallback so the import is GUI-free (set UDMT_BASE_DIR for real paths).
grep -rl 'from udmt\.gui import BASE_DIR' "$ST" --include=*.py 2>/dev/null \
  | xargs -r sed -i 's|from udmt\.gui import BASE_DIR|import os as _os; BASE_DIR = _os.environ.get("UDMT_BASE_DIR", _os.path.dirname(_os.path.abspath(__file__)))|'
remaining=$(grep -rl "udmt.gui.tabs.ST_Net" "$ST" --include=*.py 2>/dev/null | wc -l)
echo "done; files still referencing udmt.gui.tabs.ST_Net: $remaining"
