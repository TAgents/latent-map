#!/bin/sh
# Build base map v1 from scratch. Run after a restart, with the prototype server stopped
# (two copies of the model do not fit in 32GB). Takes roughly 1.5 hours; the corpus step
# can be interrupted and resumed by running this again.
set -e
cd "$(dirname "$0")"
PY=../spike/.venv/bin/python
export PYTHONPATH=../spike
if pgrep -f "prototype.*server.py|server.py" >/dev/null; then
  echo "The prototype server is running. Stop it first: pkill -f server.py"; exit 1
fi
echo "== swap check owed from the capture spike =="; sysctl -n vm.swapusage
(cd ../spike && .venv/bin/python measure.py) || echo "swap check failed; continuing"
echo "== corpus ==";  $PY generate_corpus.py v1
echo "== states ==";  $PY collect_states.py v1
echo "== map ==";     $PY build_map.py v1
echo "== names ==";   $PY name_areas.py v1
echo "== stats ==";   $PY map_stats.py v1
echo "== pictures =="; $PY draw_map.py v1
echo "done: see bundle_v1/"
