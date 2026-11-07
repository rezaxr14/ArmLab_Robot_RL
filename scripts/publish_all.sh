#!/usr/bin/env bash
# Render published media for every finished run, copy policy weights into weights/, and
# regenerate README.md. Safe to re-run: every task's media directory is rebuilt from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=".venv/bin/python"

echo "[1/3] scene gallery"
$PY -m armforge scene --task pickplace --out docs/media/scene_gallery.png

echo "[2/3] media for finished runs"
while read -r run out; do
  [ -n "${run:-}" ] || continue
  if [ -f "runs/$run/final.pt" ] && [ -d "runs/$run/milestones" ] && [ -n "$(ls -A "runs/$run/milestones" 2>/dev/null)" ]; then
    echo "  -> $run => docs/media/$out"
    rm -rf "docs/media/$out"
    $PY -m armforge publish --run "runs/$run" --out-name "$out" --cameras hero --clips 2 --episodes 30
  else
    echo "  -- skip $run (not finished)"
  fi
done <<'EOF'
reach-s0 reach
push-s0 push
pickplace-s0 pickplace
pickplace_clutter-s0 pickplace_clutter
EOF

echo "[3/3] weights + README"
for run in reach-s0 push-s0 pickplace-s0 pickplace_clutter-s0; do
  [ -d "runs/$run" ] || continue
  [ -f "runs/$run/final.pt" ] || continue
  mkdir -p "weights/$run"
  for f in best.pt final.pt; do
    if [ -f "runs/$run/$f" ]; then cp -f "runs/$run/$f" "weights/$run/"; fi
  done
  if [ -d "runs/$run/milestones" ]; then
    cp -f runs/"$run"/milestones/*.pt "weights/$run/" 2>/dev/null || true
  fi
done
$PY -m armforge readme --repo .
echo "done"
