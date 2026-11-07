#!/usr/bin/env bash
# Unattended completion pipeline:
#   1. wait for each run's final.pt/summary.json, restarting it with --resume if it dies,
#   2. render that run's media, copy its weights, regenerate README.md,
#   3. commit + push, then move on to the next run.
# Publishing incrementally means every finished task is already on GitHub even if the last
# (slowest) run is still training. Progress is appended to runs/supervise.log.
set -uo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=".venv/bin/python"
LOG="runs/supervise.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Serialise: only one supervisor may run.
exec 9>runs/supervise.lock
if ! flock -n 9; then say "another supervisor is running"; exit 0; fi

# run -> config -> published media directory
RUNS=(
  "reach-s0:configs/reach.yaml:reach"
  "push-s0:configs/push.yaml:push"
  "pickplace-s0:configs/pickplace.yaml:pickplace"
  "pickplace_clutter-s0:configs/pickplace_clutter.yaml:pickplace_clutter"
)

copy_weights() {
  local run="$1" f
  [ -d "runs/$run" ] || return 0
  mkdir -p "weights/$run"
  for f in best.pt final.pt; do
    [ -f "runs/$run/$f" ] && cp -f "runs/$run/$f" "weights/$run/"
  done
  [ -d "runs/$run/milestones" ] && cp -f runs/"$run"/milestones/*.pt "weights/$run/" 2>/dev/null
  return 0
}

publish_and_push() {
  local run="$1" out="$2"
  say "publishing $run -> docs/media/$out"
  rm -rf "docs/media/$out"
  if ! $PY -m armforge publish --run "runs/$run" --out-name "$out" --cameras hero --clips 2 --episodes 30 >> "$LOG" 2>&1; then
    say "publish failed for $run (continuing)"
    return 1
  fi
  copy_weights "$run"
  git add -f weights docs/media README.md
  if git diff --cached --quiet; then
    say "nothing to commit for $run"
  else
    git commit -m "docs: publish $out results and policy weights" >> "$LOG" 2>&1
    say "commit for $run exit=$?"
  fi
  git push origin main >> "$LOG" 2>&1
  say "push for $run exit=$?"
}

say "supervisor started"
for item in "${RUNS[@]}"; do
  run="${item%%:*}"; rest="${item#*:}"; cfg="${rest%%:*}"; out="${rest#*:}"
  while :; do
    if [ -f "runs/$run/final.pt" ] && [ -f "runs/$run/summary.json" ]; then
      say "$run complete"
      break
    fi
    if ! pgrep -f "armforge train --config $cfg --name $run --out-dir runs" >/dev/null 2>&1; then
      say "$run is not running and is incomplete -> resuming"
      mkdir -p "runs/$run"
      setsid nohup .venv/bin/armforge train --config "$cfg" --name "$run" --out-dir runs --resume >> "runs/$run/train.log" 2>&1 &
      sleep 20
    fi
    sleep 60
  done
  publish_and_push "$run" "$out"
done

say "all runs complete; final scene gallery + README"
$PY -m armforge scene --task pickplace --out docs/media/scene_gallery.png >> "$LOG" 2>&1
$PY -m armforge readme --repo . >> "$LOG" 2>&1
git add -f weights docs/media README.md
if ! git diff --cached --quiet; then
  git commit -m "docs: finalize published results, gifs and weights" >> "$LOG" 2>&1
fi
git push origin main >> "$LOG" 2>&1
say "final push exit=$?"
say "supervisor finished"
