#!/bin/bash
# Feasibility chain: one GPU job at a time, each through the shared queue, each under a RAM cap.
# usage: bash tasks/latentknockout/run_feasibility.sh <validate_run_dir> <sweep_run_dir> [steps...]
# steps: L18 L12 L6 seeds cc report   (default: all, in that order); L6 is limited by L6_MAX_GROUPS (default 4); seeds = seed 1 at L18, 4 families, reference only
set -u
cd /home/ec2-user/wt/latentknockout
source /home/ec2-user/ideating-rl-tests/common/env.sh >/dev/null 2>&1
export PYTHONPATH=/home/ec2-user/wt/latentknockout
PY=/opt/pytorch/bin/python
VAL=$1/validated.json
OUT=$2
shift 2
STEPS=${@:-L18 L12 L6 seeds cc report}
mkdir -p $OUT
run() {  # label, gb, args...
  local label=$1 gb=$2; shift 2
  echo "[$(date -u +%H:%M:%S)] start $label" >> $OUT/chain.log
  systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=2G -- /usr/bin/time -v \
    $PY -m common.gpuq run --gb $gb --heavy --label latentknockout-$label -- $PY "$@" >> $OUT/log_$label.txt 2>&1
  echo "[$(date -u +%H:%M:%S)] end $label rc=$? peakRSS_kB=$(grep 'Maximum resident' $OUT/log_$label.txt | tail -1 | awk '{print $NF}')" >> $OUT/chain.log
}
for s in $STEPS; do
  case $s in
    L12|L18) run sweep-${s} 8 -m tasks.latentknockout.sweep $VAL $OUT/cells --layers ${s#L} ;;
    L6) run sweep-L6 8 -m tasks.latentknockout.sweep $VAL $OUT/cells --layers 6 --max_groups ${L6_MAX_GROUPS:-4} ;;
    cc) run sweep-cc 8 -m tasks.latentknockout.sweep $VAL $OUT/cells --layers 18 --families country_capital --max_groups 10 ;;
    seeds) run sweep-seed1 8 -m tasks.latentknockout.sweep $VAL $OUT/cells_seeds --layers 18 --seed 1 --refonly --families city_state country_lang athlete_sport langid ;;
    report) run latent-report 8 -m tasks.latentknockout.latent_report $VAL $OUT/cells $OUT/latent_report.json --max_cells 0 ;;
  esac
done
echo "[$(date -u +%H:%M:%S)] chain done" >> $OUT/chain.log
