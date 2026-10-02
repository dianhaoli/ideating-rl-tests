#!/bin/bash
# Second chain (time box): strong search + oracle with beam width 1 and smaller pools on two more infeasible-labelled
# cells, then strict-effect oracles on 3 reference-feasible cells. One GPU job at a time.
# usage: bash tasks/latentknockout/skeptic_reproduce/run_light.sh <out_dir>
set -u
cd /home/ec2-user/wt/latentknockout
source /home/ec2-user/ideating-rl-tests/common/env.sh >/dev/null 2>&1
export PYTHONPATH=/home/ec2-user/wt/latentknockout
PY=/opt/pytorch/bin/python
OUT=$1
CELLS=runs/latentknockout/20261002T0642_sweep/cells
until grep -q "chain done" $OUT/chain.log 2>/dev/null; do sleep 10; done
run() {  # label, mode, specs...
  local label=$1 mode=$2; shift 2
  echo "[$(date -u +%H:%M:%S)] start $label" >> $OUT/chain.log
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
    $PY -m common.gpuq run --gb 9 --label latentknockout-skeptic-$label -- \
    $PY -m tasks.latentknockout.skeptic_reproduce.repro $mode $CELLS $OUT "$@" > $OUT/stdout_$label.txt 2>&1
  echo "[$(date -u +%H:%M:%S)] end $label rc=$? peakRSS_kB=$(grep 'Maximum resident' $OUT/stdout_$label.txt | tail -1 | awk '{print $NF}')" >> $OUT/chain.log
}
LK_BEAM=1 LK_POOL=30,15,15,15 run strong2 strong athlete_sport:soccer:18 country_lang:English:18
LK_BEAM=1 run strict strict city_state:Arizona:18 city_capital:Illinois:12 city_capital:Texas:18
echo "[$(date -u +%H:%M:%S)] chain2 done" >> $OUT/chain.log
