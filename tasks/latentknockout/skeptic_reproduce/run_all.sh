#!/bin/bash
# Skeptic reproduction chain: one GPU job at a time through the shared queue, each under a RAM cap.
# usage: bash tasks/latentknockout/skeptic_reproduce/run_all.sh <out_dir>
set -u
cd /home/ec2-user/wt/latentknockout
source /home/ec2-user/ideating-rl-tests/common/env.sh >/dev/null 2>&1
export PYTHONPATH=/home/ec2-user/wt/latentknockout
PY=/opt/pytorch/bin/python
OUT=$1
CELLS=runs/latentknockout/20261002T0642_sweep/cells
mkdir -p $OUT
run() {  # label, mode, specs...
  local label=$1 mode=$2; shift 2
  echo "[$(date -u +%H:%M:%S)] start $label" >> $OUT/chain.log
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
    $PY -m common.gpuq run --gb 9 --label latentknockout-skeptic-$label -- \
    $PY -m tasks.latentknockout.skeptic_reproduce.repro $mode $CELLS $OUT "$@" > $OUT/stdout_$label.txt 2>&1
  echo "[$(date -u +%H:%M:%S)] end $label rc=$? peakRSS_kB=$(grep 'Maximum resident' $OUT/stdout_$label.txt | tail -1 | awk '{print $NF}')" >> $OUT/chain.log
}
run repro repro city_capital:Illinois:12 country_lang:French:18 langid:Turkish:18 athlete_sport:golf:18 city_capital:Texas:18
run strong strong city_state:Texas:18 athlete_sport:soccer:18 country_lang:English:18
echo "[$(date -u +%H:%M:%S)] chain done" >> $OUT/chain.log
