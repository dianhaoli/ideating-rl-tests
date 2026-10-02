#!/bin/bash
# Strict-effect oracle on reference-feasible cells (after run_all.sh; one GPU job at a time).
# usage: bash tasks/latentknockout/skeptic_reproduce/run_strict.sh <out_dir>
set -u
cd /home/ec2-user/wt/latentknockout
source /home/ec2-user/ideating-rl-tests/common/env.sh >/dev/null 2>&1
export PYTHONPATH=/home/ec2-user/wt/latentknockout
PY=/opt/pytorch/bin/python
OUT=$1
until grep -q "chain done" $OUT/chain.log 2>/dev/null; do sleep 10; done
echo "[$(date -u +%H:%M:%S)] start strict" >> $OUT/chain.log
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=2G -- /usr/bin/time -v \
  $PY -m common.gpuq run --gb 9 --label latentknockout-skeptic-strict -- \
  $PY -m tasks.latentknockout.skeptic_reproduce.repro strict runs/latentknockout/20261002T0642_sweep/cells $OUT \
  city_state:Arizona:18 athlete_sport:golf:18 city_capital:Illinois:12 city_capital:Texas:18 > $OUT/stdout_strict.txt 2>&1
echo "[$(date -u +%H:%M:%S)] end strict rc=$? peakRSS_kB=$(grep 'Maximum resident' $OUT/stdout_strict.txt | tail -1 | awk '{print $NF}')" >> $OUT/chain.log
