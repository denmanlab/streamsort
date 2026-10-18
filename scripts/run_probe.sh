#!/bin/bash
# Full per-probe pipeline.  Usage: REC=jlh34 PROBE=B bash scripts/run_probe.sh [--from STAGE]
# Stages: prep sort gate bank whiten validate inject waveforms geom7 geom8 demo figures
set -o pipefail
TAG="${REC}_probe${PROBE}"; OUT="results/${TAG}"; mkdir -p "$OUT"; LOG="$OUT/pipeline.log"
FROM="${2:-prep}"; started=0
stage() { local name=$1; shift; if [ $started -eq 0 ] && [ "$name" != "$FROM" ]; then return 0; fi; started=1
  echo "=== [$TAG] $name  $(date '+%H:%M:%S')" | tee -a "$LOG"; local t0=$SECONDS
  "$@" >> "$OUT/${name}.log" 2>&1; local rc=$?
  echo "    -> rc=$rc  $((SECONDS-t0)) s" | tee -a "$LOG"; tail -2 "$OUT/${name}.log" | cut -c1-300 >> "$LOG"
  if [ $rc -ne 0 ]; then echo "STOPPED at $name" | tee -a "$LOG"; exit $rc; fi; }
stage prep      python scripts/prep_session.py
stage sort      python scripts/fit_initial_sort.py --seconds 600 --sorter kilosort4 --n-jobs 6
stage gate      python scripts/gate_match.py --tag seg600 --sorter kilosort4
stage bank      python scripts/build_template_bank.py
stage whiten    python scripts/est_whitening.py
stage validate  env SUPPRESS=1 python scripts/validate_online.py
stage inject    env N_INJ=12 python scripts/exp_artifact_inject.py
stage waveforms python scripts/diag_early_waveforms.py
stage geom7     env SUF=_sup python scripts/geometry_step7.py
stage geom8     python scripts/geometry_step8.py
stage demo      env DEMO_SPEED=1 python scripts/demo_live.py
stage figures   python scripts/make_figures.py
tar -czf "results_${TAG}.tar.gz" -C results "$TAG" && echo "=== [$TAG] bundled results_${TAG}.tar.gz" | tee -a "$LOG"
echo "=== [$TAG] DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
