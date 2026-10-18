#!/bin/bash
# Phase 1 validation for a whole recording: every probe, then the pooled population.
#   bash scripts/run_recording.sh jlh32 [A,B,C] [--skip-probes]
# Per-probe outputs: results/<rec>_probe<P>/, figures/<rec>_probe<P>/, reports/report_<rec>_probe<P>.md
# Pooled outputs:    results/<rec>_population/, figures/<rec>_population/, reports/<rec>_population_report.md
REC=$1; PROBES=${2:-A,B,C}; SKIP=$3; LOG="results/${REC}_population.log"; mkdir -p results reports
if [ "$SKIP" != "--skip-probes" ]; then
  for P in ${PROBES//,/ }; do
    REC=$REC PROBE=$P bash scripts/run_probe.sh --from prep || { echo "probe $P failed" | tee -a "$LOG"; exit 1; }
    REC=$REC PROBE=$P python scripts/make_report.py >> "$LOG" 2>&1
  done
fi
for step in "population_stage.py" "window_analysis.py" "demo_population.py" "make_population_figures.py" "make_population_report.py"; do
  echo "=== [$REC population] $step $(date '+%H:%M:%S')" | tee -a "$LOG"; t0=$SECONDS
  REC=$REC PROBES=$PROBES python scripts/$step >> "$LOG" 2>&1 || { echo "STOPPED at $step" | tee -a "$LOG"; exit 1; }
  echo "    -> $((SECONDS-t0)) s" | tee -a "$LOG"
done
echo "=== [$REC population] DONE $(date '+%H:%M:%S')" | tee -a "$LOG"
