#!/usr/bin/env bash
set -euo pipefail

MODE="overlap10"
RUNS=("fis_run1.csv" "fis_run2.csv")

REFS=(1.0 1.5 2.0 2.5 3.0 3.5)
GATES=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55)

OUTDIR="results/sweeps"
mkdir -p "$OUTDIR"

for run in "${RUNS[@]}"; do
  inpath="data/splits/fis/${run}"
  logfile="${OUTDIR}/sweep_grid_${run%.csv}.log"
  : > "$logfile"

  echo "=== Grid sweep for ${run} ===" | tee -a "$logfile"

  for w in "${REFS[@]}"; do
    for g in "${GATES[@]}"; do
      echo "" | tee -a "$logfile"
      echo ">>> run=${run} ref_weight=${w} gate_relax=${g}" | tee -a "$logfile"
      python3 src/fis_models/fis2_x7_x2x3.py \
        --input "$inpath" --mode "$MODE" \
        --ref_weight "$w" --gate_relax "$g" \
        | tee -a "$logfile"
    done
  done
done

echo ""
echo "Done. Logs saved under: $OUTDIR"
