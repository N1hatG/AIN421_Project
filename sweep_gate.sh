#!/usr/bin/env bash
set -euo pipefail

MODE="overlap10"
REF_W="3.0"
RUNS=("fis_run1.csv" "fis_run2.csv")

# Fine sweep (edit as needed)
GATES=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60)

OUTDIR="results/sweeps"
mkdir -p "$OUTDIR"

for run in "${RUNS[@]}"; do
  inpath="data/splits/fis/${run}"
  logfile="${OUTDIR}/sweep_gate_${run%.csv}_w${REF_W}.log"
  : > "$logfile"

  echo "=== Sweeping gate_relax for ${run} (ref_weight=${REF_W}) ===" | tee -a "$logfile"

  for g in "${GATES[@]}"; do
    echo "" | tee -a "$logfile"
    echo ">>> run=${run} ref_weight=${REF_W} gate_relax=${g}" | tee -a "$logfile"
    python3 src/fis_models/fis2_x7_x2x3.py \
      --input "$inpath" --mode "$MODE" \
      --ref_weight "$REF_W" --gate_relax "$g" \
      | tee -a "$logfile"
  done
done

echo ""
echo "Done. Logs saved under: $OUTDIR"
