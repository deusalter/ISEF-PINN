#!/bin/bash
# Retrain 9 missing/corrupted 7-day NeuralODE satellites
# Run with: bash retrain_missing.sh

cd "$(dirname "$0")"

SATS=(44883 57320 58536 46984 44713 44714 44715 44716 36508)
TOTAL=${#SATS[@]}
DONE=0

echo "========================================"
echo "Starting retraining of $TOTAL satellites"
echo "Start time: $(date)"
echo "========================================"

for sat in "${SATS[@]}"; do
    DONE=$((DONE + 1))
    echo ""
    echo "========================================"
    echo "[$DONE/$TOTAL] Training satellite $sat"
    echo "Start: $(date)"
    echo "========================================"

    # Skip if result already exists (in case of restart)
    if [ -f "data/gmat_results/${sat}_comparison_7day.json" ]; then
        echo "Result already exists for $sat, skipping."
        continue
    fi

    python compare_pinn_vs_sgp4.py --long-arc --sat "$sat"

    if [ -f "data/gmat_results/${sat}_comparison_7day.json" ]; then
        echo "SUCCESS: $sat completed"
    else
        echo "WARNING: $sat may have failed (no result file)"
    fi
done

echo ""
echo "========================================"
echo "All satellites processed. End time: $(date)"
echo "========================================"
echo ""
echo "Run 'python recompute_sgp4.py --long-arc' to regenerate summary."
