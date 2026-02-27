#!/bin/bash
set -e

echo "=========================================="
echo "  PHASE 1: 5-orbit PINN vs SGP4 (20 sats)"
echo "=========================================="
python compare_pinn_vs_sgp4.py 2>&1 | tee outputs/compare_5orbit.log

echo ""
echo "=========================================="
echo "  PHASE 2: Generate 7-day GMAT data"
echo "=========================================="
python generate_gmat_data.py --long-arc 2>&1 | tee outputs/generate_7day.log

echo ""
echo "=========================================="
echo "  PHASE 3: 7-day PINN vs SGP4 (20 sats)"
echo "=========================================="
python compare_pinn_vs_sgp4.py --long-arc 2>&1 | tee outputs/compare_7day.log

echo ""
echo "=========================================="
echo "  ALL PHASES COMPLETE"
echo "=========================================="
