#!/usr/bin/env bash
# launch_alphabet_arm2_probe.sh — Arm 2 sample-complexity probe phase
# 1 seed × 7 N_acq points × [L=2 acquire + L=6 readout]
# Expected: ~4.4h on 3060
set -e

cd /root/gate2/scripts
export OUT_DIR=/root/gate2/outputs

echo "=========================================="
echo "ALPHABET ACQUISITION ARM 2 (PROBE)"
echo "$(date)"
echo "=========================================="

/root/gate2/venv/bin/python -u run_alphabet_acquisition.py \
    --arm arm2 \
    --phase probe \
    --seeds 0 \
    --acq-steps 3000 \
    --rd-steps 3000 \
    --eta-enc 0.01 \
    2>&1 | tee $OUT_DIR/alphabet_arm2_probe.log

echo "=========================================="
echo "ARM 2 PROBE COMPLETE: $(date)"
echo "=========================================="
