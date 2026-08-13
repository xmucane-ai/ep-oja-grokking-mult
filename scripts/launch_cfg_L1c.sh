#!/bin/bash
# launch_cfg_L1c.sh -- Arm L1c: Fixed-encoder control (Fourier-in-depth), 10 seeds
cd /root/gate2/scripts
OUT_DIR=/root/gate2/outputs
export OUT_DIR
/root/gate2/venv/bin/python -u run_language_track_cfg.py --arm L1c --seeds 0-9 \
  --rd-steps 3000 \
  --output $OUT_DIR/cfg_L1c_fixed_enc_seeds0-9.json \
  > $OUT_DIR/cfg_L1c_fixed_enc.log 2>&1
echo "L1c DONE exit=$?" >> $OUT_DIR/cfg_L1c_fixed_enc.log
