#!/bin/bash
# launch_cfg_L1b.sh -- Arm L1b: Task B (Dyck-2), 10 seeds
cd /root/gate2/scripts
OUT_DIR=/root/gate2/outputs
export OUT_DIR
/root/gate2/venv/bin/python -u run_language_track_cfg.py --arm L1b --seeds 0-9 \
  --acq-steps 3000 --rd-steps 3000 \
  --output $OUT_DIR/cfg_L1b_dyck2_seeds0-9.json \
  > $OUT_DIR/cfg_L1b_dyck2.log 2>&1
echo "L1b DONE exit=$?" >> $OUT_DIR/cfg_L1b_dyck2.log
