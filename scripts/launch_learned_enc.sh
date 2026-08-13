#!/bin/bash
# launch_learned_enc.sh — launches the learned-encoder experiment via the slot guard.
#
# USAGE (after BOTH launch gates clear):
#   ssh root@training-container "/root/gate2/scripts/run_slot.sh /root/gate2/scripts/launch_learned_enc.sh"
#
# GATES (BINDING — from card t_f7fedbfe):
#   1. Coex1/C3 verdict (t_823511d6) must be reported
#   2. Container capacity must be < 2/2
#
# The slot guard refuses to launch when ≥2 runs active or cgroup > 85%.
# If it refuses, WAIT for a run to finish or KILL one first.

set -e
cd /root/gate2/scripts
mkdir -p /root/gate2/outputs_learned_enc

echo "=== LEARNED ENCODER L=6 (SPEC_LEARNED_ENCODER_L6_v1.2) ==="
echo "Launch time: $(date)"
echo ""

# §7.5 coverage gate FIRST (runs inside the python script too)
echo "Phase 0: coverage gate + phase 1 (add sanity, 3 seeds)"
/root/gate2/venv/bin/python -u run_learned_encoder.py \
    --phase 1 --seeds 0 1 2 --steps 3000 --eval_every 100 \
    --output /root/gate2/outputs_learned_enc/learned_enc_phase1.json \
    2>&1 | tee /root/gate2/outputs_learned_enc/learned_enc_phase1.log

echo ""
echo "Phase 2: PRIMARY — mult+add joint (10 seeds, 3000 steps)"
/root/gate2/venv/bin/python -u run_learned_encoder.py \
    --phase 2 --seeds 0 1 2 3 4 5 6 7 8 9 --steps 3000 --eval_every 100 \
    --output /root/gate2/outputs_learned_enc/learned_enc_phase2.json \
    2>&1 | tee /root/gate2/outputs_learned_enc/learned_enc_phase2.log

echo ""
echo "Phase 3: W_enc-ablated control (10 seeds, 3000 steps)"
/root/gate2/venv/bin/python -u run_learned_encoder.py \
    --phase 3 --seeds 0 1 2 3 4 5 6 7 8 9 --steps 3000 --eval_every 100 \
    --output /root/gate2/outputs_learned_enc/learned_enc_phase3.json \
    2>&1 | tee /root/gate2/outputs_learned_enc/learned_enc_phase3.log

echo ""
echo "=== ALL PHASES COMPLETE — $(date) ==="
echo "Results in /root/gate2/outputs_learned_enc/"
