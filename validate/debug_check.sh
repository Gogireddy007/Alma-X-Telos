#!/bin/zsh
# Debug-grade a checkpoint on DEBUG_harness (development data only; never sealed holdouts).
cd "${0:A:h}/.."
CK=$1; LAST=${2:-runs/alma_wm1/last.pt}; D=validation_runs/DEBUG_harness
until [ -f $CK ]; do sleep 60; done; sleep 20
rm -f $D/manifest.json $D/sealed_outputs/*
for s in sim_val sim_test sim_tanks plan_test real; do for f in inputs truth; do VALRUN=$D .venv/bin/python -c "
import sys; sys.path.insert(0,'validate'); from common import *; manifest_add('harness', RUN/'harness/${f}_${s}.npz')"; done; done
VALRUN=$D .venv/bin/python validate/predict.py --ckpt $CK --no-telos > $D/report/predict_debug.log 2>&1 || tail -20 $D/report/predict_debug.log
VALRUN=$D .venv/bin/python validate/grade.py --ckpt $CK --last $LAST 2>&1 | grep -E "^\[" | grep -E "\] (4 |5\.1|5\.6|6\.1|6\.2|6\.5\*|7 /|7\.4|9\.2|9\.7|12 |12\.1|4\.2)" | cut -c 23-300
