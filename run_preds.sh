#!/bin/zsh
# dump dev2 predictions of every candidate member for offline ensemble search (dev windows only)
cd "${0:A:h}"; PY=.venv/bin/python; LOG=runs/preds.log
run() { [ -f results/preds/$1.npz ] || $PY dev_eval.py --model alma --ckpt $2 --ctx $3 --n 256 --out results/preds/$1.json --save-preds results/preds/$1.npz > results/preds/$1.log 2>&1; echo "$1 exit $? $(date)" >> $LOG; }
run D runs/stageA/D_bucket_mix/ema_3000.pt 2048
run C runs/stageA/C_bucket_series/ema_3000.pt 2048
run B runs/stageA/B_bucket/ema_3000.pt 2048
run A runs/stageA/A_control/ema_3000.pt 2048
run E runs/stageB/E_short/ema_3000.pt 2048
run F runs/stageD/F_new/ema_3000.pt 2048
run G runs/stageD/G_new/ema_3000.pt 2048
run v2 runs/alma_v2/ema_6000.pt 2048
run v3_10k runs/alma_v3/ema_10000.pt 8192
run v3_10k_c2 runs/alma_v3/ema_10000.pt 2048
run v3ft3 runs/stageB/v3ft/ema_3000.pt 8192
run v3_16k runs/alma_v3/ema_16000.pt 8192
echo "preds done $(date)" >> $LOG
