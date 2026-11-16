#!/bin/zsh
# Queued Alma v3 forecaster: starts after wm2 training + validation finish (GPU free). Auto-resumes on crash.
cd "${0:A:h}"
until grep -q "VALIDATION EXIT\|TRAINING FAILED" runs/alma_wm2.log 2>/dev/null; do sleep 120; done
OUT=runs/alma_v3; LOG=runs/alma_v3.log
for attempt in 1 2 3 4 5; do
  [ -f $OUT/ema_20000.pt ] && break
  R=(); [ -f $OUT/last.pt ] && R=(--resume $OUT/last.pt)
  echo "=== attempt $attempt $(date) $R" >> $LOG
  .venv/bin/python train.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --layers 12 --d 384 --horizon 1024 \
      --window 4096 --bs 32 --lr 3e-4 --warmup 500 --steps 20000 --save-every 2000 --eval-every 2000 $R >> $LOG 2>&1
  sleep 30
done
echo "V3 DONE $(date)" >> $LOG
