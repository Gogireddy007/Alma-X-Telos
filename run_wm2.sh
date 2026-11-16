#!/bin/zsh
# Detached driver for Alma-WM2: train (auto-resume on crash) -> full sealed validation. Survives session restarts.
cd "${0:A:h}"
OUT=runs/alma_wm2; LOG=runs/alma_wm2.log
for attempt in 1 2 3 4 5; do
  [ -f $OUT/ema_28000.pt ] && break
  R=""; [ -f $OUT/last.pt ] && R="--resume"
  echo "=== attempt $attempt $(date) $R" >> $LOG
  .venv/bin/python train_wm2.py --out $OUT --init-from runs/alma_wm1/ema_8000.pt --base runs/alma_v2/ema_6000.pt \
      --steps 28000 --save-every 2000 $R >> $LOG 2>&1
  sleep 30
done
[ -f $OUT/ema_28000.pt ] || { echo "TRAINING FAILED $(date)" >> $LOG; exit 1; }
RUN=$(basename $(cat validation_runs/CURRENT))
make validate RUN=$RUN CKPT=$OUT/ema_28000.pt LAST=$OUT/last.pt > validation_runs/$RUN/report/make_validate.log 2>&1
echo "VALIDATION EXIT $? $(date)" >> $LOG
