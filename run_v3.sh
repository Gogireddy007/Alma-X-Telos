#!/bin/zsh
# Queued Alma v3 pipeline (starts when wm2 training + validation are done):
#   train (auto-resume) -> select best quick-dev checkpoint -> sealed forecaster validation -> full GIFT-Eval.
cd "${0:A:h}"
until grep -q "VALIDATION EXIT\|TRAINING FAILED" runs/alma_wm2.log 2>/dev/null; do sleep 120; done
OUT=runs/alma_v3; LOG=runs/alma_v3.log
for attempt in 1 2 3 4 5; do
  [ -f $OUT/ema_20000.pt ] && break
  R=""; [ -f $OUT/last.pt ] && R="--resume"
  echo "=== attempt $attempt $(date) $R" >> $LOG
  .venv/bin/python train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --steps 20000 $R >> $LOG 2>&1
  sleep 30
done
[ -f $OUT/ema_20000.pt ] || { echo "V3 TRAINING FAILED $(date)" >> $LOG; exit 1; }
BEST=$(.venv/bin/python -c "
import json
ev=[json.loads(l)['eval'] for l in open('$OUT/log.jsonl') if l.startswith('{\"eval\"')]
b=min(ev,key=lambda e:e['quick_rel_crps']); print(f\"$OUT/ema_{b['step']}.pt\")")
echo "SELECTED $BEST (best quick-dev rel-CRPS) $(date)" >> $LOG
V=$(cat validation_runs/V3_CURRENT)
echo "$BEST" > $V/prereg/selected_checkpoint.txt
VALRUN=$V .venv/bin/python validate/fc_validate.py predict --ckpt $BEST >> $V/report/fc_validate.log 2>&1
VALRUN=$V .venv/bin/python validate/fc_validate.py grade >> $V/report/fc_validate.log 2>&1
echo "V3 SEALED VALIDATION EXIT $? $(date)" >> $LOG
.venv/bin/python eval_gift.py --model alma --ckpt $BEST --context-len 8192 --device mps --out results/alma_v3_gift.json > results/alma_v3_gift.log 2>&1
echo "V3 FULL GIFT EXIT $? $(date)" >> $LOG
