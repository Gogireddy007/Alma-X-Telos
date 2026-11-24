#!/bin/zsh
# Stage E (overnight): 1) honest ensemble search over saved dev2 predictions;
# 2) v3 continued pretraining on BOTH corpora (proper data test: the 3k-step small A/B in Stage D cannot show a
#    data-repetition effect) with the best small-model recipe (bucket, mix) -> also a new ensemble member;
# 3) dump its predictions, re-run the search. Dev windows only; sealed windows untouched.
cd "${0:A:h}"; setopt NULL_GLOB
PY=.venv/bin/python; LOG=runs/stageE.log; REAL=data/pretrain,data/pretrain2
until grep -q "preds done" runs/preds.log 2>/dev/null && ! pgrep -f "model telos" >/dev/null; do sleep 60; done
echo "=== Stage E start $(date)" >> $LOG
$PY ens_search.py --route > results/ens_search_1.txt 2>&1; echo "ens_search_1 exit $? $(date)" >> $LOG
OUT=runs/stageE/v3new; mkdir -p runs/stageE
for attempt in 1 2 3 4 5; do
  [ -f $OUT/ema_10000.pt ] && break
  RS=""; [ -f $OUT/last.pt ] && RS="--resume"
  echo "--- v3new attempt $attempt $RS $(date)" >> $LOG
  $PY train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --init-v3 runs/alma_v3/ema_10000.pt \
    --steps 10000 --lr 1.5e-4 --warmup 500 --eval-every 2000 --save-every 500 --sampler bucket --loss-norm mix \
    --seed 2 --real $REAL $RS >> $OUT.train.log 2>&1
  echo "train exit $? $(date)" >> $LOG; sleep 30
done
for s in 10000 6000 2000; do
  [ -f $OUT/ema_$s.pt ] && [ ! -f results/preds/v3new_$s.npz ] && $PY dev_eval.py --model alma --ckpt $OUT/ema_$s.pt \
    --ctx 8192 --n 256 --out results/preds/v3new_$s.json --save-preds results/preds/v3new_$s.npz > results/preds/v3new_$s.log 2>&1
  echo "preds v3new_$s exit $? $(date)" >> $LOG
done
$PY ens_search.py --route > results/ens_search_2.txt 2>&1; echo "ens_search_2 exit $? $(date)" >> $LOG
$PY dev_rank.py results/dev2/*_n256.json results/preds/*.json > results/dev2/progress_summary.txt 2>/dev/null
echo "=== Stage E done $(date)" >> $LOG
