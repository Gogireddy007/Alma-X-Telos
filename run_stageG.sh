#!/bin/zsh
# Stage G: if a Stage F boost arm beats D (geo ratio < 0.995 on all 97 dev2 tasks), fine-tune v3@10k with that boost
# (1500 steps, ~2 h), dump its preds at ctx 8192 and re-run the honest ensemble search. dev2 only.
cd "${0:A:h}"; setopt NULL_GLOB; PY=.venv/bin/python; LOG=runs/stageG.log
until grep -q "Stage F done" runs/stageF.log 2>/dev/null; do sleep 60; done
TC="PEMS*,LOS_LOOP,Q-TRAFFIC,traffic_hourly,*METRO,BEIJING_SUBWAY_30MIN"; CL="alibaba*,azure*,borg*"
DEC=$($PY - <<'P'
import json, numpy as np, os
g = lambda f: {r["task"]: r["rel_crps"] for r in json.load(open(f))["per_task"]}
D = g("results/preds/D.json"); best = ("none", 9.0)
for a in ["H_boost", "H_boost5", "H_a03"]:
    f = f"results/preds/{a}.json"
    if os.path.exists(f):
        A = g(f); r = float(np.exp(np.mean([np.log(A[k] / D[k]) for k in D if k in A])))
        best = min(best, (a, r), key=lambda x: x[1])
print(best[0], f"{best[1]:.4f}")
P
)
echo "=== Stage G start $(date) best boost arm vs D: $DEC" >> $LOG
set -- ${=DEC}
if [ $(echo "$2 < 0.995" | bc) = 1 ]; then
  F=3; AL=0.5; [ $1 = H_boost5 ] && F=5; [ $1 = H_a03 ] && AL=0.3
  OUT=runs/stageG/v3boost; mkdir -p runs/stageG
  for attempt in 1 2 3; do
    [ -f $OUT/ema_1500.pt ] && break
    RS=""; [ -f $OUT/last.pt ] && RS="--resume"
    echo "--- v3boost attempt $attempt F=$F alpha=$AL $(date)" >> $LOG
    ALMA_BOOST="$TC:$F;$CL:$F" $PY train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --init-v3 runs/alma_v3/ema_10000.pt \
      --steps 1500 --lr 1e-4 --warmup 200 --eval-every 1500 --save-every 500 --sampler bucket --alpha $AL --loss-norm mix \
      --seed 3 --real data/pretrain,data/pretrain2 $RS >> $OUT.train.log 2>&1
    echo "train exit $? $(date)" >> $LOG; sleep 20
  done
  [ -f $OUT/ema_1500.pt ] && $PY dev_eval.py --model alma --ckpt $OUT/ema_1500.pt --ctx 8192 --n 256 \
    --out results/preds/v3boost.json --save-preds results/preds/v3boost.npz > results/preds/v3boost.log 2>&1
  echo "dev2 v3boost exit $? $(date)" >> $LOG
  $PY ens_search.py --route > results/ens_search_G.txt 2>&1
else
  echo "G skipped: no boost arm beat D" >> $LOG
fi
$PY fam_report.py > results/fam_report_G.txt 2>&1
$PY dev_rank.py --ref results/preds/telos.json results/preds/*.json > results/progress_morning.txt 2>&1
echo "=== Stage G done $(date)" >> $LOG
