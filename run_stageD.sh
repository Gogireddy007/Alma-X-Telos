#!/bin/zsh
# Stage D: does MORE real data help? (dev data only; sealed windows untouched)
#  D1  small-model A/B, 3k steps from v2@6k, identical to Stage A arms except --real adds data/pretrain2:
#        F_new  = A_control recipe (sqrt, token) + new data   -> compare with A_control
#        G_new  = D recipe (bucket, mix)        + new data   -> compare with D_bucket_mix
#  D2  if new data helps (best of F/G vs its control < 0.997 on all 97 tasks): v3 continued pretraining from
#      v3@10k on both corpora with that recipe, 12k steps (~12 h), checkpoints every 2k.
#  D3  dev2 on every v3 checkpoint, then ensemble of the best with D.
cd "${0:A:h}"
setopt NULL_GLOB
LOG=runs/stageD.log; PY=.venv/bin/python; D2=results/dev2; REAL=data/pretrain,data/pretrain2
dev2() { [ -f $2 ] || $PY dev_eval.py ${=1} --n 256 --out $2 > ${2%.json}.log 2>&1; echo "dev2 $2 exit $? $(date)" >> $LOG; }
until grep -q "^done:" data/prep_pretrain2.log 2>/dev/null && ! pgrep -f prep_pretrain2.py >/dev/null; do sleep 60; done
until grep -q "ensembles done" runs/stageC.log 2>/dev/null; do sleep 60; done
echo "=== Stage D start $(date) corpus2: $(grep '^done:' data/prep_pretrain2.log)" >> $LOG
mkdir -p runs/stageD
for arm in "F_new sqrt token" "G_new bucket mix"; do
  set -- ${=arm}; name=$1; out=runs/stageD/$name
  if [ ! -f $out/ema_3000.pt ]; then
    echo "--- D1 $name sampler=$2 loss=$3 $(date)" >> $LOG
    $PY train.py --out $out --init-from runs/alma_v2/ema_6000.pt --steps 3000 --bs 32 --lr 1e-4 --warmup 200 \
      --layers 8 --d 384 --window 4096 --horizon 1024 --eval-every 1000 --save-every 1000 \
      --sampler $2 --loss-norm $3 --seed 0 --real $REAL > $out.train.log 2>&1
    echo "train exit $? $(date)" >> $LOG
  fi
  dev2 "--model alma --ckpt $out/ema_3000.pt --ctx 2048" $D2/stageD_${name}_n256.json
done
DEC=$($PY - <<'P'
import json, numpy as np
g = lambda f: {r["task"]: r["rel_crps"] for r in json.load(open(f))["per_task"]}
def ratio(a, b):
    A, B = g(a), g(b)
    return float(np.exp(np.mean([np.log(A[k] / B[k]) for k in B if k in A and A[k] > 0 and B[k] > 0])))
D = "results/dev2/"
f = ratio(D + "stageD_F_new_n256.json", D + "stageA_A_control_n256.json")
gg = ratio(D + "stageD_G_new_n256.json", D + "stageA_D_bucket_mix_n256.json")
best = ("sqrt token", f) if f <= gg else ("bucket mix", gg)
print(f"{best[0]} {best[1]:.4f} F={f:.4f} G={gg:.4f}")
P
)
echo "D1 decision: $DEC $(date)" >> $LOG
set -- ${=DEC}; SMP=$1; LN=$2; R=$3
if [ $(echo "$R < 0.997" | bc) = 1 ]; then
  OUT=runs/stageD/v3new
  for attempt in 1 2 3 4 5; do
    [ -f $OUT/ema_12000.pt ] && break
    RS=""; [ -f $OUT/last.pt ] && RS="--resume"
    echo "--- D2 v3new attempt $attempt $RS sampler=$SMP loss=$LN $(date)" >> $LOG
    $PY train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --init-v3 runs/alma_v3/ema_10000.pt \
      --steps 12000 --lr 1.5e-4 --warmup 500 --eval-every 2000 --save-every 500 --sampler $SMP --loss-norm $LN \
      --seed 2 --real $REAL $RS >> $OUT.train.log 2>&1
    echo "train exit $? $(date)" >> $LOG
    sleep 30
  done
  for s in 2000 4000 6000 8000 10000 12000; do
    [ -f $OUT/ema_$s.pt ] && dev2 "--model alma --ckpt $OUT/ema_$s.pt --ctx 8192" $D2/stageD_v3new_${s}_n256.json
  done
  BV=$($PY dev_rank.py --best $D2/stageD_v3new_*_n256.json)
  CV=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['ckpt'])" $BV)
  dev2 "--model ens --members $CV@8192,runs/stageA/D_bucket_mix/ema_3000.pt@2048" $D2/ens_v3new+D_n256.json
else
  echo "D2 skipped: new data did not help the small model (ratio $R)" >> $LOG
fi
$PY dev_rank.py $D2/*_n256.json > $D2/progress_summary.txt 2>/dev/null
echo "=== Stage D done $(date)" >> $LOG
