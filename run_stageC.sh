#!/bin/zsh
# Stage C (dev data only), starts automatically after Stage B.
#  C1  are late v3 checkpoints better on unbiased dev2? (16k, 20k)
#  C2  inference ensembles: context-length ensemble of the best v3-family model; v3-family + small model.
#  C3  next training, chosen by rule: if the v3 fine-tune beat v3@10k (all-task ratio < 0.995) continue it
#      6k more steps at lr 5e-5; otherwise try the per-series loss on v3@10k for 4k steps.
#  C4  dev2 on the new run, rebuild ensembles with it, write the ranked summary.
cd "${0:A:h}"
LOG=runs/stageC.log; PY=.venv/bin/python; D2=results/dev2
until grep -q "Stage B done" runs/stageB.log 2>/dev/null; do sleep 120; done
echo "=== Stage C start $(date)" >> $LOG
dev2() { [ -f $2 ] || $PY dev_eval.py ${=1} --n 256 --out $2 > ${2%.json}.log 2>&1; echo "dev2 $2 exit $? $(date)" >> $LOG; }
ckpt_of() { $PY -c "
import json,os,sys; f=sys.argv[1]; r=json.load(open(f))
print(os.path.splitext(f)[0]+'_soup.pt' if r['model']=='soup' else r['ckpt'])" $1; }

# C1
dev2 "--model alma --ckpt runs/alma_v3/ema_16000.pt --ctx 8192" $D2/v3_16k_n256.json
dev2 "--model alma --ckpt runs/alma_v3/ema_20000.pt --ctx 8192" $D2/v3_20k_n256.json

ensembles() {  # $1 tag
  BV=$($PY dev_rank.py --best $D2/v3_*_n256.json $D2/stageB_v3ft_*_n256.json $D2/stageC_v3*_n256.json 2>/dev/null)
  BS=$($PY dev_rank.py --best $D2/stageA_D_bucket_mix_n256.json $D2/stageB_E_short_n256.json 2>/dev/null)
  CV=$(ckpt_of $BV); CS=$(ckpt_of $BS)
  echo "ensembles[$1]: best v3-family $BV -> $CV ; best small $BS -> $CS" >> $LOG
  dev2 "--model ens --members $CV@2048,$CV@4096,$CV@8192" $D2/ens_$1_ctx3_n256.json
  dev2 "--model ens --members $CV@8192,$CS@2048" $D2/ens_$1_v3+small_n256.json
  dev2 "--model ens --members $CV@8192,$CV@4096,$CS@2048" $D2/ens_$1_v3x2+small_n256.json
}
# C2
ensembles c2

# C3
CONT=$($PY -c "
import json,glob,numpy as np
ref={r['task']:r['rel_crps'] for r in json.load(open('$D2/v3_10k_n256.json'))['per_task']}
best=1e9
for f in glob.glob('$D2/stageB_v3ft_*_n256.json'):
    x={r['task']:r['rel_crps'] for r in json.load(open(f))['per_task']}
    d=[np.log(x[k]/ref[k]) for k in ref if k in x and x[k]>0 and ref[k]>0]; best=min(best,float(np.exp(np.mean(d))))
print(1 if best<0.995 else 0)")
PS=$(grep -o "B3 p_short=[0-9.]*" runs/stageB.log | tail -1 | cut -d= -f2); PS=${PS:-0.0}
if [ "$CONT" = 1 ]; then
  BF=$($PY dev_rank.py --best $D2/stageB_v3ft_*_n256.json); INIT=$(ckpt_of $BF)
  NAME=v3ft2; STEPS=6000; LR=5e-5; LN=mix
else
  INIT=runs/alma_v3/ema_10000.pt; NAME=v3ser; STEPS=4000; LR=1e-4; LN=series
fi
OUT=runs/stageC/$NAME; mkdir -p runs/stageC
echo "C3 rule CONT=$CONT -> $NAME init=$INIT steps=$STEPS lr=$LR loss=$LN p_short=$PS $(date)" >> $LOG
for attempt in 1 2 3 4; do
  [ -f $OUT/ema_$STEPS.pt ] && break
  R=""; [ -f $OUT/last.pt ] && R="--resume"
  echo "--- C3 $NAME attempt $attempt $R $(date)" >> $LOG
  $PY train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --init-v3 $INIT --steps $STEPS --lr $LR \
    --warmup 200 --eval-every 2000 --save-every 500 --sampler bucket --loss-norm $LN --p-short $PS --seed 1 $R >> $OUT.train.log 2>&1
  echo "train exit $? $(date)" >> $LOG
  sleep 30
done

# C4
L=""
for s in $(seq 2000 2000 $STEPS); do dev2 "--model alma --ckpt $OUT/ema_$s.pt --ctx 8192" $D2/stageC_${NAME}_${s}_n256.json; L="$L,$OUT/ema_$s.pt"; done
dev2 "--model soup --ckpts ${L#,} --ctx 8192" $D2/stageC_${NAME}_soup_n256.json
ensembles c4
{
  echo "# dev2 ranking vs Telos (n=256 random windows/task, seed-0 dev sample; sealed windows never used)"
  $PY dev_rank.py $D2/*_n256.json 2>/dev/null
  echo; echo "# vs v3@10k"
  $PY dev_rank.py --ref $D2/v3_10k_n256.json $D2/v3_*_n256.json $D2/stageB_v3ft_*_n256.json $D2/stageC_*_n256.json $D2/ens_*_n256.json 2>/dev/null
} > $D2/progress_summary.txt
echo "=== Stage C done $(date)" >> $LOG
