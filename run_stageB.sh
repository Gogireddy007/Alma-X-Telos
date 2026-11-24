#!/bin/zsh
# Stage B (dev data only; sealed windows untouched).
#  B1  E_short: arm D recipe (bucket sampler + mixed loss) + 15% short ETS synthetic series, 3k steps from v2@6k.
#  B2  v3@10k and a v3 checkpoint soup (8k,10k,12k) on dev2 n=256.
#  B3  port the best recipe to v3: fine-tune v3@10k for 4k steps (bucket + mix [+ short if E beats D]).
#  B4  dev2 on the fine-tune checkpoints and their soup; summary vs Telos / v3@10k.
cd "${0:A:h}"
LOG=runs/stageB.log; PY=.venv/bin/python
echo "=== Stage B start $(date)" >> $LOG
dev2() { [ -f $2 ] || $PY dev_eval.py ${=1} --n 256 --out $2 > ${2%.json}.log 2>&1; echo "dev2 $2 exit $? $(date)" >> $LOG; }

# B1
out=runs/stageB/E_short
if [ ! -f $out/ema_3000.pt ]; then
  echo "--- B1 E_short $(date)" >> $LOG
  $PY train.py --out $out --init-from runs/alma_v2/ema_6000.pt --steps 3000 --bs 32 --lr 1e-4 --warmup 200 \
    --layers 8 --d 384 --window 4096 --horizon 1024 --eval-every 1000 --save-every 1000 \
    --sampler bucket --loss-norm mix --p-short 0.15 --seed 0 > $out.train.log 2>&1
  echo "train exit $? $(date)" >> $LOG
fi
dev2 "--model alma --ckpt $out/ema_3000.pt --ctx 2048" results/dev2/stageB_E_short_n256.json

# B2
dev2 "--model alma --ckpt runs/alma_v3/ema_10000.pt --ctx 8192" results/dev2/v3_10k_n256.json
dev2 "--model soup --ckpts runs/alma_v3/ema_8000.pt,runs/alma_v3/ema_10000.pt,runs/alma_v3/ema_12000.pt --ctx 8192" results/dev2/v3_soup_8_10_12_n256.json

# B3: include short series only if E beats D overall on dev2
PS=$($PY -c "
import json,numpy as np
g=lambda f:{r['task']:r['rel_crps'] for r in json.load(open(f))['per_task']}
E,D=g('results/dev2/stageB_E_short_n256.json'),g('results/dev2/stageA_D_bucket_mix_n256.json')
d=np.array([np.log(E[k]/D[k]) for k in D if k in E and E[k]>0 and D[k]>0]); print(0.15 if d.mean()<0 else 0.0)")
echo "B3 p_short=$PS $(date)" >> $LOG
OUT=runs/stageB/v3ft
for attempt in 1 2 3 4; do
  [ -f $OUT/ema_4000.pt ] && break
  R=""; [ -f $OUT/last.pt ] && R="--resume"
  echo "--- B3 v3ft attempt $attempt $R $(date)" >> $LOG
  $PY train_v3.py --out $OUT --init-from runs/alma_v2/ema_6000.pt --init-v3 runs/alma_v3/ema_10000.pt \
    --steps 4000 --lr 1e-4 --warmup 200 --eval-every 1000 --save-every 500 \
    --sampler bucket --loss-norm mix --p-short $PS --seed 0 $R >> $OUT.train.log 2>&1
  echo "train exit $? $(date)" >> $LOG
  sleep 30
done

# B4
for s in 2000 3000 4000; do dev2 "--model alma --ckpt $OUT/ema_$s.pt --ctx 8192" results/dev2/stageB_v3ft_${s}_n256.json; done
dev2 "--model soup --ckpts $OUT/ema_2000.pt,$OUT/ema_3000.pt,$OUT/ema_4000.pt --ctx 8192" results/dev2/stageB_v3ft_soup_n256.json
{
  for f in results/dev2/stageB_*_n256.json results/dev2/v3_10k_n256.json results/dev2/v3_soup_8_10_12_n256.json; do
    [ -f $f ] || continue
    echo "### $f vs Telos"; $PY dev_compare.py $f results/dev2/telos_n256.json
  done
  for f in results/dev2/stageB_v3ft_*_n256.json results/dev2/v3_soup_8_10_12_n256.json; do
    [ -f $f ] || continue
    echo "### $f vs v3@10k"; $PY dev_compare.py $f results/dev2/v3_10k_n256.json
  done
} > results/dev2/stageB_summary.txt 2>&1
echo "=== Stage B done $(date)" >> $LOG
