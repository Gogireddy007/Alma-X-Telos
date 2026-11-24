#!/bin/zsh
# Stage A recipe search (dev data only). Each arm: 3,000 steps from v2@6k, identical except the named change.
cd "${0:A:h}"
LOG=runs/stageA.log
while pgrep -f dev_eval.py >/dev/null; do sleep 30; done
echo "=== Stage A start $(date)" >> $LOG
for arm in "A_control sqrt token" "B_bucket bucket token" "C_bucket_series bucket series" "D_bucket_mix bucket mix"; do
  set -- ${=arm}
  name=$1; smp=$2; ln=$3; out=runs/stageA/$name
  if [ -f $out/ema_3000.pt ]; then echo "skip $name" >> $LOG; continue; fi
  echo "--- $name sampler=$smp loss=$ln $(date)" >> $LOG
  .venv/bin/python train.py --out $out --init-from runs/alma_v2/ema_6000.pt --steps 3000 --bs 32 --lr 1e-4 \
    --warmup 200 --layers 8 --d 384 --window 4096 --horizon 1024 --eval-every 1000 --save-every 1000 \
    --sampler $smp --loss-norm $ln --seed 0 > $out.train.log 2>&1
  echo "train exit $? $(date)" >> $LOG
  .venv/bin/python dev_eval.py --model alma --ckpt $out/ema_3000.pt --ctx 2048 --n 256 \
    --out results/dev2/stageA_${name}_n256.json > results/dev2/stageA_${name}_n256.log 2>&1
  echo "dev2 exit $? $(date)" >> $LOG
done
echo "=== Stage A done $(date)" >> $LOG
