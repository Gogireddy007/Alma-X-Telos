#!/bin/zsh
# When Alma v1 finishes: rename its run to alma_v1, run the full GIFT-Eval on its final EMA (CPU),
# and start Alma v2 on the GPU (warm start, 1024-step horizon, 4096-step windows) with checkpoint tracking.
cd "${0:A:h}"
while pgrep -f "train.py --out runs/meridian_s" >/dev/null; do sleep 30; done
while pgrep -f "track.sh runs/meridian_s" >/dev/null; do sleep 15; done  # let it score the final checkpoint
mv runs/meridian_s runs/alma_v1 && mv runs_meridian_s.log runs/alma_v1/train.log
mv results/track_meridian_s.log results/track_alma_v1.log 2>/dev/null
mv results/curve_meridian_s.jsonl results/curve_alma_v1.jsonl 2>/dev/null
for f in results/quick_meridian_s_*.json(N); do mv $f ${f/meridian_s/alma_v1}; done
echo "V1_DONE $(date)"
final=runs/alma_v1/ema_20000.pt
[ -f $final ] || { echo "V1_FINAL_MISSING"; exit 1; }

nohup .venv/bin/python train.py --out runs/alma_v2 --init-from $final --window 4096 --horizon 1024 \
  --bs 32 --steps 10000 --lr 3e-4 --warmup 300 --workers 6 --eval-every 2000 > runs/alma_v2.log 2>&1 &
echo "V2_STARTED $(date)"
sleep 120
nohup ./track.sh runs/alma_v2 > results/track_alma_v2.log 2>&1 &

OMP_NUM_THREADS=3 .venv/bin/python eval_gift.py --model alma --ckpt $final --max-windows 1000 \
  --out results/alma_v1_final_gift.json > results/alma_v1_final_gift.log 2>&1
echo "V1_FULL_EVAL_DONE $(grep -E "skill_pct" results/alma_v1_final_gift.log | tr -d "\n ")"
