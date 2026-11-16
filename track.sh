#!/bin/zsh
# Scores every new EMA checkpoint of a run on the quick GIFT proxy (symmetric averaging) -> results/curve_<run>.jsonl
# usage: ./track.sh runs/meridian_s
run=$1; name=${run:t}; out=results/curve_${name}.jsonl
touch $out
while true; do
  for ck in $(ls $run/ema_*.pt 2>/dev/null | sort -t_ -k3 -n); do
    step=${${ck:t:r}#ema_}
    grep -q "\"step\": $step," $out && continue
    r=$(OMP_NUM_THREADS=3 .venv/bin/python quick_eval.py --model alma --ckpt $ck --out results/quick_${name}_${step}.json 2>/dev/null | tail -1)
    echo "{\"step\": $step, \"quick\": $r}" >> $out
    echo "CKPT step=$step $r"
  done
  pgrep -f "train.py --out $run" >/dev/null || { echo "TRAINING_EXITED"; exit 0; }
  sleep 60
done
