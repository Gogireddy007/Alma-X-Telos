#!/bin/zsh
# Stage F: targeted fixes for the sealed gap (traffic LOOP_SEATTLE, cloud bitbrains, short yearly/quarterly).
# Small model, 3k steps from v2@6k, D recipe (bucket sampler, mix loss), both corpora. dev2 only; sealed untouched.
#  H_seed  = D recipe, seed 1, old data         -> noise floor for D (and an ensemble member)
#  H_boost = D recipe + traffic/cloud x3        -> targeted
#  H_boost5= traffic/cloud x5
#  H_a03   = boost x3 + alpha 0.3 (lifts yearly/quarterly buckets)
cd "${0:A:h}"; setopt NULL_GLOB
PY=.venv/bin/python; LOG=runs/stageF.log; mkdir -p runs/stageF
TC="PEMS*,LOS_LOOP,Q-TRAFFIC,traffic_hourly,*METRO,BEIJING_SUBWAY_30MIN"; CL="alibaba*,azure*,borg*"
arm() {  # name real alpha seed boost
  out=runs/stageF/$1
  if [ ! -f $out/ema_3000.pt ]; then
    echo "--- $1 real=$2 alpha=$3 seed=$4 boost=$5 $(date)" >> $LOG
    ALMA_BOOST="$5" $PY train.py --out $out --init-from runs/alma_v2/ema_6000.pt --steps 3000 --bs 32 --lr 1e-4 --warmup 200 \
      --layers 8 --d 384 --window 4096 --horizon 1024 --eval-every 1000 --save-every 1000 \
      --sampler bucket --alpha $3 --loss-norm mix --seed $4 --real $2 > $out.train.log 2>&1
    echo "train exit $? $(date)" >> $LOG
  fi
  [ -f results/preds/$1.npz ] || $PY dev_eval.py --model alma --ckpt $out/ema_3000.pt --ctx 2048 --n 256 \
      --out results/preds/$1.json --save-preds results/preds/$1.npz > results/preds/$1.log 2>&1
  echo "dev2 $1 exit $? $(date)" >> $LOG
  $PY dev_rank.py --ref results/preds/telos.json results/preds/D.json results/preds/$1.json >> $LOG 2>&1
}
B="data/pretrain,data/pretrain2"
arm H_boost $B 0.5 0 "$TC:3;$CL:3"
arm H_seed data/pretrain 0.5 1 ""
arm H_boost5 $B 0.5 0 "$TC:5;$CL:5"
arm H_a03 $B 0.3 0 "$TC:3;$CL:3"
$PY ens_search.py --route > results/ens_search_F.txt 2>&1; echo "ens_search_F exit $? $(date)" >> $LOG
$PY fam_report.py > results/fam_report_F.txt 2>&1
echo "=== Stage F done $(date)" >> $LOG
