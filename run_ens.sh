#!/bin/zsh
# Inference ensembles on dev2 (quantile averaging). Members: v3@10k (best v3-family on dev2) and D (best small).
cd "${0:A:h}"
LOG=runs/stageC.log; PY=.venv/bin/python; D2=results/dev2
V=runs/alma_v3/ema_10000.pt; S=runs/stageA/D_bucket_mix/ema_3000.pt; F=runs/stageB/v3ft/ema_4000.pt
dev2() { [ -f $2 ] || $PY dev_eval.py ${=1} --n 256 --out $2 > ${2%.json}.log 2>&1; echo "dev2 $2 exit $? $(date)" >> $LOG; }
echo "=== ensembles start $(date)" >> $LOG
dev2 "--model ens --members $V@8192,$S@2048" $D2/ens_v3+D_n256.json
dev2 "--model ens --members $V@8192,$V@2048" $D2/ens_v3ctx2_n256.json
dev2 "--model ens --members $V@8192,$F@8192,$S@2048" $D2/ens_v3+v3ft+D_n256.json
dev2 "--model ens --members $V@8192,$S@2048,runs/stageA/C_bucket_series/ema_3000.pt@2048" $D2/ens_v3+D+C_n256.json
echo "=== ensembles done $(date)" >> $LOG
