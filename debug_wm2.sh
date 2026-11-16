#!/bin/zsh
# Development-data (DEBUG_harness) checks of wm2 checkpoints during training. Never touches sealed holdouts.
cd "${0:A:h}"
for s in 4000 12000 20000; do
  CK=runs/alma_wm2/ema_$s.pt
  until [ -f $CK ]; do sleep 120; done; sleep 60
  echo "## step $s ($(date))" >> runs/alma_wm2_debug.md
  zsh validate/debug_check.sh $CK runs/alma_wm2/last.pt >> runs/alma_wm2_debug.md 2>&1
done
