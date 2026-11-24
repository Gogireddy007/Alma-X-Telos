#!/bin/zsh
# Resume after pause.sh. If the processes are still alive (Mac only slept), continue them exactly where they froze.
# If they are gone (reboot / shutdown), relaunch the drivers; training resumes from runs/alma_wm2/last.pt.
cd "${0:A:h}"
P=$(pgrep -f "run_wm2.sh|train_wm2.py|debug_wm2.sh|run_v3.sh|train_v3.py|fc_validate.py|eval_gift.py|validate/(predict|grade|static_tests).py")
if [ -n "$P" ]; then
  W=$(for p in ${=P}; do pgrep -P $p; done)
  kill -CONT ${=W} ${=P} 2>/dev/null
  echo "RESUMED $(date) (in-memory, no steps lost)" >> runs/pause.log; echo "resumed $(echo $P $W | wc -w) processes in place"
else
  echo "RELAUNCH $(date) (processes were gone; restarting from last.pt)" >> runs/pause.log
  if [ ! -f runs/alma_v3/ema_20000.pt ] && ! grep -q "VALIDATION EXIT" runs/alma_wm2.log; then
    nohup caffeinate -is ./run_wm2.sh >/dev/null 2>&1 &
    # remaining development checks only (skip ones already in the debug report)
    S=""; for s in 4000 12000 20000; do grep -q "## step $s " runs/alma_wm2_debug.md 2>/dev/null || S="$S $s"; done
    sed "s/for s in 4000 12000 20000/for s in$S/" debug_wm2.sh > runs/_debug_wm2_resume.sh
    [ -n "$S" ] && nohup caffeinate -is zsh runs/_debug_wm2_resume.sh >/dev/null 2>&1 &
  fi
  nohup caffeinate -is ./run_v3.sh >/dev/null 2>&1 &
  echo "relaunched drivers"
fi
