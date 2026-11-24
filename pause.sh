#!/bin/zsh
# Freeze the whole pipeline in memory (SIGSTOP). Nothing is killed, so resume.sh continues at the exact step.
cd "${0:A:h}"
P=$(pgrep -f "run_wm2.sh|train_wm2.py|debug_wm2.sh|run_v3.sh|train_v3.py|fc_validate.py|eval_gift.py|validate/(predict|grade|static_tests).py")
W=$(for p in ${=P}; do pgrep -P $p; done)   # data-loader workers / sleeps
[ -z "$P" ] && { echo "nothing running"; exit 0; }
kill -STOP ${=P} ${=W} 2>/dev/null
echo "PAUSED $(date) pids: $P $W" | tr '\n' ' ' >> runs/pause.log; echo >> runs/pause.log
echo "paused: $(echo $P $W | wc -w) processes"; tail -1 runs/alma_wm2/log.jsonl | cut -c1-40
