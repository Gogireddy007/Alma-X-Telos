#!/bin/zsh
# Resume the paused Stage F (skips finished arms/evals) and then Stage G; keeps the Mac awake while running.
cd "${0:A:h}"
nohup ./run_stageF.sh >/dev/null 2>&1 &
F=$!; nohup ./run_stageG.sh >/dev/null 2>&1 &
G=$!; nohup caffeinate -i -s -w $G >/dev/null 2>&1 &
echo "resumed: stageF pid $F, stageG pid $G"
