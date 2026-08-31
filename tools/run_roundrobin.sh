#!/bin/bash
# intra-JS-level ring to separate L1..L7
cd /data00/home/lingchen.judy/self/alphazero-gomoku
W=tools/crossmatch.py; O=runtime_v2/cross2
PAIRS=("1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "7 1" "1 3" "2 5" "4 7" "1 6" "3 7")
N=0
for PAIR in "${PAIRS[@]}"; do
  read -r A B <<< "$PAIR"
  N=$((N+1))
  G=16; if [ "$A" = "4" ] || [ "$B" = "4" ]; then G=8; fi
  (
    taskset -c 0-47 python3 $W --home "js|$A" --away "js|$B" --games $G --workers 12 \
      --seed $((1700+N)) --out $O/rr_${A}v${B}.jsonl > $O/rr_${A}v${B}.log 2>&1
  ) &
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
done
wait
echo RR_DONE > $O/rr.done
