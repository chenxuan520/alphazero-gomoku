#!/bin/bash
# Full rating campaign v2: three tracks on disjoint core ranges.
set -u
cd /data00/home/lingchen.judy/self/alphazero-gomoku
W=tools/crossmatch.py
R=/tmp/rapfi/Rapfi/build/pbrain-rapfi
O=runtime_v2/cross2
M=runtime_v2/candidates
B80=$M/k1000-best80-19ea8c34.net
B85=$M/k1000-checkpoint-best-85-1b4ea60a.net
mkdir -p $O

trackA() {  # rapfi budget ladder, cores 0-15
  taskset -c 0-15 python3 $W --home "rapfi|$R|turnms=150"  --away "rapfi|$R|turnms=500"  --games 24 --workers 8 --seed 101 --out $O/i1_150v500.jsonl  > $O/i1.log 2>&1
  taskset -c 0-15 python3 $W --home "rapfi|$R|turnms=500"  --away "rapfi|$R|turnms=1500" --games 24 --workers 8 --seed 102 --out $O/i2_500v1500.jsonl > $O/i2.log 2>&1
  taskset -c 0-15 python3 $W --home "rapfi|$R|turnms=1500" --away "rapfi|$R|turnms=5000" --games 24 --workers 8 --seed 103 --out $O/i3_1500v5000.jsonl> $O/i3.log 2>&1
  echo A > $O/tierA2.done
}

trackB() {  # best80 placement vs rapfi, cores 16-31
  for SIMS in 48 96 512; do
    case $SIMS in
      48)  BUD="150 500 1500";;
      96)  BUD="150 500 1500";;
      512) BUD="500 1500 5000";;
    esac
    for T in $BUD; do
      taskset -c 16-31 python3 $W --home "az|$B80|sims=$SIMS|threads=1" --away "rapfi|$R|turnms=$T" \
        --games 24 --workers 8 --seed $((200+SIMS*10+T/100)) --out $O/b80_s${SIMS}_r${T}.jsonl > $O/b80_s${SIMS}_r${T}.log 2>&1
    done
  done
  echo B > $O/tierB2.done
}

trackC() {  # best85 spot-check + JS levels vs rapfi, cores 32-47
  taskset -c 32-47 python3 $W --home "az|$B85|sims=48|threads=1" --away "rapfi|$R|turnms=150" \
    --games 24 --workers 8 --seed 501 --out $O/b85_s48_r150.jsonl > $O/b85_s48_r150.log 2>&1
  taskset -c 32-47 python3 $W --home "az|$B85|sims=48|threads=1" --away "rapfi|$R|turnms=500" \
    --games 24 --workers 8 --seed 502 --out $O/b85_s48_r500.jsonl > $O/b85_s48_r500.log 2>&1
  for L in 1 2 3 4 5; do
    taskset -c 32-47 python3 $W --home "js|$L" --away "rapfi|$R|turnms=150" \
      --games 24 --workers 8 --seed $((600+L)) --out $O/v${L}_r150.jsonl > $O/v${L}_r150.log 2>&1
  done
  taskset -c 32-47 python3 $W --home "js|6" --away "rapfi|$R|turnms=150" \
    --games 24 --workers 8 --seed 606 --out $O/v6_r150.jsonl > $O/v6_r150.log 2>&1
  taskset -c 32-47 python3 $W --home "js|7" --away "rapfi|$R|turnms=500" \
    --games 24 --workers 6 --seed 607 --out $O/v7_r500.jsonl > $O/v7_r500.log 2>&1
  echo C > $O/tierC2.done
}

trackA & TA=$!
trackB & TB=$!
trackC & TC=$!
wait $TA $TB $TC
echo ALL_DONE > $O/all2.done
