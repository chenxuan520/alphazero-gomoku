#!/usr/bin/env bash
# Exit criteria: whether the k1000 dead-zone snapshots (110..146) accidentally
# improved over final_iter440 somewhere in the unexplored tail.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AZ_BIN="$ROOT_DIR/bin/alphazero"
BASE_NET="$ROOT_DIR/runtime/final_iter440.net"
OUT_DIR="$ROOT_DIR/runtime_v2/cross2"
SUMMARY="$OUT_DIR/k1000_deadzone_scan.log"

# archive candidates that were never externally measured / promoted
ARCHIVES=(
  "$ROOT_DIR/runtime_v2/candidates/k1000-checkpoint-best-90-317583ce.net"
  "$ROOT_DIR/runtime_v2/candidates/k1000-best105-beb36e39.net"
  "$ROOT_DIR/runtime_v2/candidates/k1000-latest123-7d8f00ab.net"
)

mkdir -p "$OUT_DIR"
echo "# k1000 dead-zone scan $(date -Is)" | tee -a "$SUMMARY"
for net in "${ARCHIVES[@]}"; do
  tag="$(basename "$net" .net)"
  for sims in 48 96; do
    for seed in 4242 4343; do
      log="$OUT_DIR/${tag}_vs_440_s${sims}_s${seed}.log"
      "$AZ_BIN" arena \
        --model-a "$net" --model-b "$BASE_NET" \
        --games 16 --sims "$sims" --workers 2 \
        --max-moves 180 --seed "$seed" --temp-moves 8 \
        --dir-eps 0.02 --dir-alpha 0.03 --deterministic-games 1 --reuse-tree 1 \
        > "$log" 2>&1
      tail -1 "$log" | tee -a "$SUMMARY"
    done
  done
done
