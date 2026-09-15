#!/usr/bin/env bash
# Low-budget VCF vs Rapfi sweep, the browser-only useful regime (48/96 sims).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CROSS="$ROOT_DIR/tools/crossmatch.py"
MODEL="$ROOT_DIR/runtime_v2/candidates/k1000-best80-19ea8c34.net"
RAPFI="/tmp/rapfi/Rapfi/build/pbrain-rapfi"
OUT_DIR="$ROOT_DIR/runtime_v2/cross2"
SUMMARY="$OUT_DIR/vcf_lowsims_scan.log"

SIMS=("48" "96")
TURNMS=("500" "1500")
VCF="20000"
GAMES=24
WORKERS=3
SEED="3214"

mkdir -p "$OUT_DIR"
echo "# low-sims VCF vs Rapfi $(date -Is)" | tee -a "$SUMMARY"
for s in "${SIMS[@]}"; do
  for r in "${TURNMS[@]}"; do
    out="$OUT_DIR/b80_s${s}_vcf_r${r}.jsonl"
    log="$OUT_DIR/b80_s${s}_vcf_r${r}.log"
    python3 "$CROSS" \
      --home "az|$MODEL|sims=$s|threads=1|vcf=$VCF" \
      --away "rapfi|$RAPFI|turnms=$r" \
      --games "$GAMES" --workers "$WORKERS" --seed "$SEED" \
      --out "$out" > "$log" 2>&1 || true
    tail -1 "$log" | tee -a "$SUMMARY"
  done
done
