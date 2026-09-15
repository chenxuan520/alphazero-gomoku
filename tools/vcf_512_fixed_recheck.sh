#!/usr/bin/env bash
# Rebuild the §11 512-sims VCF evidence after the root-pin fix.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CROSS="$ROOT_DIR/tools/crossmatch.py"
MODEL="$ROOT_DIR/runtime_v2/candidates/k1000-best80-19ea8c34.net"
RAPFI="/tmp/rapfi/Rapfi/build/pbrain-rapfi"
OUT_DIR="$ROOT_DIR/runtime_v2/cross2"
SUMMARY="$OUT_DIR/vcf_512_fixed_recheck.log"

mkdir -p "$OUT_DIR"
for r in 500 1500; do
  out="$OUT_DIR/b80_s512_vcf_fixed_r${r}.jsonl"
  log="$OUT_DIR/b80_s512_vcf_fixed_r${r}.log"
  python3 "$CROSS" \
    --home "az|$MODEL|sims=512|threads=1|vcf=20000" \
    --away "rapfi|$RAPFI|turnms=$r" \
    --games 48 --workers 2 --seed 9187 \
    --out "$out" > "$log" 2>&1 || true
  tail -2 "$log" | tee -a "$SUMMARY"
done
