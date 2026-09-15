#!/usr/bin/env python3
"""Convert crossmatch jsonl games into a ReplayBuffer binary for --init-buffer.

Each position becomes one sample with:
- planes: Gomoku::EncodeInto semantics from the current mover's perspective,
- policy: one-hot of the actually played move (supervised imitation),
- value: final outcome z from the mover's perspective (+1/-1/0), shifted so
  that opening noise and draw caps already encode uncertainty.

Layout matches C++ ReplayBuffer::Save: two size_t (size, write_pos), then
raw float arrays [4*225 planes][225 policy][1 value] per sample.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

CELLS = 225
PLANES = 4


def encode(board: list[int], current: int, last: int) -> list[float]:
    out = [0.0] * (PLANES * CELLS)
    for cell, stone in enumerate(board):
        if stone == current:
            out[cell] = 1.0
        elif stone == -current:
            out[CELLS + cell] = 1.0
    if last >= 0:
        out[2 * CELLS + last] = 1.0
    if current == 1:
        for i in range(CELLS):
            out[3 * CELLS + i] = 1.0
    return out


def convert(files: list[str], out_path: Path, max_positions: int) -> int:
    records: list[tuple[list[float], list[float], float]] = []
    for name in files:
        with open(name) as fh:
            for line in fh:
                rec = json.loads(line)
                result = rec.get("result")
                if result == -99 or not rec.get("moves"):
                    continue
                board = [0] * CELLS
                last = -1
                mover = 1  # black
                for row, col in rec["moves"]:
                    action = int(row) * 15 + int(col)
                    if len(records) < max_positions:
                        planes = encode(board, mover, last)
                        policy = [0.0] * CELLS
                        policy[action] = 1.0
                        if result == 0:
                            z = 0.0
                        elif result == mover:
                            z = 1.0
                        else:
                            z = -1.0
                        records.append((planes, policy, z))
                    board[action] = mover
                    last = action
                    mover = -mover
    with out_path.open("wb") as out:
        out.write(struct.pack("<Q", len(records)))
        out.write(struct.pack("<Q", len(records)))
        for planes, policy, z in records:
            out.write(struct.pack(f"<{len(planes)}f", *planes))
            out.write(struct.pack(f"<{len(policy)}f", *policy))
            out.write(struct.pack("<f", z))
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-positions", type=int, default=200000)
    args = ap.parse_args()
    n = convert(args.files, Path(args.out), args.max_positions)
    print(f"wrote {n} samples -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
