#!/usr/bin/env python3
"""Batch root-value calibration from archived crossmatch games.

Replays recorded move lists through `bin/az_model_probe`, then aligns the
current-side value prediction with each game's actual result. This uses only
the repository's own model snapshots and self-play-derived networks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "runtime_v2/candidates/k1000-best80-19ea8c34.net"
DEFAULT_FILES = [
    "runtime_v2/cross2/b80_s48_r500.jsonl",
    "runtime_v2/cross2/b80_s96_r500.jsonl",
    "runtime_v2/cross2/b80_s512_r500.jsonl",
    "runtime_v2/cross2/b80_s48_r1500.jsonl",
    "runtime_v2/cross2/b80_s96_r1500.jsonl",
    "runtime_v2/cross2/b80_s512_r1500.jsonl",
    "runtime_v2/cross2/b80_2000_r500.jsonl",
]


def clean_games(files: list[str], max_games: int, max_moves: int) -> list[dict]:
    games = []
    for name in files:
        path = ROOT / name
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("result") == -99:
                    continue
                moves = rec.get("moves") or []
                if not moves or len(moves) > max_moves:
                    continue
                games.append({
                    "source": name,
                    "home_spec": rec.get("home_spec", ""),
                    "away_spec": rec.get("away_spec", ""),
                    "home_black": bool(rec.get("home_black")),
                    "result": rec.get("result"),
                    "winner_side": rec.get("winner_side", "draw"),
                    "moves": [(int(r) * 15 + int(c)) for r, c in moves],
                })
                if len(games) >= max_games:
                    return games
    return games


def probe_game(model: str, game: dict) -> list[dict]:
    """Replay one game prefix-by-prefix. The probe binary stays external: that
    keeps this tool honest to the same checkpoint used by the browser engine,
    at the price of one process launch per ply.
    """
    probe = ROOT / "bin" / "az_model_probe"
    rows = []
    moves = game["moves"]
    for ply in range(1, len(moves) + 1):
        prefix = ",".join(str(x) for x in moves[:ply])
        proc = subprocess.run(
            [str(probe), str(model), prefix],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=120)
        if proc.returncode != 0:
            break
        rec = json.loads(proc.stdout)

        mover = 1 if (ply - 1) % 2 == 0 else -1
        result = game["result"]
        z = 0.0 if result == 0 else (1.0 if result == mover else -1.0)
        rows.append({
            "source": game["source"],
            "ply": ply,
            "value": rec["value"],
            "current_player": rec["current_player"],
            "mover": mover,
            "z": z,
            "result": result,
        })
    return rows


def bucket(rows: list[dict]) -> dict:
    buckets = {}
    ece = brier = 0.0
    for row in rows:
        p = max(0.0, min(1.0, (row["value"] + 1.0) / 2.0))
        y = (row["z"] + 1.0) / 2.0
        b = min(19, int(p * 20))
        slot = buckets.setdefault(b, [0, 0.0, 0.0])
        slot[0] += 1
        slot[1] += p
        slot[2] += y
        ece += abs(p - y)
        brier += (p - y) ** 2
    table = []
    for b, (n, ps, ys) in sorted(buckets.items()):
        table.append({
            "bucket": b,
            "range": [b / 20, (b + 1) / 20],
            "n": n,
            "pred": ps / n,
            "actual": ys / n,
            "actual_minus_pred": ys / n - ps / n,
        })
    return {
        "n": len(rows),
        "ece": ece / max(1, len(rows)),
        "brier": brier / max(1, len(rows)),
        "buckets": table,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--max-games", type=int, default=120)
    ap.add_argument("--max-moves", type=int, default=180)
    ap.add_argument("--out", default="runtime_v2/cross2/value_calibration.jsonl")
    args = ap.parse_args()

    games = clean_games(args.files, args.max_games, args.max_moves)
    if not games:
        print("no eligible games", file=sys.stderr)
        return 2
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    rows = []
    for game in games:
        rows.extend(probe_game(str(model_path), game))
    summary = bucket(rows)
    out = ROOT / args.out
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = Path(str(out) + ".summary.json")
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
