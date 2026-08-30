#!/usr/bin/env python3
"""Bradley-Terry Elo over crossmatch jsonl series (draw=0.5).

Each match file contains games between exactly two sides. Sides are keyed by
their engine spec string, so cross-series files reuse the same anchor nodes.

Usage: python3 tools/rating.py runtime_v2/cross/*.jsonl [--anchor 'rapfi|..|turnms=500' 2716]
"""
import argparse
import json
import math
import sys
from collections import defaultdict


def side_key(spec, which):
    return spec


def elo_diff_from_score(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--anchor", nargs=2, metavar=("SPEC", "ELO"))
    args = ap.parse_args()

    wins = defaultdict(float)  # wins[A][B] += 1 (or 0.5)
    games = defaultdict(int)
    for f in args.files:
        file_home = file_away = None
        with open(f) as fh:
            for ln in fh:
                r = json.loads(ln)
                home = r.get("home_spec") or file_home
                away = r.get("away_spec") or file_away
                if home is None:
                    # legacy log-based fallback
                    log = f[:-6] + ".log"
                    try:
                        with open(log) as lf:
                            for line in lf:
                                if line.startswith("[cmd]"):
                                    specs = line[5:].strip().split("  vs  ")
                                    file_home, file_away = specs
                                    home, away = file_home, file_away
                                    break
                    except FileNotFoundError:
                        pass
                if home is None or away is None:
                    continue
                res = r.get("result")
                if res is None or res == -99:
                    continue
                hb = r["home_black"]
                black_name, white_name = (home, away) if hb else (away, home)
                games[(black_name, white_name)] += 1
                if res == 1:
                    wins[black_name][white_name] += 1.0
                elif res == -1:
                    wins[white_name][black_name] += 1.0
                else:
                    wins[black_name][white_name] += 0.5
                    wins[white_name][black_name] += 0.5

    # symmetric pair matrix (do double-count — referee alternates colors)
    players = sorted({p for k in wins for p in (k[0], k[1])})
    M = defaultdict(float)  # M[A][B] = games between A and B
    S = defaultdict(float)  # S[A][B] = A's points vs B
    for (a, b), v in wins.items():
        M[a][b] += games[(a, b)]
        S[a][b] += v
    # The above only filled one direction pair; complete both directions.
    for (a, b), v in list(wins.items()):
        if not M[b][a]:
            M[b][a] = 0.0
            S[b][a] = 0.0

    # Bradley-Terry (Hunter's MM) with draws already folded into S.
    rating = {p: 1.0 for p in players}
    for _ in range(10000):
        new = {}
        max_delta = 0.0
        for i in players:
            num = sum(S[i][j] for j in players if j != i)
            den = 0.0
            for j in players:
                if j == i or M[i][j] == 0:
                    continue
                den += M[i][j] / (rating[i] + rating[j])
            if den == 0:
                new[i] = rating[i]
                continue
            new[i] = num / den
            max_delta = max(max_delta, abs(math.log(max(new[i], 1e-12) /
                                                      max(rating[i], 1e-12))))
        rating = new
        if max_delta < 1e-9:
            break

    elo = {p: 400.0 * math.log10(rating[p]) for p in players}
    if args.anchor:
        spec, val = args.anchor
        if spec in elo:
            shift = float(val) - elo[spec]
            for p in elo:
                elo[p] += shift
    print(f"{'rating':>7}  player")
    for p in sorted(elo, key=elo.get, reverse=True):
        ng = sum(games[(p, q)] for q in players if (p, q) in games)
        print(f"{elo[p]:7.0f}  {p}   ({ng} games)")


if __name__ == "__main__":
    main()
