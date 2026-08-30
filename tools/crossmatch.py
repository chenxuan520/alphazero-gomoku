#!/usr/bin/env python3
"""Cross-engine match referee for alphazero-gomoku.

Speaks the piskvork protocol to any engine binary (Rapfi, or our own
`alphazero piskvork`), and JSON-lines to the game-old JS levels through
`node tools/js_engine/engine.js`.

Match setup:
  python3 tools/crossmatch.py \
      --home 'az|runtime_v2/candidates/k1000-best80-19ea8c34.net|sims=48|threads=1' \
      --away 'rapfi|/tmp/rapfi/Rapfi/build/pbrain-rapfi|turnms=1000' \
      --games 24 --workers 12 --seed 1234 --out runtime_v2/cross/az48_vs_rapfi_1s.jsonl

Colors alternate; openings rotate over a small fixed pool to diversify games.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

BOARD = 15


def build_openings(seed=20260830, count=64):
    """Deterministic engine-varied opening book: first two moves within the
    Chebyshev-2 neighborhood of center, filtered/deduplicated."""
    import random as _r
    rng = _r.Random(seed)
    seen = set()
    out = []
    c = BOARD // 2
    while len(out) < count:
        f = (c + rng.randint(-2, 2), c + rng.randint(-2, 2))
        cands = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = f[0] + dr, f[1] + dc
                if 0 <= nr < BOARD and 0 <= nc < BOARD and (nr, nc) != f and \
                        (abs(nr - c) <= 3 and abs(nc - c) <= 3):
                    cands.append((nr, nc))
        rng.shuffle(cands)
        s = cands[0]
        key = (f, s)
        if key in seen:
            continue
        seen.add(key)
        out.append([f, s])
    return out


OPENINGS = build_openings()
AZ_BIN = os.path.join(os.path.dirname(__file__), "..", "bin", "alphazero")
JS_ENGINE = os.path.join(os.path.dirname(__file__), "js_engine", "engine.js")


class PiskvorkSide:
    """piskvork-protocol engine (also our own `alphazero piskvork`)."""

    def __init__(self, exe, args, cwd=None, name="pk"):
        self.proc = subprocess.Popen(
            [exe, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=cwd, text=True, bufsize=1)
        self.name = name
        self.send("start 15")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self, timeout=600):
        # blocking read with timeout via dedicated reader thread
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{self.name}: move timeout")

    def init_reader(self):
        self._q = queue.Queue()

        def rd():
            for ln in self.proc.stdout:
                ln = ln.strip()
                if ln:
                    self._q.put(ln)

        self._t = threading.Thread(target=rd, daemon=True)
        self._t.start()

    def start_io(self):
        self.init_reader()

    def play_move(self):
        """After send(begin/turn), fetch the engine's 'x,y' reply."""
        while True:
            ln = self.readline()
            up = ln.upper()
            if up.startswith(("MESSAGE", "INFO", "DEBUG", "ERROR", "UNKNOWN")):
                continue
            if "," in ln:
                x, y = ln.split(",")[:2]
                return int(x), int(y)

    def close(self):
        try:
            self.send("end")
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class RandomSide:
    """Uniform-random legal moves — absolute-zero anchor for Elo ladders."""

    def __init__(self, seed):
        import random
        self.rng = random.Random(seed)
        self.board = None
        self.me = None

    def begin_game(self, me):
        self.me = me
        self.board = [[0] * BOARD for _ in range(BOARD)]

    def apply(self, x, y, who):
        self.board[x][y] = who

    def play_move(self):
        empties = [(r, c) for r in range(BOARD) for c in range(BOARD)
                   if self.board[r][c] == 0]
        return self.rng.choice(empties)

    def close(self):
        pass


class JsonSide:
    """game-old JS levels wrapper (JSON-lines)."""

    def __init__(self, level):
        self.level = level
        self.proc = subprocess.Popen(
            ["node", JS_ENGINE], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1, cwd=os.path.dirname(JS_ENGINE) + "/../..")
        self.board = [[0] * BOARD for _ in range(BOARD)]
        self.me = None

    # called at game start; `me` is 1 when this side plays black (moves first)
    def begin_game(self, me):
        self.me = me
        self.board = [[0] * BOARD for _ in range(BOARD)]

    def apply(self, x, y, who):
        self.board[x][y] = who

    def play_move(self):
        req = {"board": self.board, "me": self.me, "level": self.level}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        # some levels log noise lines before the JSON reply; skip them
        while True:
            ln = self.proc.stdout.readline()
            if not ln:
                raise RuntimeError(f"level {self.level}: engine EOF")
            ln = ln.strip()
            if ln.startswith("{"):
                break
        resp = json.loads(ln)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["x"], resp["y"]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def parse_side(spec):
    parts = spec.split("|")
    kind = parts[0]
    return kind, parts[1:]


def spawn(spec, tag):
    kind, parts = parse_side(spec)
    if kind == "az":
        model = parts[0]
        sims = "100"; threads = "1"
        for p in parts[1:]:
            k, v = p.split("=", 1)
            if k == "sims": sims = v
            if k == "threads": threads = v
        side = PiskvorkSide(AZ_BIN, ["piskvork", "--model", model,
                                     "--sims", sims, "--threads", threads],
                            name=f"az(s{sims})", cwd=os.path.join(
                                os.path.dirname(AZ_BIN), ".."))
        side.start_io()
        return side
    if kind == "rapfi":
        exe = parts[0]
        turnms = "1000"
        for p in parts[1:]:
            k, v = p.split("=", 1)
            if k == "turnms": turnms = v
        side = PiskvorkSide(exe, [], name=f"rapfi({turnms}ms)",
                            cwd=os.path.dirname(exe))
        side.start_io()
        side.send(f"info timeout_turn {turnms}")
        side.send("info timeout_match 300000")
        side.send("info max_memory 268435456")
        return side
    if kind == "js":
        return JsonSide(int(parts[0]))
    if kind == "rnd":
        seed = 1
        for p in parts[1:]:
            k, v = p.split("=", 1)
            if k == "seed": seed = int(v)
        return RandomSide(seed)
    raise ValueError(f"bad engine kind: {kind}")


def run_game(home_spec, away_spec, opening, home_black, log_q):
    home = spawn(home_spec, "home")
    away = spawn(away_spec, "away")
    black, white = (home, away) if home_black else (away, home)
    names = {"black": "home" if home_black else "away",
             "white": "away" if home_black else "home"}
    try:
        if not isinstance(black, PiskvorkSide):
            black.begin_game(1)
        if not isinstance(white, PiskvorkSide):
            white.begin_game(2)
        moves = []  # (row, col)
        times = {"black": 0.0, "white": 0.0}
        board_sent = {id(home): False, id(away): False}
        lbd = [[0] * BOARD for _ in range(BOARD)]
        move_no = 0
        result = 0

        def push_board_to(side):
            """Sync a piskvork engine to the full current history once."""
            if not isinstance(side, PiskvorkSide):
                return
            side.send("board")
            for mr, mc in moves:
                # field: 1 if this side owns the stone
                stone = lbd[mr][mc]
                owns_black = (side is home and home_black) or \
                    (side is away and not home_black)
                field = stone if owns_black else (3 - stone)
                side.send(f"{mc},{mr},{field}")
            side.send("done")
            board_sent[id(side)] = True

        while True:
            who = "black" if move_no % 2 == 0 else "white"
            side = black if who == "black" else white
            t0 = time.time()
            if move_no < len(opening):
                row, col = opening[move_no]
            else:
                if isinstance(side, PiskvorkSide):
                    if move_no == 0:
                        side.send("begin")
                        board_sent[id(side)] = True
                    elif not board_sent[id(side)]:
                        # engine will reply right after `done`
                        push_board_to(side)
                    else:
                        pr, pc = moves[-1]  # last move was opponent's
                        side.send(f"turn {pc},{pr}")  # piskvork x=col,y=row
                    x, y = side.play_move()
                    row, col = y, x  # piskvork x=col,y=row → row,col
                else:
                    row, col = side.play_move()
            dt = time.time() - t0
            times[who] += dt
            if not (0 <= row < BOARD and 0 <= col < BOARD) or lbd[row][col] != 0:
                result = -99
                break
            lbd[row][col] = 1 if who == "black" else 2
            for s in (home, away):
                if not isinstance(s, PiskvorkSide):
                    s.apply(row, col, 1 if who == "black" else 2)
            moves.append((row, col))
            move_no += 1
            if check_win(lbd, row, col):
                result = 1 if who == "black" else -1
                break
            if move_no >= BOARD * BOARD:
                result = 0
                break

        return {
            "home_spec": home_spec, "away_spec": away_spec,
            "home_black": home_black, "opening": opening, "result": result,
            "winner_side": names["black"] if result == 1 else
                (names["white"] if result == -1 else "draw"),
            "moves": moves,
            "time_black_s": round(times["black"], 3),
            "time_white_s": round(times["white"], 3),
        }
    finally:
        home.close()
        away.close()


def check_win(board, r, c):
    who = board[r][c]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            k = 1
            while 0 <= r + s * k * dr < BOARD and 0 <= c + s * k * dc < BOARD \
                    and board[r + s * k * dr][c + s * k * dc] == who:
                n += 1
                k += 1
        if n >= 5:
            return True
    return False


def worker_loop(q, log_q, out_fh, counters, lock):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        game_idx, home_spec, away_spec, opening, home_black = job
        try:
            rec = run_game(home_spec, away_spec, opening, home_black, log_q)
        except Exception as e:
            rec = {"error": str(e), "home_black": home_black, "opening": opening,
                   "result": -99}
        with lock:
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            counters["done"] += 1
            print(f"\r[crossmatch] {counters['done']}/{counters['total']}",
                  end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    q = queue.Queue()
    for i in range(args.games):
        opening = OPENINGS[(args.seed + i) % len(OPENINGS)]
        q.put((i, args.home, args.away, opening, i % 2 == 0))

    counters = {"done": 0, "total": args.games}
    with open(args.out, "w") as fh:
        lock = threading.Lock()
        threads = []
        for _ in range(args.workers):
            t = threading.Thread(target=worker_loop,
                                 args=(q, None, fh, counters, lock))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    print()
    # summary
    win = loss = draw = err = 0
    with open(args.out) as fh:
        for ln in fh:
            r = json.loads(ln)
            w = r.get("result")
            if w == -99:
                err += 1
            elif r.get("winner_side") == "home":
                win += 1
            elif r.get("winner_side") == "away":
                loss += 1
            else:
                draw += 1
    total = max(1, win + loss + draw)
    print(f"{args.home}  vs  {args.away}")
    print(f"home W/L/D = {win}/{loss}/{draw}, errors={err}, "
          f"home-win-rate={win / total:.3f}")


if __name__ == "__main__":
    main()
