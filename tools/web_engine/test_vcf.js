// VCF JS port regression vs the C++ unit cases (src/game/vcf).
// Run: node tools/web_engine/test_vcf.js

const AZ = require("./engine.js");

const SIZE = 15, CELLS = SIZE * SIZE;
let checks = 0, failures = 0;
function CHECK(cond, label) {
  checks++;
  if (!cond) { failures++; console.log("FAIL", label); }
}
function parseBoard(rows) {
  const b = new Int8Array(CELLS);
  rows.forEach((row, r) => {
    for (let c = 0; c < Math.min(SIZE, row.length); c++) {
      if (row[c] === "X") b[r * SIZE + c] = 1;
      if (row[c] === "O") b[r * SIZE + c] = -1;
    }
  });
  return b;
}
const A = (r, c) => r * SIZE + c;

// 1. Immediate five: black XXXX at row 7 cols 3-6.
{
  const b = parseBoard(["...............","...............","...............",
    "...O...........","...............","...............","...............",
    "...XXXX........","...............","..............O","...............",
    "...............","...............","...............","..............."]);
  const s = new AZ.VcfSolver();
  CHECK(s.solve(b, 1, 100000), "immediate five solves");
  const mv = s.findWinningMove(b, 1, 100000);
  CHECK(mv === A(7, 2) || mv === A(7, 7), "first move is a five point");
  CHECK(!s.solve(b, -1, 100000), "white cannot be proven");
  CHECK(!s.undecided, "no undecided flag");
}

// 2. Open four next move wins (two five-points vs single reply).
{
  const b = parseBoard(["...............","...............","..O............",
    "...............","...............","...............","...............",
    ".....XXX.......","...............","...............","...........O...",
    "...............","...............","...............","..............."]);
  const s = new AZ.VcfSolver();
  CHECK(s.solve(b, 1, 200000), "open-four chain solves");
  const mv = s.findWinningMove(b, 1, 200000);
  CHECK(mv === A(7, 4) || mv === A(7, 8), "root is the four move");
}

// 3. No false positives on a fully blocked three.
{
  const b = parseBoard(["...............","...............","...............",
    ".......O.......",".....OXXXO.....","...............","...............",
    "...............","..O.........O..","...............","...............",
    "...............","...............","...............","..............."]);
  const s = new AZ.VcfSolver();
  CHECK(!s.solve(b, 1, 200000), "blocked three must not solve");
  CHECK(!s.undecided, "decided");
}

// 4. Defender's block also creating a defender four refutes the chain.
{
  const b = parseBoard(["...............","...............","...............",
    "...............","...............","...............","...............",
    ".OOO.XXX.......","...............","...............","...............",
    "...............","...............","...............","..............."]);
  const s = new AZ.VcfSolver();
  CHECK(!s.solve(b, 1, 200000), "counter-four defense must not solve");
}

// 5. Budget exhaustion is reported, never upgraded to a fake win.
{
  const b = parseBoard(["...............","...............","...............",
    "...............","...............","...............","....XXX........",
    "...............","....X..........","...............","......O........",
    "...............","...............","...............","..............."]);
  const s = new AZ.VcfSolver();
  CHECK(!s.solve(b, 1, 1), "budget 1 cannot solve");
  CHECK(s.undecided, "undecided flag set on budget");
}

console.log(`${checks} checks, ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
