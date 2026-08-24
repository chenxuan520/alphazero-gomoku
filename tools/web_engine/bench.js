// Node bench + exactness check for engine.js forward optimizations.
const fs = require("fs");
const path = require("path");
eval(fs.readFileSync(__dirname + "/engine.js", "utf8"));
const AZ = globalThis.AlphaZeroGomoku;
eval(fs.readFileSync(__dirname + "/opt.js", "utf8"));

function loadModel(f) {
  const buf = fs.readFileSync(f);
  return AZ.parseModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

function randomBoard(rng, moves) {
  let st = AZ.createState();
  for (let i = 0; i < moves; i++) {
    const acts = AZ.candidateActions(st.board);
    const a = acts[Math.floor(rng() * acts.length)];
    AZ.applyMove(st, a);
  }
  return st;
}

function lcg(seed) { let s = seed >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; }

function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

const A = loadModel(__dirname + "/models/latest30.net");
const B = loadModel(__dirname + "/models/iter440.net");

const positions = [];
for (let i = 0; i < 4; i++) positions.push(randomBoard(lcg(1000 + i), 6 + i * 7));

// exactness: reload fresh copies (opt folds its own model's weights in-place)
function checkForward(fn, refPath, tag) {
  const mR = loadModel(refPath), mO = loadModel(refPath);
  for (const st of positions) {
    const ref = AZ.forward(mR, st.board, st.currentPlayer, st.lastAction);
    const got = fn(mO, st.board, st.currentPlayer, st.lastAction);
    const dp = maxDiff(ref.policyLogits, got.policyLogits), dv = Math.abs(ref.value - got.value);
    if (dp > 5e-4 || dv > 5e-4) throw new Error(`${tag} mismatch: dp=${dp} dv=${dv}`);
  }
  console.log(tag, "exact-ok");
}

function time(fn, model, iters) {
  const st = positions[2];
  for (let i = 0; i < 3; i++) fn(model, st.board, st.currentPlayer, st.lastAction); // warm
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(model, st.board, st.currentPlayer, st.lastAction);
  const dt = (performance.now() - t0) / iters;
  return dt;
}

const iters = parseInt(process.argv[2] || "20", 10);
for (const [name, model] of [["64x8", A], ["32x4", B]]) {
  const t = time(AZ.forward, model, iters);
  console.log(`${name} stock forward: ${t.toFixed(1)} ms`);
}
if (globalThis.optForward) {
  checkForward(globalThis.optForward, __dirname + "/models/latest30.net", "opt 64x8");
  checkForward(globalThis.optForward, __dirname + "/models/iter440.net", "opt 32x4");
  for (const [name, model] of [["64x8", A], ["32x4", B]]) {
    const t = time(globalThis.optForward, model, iters);
    console.log(`${name} opt forward:   ${t.toFixed(1)} ms`);
  }
  // NOTE: A/B got folded by the above timing calls; stock timings above already captured.
}
