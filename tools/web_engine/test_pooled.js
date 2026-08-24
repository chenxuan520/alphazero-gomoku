// test_pooled.js — invariants + timing + parity for the pooled search.
const fs = require("fs");
const path = require("path");
const { Worker } = require("worker_threads");

eval(fs.readFileSync(__dirname + "/engine2.js", "utf8"));
const AZ = globalThis.AlphaZeroGomoku;

const ENGINE = __dirname + "/engine2.js";
function makeNodePool(modelBytes, count) {
  return AZ.createSearchPool(modelBytes, count, () => {
    const w = new Worker(__dirname + "/pool_worker_node.js", { workerData: { enginePath: ENGINE } });
    return {
      post: (m) => w.postMessage(m),
      onMessage: (fn) => w.on("message", fn),
      terminate: () => w.terminate(),
    };
  });
}

function loadBytes(f) {
  const b = fs.readFileSync(f);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
function loadModel(f) { return AZ.parseModel(loadBytes(f)); }

function lcg(seed) { let s = seed >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; }

async function main() {
  const bytes64 = loadBytes(__dirname + "/models/latest30.net");
  const model64 = AZ.parseModel(loadBytes(__dirname + "/models/latest30.net"));

  // --- invariants on a mid-game position ---
  let st = AZ.createState();
  const rng = lcg(4242);
  for (let i = 0; i < 9; i++) {
    const acts = AZ.candidateActions(st.board, st.moveCount);
    AZ.applyMove(st, acts[Math.floor(rng() * acts.length)]);
  }

  for (const W of [2, 4, 8]) {
    const pool = makeNodePool(bytes64, W);
    await pool.ready();
    for (const sims of [24, 48]) {
      // fresh tree each time (no session) so root.n should equal sims exactly
      const r = await AZ.searchPooled(pool, st, { simulations: sims });
      let sum = 0;
      for (const v of r.visits) sum += v.n;
      if (r.root.n !== sims || sum !== sims)
        throw new Error(`invariant broken W=${W} sims=${sims}: root.n=${r.root.n} sum=${sum}`);
      if (!Number.isInteger(r.action) || r.action < 0 || r.action >= 225)
        throw new Error("bad action");
    }
    pool.terminate();
    console.log(`invariants OK W=${W}`);
  }

  // --- session reuse path ---
  {
    const pool = makeNodePool(bytes64, 4);
    await pool.ready();
    const session = new AZ.SearchSession();
    const s0 = AZ.createState();
    const r1 = await AZ.searchPooled(pool, s0, { simulations: 16, session });
    AZ.applyMove(s0, r1.action); session.advance(r1.action);
    const s1 = s0;
    const r2 = await AZ.searchPooled(pool, s1, { simulations: 16, session });
    if (!r2.reused) throw new Error("session reuse broken: not reused");
    if (r2.root.n !== r2.inheritedVisits + 16)
      throw new Error(`session reuse broken: root.n=${r2.root.n} inherited=${r2.inheritedVisits}`);
    console.log("session reuse OK inherited=", r2.inheritedVisits, "root.n=", r2.root.n);
    pool.terminate();
  }

  // --- speed: pooled vs stock on 64x8 @48 sims ---
  const pos = AZ.createState();
  for (const a of [112, 113, 98, 114]) AZ.applyMove(pos, a);
  // stock warm + time
  await AZ.search(model64, pos, { simulations: 8 });
  let t0 = performance.now();
  await AZ.search(model64, pos, { simulations: 48 });
  const stockMs = performance.now() - t0;
  for (const W of [2, 4, 8]) {
    const pool = makeNodePool(bytes64, W);
    await pool.ready();
    await AZ.searchPooled(pool, pos, { simulations: 8 }); // warm
    t0 = performance.now();
    await AZ.searchPooled(pool, pos, { simulations: 48 });
    const ms = performance.now() - t0;
    console.log(`48 sims: stock=${stockMs.toFixed(0)}ms  pooled W=${W}: ${ms.toFixed(0)}ms`);
    pool.terminate();
  }
  process.exit(0);
}

main().catch((e) => { console.error("FAIL", e); process.exit(1); });
