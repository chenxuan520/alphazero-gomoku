// A/B arena: stock 48-sims vs jueyi config (96sims + VCF) on the same model.
// Run: node test_strength.js [games]
const AZ = require("./engine.js");

const MANIFEST = "https://azgomoku.011203.xyz/channels/stable.json";
const CELLS = 225;

function applyGameMove(state, action) {
  return AZ.applyMove(state, action);
}

async function engineMove(api, model, state, sims, vcf) {
  const opts = {
    simulations: sims, cPuct: 0.8, fpu: 0, yieldEvery: 4,
  };
  if (vcf) { opts.vcf = vcf; opts.vcfLeafNodes = 20000; }
  const r = await api.search(model, state, opts);
  return r.action;
}

async function playGame(model, seedOffset, firstJueyi) {
  const state = AZ.createState(new Int8Array(CELLS), 1, -1);
  let moves = 0;
  // cheap deterministic opening variety: first move jittered near center
  while (state.result === 0 && moves < 160) {
    const me = state.currentPlayer;
    const isJueyi = (me === 1) === firstJueyi;
    let action;
    if (moves === 0) {
      action = 112 + seedOffset; // open near center, seeds vary openings
      if (!AZ.applyMove(state, action)) action = 112, AZ.applyMove(state, 112);
    } else {
      action = await engineMove(AZ, model, state, isJueyi ? 96 : 48,
                                isJueyi ? 1 : 0);
    }
    if (action < 0) break;
    moves++;
  }
  return state.result; // 1 black, -1 white, 2 draw (0=leeft unfinished)
}

(async () => {
  const games = parseInt(process.argv[2] || "8", 10);
  const resp = await fetch(MANIFEST);
  const manifest = await resp.json();
  const wresp = await fetch(manifest.file);
  const bytes = await wresp.arrayBuffer();
  const model = AZ.parseModel(bytes);
  console.log("model loaded");

  let jW = 0, sW = 0, draws = 0;
  for (let g = 0; g < games; g++) {
    const firstJueyi = g % 2 === 0;
    const t0 = Date.now();
    const result = await playGame(model, g, firstJueyi);
    const ms = Date.now() - t0;
    let outcome = "draw";
    if (result !== 0 && result !== 2) {
      const winnerIsJueyi = (result === 1) === firstJueyi;
      winnerIsJueyi ? jW++ : sW++;
      outcome = winnerIsJueyi ? "jueyi" : "stock48";
    }
    console.log(`game ${g}: result=${result} (${outcome}) ${ms}ms`);
  }
  console.log(`stock48: ${sW}  jueyi96+vcf: ${jW}  draws: ${draws}`);
})();
