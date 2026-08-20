'use strict';

const fs = require('fs');
const path = require('path');

const FRONTEND = process.env.GAME_OLD_FRONTEND ||
  path.resolve(__dirname, '../../../game-old/gobang-web/frontend');
if (!fs.existsSync(FRONTEND)) {
  throw new Error(
    'game-old frontend not found; set GAME_OLD_FRONTEND=/path/to/gobang-web/frontend');
}
const SIZE = 15;
globalThis.BOARD_SIZE = SIZE;
globalThis.gameState = {board: null};

globalThis.checkWin = function (x, y, player, board) {
  board = board || globalThis.gameState.board;
  for (const [dx, dy] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    let count = 1;
    for (const sign of [1, -1]) {
      for (let step = 1; step < 5; step++) {
        const nx = x + sign * step * dx;
        const ny = y + sign * step * dy;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE ||
            board[nx][ny] !== player) break;
        count++;
      }
    }
    if (count >= 5) return true;
  }
  return false;
};

function loadSource(file) {
  (0, eval)(fs.readFileSync(file, 'utf8'));
}

globalThis.fetch = async function (url) {
  return {
    ok: true,
    status: 200,
    json: async () => JSON.parse(fs.readFileSync(url, 'utf8')),
  };
};

loadSource(path.join(__dirname, 'script_ai_extract.js'));
loadSource(path.join(FRONTEND, 'master_ai.js'));
loadSource(path.join(FRONTEND, 'teacher_ai.js'));
loadSource(path.join(FRONTEND, 'mcts_ai.js'));

function seededRandom(seed) {
  let state = seed >>> 0;
  return function () {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

async function pick(level, me) {
  switch (level) {
    case 1: return basicAI(me);
    case 2: return defenseAI(me);
    case 3: return attackAI(me);
    case 4: return expertAI(me);
    case 5: return MasterAI.pick(gameState.board, me);
    case 6: return TeacherAI.pick(gameState.board, me, {depth: 3, topK: 20});
    case 7: return MctsAI.pick(gameState.board, me,
                              {simulations: 200, cPuct: 2.0,
                               hybridAttack: 0.5});
    default: throw new Error('unknown level ' + level);
  }
}

async function play(blackLevel, whiteLevel, seed) {
  const oldRandom = Math.random;
  Math.random = seededRandom(seed);
  try {
    const board = Array.from({length: SIZE}, () => Array(SIZE).fill(0));
    gameState.board = board;
    let player = 1;
    for (let moveCount = 0; moveCount < SIZE * SIZE; moveCount++) {
      const level = player === 1 ? blackLevel : whiteLevel;
      const move = await pick(level, player);
      if (!move || !Number.isInteger(move.x) || !Number.isInteger(move.y) ||
          move.x < 0 || move.x >= SIZE || move.y < 0 || move.y >= SIZE ||
          board[move.x][move.y] !== 0) {
        return player === 1 ? 2 : 1; // illegal/no move loses by forfeit
      }
      board[move.x][move.y] = player;
      if (checkWin(move.x, move.y, player, board)) return player;
      player = 3 - player;
    }
    return 0;
  } finally {
    Math.random = oldRandom;
  }
}

async function main() {
  const a = Number(process.argv[2]);
  const b = Number(process.argv[3]);
  const games = Number(process.argv[4] || 6);
  const baseSeed = Number(process.argv[5] || 4242);
  if (!(a >= 1 && a <= 7 && b >= 1 && b <= 7 && a !== b && games > 0)) {
    throw new Error('usage: node rank_pair.js LEVEL_A LEVEL_B GAMES [SEED]');
  }
  await MasterAI.loadModel(path.join(FRONTEND, 'master_model.json'));
  let aWins = 0, bWins = 0, draws = 0;
  let aBlackWins = 0, aWhiteWins = 0;
  for (let game = 0; game < games; game++) {
    const aBlack = game % 2 === 0;
    const result = await play(aBlack ? a : b, aBlack ? b : a,
                              baseSeed + game);
    if (result === 0) draws++;
    else if ((result === 1) === aBlack) {
      aWins++;
      if (aBlack) aBlackWins++; else aWhiteWins++;
    } else {
      bWins++;
    }
  }
  console.log(JSON.stringify({a, b, games, aWins, bWins, draws,
                              aBlackWins, aWhiteWins}));
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
