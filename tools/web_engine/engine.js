(function (global) {
  "use strict";

  const SIZE = 15;
  const CELLS = SIZE * SIZE;
  const PLANES = 4;

  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }

  class BinaryReader {
    constructor(buffer) {
      this.buffer = buffer;
      this.view = new DataView(buffer);
      this.offset = 0;
    }
    bytes(count) {
      const out = new Uint8Array(this.buffer, this.offset, count);
      this.offset += count;
      return out;
    }
    u32() {
      const value = this.view.getUint32(this.offset, true);
      this.offset += 4;
      return value;
    }
    i32() {
      const value = this.view.getInt32(this.offset, true);
      this.offset += 4;
      return value;
    }
    f32() {
      const value = this.view.getFloat32(this.offset, true);
      this.offset += 4;
      return value;
    }
    vector(expected) {
      const low = this.view.getUint32(this.offset, true);
      const high = this.view.getUint32(this.offset + 4, true);
      this.offset += 8;
      const count = high * 4294967296 + low;
      assert(count === expected, `vector size mismatch: expected ${expected}, got ${count}`);
      const copy = new Float32Array(expected);
      copy.set(new Float32Array(this.buffer, this.offset, expected));
      this.offset += expected * 4;
      return copy;
    }
  }

  function packConv(weight, outChannels, inChannels, kernel) {
    const kernelSize = inChannels * kernel * kernel;
    const packed = new Float32Array(kernelSize * outChannels);
    for (let out = 0; out < outChannels; out++) {
      const sourceBase = out * kernelSize;
      for (let k = 0; k < kernelSize; k++) {
        packed[k * outChannels + out] = weight[sourceBase + k];
      }
    }
    return packed;
  }

  function readConv(reader, inChannels, outChannels, kernel, padding) {
    const weight = reader.vector(outChannels * inChannels * kernel * kernel);
    const bias = reader.vector(outChannels);
    return {
      inChannels,
      outChannels,
      kernel,
      padding,
      weight,
      packed: packConv(weight, outChannels, inChannels, kernel),
      bias,
    };
  }

  function readNorm(reader, channels, epsilon) {
    const gamma = reader.vector(channels);
    const beta = reader.vector(channels);
    const mean = reader.vector(channels);
    const variance = reader.vector(channels);
    const scale = new Float32Array(channels);
    const bias = new Float32Array(channels);
    for (let c = 0; c < channels; c++) {
      scale[c] = gamma[c] / Math.sqrt(variance[c] + epsilon);
      bias[c] = beta[c] - mean[c] * scale[c];
    }
    return { gamma, beta, mean, variance, scale, bias };
  }

  function readLinear(reader, inputDim, outputDim) {
    return {
      inputDim,
      outputDim,
      weight: reader.vector(inputDim * outputDim),
      bias: reader.vector(outputDim),
    };
  }

  function parseModel(buffer) {
    const reader = new BinaryReader(buffer);
    const magic = String.fromCharCode(...reader.bytes(8));
    assert(magic === "XQPVRN01", `invalid model magic: ${magic}`);
    const version = reader.u32();
    assert(version === 1, `unsupported model version: ${version}`);
    const config = {
      inputChannels: reader.i32(),
      height: reader.i32(),
      width: reader.i32(),
      trunkChannels: reader.i32(),
      residualBlocks: reader.i32(),
      policyChannels: reader.i32(),
      policySize: reader.i32(),
      valueChannels: reader.i32(),
      valueHidden: reader.i32(),
      threadNum: reader.i32(),
      seed: reader.i32(),
      epsilon: reader.f32(),
      momentum: reader.f32(),
    };
    assert(config.inputChannels === PLANES && config.height === SIZE && config.width === SIZE,
      "model board/input shape mismatch");
    assert(config.policySize === CELLS, "model policy size mismatch");

    const model = { config, blocks: [] };
    model.stemConv = readConv(reader, config.inputChannels, config.trunkChannels, 3, 1);
    model.stemNorm = readNorm(reader, config.trunkChannels, config.epsilon);
    for (let i = 0; i < config.residualBlocks; i++) {
      model.blocks.push({
        conv1: readConv(reader, config.trunkChannels, config.trunkChannels, 3, 1),
        norm1: readNorm(reader, config.trunkChannels, config.epsilon),
        conv2: readConv(reader, config.trunkChannels, config.trunkChannels, 3, 1),
        norm2: readNorm(reader, config.trunkChannels, config.epsilon),
      });
    }
    model.policyConv = readConv(reader, config.trunkChannels, config.policyChannels, 1, 0);
    model.policyNorm = readNorm(reader, config.policyChannels, config.epsilon);
    model.policyLinear = readLinear(reader, config.policyChannels * CELLS, config.policySize);
    model.valueConv = readConv(reader, config.trunkChannels, config.valueChannels, 1, 0);
    model.valueNorm = readNorm(reader, config.valueChannels, config.epsilon);
    model.valueHidden = readLinear(reader, config.valueChannels * CELLS, config.valueHidden);
    model.valueOutput = readLinear(reader, config.valueHidden, 1);
    assert(reader.offset === buffer.byteLength,
      `trailing model bytes: ${buffer.byteLength - reader.offset}`);
    return model;
  }

  // Activations use [position, channel] (NHWC without batch) for contiguous
  // output-channel updates in the convolution inner loop.
  function convAffine(input, H, W, layer, norm, relu) {
    const C_in = layer.inChannels, C_out = layer.outChannels;
    const K = layer.kernel, pad = layer.padding;
    const packed = layer.packed, bias = norm.bias;
    const out = new Float32Array(H * W * C_out);

    const r0 = pad, r1 = H - pad, c0 = pad, c1 = W - pad;
    for (let r = 0; r < H; r++) {
      const rowMid = r >= r0 && r < r1;
      for (let c = 0; c < W; c++) {
        const ob = (r * W + c) * C_out;
        for (let oc = 0; oc < C_out; oc++) out[ob + oc] = bias[oc];

        if (rowMid && c >= c0 && c < c1) {
          // interior: all K*K taps in-bounds, oc unrolled by 4
          const iCol = ((r - pad) * W + (c - pad)) * C_in;  // top-left of window
          for (let ic = 0; ic < C_in; ic++) {
            let wb = ic * K * K * C_out;
            for (let kr = 0; kr < K; kr++) {
              const iRow = iCol + kr * W * C_in;
              for (let kc = 0; kc < K; kc++, wb += C_out) {
                const v = input[iRow + kc * C_in + ic];
                let oc = 0;
                for (; oc + 3 < C_out; oc += 4) {
                  out[ob + oc]     += v * packed[wb + oc];
                  out[ob + oc + 1] += v * packed[wb + oc + 1];
                  out[ob + oc + 2] += v * packed[wb + oc + 2];
                  out[ob + oc + 3] += v * packed[wb + oc + 3];
                }
                for (; oc < C_out; oc++) out[ob + oc] += v * packed[wb + oc];
              }
            }
          }
        } else {
          // border: original checked path (scale folded already)
          let kp = 0;
          for (let ic = 0; ic < C_in; ic++) {
            for (let kr = 0; kr < K; kr++) {
              const ir = r + kr - pad;
              for (let kc = 0; kc < K; kc++, kp++) {
                const jc = c + kc - pad;
                if (ir < 0 || ir >= H || jc < 0 || jc >= W) continue;
                const v = input[(ir * W + jc) * C_in + ic];
                const wb = kp * C_out;
                for (let oc = 0; oc < C_out; oc++) out[ob + oc] += v * packed[wb + oc];
              }
            }
          }
        }
        if (relu) for (let oc = 0; oc < C_out; oc++) { const x = out[ob + oc]; if (x < 0) out[ob + oc] = 0; }
      }
    }
    return out;
  }

  function residualForward(input, config, block) {
    const first = convAffine(input, config.height, config.width, block.conv1, block.norm1, true);
    const second = convAffine(first, config.height, config.width, block.conv2, block.norm2, false);
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) out[i] = Math.max(0, input[i] + second[i]);
    return out;
  }

  function flattenNCHW(nhwc, channels) {
    const flat = new Float32Array(CELLS * channels);
    for (let channel = 0; channel < channels; channel++) {
      for (let pos = 0; pos < CELLS; pos++) flat[channel * CELLS + pos] = nhwc[pos * channels + channel];
    }
    return flat;
  }

  function linear(input, layer, relu) {
    const out = new Float32Array(layer.outputDim);
    for (let output = 0; output < layer.outputDim; output++) {
      let sum = layer.bias[output];
      const base = output * layer.inputDim;
      for (let inputIndex = 0; inputIndex < layer.inputDim; inputIndex++) {
        sum += input[inputIndex] * layer.weight[base + inputIndex];
      }
      out[output] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  function encodeBoard(board, currentPlayer, lastAction) {
    const input = new Float32Array(CELLS * PLANES); // temporary NCHW
    for (let cell = 0; cell < CELLS; cell++) {
      if (board[cell] === currentPlayer) input[cell] = 1;
      else if (board[cell] === -currentPlayer) input[CELLS + cell] = 1;
    }
    if (lastAction >= 0) input[2 * CELLS + lastAction] = 1;
    if (currentPlayer === 1) input.fill(1, 3 * CELLS, 4 * CELLS);
    const nhwc = new Float32Array(CELLS * PLANES);
    for (let pos = 0; pos < CELLS; pos++) {
      for (let channel = 0; channel < PLANES; channel++) {
        nhwc[pos * PLANES + channel] = input[channel * CELLS + pos];
      }
    }
    return nhwc;
  }


  function foldModel(model) {
    if (model.__folded) return model;
    const fold = (layer, norm) => {
      const p = layer.packed, s = norm.scale, co = layer.outChannels;
      for (let kp = 0, kpn = p.length / co; kp < kpn; kp++)
        for (let oc = 0; oc < co; oc++) p[kp * co + oc] *= s[oc];
    };
    fold(model.stemConv, model.stemNorm);
    for (const b of model.blocks) { fold(b.conv1, b.norm1); fold(b.conv2, b.norm2); }
    fold(model.policyConv, model.policyNorm);
    fold(model.valueConv, model.valueNorm);
    model.__folded = true;
    return model;
  }

  function forward(model, board, currentPlayer, lastAction) {
    foldModel(model);
    const c = model.config;
    let trunk = convAffine(encodeBoard(board, currentPlayer, lastAction), SIZE, SIZE,
      model.stemConv, model.stemNorm, true);
    for (const block of model.blocks) trunk = residualForward(trunk, c, block);

    const policyTensor = convAffine(trunk, SIZE, SIZE, model.policyConv, model.policyNorm, true);
    const policyLogits = linear(flattenNCHW(policyTensor, c.policyChannels), model.policyLinear, false);

    const valueTensor = convAffine(trunk, SIZE, SIZE, model.valueConv, model.valueNorm, true);
    const hidden = linear(flattenNCHW(valueTensor, c.valueChannels), model.valueHidden, true);
    const rawValue = linear(hidden, model.valueOutput, false)[0];
    return { policyLogits, value: Math.tanh(rawValue) };
  }

  function legalSoftmax(logits, board) {
    const policy = new Float32Array(CELLS);
    let max = -Infinity;
    for (let a = 0; a < CELLS; a++) if (board[a] === 0 && logits[a] > max) max = logits[a];
    let sum = 0;
    for (let a = 0; a < CELLS; a++) {
      if (board[a] !== 0) continue;
      const value = Math.exp(logits[a] - max);
      policy[a] = value;
      sum += value;
    }
    if (sum > 0) for (let a = 0; a < CELLS; a++) policy[a] /= sum;
    return policy;
  }

  function candidateActions(board, moveCount, radius = 2) {
    if (moveCount === 0) return [Math.floor(CELLS / 2)];
    const result = [];
    for (let row = 0; row < SIZE; row++) {
      for (let col = 0; col < SIZE; col++) {
        const cell = row * SIZE + col;
        if (board[cell] !== 0) continue;
        let near = false;
        for (let dr = -radius; dr <= radius && !near; dr++) {
          for (let dc = -radius; dc <= radius && !near; dc++) {
            const r = row + dr, c = col + dc;
            if (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[r * SIZE + c] !== 0) near = true;
          }
        }
        if (near) result.push(cell);
      }
    }
    if (result.length === 0) for (let a = 0; a < CELLS; a++) if (board[a] === 0) result.push(a);
    return result;
  }

  function checkWin(board, action) {
    const row = Math.floor(action / SIZE), col = action % SIZE, color = board[action];
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of directions) {
      let count = 1;
      for (const sign of [-1, 1]) {
        for (let step = 1; step < 5; step++) {
          const r = row + sign * dr * step, c = col + sign * dc * step;
          if (r < 0 || r >= SIZE || c < 0 || c >= SIZE || board[r * SIZE + c] !== color) break;
          count++;
        }
      }
      if (count >= 5) return true;
    }
    return false;
  }

  function cloneState(state) {
    return {
      board: new Int8Array(state.board),
      currentPlayer: state.currentPlayer,
      moveCount: state.moveCount,
      lastAction: state.lastAction,
      result: state.result,
    };
  }

  function applyMove(state, action) {
    if (state.result !== 0 || action < 0 || action >= CELLS || state.board[action] !== 0) return false;
    state.board[action] = state.currentPlayer;
    state.moveCount++;
    state.lastAction = action;
    if (checkWin(state.board, action)) state.result = state.currentPlayer;
    else if (state.moveCount === CELLS) state.result = 2;
    else state.currentPlayer = -state.currentPlayer;
    return true;
  }

  function newNode() {
    return { edges: null, n: 0, w: 0 };
  }

  function sameState(left, right) {
    if (!left || !right || left.currentPlayer !== right.currentPlayer ||
        left.moveCount !== right.moveCount || left.lastAction !== right.lastAction ||
        left.result !== right.result) return false;
    for (let i = 0; i < CELLS; i++) {
      if (left.board[i] !== right.board[i]) return false;
    }
    return true;
  }

  class SearchSession {
    constructor(options = {}) {
      this.maxNodes = Math.max(1, options.maxNodes || 12000);
      // A legal root can contain every board action. The edge budget must
      // always accommodate one usable root even when callers request less.
      this.maxEdges = Math.max(CELLS, options.maxEdges || 250000);
      this.epoch = 0;
      this.reset();
    }

    reset() {
      this.epoch++;
      this.root = null;
      this.rootState = null;
      this.nodeCount = 0;
      this.edgeCount = 0;
      this.budgetExhausted = false;
    }

    begin(rootState) {
      this.epoch++;
      const reused = this.root != null && sameState(this.rootState, rootState);
      if (!reused) {
        this.root = newNode();
        this.rootState = cloneState(rootState);
        this.nodeCount = 1;
        this.edgeCount = 0;
      }
      // Matches C++ Search(): exhaustion applies to one search only. If the
      // retained tree has no room, this search will set it again and the next
      // real-move advance drops the cache.
      this.budgetExhausted = false;
      return { root: this.root, token: this.epoch, reused };
    }

    cancelled(token, externalStop) {
      return token !== this.epoch || (externalStop && externalStop());
    }

    allocateNode() {
      if (this.nodeCount >= this.maxNodes) {
        this.budgetExhausted = true;
        return null;
      }
      this.nodeCount++;
      return newNode();
    }

    reserveEdges(count) {
      if (this.edgeCount + count > this.maxEdges) {
        this.budgetExhausted = true;
        return false;
      }
      this.edgeCount += count;
      return true;
    }

    recountReachable() {
      if (!this.root) return false;
      let nodes = 0, edges = 0;
      const stack = [this.root];
      const seen = new Set();
      while (stack.length) {
        const node = stack.pop();
        if (seen.has(node)) continue;
        seen.add(node);
        nodes++;
        if (nodes > this.maxNodes) return false;
        if (!node.edges) continue;
        edges += node.edges.length;
        if (edges > this.maxEdges) return false;
        for (const edge of node.edges) if (edge.child) stack.push(edge.child);
      }
      this.nodeCount = nodes;
      this.edgeCount = edges;
      return true;
    }

    advance(action) {
      const exhausted = this.budgetExhausted;
      if (!this.root || !this.rootState || !applyMove(this.rootState, action)) {
        this.reset();
        return false;
      }
      const edge = this.root.edges && this.root.edges.find((item) => item.action === action);
      if (!edge) {
        this.reset();
        return false;
      }
      if (!edge.child) edge.child = this.allocateNode();
      if (!edge.child) {
        this.reset();
        return false;
      }
      this.root = edge.child;
      this.epoch++;
      if (exhausted) {
        this.reset();
        return false;
      }
      if (!this.recountReachable()) {
        this.reset();
        return false;
      }
      return true;
    }

    align(state) {
      if (sameState(this.rootState, state)) return true;
      if (!this.rootState || state.moveCount !== this.rootState.moveCount + 1 ||
          state.lastAction < 0) return false;
      const next = cloneState(this.rootState);
      if (!applyMove(next, state.lastAction) || !sameState(next, state)) return false;
      return this.advance(state.lastAction);
    }
  }

  function evaluate(model, state) {
    const output = forward(model, state.board, state.currentPlayer, state.lastAction);
    return { policy: legalSoftmax(output.policyLogits, state.board), value: output.value };
  }

  function expand(node, model, state, session) {
    const result = evaluate(model, state);
    const actions = candidateActions(state.board, state.moveCount);
    if (session && !session.reserveEdges(actions.length)) return result.value;
    let sum = 0;
    for (const action of actions) sum += result.policy[action];
    node.edges = actions.map((action) => ({
      action,
      prior: sum > 0 ? result.policy[action] / sum : 0,
      n: 0,
      w: 0,
      child: null,
    }));
    return result.value;
  }

  function terminalValue(state) {
    return state.result === 2 ? 0 : -1;
  }

  async function search(model, rootState, options = {}) {
    const simulations = Math.max(1, options.simulations || 60);
    const cPuct = options.cPuct == null ? 1.5 : options.cPuct;
    const fpu = options.fpu || 0;
    const yieldEvery = Math.max(1, options.yieldEvery || 4);
    const session = options.session || null;
    if (session && !sameState(session.rootState, rootState)) session.align(rootState);
    const started = session ? session.begin(rootState) :
      { root: newNode(), token: 0, reused: false };
    const root = started.root;
    const inheritedVisits = root.n;
    if (root.edges == null) expand(root, model, cloneState(rootState), session);

    for (let sim = 0; sim < simulations; sim++) {
      if ((session && session.cancelled(started.token, options.shouldStop)) ||
          (!session && options.shouldStop && options.shouldStop())) {
        return { cancelled: true, action: -1, visits: [], root };
      }
      if (session && session.budgetExhausted) break;
      const state = cloneState(rootState);
      let node = root;
      const path = [];
      let leafValue;
      while (true) {
        if (state.result !== 0) {
          leafValue = terminalValue(state);
          break;
        }
        if (node.edges == null) {
          leafValue = expand(node, model, state, session);
          break;
        }
        const parentQ = node.n > 0 ? node.w / node.n : 0;
        const sqrtN = Math.sqrt(Math.max(1, node.n));
        let best = node.edges[0], bestScore = -Infinity;
        for (const edge of node.edges) {
          const q = edge.n > 0 ? edge.w / edge.n : parentQ - fpu;
          const u = cPuct * edge.prior * sqrtN / (1 + edge.n);
          const score = q + u;
          if (score > bestScore) {
            bestScore = score;
            best = edge;
          }
        }
        path.push([node, best]);
        applyMove(state, best.action);
        if (best.child == null) {
          best.child = session ? session.allocateNode() : newNode();
          if (best.child == null) {
            leafValue = 0;
            break;
          }
        }
        node = best.child;
      }
      let value = leafValue;
      for (let i = path.length - 1; i >= 0; i--) {
        value = -value;
        const [parent, edge] = path[i];
        edge.n++;
        edge.w += value;
        parent.n++;
        parent.w += value;
      }
      if ((sim + 1) % yieldEvery === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        if ((session && session.cancelled(started.token, options.shouldStop)) ||
            (!session && options.shouldStop && options.shouldStop())) {
          return { cancelled: true, action: -1, visits: [], root };
        }
      }
    }

    assert(root.edges && root.edges.length, "search root has no candidate moves");
    let best = root.edges[0];
    for (const edge of root.edges) if (edge.n > best.n) best = edge;
    return {
      action: best.action,
      visits: root.edges.map((edge) => ({ action: edge.action, n: edge.n, q: edge.n ? edge.w / edge.n : 0, p: edge.prior })),
      root,
      reused: started.reused,
      inheritedVisits,
      nodeCount: session ? session.nodeCount : null,
      edgeCount: session ? session.edgeCount : null,
      budgetExhausted: session ? session.budgetExhausted : false,
    };
  }

  async function load(manifestUrl) {
    const manifestResponse = await fetch(manifestUrl);
    assert(manifestResponse.ok, `manifest HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
    const weightsResponse = await fetch(manifest.file);
    assert(weightsResponse.ok, `weights HTTP ${weightsResponse.status}`);
    const buffer = await weightsResponse.arrayBuffer();
    if (manifest.sha256) {
      assert(global.crypto && global.crypto.subtle, "WebCrypto SHA-256 support required");
      const digest = await global.crypto.subtle.digest("SHA-256", buffer);
      const actual = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
      assert(actual === manifest.sha256, `weights SHA-256 mismatch: ${actual}`);
    }
    const model = parseModel(buffer);
    model.manifest = manifest;
    return model;
  }


  function expandWithPolicy(node, state, policy, session) {
    const actions = candidateActions(state.board, state.moveCount);
    if (session && !session.reserveEdges(actions.length)) return null;
    let sum = 0;
    for (const action of actions) sum += policy[action];
    node.edges = actions.map((action) => ({
      action,
      prior: sum > 0 ? policy[action] / sum : 0,
      n: 0,
      w: 0,
      child: null,
    }));
    return node.edges;
  }

  // pool adapter: { post(msg), onMessage(fn), terminate() } per worker.
  function createSearchPool(modelBytes, workerCount, makeWorker) {
    const workers = [];
    let readyLeft = workerCount;
    let terminated = false;
    const idle = [];
    const waiters = new Map();
    let readyResolve;
    const readyPromise = new Promise((resolve) => { readyResolve = resolve; });
    for (let i = 0; i < workerCount; i++) {
      const w = makeWorker();
      w.onMessage((m) => {
        if (m.ok) {
          idle.push(w);
          if (--readyLeft === 0) readyResolve();
          return;
        }
        idle.push(w);
        const resolve = waiters.get(m.id);
        waiters.delete(m.id);
        if (resolve) resolve(m);
      });
      workers.push(w);
    }
    const init = { type: "init", id: 0, bytes: modelBytes };
    for (const w of workers) w.post(init);
    return {
      ready: () => readyPromise,
      size: () => workerCount,
      take: () => (terminated ? null : idle.pop() || null),
      release: (w) => { if (!terminated) idle.push(w); },
      exec: (w, msg) => new Promise((resolve) => {
        waiters.set(msg.id, resolve);
        w.post(msg);
      }),
      terminate: () => { terminated = true; for (const w of workers) w.terminate(); },
    };
  }

  async function searchPooled(pool, rootState, options = {}) {
    const simulations = Math.max(1, options.simulations || 60);
    const cPuct = options.cPuct == null ? 1.5 : options.cPuct;
    const fpu = options.fpu || 0;
    const session = options.session || null;
    if (session && !sameState(session.rootState, rootState)) session.align(rootState);
    const started = session ? session.begin(rootState) :
      { root: newNode(), token: 0, reused: false };
    const root = started.root;
    const inheritedVisits = root.n;

    const pending = new Map();
    const resolved = [];
    let notify = null;
    let done = 0, seq = 0;
    let cancelled = false;
    const inFlight = new Set();

    if (root.edges == null) {
      const worker = pool.take();
      if (!worker) throw new Error("pool has no worker for root expansion");
      const id = ++seq;
      const m = await pool.exec(worker, {
        type: "eval", id,
        board: rootState.board, currentPlayer: rootState.currentPlayer,
        lastAction: rootState.lastAction,
      });
      expandWithPolicy(root, rootState, m.policy, session);
    }

    const checkStop = () =>
      (session && session.cancelled(started.token, options.shouldStop)) ||
      (!session && options.shouldStop && options.shouldStop());

    const backprop = (path, leafValue, sign) => {
      let value = leafValue;
      for (let i = path.length - 1; i >= 0; i--) {
        value = -value;
        const [parent, edge] = path[i];
        edge.n += sign;
        edge.w += sign * value;
        parent.n += sign;
        parent.w += sign * value;
      }
    };

    while (done < simulations) {
      if (checkStop()) { cancelled = true; break; }
      if (session && session.budgetExhausted) break;

      while (done + pending.size < simulations) {
        if (session && session.budgetExhausted) break;
        const worker = pool.take();
        if (!worker) break;
        const state = cloneState(rootState);
        let node = root;
        const path = [];
        let needEval = false, evalNode = null, termValue = 0, busyLeaf = false;
        while (true) {
          if (state.result !== 0) { termValue = terminalValue(state); break; }
          if (node.edges == null) {
            if (inFlight.has(node)) { busyLeaf = true; break; }
            needEval = true; evalNode = node; break;
          }
          const parentQ = node.n > 0 ? node.w / node.n : 0;
          const sqrtN = Math.sqrt(Math.max(1, node.n));
          let best = node.edges[0], bestScore = -Infinity;
          for (const edge of node.edges) {
            const q = edge.n > 0 ? edge.w / edge.n : parentQ - fpu;
            const u = cPuct * edge.prior * sqrtN / (1 + edge.n);
            const score = q + u;
            if (score > bestScore) { bestScore = score; best = edge; }
          }
          path.push([node, best]);
          applyMove(state, best.action);
          if (best.child == null) {
            best.child = session ? session.allocateNode() : newNode();
            if (best.child == null) { termValue = 0; break; }
          }
          node = best.child;
        }
        if (busyLeaf) {
          pool.release(worker);
          break;
        }
        if (needEval) {
          backprop(path, -1, 1); // virtual loss: pretend loss while in flight
          inFlight.add(evalNode);
          const id = ++seq;
          pending.set(id, true);
          pool.exec(worker, {
            type: "eval", id,
            board: state.board, currentPlayer: state.currentPlayer,
            lastAction: state.lastAction,
          }).then((m) => {
            resolved.push({ id, path, node: evalNode, state, policy: m.policy, value: m.value });
            if (notify) { const n = notify; notify = null; n(); }
          });
        } else {
          pool.release(worker);
          backprop(path, termValue, 1);
          done++;
        }
      }

      while (resolved.length && done < simulations) {
        const leaf = resolved.shift();
        if (!pending.delete(leaf.id)) continue; // stale after cancel-drain
        inFlight.delete(leaf.node);
        backprop(leaf.path, -1, -1); // undo virtual loss
        expandWithPolicy(leaf.node, leaf.state, leaf.policy, session);
        backprop(leaf.path, leaf.value, 1);
        done++;
      }
      if (done < simulations && pending.size > 0 && resolved.length === 0) {
        await new Promise((resolve) => { notify = resolve; });
      }
    }

    if (cancelled) {
      while (pending.size > 0) {
        if (resolved.length === 0) await new Promise((resolve) => { notify = resolve; });
        while (resolved.length) {
          const leaf = resolved.shift();
          if (!pending.delete(leaf.id)) continue;
          inFlight.delete(leaf.node);
          backprop(leaf.path, -1, -1); // restore tree consistency for session reuse
        }
      }
      return { cancelled: true, action: -1, visits: [], root };
    }

    assert(root.edges && root.edges.length, "search root has no candidate moves");
    let best = root.edges[0];
    for (const edge of root.edges) if (edge.n > best.n) best = edge;
    return {
      action: best.action,
      visits: root.edges.map((edge) => ({ action: edge.action, n: edge.n, q: edge.n ? edge.w / edge.n : 0, p: edge.prior })),
      root,
      reused: started.reused,
      inheritedVisits,
    };
  }

  const api = {
    SIZE,
    CELLS,
    parseModel,
    load,
    forward,
    legalSoftmax,
    candidateActions,
    checkWin,
    applyMove,
    search,
    SearchSession,
    searchPooled,
    createSearchPool,
    expandWithPolicy,
    createState(board, currentPlayer, lastAction = -1) {
      const cells = board instanceof Int8Array ? new Int8Array(board) : Int8Array.from(board || new Array(CELLS).fill(0));
      let moveCount = 0;
      for (const value of cells) if (value !== 0) moveCount++;
      return { board: cells, currentPlayer: currentPlayer || (moveCount % 2 === 0 ? 1 : -1), moveCount, lastAction, result: 0 };
    },
  };

  global.AlphaZeroGomoku = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);

// ---------- worker-parallel search (virtual-loss batching) ----------
// Protocol: worker init {type:"init", bytes} -> model; eval request
// {type:"eval", id, board, currentPlayer, lastAction}; response
// {id, policy: Float32Array(CELLS), value}.
// The same file runs in the page and inside a dedicated worker.

(function (global) {
  const isWorker = typeof global.window === "undefined" &&
    typeof global.importScripts === "function";
  if (isWorker) {
    let model = null;
    global.onmessage = (e) => {
      const m = e.data;
      if (m.type === "init") {
        model = global.AlphaZeroGomoku.parseModel(m.bytes);
        global.postMessage({ id: m.id, ok: true });
        return;
      }
      if (m.type === "eval") {
        const AZ = global.AlphaZeroGomoku;
        const out = AZ.forward(model, m.board, m.currentPlayer, m.lastAction);
        const policy = AZ.legalSoftmax(out.policyLogits, m.board);
        global.postMessage({ id: m.id, policy, value: out.value }, [policy.buffer]);
      }
    };
    return;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
