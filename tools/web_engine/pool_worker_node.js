// pool_worker_node.js — node:worker_threads adapter emulating the browser worker protocol.
const { parentPort, workerData } = require("worker_threads");
const fs = require("fs");
const self = globalThis;
self.importScripts = () => {};
self.postMessage = (msg) => parentPort.postMessage(msg);
parentPort.on("message", (m) => self.onmessage({ data: m }));
self.window = undefined;
eval(fs.readFileSync(workerData.enginePath, "utf8"));
