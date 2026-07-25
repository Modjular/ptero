// Main-thread handle for the scratch kernel Worker.
import { catalogue } from "../registry.js";

let worker = null;
let booting = null;
let seq = 0;
const pending = new Map();

function send(cmd, payload) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, payload });
  });
}

/**
 * Boot the scratch kernel. Deliberately not called at page load — a session where
 * nobody talks to the agent should never pay for a second Pyodide.
 */
export function ensureScratch() {
  if (booting) return booting;
  booting = (async () => {
    worker = new Worker(new URL("./scratch-worker.js", import.meta.url));
    worker.onmessage = (e) => {
      const { id, ok, result, error } = e.data;
      const p = pending.get(id);
      if (!p) return;
      pending.delete(id);
      ok ? p.resolve(result) : p.reject(new Error(error));
    };
    worker.onerror = (e) => {
      for (const [, p] of pending) p.reject(new Error(e.message || "scratch worker crashed"));
      pending.clear();
    };
    await send("boot", {
      shimBase: new URL("../shims/", import.meta.url).href,
      catalogue: JSON.stringify(catalogue()),
    });
  })();
  return booting;
}

export async function testInScratch({ code, vars }) {
  await ensureScratch();
  return send("test", { code, vars });
}

export async function resetScratch() {
  if (!booting) return;
  await ensureScratch();
  return send("reset");
}

export function isScratchUp() { return booting !== null; }
