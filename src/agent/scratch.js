// Main-thread handle for the scratch kernel Worker.
import { catalogue } from "../registry.js";
import { getWorkspaceHandle } from "../kernel.js";

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
      wheelBase: new URL("../../wheels/", import.meta.url).href,
      catalogue: JSON.stringify(catalogue()),
    });
  })();
  return booting;
}

export async function testInScratch({ code, vars }) {
  await ensureScratch();
  // Handing the handle over on every call (rather than once, at boot) is what lets a
  // folder chosen mid-conversation, or picked before the agent ever boots the worker,
  // show up without extra plumbing. The worker copies from it read-only; see
  // scratch-worker.js's mountWorkspace.
  return send("test", { code, vars, workspaceHandle: getWorkspaceHandle() });
}

// NOTE: not called anywhere today. It clears the scratch worker's globals for a fresh
// start (scratch-worker.js's `reset`), which the worker's own comment says should be
// possible "when the agent changes approach" — but no caller was ever wired up. Kept
// as-is rather than deleted: this looks like an unfinished hook, not dead code.
export async function resetScratch() {
  if (!booting) return;
  await ensureScratch();
  return send("reset");
}

export function isScratchUp() { return booting !== null; }
