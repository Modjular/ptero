// The email agent's compute: claims notebook jobs from the email-agent Worker, runs each in a fresh
// notebook.html tab with real WebGPU, and reports the cells' outcomes, figures, and written files back.
//
//   RUNNER_TOKEN=… node tools/runner.mjs --api https://email.ptero.example
//   node tools/runner.mjs --api http://localhost:8787 --once        # drain the queue, then exit
//
// Requires the same static server as drive.mjs (`python3 -m http.server 8765` on the repo root) and the same
// Chrome flags: --chrome / CHROME_PATH, --headful, --swiftshader.
//
// A job is leased, not taken: if this process dies, the Worker hands the job out again once the lease
// expires (20 min), and gives up after three tries. So a failure of the runner itself — the notebook not
// booting, the API unreachable — is never reported as the job's failure: the agent would "fix" cells that
// were fine. Only what the cells did is reported.
import * as notebook from "./notebook.mjs";

const args = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};
const has = (name) => args.includes(name);

const API = (flag("--api", process.env.PTERO_API) ?? "").replace(/\/$/, "");
const TOKEN = process.env.RUNNER_TOKEN;
const BASE = flag("--base", "http://localhost:8765");
const POLL_MS = Number(flag("--poll", 15)) * 1000;
// Under the Worker's 20-minute lease, so a slow job reports before its lease can be handed to someone else.
const RUN_TIMEOUT_MS = Number(flag("--job-timeout", 15 * 60)) * 1000;
const BOOT_TIMEOUT_MS = 5 * 60_000;
const OUTPUT_LIMIT = 4000;

if (!API || !TOKEN) {
  console.error("usage: RUNNER_TOKEN=… node tools/runner.mjs --api <email-agent URL> [--once]");
  process.exit(2);
}

/**
 * Pyodide's package loader writes its progress into the cell output ("Loading numpy, …", "numpy already
 * loaded from default channel"). The agent pays for every line it reads, and these say nothing about the
 * cell, so they never reach the report.
 */
const LOADER_NOISE = [
  /^No new packages to load$/,
  /^[\w.-]+ already loaded from default channel$/,
  /^Didn't find package [\w.-]+/,
  /APIVersionWarning: Unsupported API minor version/,
  /^\s+project_detail = from_project_details_html\(data, pkgname\)$/,
];
const PACKAGES = /^Load(?:ing|ed) ([\w.-]+(?:, [\w.-]+)*)$/;
const cleanOutput = (text) => {
  const lines = text.split("\n");
  // "Loading a, b" is the loader only if "Loaded a, b" follows; a cell printing "Loading data" stays.
  const loaded = new Set(lines.filter((l) => l.startsWith("Loaded ")).map((l) => l.match(PACKAGES)?.[1]));
  return lines
    .filter((line) => {
      const packages = line.match(PACKAGES)?.[1];
      if (packages !== undefined && loaded.has(packages)) return false;
      return !LOADER_NOISE.some((noise) => noise.test(line));
    })
    .join("\n")
    .trim();
};

/** Names the Worker will store: one path segment, nothing that climbs out of the job's prefix. */
const safeName = (name) => name.length > 0 && name.length <= 200 && !/[/\\]|^\.\.?$/.test(name);

async function api(method, path, { body, lease, ok = [200] } = {}) {
  const headers = { Authorization: `Bearer ${TOKEN}` };
  if (lease) headers["X-Lease"] = lease;
  const response = await fetch(`${API}${path}`, { method, headers, body });
  if (!ok.includes(response.status)) {
    throw new Error(`${method} ${path}: ${response.status} ${await response.text().catch(() => "")}`);
  }
  return response;
}

/** Thrown for anything the cells did not cause; the job is left to its lease. */
class RunnerFault extends Error {}

async function runJob(browser, job) {
  const jobPath = `/runner/jobs/${encodeURIComponent(job.id)}`;
  const opts = { lease: job.lease };
  const log = (msg) => console.log(`[${job.id}] ${msg}`);
  log(`attempt ${job.attempt}: ${job.cells.length} cells on ${job.files.join(", ") || "no files"}`);

  let page;
  try {
    try {
      ({ page } = await notebook.openNotebook(browser, `${BASE}/notebook.html`, { timeout: BOOT_TIMEOUT_MS }));
      for (const name of job.files) {
        const bytes = new Uint8Array(await (await api("GET", `${jobPath}/files/${encodeURIComponent(name)}`, opts)).arrayBuffer());
        await notebook.stage(page, name, bytes);
        log(`staged ${name} (${(bytes.length / 1e6).toFixed(1)} MB)`);
      }
      await notebook.appendCells(page, job.cells, "agent");
    } catch (error) {
      throw new RunnerFault(error.message);
    }

    let error;
    try {
      await notebook.runAll(page, { timeout: RUN_TIMEOUT_MS });
    } catch {
      // The cells are still running: what they did so far is the report, and the timeout is their failure.
      error = `the notebook was still running after ${RUN_TIMEOUT_MS / 60_000} minutes`;
    }

    const cells = await notebook.readCells(page);
    const artifacts = [];
    const upload = async (name, bytes) => {
      if (!safeName(name)) return log(`skipped artifact with unsafe name ${JSON.stringify(name)}`);
      await api("PUT", `${jobPath}/artifacts/${encodeURIComponent(name)}`, { ...opts, body: bytes });
      artifacts.push(name);
    };
    const reported = [];
    for (const cell of cells) {
      const figures = [];
      for (const [k, b64] of cell.figures.entries()) {
        const name = `cell-${cell.i + 1}-fig-${k + 1}.png`;
        await upload(name, Buffer.from(b64, "base64"));
        figures.push(name);
      }
      // A cell after the first failure keeps its idle state: say so plainly rather than leak the UI's term.
      const state = cell.state === "done" || cell.state === "error" ? cell.state : "not run";
      reported.push({ state, text: cleanOutput(cell.out).slice(-OUTPUT_LIMIT), figures });
    }
    if (error === undefined) {
      for (const name of await notebook.producedFiles(page)) await upload(name, await notebook.readFile(page, name));
    }
    // The notebook as run: the deliverable the customer will open.
    await upload("notebook.json", JSON.stringify({ files: job.files, cells: job.cells }, null, 2));

    const ok = error === undefined && cells.length > 0 && cells.every((cell) => cell.state === "done");
    await api("POST", `${jobPath}/report`, {
      ...opts,
      body: JSON.stringify({ ok, error, cells: reported, artifacts }),
      // 409: the lease ran out under us and the job went to another runner; its report wins.
      ok: [200, 409],
    });
    log(`${ok ? "done" : "failed"}: ${cells.map((c) => c.state).join(" ")}; ${artifacts.length} artifacts`);
  } finally {
    await page?.close().catch(() => {});
  }
}

async function main() {
  let browser;
  const ensureBrowser = async () => {
    if (browser?.connected) return browser;
    browser = await notebook.launch({
      chrome: flag("--chrome", process.env.CHROME_PATH || undefined),
      headful: has("--headful"),
      swiftshader: has("--swiftshader"),
    });
    return browser;
  };

  console.log(`runner polling ${API} every ${POLL_MS / 1000}s`);
  try {
    for (;;) {
      let job;
      try {
        const response = await api("POST", "/runner/claim", { ok: [200, 204] });
        job = response.status === 200 ? await response.json() : null;
      } catch (error) {
        console.error(`claim failed: ${error.message}`);
      }
      if (!job) {
        if (has("--once")) return;
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        continue;
      }
      try {
        await runJob(await ensureBrowser(), job);
      } catch (error) {
        console.error(`[${job.id}] runner fault, leaving the job to its lease: ${error.message}`);
        if (!(error instanceof RunnerFault)) await browser?.close().catch(() => {});
      }
    }
  } finally {
    await browser?.close().catch(() => {});
  }
}

await main();
