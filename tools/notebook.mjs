// Drive notebook.html headlessly: boot it, stage input files, add cells, run them, and read back what they
// produced. Shared by drive.mjs (the regression run) and runner.mjs (email-agent jobs), so both exercise the
// notebook exactly the way a person clicking "Run all" does.
import puppeteer from "puppeteer-core";

const MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/**
 * Chrome flags for WebGPU. ANGLE-on-Metal is what makes WebGPU work in headless Chrome on macOS; without it
 * the page boots but navigator.gpu is missing and every model call fails. `swiftshader` is the CPU fallback
 * for machines with no usable GPU (CI, containers): correct, and very slow.
 */
export function webgpuArgs({ swiftshader = false } = {}) {
  const args = ["--enable-unsafe-webgpu", "--no-sandbox"];
  if (swiftshader) args.push("--enable-unsafe-swiftshader", "--use-webgpu-adapter=swiftshader");
  else if (process.platform === "darwin") args.push("--use-angle=metal");
  else args.push("--enable-features=Vulkan");
  return args;
}

export function launch({ chrome = process.env.CHROME_PATH || MAC_CHROME, headful = false, swiftshader = false } = {}) {
  return puppeteer.launch({
    executablePath: chrome,
    headless: !headful,
    args: webgpuArgs({ swiftshader }),
    // A cell that holds the page's main thread (Python is single-threaded, and so is a big segmentation's
    // CPU side) blocks every DevTools call until it yields. Puppeteer's default 180 s per call then kills a
    // run that was fine. The waits below carry their own timeouts, so none is needed here.
    protocolTimeout: 0,
  });
}

/**
 * Open the notebook and wait for the kernel. `fresh` drops the saved cells first, so a run starts from an empty
 * notebook rather than whatever a previous session left in localStorage. `onPage` runs before the first
 * navigation, for request interception and the like.
 */
export async function openNotebook(browser, url, { timeout = 300_000, fresh = true, onPage } = {}) {
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(m.text()));
  page.on("pageerror", (e) => logs.push("PAGEERROR " + e.message));
  if (onPage) await onPage(page);
  await page.goto(url, { waitUntil: "domcontentloaded" });
  if (fresh) {
    await page.evaluate(() => localStorage.removeItem("ptero-cells"));
    await page.reload({ waitUntil: "domcontentloaded" });
  }
  try {
    await page.waitForFunction(() => document.getElementById("stat")?.textContent === "ready", { timeout, polling: 500 });
  } catch (error) {
    const stat = await page.evaluate(() => document.getElementById("stat")?.textContent).catch(() => "?");
    throw new Error(`notebook did not boot (status: ${stat}): ${error.message}`);
  }
  return { page, logs };
}

/** Banners the notebook shows at boot, such as a missing-WebGPU warning. */
export function banners(page) {
  return page.evaluate(() => [...document.querySelectorAll(".banner")].map((b) => b.textContent));
}

/**
 * Write a file into the Pyodide FS working directory. The folder picker needs a real user gesture, so a
 * headless run has no other way to supply an image. Bytes cross as base64 in chunks, which keeps each
 * DevTools message small for multi-hundred-MB stacks.
 */
export async function stage(page, name, bytes) {
  const CHUNK = 8 << 20;
  await page.evaluate((name) => { globalThis.__pteroStage = { name, parts: [] }; }, name);
  for (let at = 0; at < bytes.length; at += CHUNK) {
    const b64 = Buffer.from(bytes.subarray(at, at + CHUNK)).toString("base64");
    await page.evaluate((b64) => {
      const bin = atob(b64);
      const part = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) part[i] = bin.charCodeAt(i);
      globalThis.__pteroStage.parts.push(part);
    }, b64);
  }
  await page.evaluate(() => {
    const { name, parts } = globalThis.__pteroStage;
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { all.set(p, at); at += p.length; }
    globalThis.pyodide.FS.writeFile(name, all);
    delete globalThis.__pteroStage;
  });
}

/** Stage a file the page can fetch itself, such as one served from the repo root. */
export function stageUrl(page, url, name) {
  return page.evaluate(async (url, name) => {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    globalThis.pyodide.FS.writeFile(name, buf);
  }, url, name);
}

export function cellCount(page) {
  return page.evaluate(() => document.querySelectorAll(".cell").length);
}

export function appendCells(page, sources, author = "user") {
  return page.evaluate(async (sources, author) => {
    const cellsMod = await import("./src/notebook/cells.js");
    for (const src of sources) cellsMod.appendCell(src, author);
  }, sources, author);
}

/** Click "Run all" and wait for it to finish; the notebook stops at the first failing cell. */
export async function runAll(page, { timeout = 300_000 } = {}) {
  await page.click("#runall");
  await page.waitForFunction(() => !document.getElementById("runall").disabled, { timeout, polling: 1000 });
}

/**
 * Every cell's outcome. `state` is "done", "error", or whatever a cell that never ran keeps; `figures` are
 * the PNGs the kernel captured from figures the cell left open, base64-encoded.
 */
export function readCells(page) {
  return page.evaluate(() =>
    [...document.querySelectorAll(".cell")].map((el, i) => ({
      i,
      state: el.dataset.state,
      author: el.dataset.author,
      took: el.querySelector(".cellstat")?.textContent || "",
      out: (el.querySelector(".cellout")?.textContent || "").trim(),
      figures: [...el.querySelectorAll(".cellout-fig")].map((img) => img.src.replace(/^data:image\/png;base64,/, "")),
      table: !!el.querySelector(".cellout-html table"),
    })));
}

/** Files a cell run wrote, as the notebook's download bar lists them. */
export function producedFiles(page) {
  return page.evaluate(() =>
    [...document.querySelectorAll("#downloads .download")].map((a) => a.textContent.replace(/^⤓\s*/, "")));
}

export async function readFile(page, name) {
  const b64 = await page.evaluate((name) => {
    const bytes = globalThis.pyodide.FS.readFile(name);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  }, name);
  return Buffer.from(b64, "base64");
}
