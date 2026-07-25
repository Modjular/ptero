// Headless regression driver for the notebook.
//
// Boots notebook.html in Chrome with WebGPU, optionally stages input files into the
// Pyodide filesystem (headless can't use the folder picker), runs every cell, and
// reports each cell's status and output. Exits nonzero if any cell errors.
//
//   node tools/drive.mjs                       # run the seeded cells as-is
//   node tools/drive.mjs --stage demo/images/Composite.tif
//   node tools/drive.mjs --url http://localhost:8765/notebook.html --keep
//
// Requires a static server on the repo root and puppeteer-core:
//   python3 -m http.server 8765 &
//   npm install
import { spawn } from "node:child_process";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};
const has = (name) => args.includes(name);

const BASE = flag("--base", "http://localhost:8765");
const URL_ = flag("--url", `${BASE}/notebook.html`);
const TIMEOUT = Number(flag("--timeout", 300)) * 1000;
const STAGE = args.filter((a, i) => args[i - 1] === "--stage");

const CHROME = flag("--chrome", process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: !has("--headful"),
  // ANGLE-on-Metal is what makes WebGPU work in headless Chrome on macOS; without
  // these the page boots but navigator.gpu is missing and every model call fails.
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"],
});

let failed = false;
try {
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(m.text()));
  page.on("pageerror", (e) => { logs.push("PAGEERROR " + e.message); failed = true; });

  console.log(`→ ${URL_}`);
  await page.goto(URL_, { waitUntil: "domcontentloaded" });

  // Start from the example pipeline rather than whatever a previous manual session
  // left in localStorage, so a run is reproducible.
  if (!has("--keep")) {
    await page.evaluate(() => localStorage.removeItem("ptero-cells"));
    await page.reload({ waitUntil: "domcontentloaded" });
  }

  await page.waitForFunction(
    () => document.getElementById("stat")?.textContent === "ready", { timeout: TIMEOUT });
  console.log("✓ booted");

  const banners = await page.evaluate(() =>
    [...document.querySelectorAll(".banner")].map((b) => b.textContent));
  for (const b of banners) console.log("! " + b);

  // Stage input files into the Pyodide FS — the folder picker needs a real user
  // gesture, so a headless run has no other way to supply an image.
  for (const rel of STAGE) {
    const name = rel.split("/").pop();
    await page.evaluate(async (url, name) => {
      const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
      window.pyodide.FS.writeFile(name, buf);
    }, `${BASE}/${rel}`, name);
    console.log(`✓ staged ${name}`);
  }

  console.log("\n→ run all");
  await page.click("#runall");
  await page.waitForFunction(
    () => !document.getElementById("runall").disabled, { timeout: TIMEOUT });

  const cells = await page.evaluate(() =>
    [...document.querySelectorAll(".cell")].map((el, i) => ({
      i,
      state: el.dataset.state,
      author: el.dataset.author,
      took: el.querySelector(".cellstat")?.textContent || "",
      out: (el.querySelector(".cellout")?.textContent || "").trim().slice(0, 600),
      figs: el.querySelectorAll(".cellout-fig").length,
      table: !!el.querySelector(".cellout-html table"),
    })));

  console.log("");
  for (const c of cells) {
    const mark = c.state === "done" ? "✓" : c.state === "error" ? "✗" : "·";
    const extras = [c.figs ? `${c.figs} figure(s)` : null, c.table ? "table" : null]
      .filter(Boolean).join(", ");
    console.log(`${mark} cell ${c.i} [${c.state}] ${c.took}${extras ? "  " + extras : ""}`);
    if (c.out) console.log("    " + c.out.replace(/\n/g, "\n    "));
    if (c.state === "error") failed = true;
    if (c.state !== "done" && c.state !== "error") failed = true;
  }

  const files = await page.evaluate(() =>
    [...document.querySelectorAll("#downloads .download")].map((a) => a.textContent));
  if (files.length) console.log("\nproduced: " + files.join(", "));

  if (has("--logs")) {
    console.log("\n--- browser console ---");
    for (const l of logs) console.log("  " + l);
  }
} finally {
  await browser.close();
}

console.log(failed ? "\nFAILED" : "\nOK");
process.exit(failed ? 1 : 0);
