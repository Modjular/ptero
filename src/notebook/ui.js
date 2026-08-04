// Notebook chrome: theme, workspace folder, toolbar, downloads, boot orchestration.
import * as kernel from "../kernel.js";
import * as cells from "./cells.js";

const $ = (id) => document.getElementById(id);

// ---- theme ---------------------------------------------------------------------------
let isDark = (() => {
  const stored = localStorage.getItem("ptero-theme");
  if (stored) return stored === "dark";
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true;
})();

function applyTheme(dark) {
  isDark = dark;
  document.body.classList.toggle("theme-light", !dark);
  $("themetoggle").textContent = dark ? "🌙" : "☀️";
  $("themetoggle").title = dark ? "Switch to light theme" : "Switch to dark theme";
  localStorage.setItem("ptero-theme", dark ? "dark" : "light");
  cells.applyTheme(dark);
}

// ---- workspace + downloads -------------------------------------------------------------
// Files the notebook produced, offered as downloads. Without a mounted folder this is
// the only way results leave the page, so it is not optional chrome.
const PRODUCED = /\.(csv|tif|tiff|png|json|txt|xlsx)$/i;
let knownFiles = new Set();

async function refreshWorkspace() {
  const names = await kernel.syncWorkspace();
  const info = $("workspace-info");
  if (kernel.hasWorkspace()) {
    info.textContent = names.length
      ? `Workspace: ${names.join(", ")}`
      : "Workspace is empty — copy an image into that folder, then hit ↻.";
  } else {
    info.textContent = "No workspace folder. Cells run against an in-memory filesystem; " +
      "anything they write shows up below as a download.";
  }

  const dl = $("downloads");
  dl.replaceChildren();
  for (const name of names) {
    if (!PRODUCED.test(name) || !knownFiles.has(name)) continue;
    const a = document.createElement("a");
    a.className = "download";
    a.textContent = `⤓ ${name}`;
    a.href = URL.createObjectURL(new Blob([kernel.readFile(name)]));
    a.download = name;
    dl.appendChild(a);
  }
}

// Only offer downloads for files a *cell run* produced. Diffing the listing across
// each run (rather than once against boot) means mounting a folder of images, or
// dropping a new file in mid-session, never makes those inputs look like results.
let filesBeforeRun = new Set();

async function beforeRun() {
  filesBeforeRun = new Set(await kernel.syncWorkspace());
}

async function noteNewFiles() {
  for (const n of await kernel.syncWorkspace()) {
    if (!filesBeforeRun.has(n)) knownFiles.add(n);
  }
}

async function chooseWorkspace() {
  if (!window.showDirectoryPicker) {
    setStatus("folder picker needs Chrome or Edge");
    return;
  }
  let handle;
  try {
    handle = await window.showDirectoryPicker({ mode: "readwrite" });
  } catch {
    return;   // user cancelled
  }
  if (await handle.requestPermission({ mode: "readwrite" }) !== "granted") {
    setStatus("read/write access denied");
    return;
  }
  await kernel.mountWorkspace(handle);
  $("choosews").textContent = `📁 ${handle.name}`;
  // A newly mounted folder's existing contents are inputs, not results.
  knownFiles = new Set();
  await refreshWorkspace();
}

// ---- status ------------------------------------------------------------------------------
function setStatus(msg) { $("stat").textContent = msg; }

// ---- boot ---------------------------------------------------------------------------------
export async function start() {
  cells.mount($("cells"), { dark: isDark });
  cells.setRunHooks({
    before: beforeRun,
    after: async () => { await noteNewFiles(); await refreshWorkspace(); },
  });

  $("themetoggle").addEventListener("click", () => applyTheme(!isDark));
  $("choosews").addEventListener("click", chooseWorkspace);
  $("refreshws").addEventListener("click", refreshWorkspace);
  $("addcell").addEventListener("click", () => {
    const c = cells.appendCell("");
    c.view.focus();
  });
  $("runall").addEventListener("click", async () => {
    $("runall").disabled = true;
    try { await cells.runAll(); } finally { $("runall").disabled = false; }
  });
  applyTheme(isDark);

  if (!navigator.gpu) {
    setStatus("no WebGPU");
    document.body.prepend(banner(
      "This page needs WebGPU. Use Chrome or Edge (or Safari 18+) — segmentation cells " +
      "will fail without it, though the rest of the Python still runs."));
  }

  setStatus("booting Python…");
  try {
    await kernel.boot({ stdout: (m) => console.log(m) });
  } catch (e) {
    setStatus("boot failed");
    document.body.prepend(banner(`Couldn't start the Python kernel: ${e.message}`));
    throw e;
  }
  // Restore the previous session's cells, if any — a fresh notebook starts empty.
  const saved = cells.loadSavedCells();
  if (saved) {
    for (const { source, author } of saved) cells.appendCell(source, author);
  }

  // Loaded dynamically so a session that never opens the assistant doesn't pay for it.
  const { mountChat } = await import("../agent/chat.js");
  mountChat();

  setStatus("ready");
  await refreshWorkspace();
}

function banner(text, kind = "error") {
  const d = document.createElement("div");
  d.className = `banner banner-${kind}`;
  d.textContent = text;
  return d;
}
