// The chat surface: a side pane, the settings popover, and the UI object the agent
// talks back through.
import { Agent } from "./agent.js";
import {
  PROVIDERS, getProviderId, setProviderId, provider,
  getKey, setKey, hasKey, getModel, setModel, listModels,
} from "./llm.js";
import * as cellsMod from "../notebook/cells.js";

const $ = (id) => document.getElementById(id);

let agent = null;
let logEl, inputEl, sendBtn, paneEl;

// ---- rendering -----------------------------------------------------------------------
function add(cls, text) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  return el;
}

// Minimal markdown: fenced code and inline `code` are the only things the agent
// reliably emits that plain text would mangle. Everything is inserted as text nodes.
function renderProse(el, raw) {
  el.replaceChildren();
  for (const [i, part] of raw.split(/```(?:\w+\n|\n)?/).entries()) {
    if (!part) continue;
    if (i % 2) {
      const pre = document.createElement("pre");
      pre.className = "msg-code";
      pre.textContent = part.replace(/\n$/, "");
      el.appendChild(pre);
    } else {
      for (const [j, chunk] of part.split("`").entries()) {
        if (!chunk) continue;
        if (j % 2) {
          const c = document.createElement("code");
          c.textContent = chunk;
          el.appendChild(c);
        } else {
          el.appendChild(document.createTextNode(chunk));
        }
      }
    }
  }
  logEl.scrollTop = logEl.scrollHeight;
}

// The activity chips. Deliberately terse: the point is that the user can see the agent
// is working and roughly on what, not that they read a debug log. Failed drafts and
// tracebacks never come through here — that is the context-protection rule.
function chip(text, cell) {
  const el = document.createElement("div");
  el.className = "chip";
  el.textContent = text;
  if (cell) {
    el.classList.add("chip-link");
    el.addEventListener("click", () => cellsMod.highlight(cell));
  }
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
}

// ---- the UI object the agent drives ---------------------------------------------------
function makeUI() {
  return {
    note: (text, cell) => chip(text, cell),
    error: (text) => add("err", text),
    done: () => {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
      inputEl.disabled = false;
      inputEl.focus();
    },

    // Streaming prose: one bubble that grows as tokens arrive.
    stream: () => {
      let el = null, buf = "";
      return {
        push(t) {
          buf += t;
          if (!el) el = add("bot", "");
          renderProse(el, buf);
        },
        close() { if (el && !buf.trim()) el.remove(); },
      };
    },

    // ask_user: render the question with optional buttons and block until answered.
    ask: (question, options) => new Promise((resolve) => {
      const wrap = document.createElement("div");
      wrap.className = "msg bot ask";
      const q = document.createElement("div");
      q.textContent = question;
      wrap.appendChild(q);

      // Clearing pendingAnswer belongs here, in the one place every path funnels
      // through — not in the typed-answer handler. Clicking an option button
      // otherwise leaves the pane still believing it is waiting for an answer, and
      // the user's next message gets routed into an already-resolved promise and
      // silently dropped.
      const finish = (answer) => {
        pendingAnswer = null;
        inputEl.placeholder = defaultPlaceholder;
        wrap.querySelector(".ask-opts")?.remove();
        add("me", answer);
        resolve(answer);
      };

      if (options?.length) {
        const opts = document.createElement("div");
        opts.className = "ask-opts";
        for (const o of options) {
          const b = document.createElement("button");
          b.className = "btn-subtle";
          b.textContent = o;
          b.addEventListener("click", () => finish(o));
          opts.appendChild(b);
        }
        wrap.appendChild(opts);
      }
      logEl.appendChild(wrap);
      logEl.scrollTop = logEl.scrollHeight;

      // Typing an answer works too, whether or not there are buttons.
      inputEl.disabled = false;
      inputEl.placeholder = "Answer…";
      inputEl.focus();
      pendingAnswer = finish;
    }),
  };
}

let pendingAnswer = null;
const defaultPlaceholder = "Describe what you want to measure…";

// ---- input --------------------------------------------------------------------------
function submit() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  autosize();

  // An answer to ask_user resolves the agent's pending promise instead of starting a
  // new turn — the agent is mid-thought and waiting.
  if (pendingAnswer) { pendingAnswer(text); return; }

  add("me", text);
  sendBtn.disabled = true;
  sendBtn.textContent = "…";
  inputEl.disabled = true;
  agent.turn(text);
}

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
}

// ---- settings -------------------------------------------------------------------------
// The dialog edits one provider at a time. Switching the dropdown swaps the key and
// model fields to that provider's stored values rather than carrying the previous
// provider's text across, so nothing gets saved under the wrong provider.
let editingProvider = null;

function showProviderFields(id) {
  editingProvider = id;
  const p = PROVIDERS[id];
  $("apikey").value = getKey(id);
  $("apikey").placeholder = p.keyPlaceholder;
  $("modelname").value = getModel(id);
  $("modellist").replaceChildren();
  $("keyhint").innerHTML = "";
  $("keyhint").append(
    document.createTextNode(
      `Stored in this browser only and sent straight from this page to ${p.host}. ` +
      `Anything else running on this origin can read it, so use a key you're willing ` +
      `to scope narrowly and rotate. `),
  );
  const a = document.createElement("a");
  a.href = p.keyUrl; a.target = "_blank"; a.rel = "noreferrer";
  a.textContent = "Get a key";
  $("keyhint").append(a);
  $("modelhint").textContent = "Keys and model choices are remembered per provider.";
}

function openSettings() {
  const sel = $("providersel");
  sel.replaceChildren();
  for (const [id, p] of Object.entries(PROVIDERS)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = p.label + (hasKey(id) ? " ✓" : "");
    sel.appendChild(opt);
  }
  sel.value = getProviderId();
  showProviderFields(sel.value);
  $("settings").showModal();
}

// Ask the provider what models the key can actually reach, rather than shipping a
// hardcoded list that goes stale. Needs the key, so it saves whatever is typed first.
async function loadModelList() {
  const btn = $("loadmodels");
  const typed = $("apikey").value;
  if (!typed.trim()) { $("modelhint").textContent = "Enter a key first."; return; }
  setKey(typed, editingProvider);
  const previous = getProviderId();
  setProviderId(editingProvider);
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const models = await listModels();
    $("modellist").replaceChildren();
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.label = m.label;
      $("modellist").appendChild(opt);
    }
    $("modelhint").textContent =
      `${models.length} model${models.length === 1 ? "" : "s"} available — click the ` +
      `field to pick one.`;
  } catch (e) {
    $("modelhint").textContent = e.message;
  } finally {
    setProviderId(previous);
    btn.disabled = false;
    btn.textContent = "List models";
  }
}

function saveSettings() {
  setKey($("apikey").value, editingProvider);
  setModel($("modelname").value, editingProvider);
  setProviderId(editingProvider);
  $("settings").close();
  updateGate();
  greetIfEmpty();
}

function updateGate() {
  const ready = hasKey();
  inputEl.disabled = !ready;
  sendBtn.disabled = !ready;
  inputEl.placeholder = ready
    ? defaultPlaceholder
    : "Add an API key under ⚙ to start";
  $("chatsettings").title = ready
    ? `${provider().label} · ${getModel()}`
    : "Add an API key";
}

// ---- mount ---------------------------------------------------------------------------
export function mountChat({ syncCalls = true } = {}) {
  paneEl = $("chat");
  logEl = $("chatlog");
  inputEl = $("chatinput");
  sendBtn = $("chatsend");

  agent = new Agent(makeUI());
  agent.syncCalls = syncCalls;

  sendBtn.addEventListener("click", submit);
  inputEl.addEventListener("input", autosize);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  $("chatsettings").addEventListener("click", openSettings);
  $("settingssave").addEventListener("click", (e) => { e.preventDefault(); saveSettings(); });
  $("loadmodels").addEventListener("click", loadModelList);
  $("providersel").addEventListener("change", (e) => {
    // Save what was typed for the provider being left, so switching away and back
    // doesn't lose a freshly pasted key.
    setKey($("apikey").value, editingProvider);
    setModel($("modelname").value, editingProvider);
    showProviderFields(e.target.value);
  });
  $("chatreset").addEventListener("click", () => {
    agent.stop();
    agent.reset();
    pendingAnswer = null;
    inputEl.placeholder = defaultPlaceholder;
    logEl.replaceChildren();
    greet();
  });
  $("chattoggle").addEventListener("click", () => {
    document.body.classList.toggle("chat-open");
  });

  updateGate();
  greet();
}

function greet() {
  add("bot", hasKey()
    ? "Tell me what you're looking at and what you want to measure. I'll ask about " +
      "anything I can't work out myself, test the code before you see it, and put " +
      "working cells in your notebook for you to run."
    : "I need an API key before I can help — click ⚙ above and pick a provider " +
      "(Anthropic or Google Gemini). It's stored in this browser only, and calls go " +
      "straight from this page to the provider.");
}

// After a key is added, replace the "I need a key" opener rather than stacking a
// second greeting under it.
function greetIfEmpty() {
  if (logEl.children.length !== 1) return;
  logEl.replaceChildren();
  greet();
}
