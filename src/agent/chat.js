// The chat surface: a side pane, the settings popover, and the UI object the agent
// talks back through.
import { Agent } from "./agent.js";
import {
  PROVIDERS, getProviderId, setProviderId, provider,
  getKey, setKey, hasKey, getModel, setModel, getBaseUrl, setBaseUrl, listModels,
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

// Minimal markdown: fenced code, inline `code`, **bold**, and *italics*.
// Everything is safely inserted as text nodes to prevent XSS.
//
// `.msg` renders with `white-space: pre-wrap`, so a leading/trailing blank line in the
// model's raw text — a completely ordinary way for it to format a reply — shows up as
// real vertical space in the bubble, not CSS padding. Blank-line runs are trimmed at
// the outer edges (never internally, which would eat intentional paragraph breaks).
function renderProse(el, raw) {
  el.replaceChildren();
  const trimmed = raw.replace(/^\n+/, "").replace(/\n+$/, "");
  
  // 1. Split by fenced code blocks
  for (const [i, part] of trimmed.split(/```(?:\w+\n|\n)?/).entries()) {
    if (!part) continue;
    
    if (i % 2) {
      const pre = document.createElement("pre");
      pre.className = "msg-code";
      pre.textContent = part.replace(/^\n+/, "").replace(/\n+$/, "");
      el.appendChild(pre);
    } else {
      
      // 2. Split by inline code
      for (const [j, chunk] of part.split("`").entries()) {
        if (!chunk) continue;
        
        if (j % 2) {
          const c = document.createElement("code");
          c.textContent = chunk;
          el.appendChild(c);
        } else {
          
          // 3. Split plain text by bold/italic markers
          // The capturing group ( ) ensures the matched delimiters are kept in the array
          const textTokens = chunk.split(/(\*\*[\s\S]+?\*\*|__[\s\S]+?__|\*[\s\S]+?\*|_[\s\S]+?_)/);
          
          for (const token of textTokens) {
            if (!token) continue;
            
            if (token.startsWith("**") && token.endsWith("**") || 
                token.startsWith("__") && token.endsWith("__")) {
              const strong = document.createElement("strong");
              strong.textContent = token.slice(2, -2); // Strip the 2 delimiter chars
              el.appendChild(strong);
            } 
            else if (token.startsWith("*") && token.endsWith("*") || 
                     token.startsWith("_") && token.endsWith("_")) {
              const em = document.createElement("em");
              em.textContent = token.slice(1, -1); // Strip the 1 delimiter char
              el.appendChild(em);
            } 
            else {
              el.appendChild(document.createTextNode(token));
            }
          }
        }
      }
    }
  }
  
  // Assuming logEl is defined in your broader scope
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
  return el;
}

// A capture_view preview, shown to the user as a thumbnail. This is the one tool
// result that deliberately reaches the transcript: it is an image the agent already
// looked at, not debugging noise, so it does not violate the context-protection rule.
// Clicking opens the full-size preview in a lightbox.
function renderCapture({ image, meta = {}, title }) {
  const wrap = document.createElement("div");
  wrap.className = "capture";

  const img = document.createElement("img");
  img.className = "capture-thumb";
  img.loading = "lazy";
  img.src = `data:image/png;base64,${image}`;
  img.alt = title || meta.kind || "capture preview";
  img.title = "Click to enlarge";
  img.addEventListener("click", () => openLightbox(img.src, title || meta.kind || "capture"));
  wrap.appendChild(img);

  const bits = [
    title,
    meta.kind,
    meta.shape && `shape ${meta.shape}`,
    meta.dtype,
    meta.min != null && meta.max != null &&
      `${Number(meta.min).toFixed(1)}–${Number(meta.max).toFixed(1)}`,
    meta.scale_bar && `scale bar ${meta.scale_bar}`,
  ].filter(Boolean);
  if (bits.length) {
    const cap = document.createElement("div");
    cap.className = "capture-cap";
    cap.textContent = bits.join(" · ");
    wrap.appendChild(cap);
  }

  logEl.appendChild(wrap);
  logEl.scrollTop = logEl.scrollHeight;
}

// One reusable dialog rather than one per capture — a transcript can accumulate many
// thumbnails, and each only ever needs the single full-size view.
let lightboxEl = null;
function openLightbox(src, caption) {
  if (!lightboxEl) {
    lightboxEl = document.createElement("dialog");
    lightboxEl.className = "chat-lightbox";
    const big = document.createElement("img");
    big.className = "chat-lightbox-img";
    const noteEl = document.createElement("div");
    noteEl.className = "chat-lightbox-cap";
    lightboxEl.append(big, noteEl);
    lightboxEl.addEventListener("click", () => lightboxEl.close());
    document.body.appendChild(lightboxEl);
  }
  lightboxEl.querySelector(".chat-lightbox-img").src = src;
  lightboxEl.querySelector(".chat-lightbox-cap").textContent = caption;
  lightboxEl.showModal();
}

// A placeholder pill for the gap before any real signal exists — right after a turn
// starts, or between one tool result and the model's next move. Any actual signal
// (a note, streamed text, a question) supersedes it — but rather than vanishing, it
// freezes in place as "Thought for Ns", so the transcript keeps a quiet record of
// how long each step took instead of the indicator just disappearing.
let thinkingEl = null;
let thinkingStart = 0;
function clearThinking() {
  if (!thinkingEl) return;
  const secs = ((performance.now() - thinkingStart) / 1000).toFixed(1);
  thinkingEl.textContent = `Thought for ${secs}s`;
  thinkingEl.classList.remove("chip-thinking");
  thinkingEl = null;
}
function showThinking() {
  if (thinkingEl) return;
  thinkingStart = performance.now();
  thinkingEl = chip("thinking…");
  thinkingEl.classList.add("chip-thinking");
}

// test_in_scratch reports its start and outcome as two separate notes (tools.js and
// the recursion-cap test both key off that exact pair), so the duration can't be
// merged into one chip. Instead, time the gap between the two and append it to the
// outcome note only — the start note's text and count stay untouched.
let scratchStart = null;
function timedNote(text) {
  if (/^testing in scratch/.test(text)) {
    scratchStart = performance.now();
    return text;
  }
  if (scratchStart != null && /^scratch test (passed|failed)/.test(text)) {
    const secs = ((performance.now() - scratchStart) / 1000).toFixed(1);
    scratchStart = null;
    return `${text} · ${secs}s`;
  }
  return text;
}

// ---- the UI object the agent drives ---------------------------------------------------
function makeUI() {
  return {
    note: (text, cell) => { clearThinking(); chip(timedNote(text), cell); },
    capture: ({ image, meta, title }) => { clearThinking(); renderCapture({ image, meta, title }); },
    error: (text) => { clearThinking(); add("err", text); },
    thinking: () => showThinking(),
    done: () => {
      clearThinking();
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
      inputEl.disabled = false;
      inputEl.focus();
    },

    // Streaming prose: one bubble that grows as tokens arrive. A fast local model can
    // emit far more SSE chunks per second than the eye can register, and each push
    // used to force a full re-render and re-scroll — coalescing to one paint per
    // animation frame keeps it visually smooth without dropping any text.
    stream: () => {
      let el = null, buf = "", raf = null;
      const paint = () => {
        raf = null;
        if (!el) el = add("bot", "");
        renderProse(el, buf);
      };
      return {
        push(t) {
          clearThinking();
          buf += t;
          if (raf == null) raf = requestAnimationFrame(paint);
        },

        // The model can start streaming a tool call's arguments the instant its last
        // sentence ends, with no gap for us to detect — so without care the "thinking…"
        // chip lands in the *same* paint as the tail of the sentence, and the two read
        // as one garbled event instead of "it finished talking, then it paused". Flush
        // whatever prose is still buffered right now, synchronously, so the sentence is
        // committed to the DOM in full — then push the chip to the next frame so the
        // browser actually paints the finished sentence on its own before the chip
        // appears next to it.
        toolStarting() {
          if (raf != null) { cancelAnimationFrame(raf); paint(); }
          requestAnimationFrame(showThinking);
        },

        close() {
          if (raf != null) { cancelAnimationFrame(raf); paint(); }
          if (el && !buf.trim()) el.remove();
        },
      };
    },

    // ask_user: render the question with optional buttons and block until answered.
    ask: (question, options) => new Promise((resolve) => {
      clearThinking();
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
      (p.keyOptional
        ? `Optional — leave blank if ${p.host} doesn't check auth. `
        : "") +
      `Stored in this browser only and sent straight from this page to ${p.host}. ` +
      `Anything else running on this origin can read it, so use a key you're willing ` +
      `to scope narrowly and rotate. `),
  );
  const a = document.createElement("a");
  a.href = p.keyUrl; a.target = "_blank"; a.rel = "noreferrer";
  a.textContent = "Get a key";
  $("keyhint").append(a);
  $("apikeylabel").textContent = p.keyOptional ? "API key (optional)" : "API key";
  $("baseurlfield").hidden = !p.defaultBaseUrl;
  if (p.defaultBaseUrl) {
    $("baseurl").value = getBaseUrl(id);
    $("baseurl").placeholder = p.defaultBaseUrl;
  }
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
  if (!typed.trim() && !PROVIDERS[editingProvider].keyOptional) {
    $("modelhint").textContent = "Enter a key first.";
    return;
  }
  setKey(typed, editingProvider);
  if (PROVIDERS[editingProvider].defaultBaseUrl) setBaseUrl($("baseurl").value, editingProvider);
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
  if (PROVIDERS[editingProvider].defaultBaseUrl) setBaseUrl($("baseurl").value, editingProvider);
  setProviderId(editingProvider);
  $("settings").close();
  updateGate();
  greetIfEmpty();
}

function updateGate() {
  const ready = provider().keyOptional || hasKey();
  inputEl.disabled = !ready;
  sendBtn.disabled = !ready;
  inputEl.placeholder = ready
    ? defaultPlaceholder
    : "Add an API key under ⚙ to start";
  $("chatsettings").title = ready
    ? `${provider().label} · ${getModel()}`
    : "Add an API key";
}

const CHAT_WIDTH_KEY = "ptero-chat-width";
const CHAT_WIDTH_MIN = 400;
function chatWidthMax() { return Math.max(CHAT_WIDTH_MIN, Math.min(800, window.innerWidth * 0.6)); }

function applyChatWidth(px) {
  const clamped = Math.min(Math.max(px, CHAT_WIDTH_MIN), chatWidthMax());
  document.documentElement.style.setProperty("--chat-width", `${clamped}px`);
  return clamped;
}

function initResize() {
  const handle = $("chatresize");
  const saved = parseFloat(localStorage.getItem(CHAT_WIDTH_KEY));
  if (saved) applyChatWidth(saved);

  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    startX = e.clientX;
    startW = paneEl.getBoundingClientRect().width;
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add("resizing-chat");
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    // The pane is on the right, so dragging the handle left (negative dx) grows it.
    applyChatWidth(startW - (e.clientX - startX));
  });
  const stopDrag = () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing-chat");
    localStorage.setItem(CHAT_WIDTH_KEY,
      parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--chat-width")));
  };
  handle.addEventListener("pointerup", stopDrag);
  handle.addEventListener("pointercancel", stopDrag);
}

// ---- mount ---------------------------------------------------------------------------
export function mountChat() {
  paneEl = $("chat");
  logEl = $("chatlog");
  inputEl = $("chatinput");
  sendBtn = $("chatsend");

  agent = new Agent(makeUI());

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
  initResize();

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
