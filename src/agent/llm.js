// Model providers, called directly from the page.
//
// Three are supported: Anthropic, Google Gemini, and an OpenAI-compatible adapter that
// defaults to DeepInfra but can be pointed at any OpenAI-compatible Chat Completions
// host (including a local server) via the Base URL field in ⚙. All three allow
// browser-origin calls with a user-supplied key — Anthropic needs an explicit opt-in
// header, Gemini and DeepInfra serve CORS by default; a custom endpoint must serve CORS
// itself, since there is no proxy either way.
// That keeps the "serve a folder and open a page" story intact, at the cost of the key
// being readable by anything on this origin. Two consequences are handled rather than
// merely noted:
//   * keys are read from localStorage at call time and never parked on `window`;
//   * pandas HTML output is sanitised before it reaches the DOM (notebook/cells.js),
//     because a DataFrame built from an untrusted file is otherwise a path to them.
//
// The conversation format used everywhere above this module is Anthropic's — content
// blocks, `tool_use`, `tool_result`. It is the most expressive of the three, so Gemini
// and OpenAI/DeepInfra each translate on the way out and normalise their responses back
// into the same shape. agent.js never learns which provider is in use.

export class LLMError extends Error {}

const PROVIDER_STORE = "ptero-provider";
const keyStore = (id) => `ptero-key-${id}`;
const modelStore = (id) => `ptero-model-${id}`;
const baseUrlStore = (id) => `ptero-baseurl-${id}`;

// Storage went per-provider when Gemini was added. Carry over anything saved under the
// single-provider names so an existing key doesn't silently vanish on upgrade.
(function migrateLegacyKeys() {
  const moves = [["ptero-anthropic-key", keyStore("anthropic")],
                 ["ptero-model", modelStore("anthropic")]];
  for (const [from, to] of moves) {
    const v = localStorage.getItem(from);
    if (v && !localStorage.getItem(to)) localStorage.setItem(to, v);
    if (v) localStorage.removeItem(from);
  }
})();

// ---- settings ----------------------------------------------------------------------
// Keys and model choices are stored per provider, so switching back and forth doesn't
// make you re-paste anything.
export function getProviderId() {
  const id = localStorage.getItem(PROVIDER_STORE);
  return PROVIDERS[id] ? id : "anthropic";
}
export function setProviderId(id) {
  if (PROVIDERS[id]) localStorage.setItem(PROVIDER_STORE, id);
}
export function provider() { return PROVIDERS[getProviderId()]; }

export function getKey(id = getProviderId()) { return localStorage.getItem(keyStore(id)) || ""; }
export function setKey(k, id = getProviderId()) {
  k?.trim() ? localStorage.setItem(keyStore(id), k.trim()) : localStorage.removeItem(keyStore(id));
}
export function hasKey(id = getProviderId()) { return !!getKey(id); }

export function getModel(id = getProviderId()) {
  return localStorage.getItem(modelStore(id)) || PROVIDERS[id].defaultModel;
}
export function setModel(m, id = getProviderId()) {
  m?.trim() ? localStorage.setItem(modelStore(id), m.trim()) : localStorage.removeItem(modelStore(id));
}

// Thinking effort is one conceptual control, so it's stored globally rather than
// per-provider — each provider's send() below maps this level onto its own knob
// (Anthropic's effort enum, Gemini's thinking budget, OpenAI's reasoning_effort). The
// levels are deliberately abstract: a raw token budget can't be shared across providers
// (Anthropic rejects budget_tokens on current models outright), an ordered level can.
const EFFORT_STORE = "ptero-effort";
export const EFFORT_LEVELS = ["off", "low", "medium", "high"];
export function getEffort() {
  const v = localStorage.getItem(EFFORT_STORE);
  return EFFORT_LEVELS.includes(v) ? v : "medium";
}
export function setEffort(level) {
  if (EFFORT_LEVELS.includes(level)) localStorage.setItem(EFFORT_STORE, level);
}

// Only providers that opt in (currently just `openai`, via `defaultBaseUrl`) use this —
// Anthropic and Gemini have fixed hosts.
export function getBaseUrl(id = getProviderId()) {
  return localStorage.getItem(baseUrlStore(id)) || PROVIDERS[id].defaultBaseUrl;
}
export function setBaseUrl(u, id = getProviderId()) {
  u?.trim() ? localStorage.setItem(baseUrlStore(id), u.trim()) : localStorage.removeItem(baseUrlStore(id));
}

function requireKey() {
  const p = provider();
  if (!hasKey()) throw new LLMError(`No ${p.label} API key set — open ⚙ and paste one.`);
  return getKey();
}

// Both APIs report failures differently enough to be worth normalising once.
async function checkResponse(res, label) {
  if (res.ok) return;
  const body = await res.text().catch(() => "");
  let detail = body.slice(0, 300);
  try { detail = JSON.parse(body).error?.message ?? detail; } catch { /* keep raw */ }
  // Gemini answers an invalid key with 400 INVALID_ARGUMENT rather than 401, so match
  // on the message too — a bad key is the likeliest first-run failure and deserves to
  // say so plainly instead of surfacing as a generic 400.
  if (res.status === 401 || res.status === 403 || /API key not valid|API_KEY_INVALID/i.test(detail)) {
    throw new LLMError(`${label} rejected the API key. Check it under ⚙.`);
  }
  if (res.status === 429) throw new LLMError("Rate limited (429). Wait a moment and retry.");
  throw new LLMError(`${label} error ${res.status}: ${detail}`);
}

// Reads an SSE body and hands each parsed `data:` payload to `onEvent`.
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      let ev;
      try { ev = JSON.parse(payload); } catch { continue; }
      onEvent(ev);
    }
  }
}

// ---- Anthropic ------------------------------------------------------------------------
const anthropic = {
  id: "anthropic",
  label: "Anthropic",
  defaultModel: "claude-opus-5",
  keyPlaceholder: "sk-ant-…",
  keyUrl: "https://console.anthropic.com/settings/keys",
  host: "api.anthropic.com",

  headers() {
    return {
      "content-type": "application/json",
      "x-api-key": requireKey(),
      "anthropic-version": "2023-06-01",
      // Without this the API refuses browser-origin requests outright.
      "anthropic-dangerous-direct-browser-access": "true",
    };
  },

  async listModels() {
    const res = await fetch("https://api.anthropic.com/v1/models?limit=100",
                            { headers: this.headers() });
    await checkResponse(res, "Anthropic");
    const { data } = await res.json();
    return data.map(m => ({ id: m.id, label: m.display_name || m.id }));
  },

  async send({ system, messages, tools, onText, onToolStart, signal, maxTokens, effort }) {
    // Extended thinking is expressed through the modern adaptive+effort surface, never
    // `budget_tokens` (removed on current models — sending it is a 400). "off" disables
    // thinking outright; any other level runs adaptive with the matching effort bucket.
    // Thinking output is deliberately not requested for display (default is empty
    // thinking text): the reasoning must never reach the transcript, only steer depth.
    const thinking = effort === "off"
      ? { type: "disabled" }
      : { type: "adaptive" };
    const body = {
      model: getModel(), max_tokens: maxTokens, system, messages, tools, thinking,
      stream: true,
    };
    if (effort !== "off") body.output_config = { effort };
    let res;
    try {
      res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal,
      });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new LLMError(`Couldn't reach the Anthropic API: ${e.message}`);
    }
    await checkResponse(res, "Anthropic");

    // Blocks arrive as start/delta/stop events; text and tool-input JSON both stream
    // in fragments, so each block is accumulated and parsed only once complete.
    const blocks = [];
    let stopReason = null;
    await readSSE(res, (ev) => {
      if (ev.type === "content_block_start") {
        const b = ev.content_block;
        // Thinking blocks (and their redacted form) must be captured and carried back
        // in history verbatim — the API rejects a tool-use turn on replay if the
        // thinking that produced it was stripped. They accumulate their own deltas
        // (`thinking_delta` / `signature_delta`) but never stream to onText, so the
        // reasoning is preserved for the model yet stays out of the transcript.
        if (b.type === "thinking") blocks[ev.index] = { type: "thinking", thinking: b.thinking || "", signature: b.signature || "" };
        else if (b.type === "redacted_thinking") blocks[ev.index] = { type: "redacted_thinking", data: b.data };
        else blocks[ev.index] = b.type === "tool_use"
          ? { type: "tool_use", id: b.id, name: b.name, _json: "" }
          : { type: "text", text: "" };
        // The model's tool-call arguments (e.g. a whole scratch-test draft) can take a
        // few seconds to stream as `input_json_delta` chunks, and none of that text is
        // shown anywhere — onText only fires for text blocks. Without this, the UI goes
        // silent the instant the preceding prose block ends, which reads as a freeze.
        if (b.type === "tool_use") onToolStart?.();
      } else if (ev.type === "content_block_delta") {
        const b = blocks[ev.index];
        if (!b) return;
        if (ev.delta.type === "text_delta") { b.text += ev.delta.text; onText?.(ev.delta.text); }
        else if (ev.delta.type === "input_json_delta") b._json += ev.delta.partial_json;
        else if (ev.delta.type === "thinking_delta") b.thinking += ev.delta.thinking;
        else if (ev.delta.type === "signature_delta") b.signature += ev.delta.signature;
      } else if (ev.type === "content_block_stop") {
        const b = blocks[ev.index];
        if (b?.type === "tool_use") {
          try {
            b.input = b._json ? JSON.parse(b._json) : {};
          } catch {
            // A truncated tool call is recoverable: the loop reports it back as a tool
            // error and the model retries, rather than the whole turn dying.
            b.input = {};
            b._malformed = true;
          }
          delete b._json;
        }
      } else if (ev.type === "message_delta") {
        stopReason = ev.delta?.stop_reason ?? stopReason;
      } else if (ev.type === "error") {
        throw new LLMError(ev.error?.message || "stream error");
      }
    });
    // A text block can open (content_block_start) and never receive a delta — some
    // models emit an empty text block ahead of a tool-only turn. That's fine as a
    // response, but agent.js pushes reply.content straight into history, and Anthropic
    // rejects an empty text block on the *next* request with "text content blocks must
    // be non-empty". Drop them here so they never reach history.
    const content = blocks.filter(b => b && !(b.type === "text" && b.text === ""));
    return { content, stopReason };
  },
};

// ---- Gemini ----------------------------------------------------------------------------
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";

// Gemini's function declarations take an OpenAPI-flavoured subset of JSON Schema and
// reject vocabulary it doesn't know (notably `additionalProperties` and `$schema`).
// Strip those recursively rather than hand-maintaining a second copy of every schema.
function geminiSchema(node) {
  if (Array.isArray(node)) return node.map(geminiSchema);
  if (!node || typeof node !== "object") return node;
  const out = {};
  for (const [k, v] of Object.entries(node)) {
    if (k === "additionalProperties" || k === "$schema") continue;
    if (k === "properties") {
      out.properties = Object.fromEntries(
        Object.entries(v).map(([pk, pv]) => [pk, geminiSchema(pv)]));
    } else if (k === "items") {
      out.items = geminiSchema(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

function geminiTools(tools) {
  if (!tools?.length) return undefined;
  return [{
    functionDeclarations: tools.map(t => {
      const decl = { name: t.name, description: t.description };
      // A parameters object with no properties is rejected; omit it entirely for
      // no-argument tools like inspect_user_kernel.
      if (Object.keys(t.input_schema?.properties ?? {}).length) {
        decl.parameters = geminiSchema(t.input_schema);
      }
      return decl;
    }),
  }];
}

// Anthropic-shaped history -> Gemini `contents`. Tool results in Gemini are keyed by
// function *name*, not by call id, so the id->name mapping is rebuilt while walking
// the conversation in order.
function geminiContents(messages) {
  const nameById = new Map();
  const contents = [];
  for (const msg of messages) {
    const parts = [];
    if (typeof msg.content === "string") {
      if (msg.content) parts.push({ text: msg.content });
    } else for (const block of msg.content) {
      if (block.type === "text") {
        if (block.text) parts.push({ text: block.text });
      } else if (block.type === "tool_use") {
        nameById.set(block.id, block.name);
        parts.push({ functionCall: { name: block.name, args: block.input ?? {} } });
      } else if (block.type === "tool_result") {
        parts.push({
          functionResponse: {
            name: nameById.get(block.tool_use_id) ?? "unknown",
            // The response must be an object, so a plain string result gets wrapped.
            response: block.is_error
              ? { error: String(block.content) }
              : { result: String(block.content) },
          },
        });
      }
    }
    if (!parts.length) continue;
    const role = msg.role === "assistant" ? "model" : "user";
    // Gemini expects multiturn requests to alternate user/model and can reject
    // consecutive same-role entries. Our history legitimately produces them — a
    // tool_result (user) immediately followed by the recursion cap's steering
    // message (also user) — so adjacent same-role turns are merged rather than sent
    // as-is. Anthropic tolerates the same history unchanged, which is why this is
    // the adapter's problem and not the agent loop's.
    const prev = contents[contents.length - 1];
    if (prev?.role === role) prev.parts.push(...parts);
    else contents.push({ role, parts });
  }
  return contents;
}

const gemini = {
  id: "gemini",
  label: "Google Gemini",
  defaultModel: "gemini-2.5-pro",
  keyPlaceholder: "AIza…",
  keyUrl: "https://aistudio.google.com/apikey",
  host: "generativelanguage.googleapis.com",

  async listModels() {
    const res = await fetch(`${GEMINI_BASE}/models?pageSize=200`,
                            { headers: { "x-goog-api-key": requireKey() } });
    await checkResponse(res, "Gemini");
    const { models = [] } = await res.json();
    return models
      // Embedding and legacy models can't hold a tool-using conversation.
      .filter(m => m.supportedGenerationMethods?.includes("generateContent"))
      .map(m => ({ id: m.name.replace(/^models\//, ""), label: m.displayName || m.name }));
  },

  async send({ system, messages, tools, onText, onToolStart, signal, maxTokens, effort }) {
    const model = getModel();
    // Gemini is the one provider whose thinking knob really is a token budget. Keep the
    // per-level budgets modest and well under maxOutputTokens (raised below when on) so
    // reasoning can't starve the visible answer. "off" omits thinkingConfig, leaving the
    // model at its own default — 2.5 Pro can't be forced to zero, so this is the closest
    // honest mapping. includeThoughts is left off: thoughts must never reach the UI.
    const budgets = { low: 512, medium: 2048, high: 8192 };
    const generationConfig = { maxOutputTokens: maxTokens };
    if (effort !== "off") {
      generationConfig.thinkingConfig = { thinkingBudget: budgets[effort] };
      generationConfig.maxOutputTokens = maxTokens + budgets[effort];
    }
    let res;
    try {
      res = await fetch(
        `${GEMINI_BASE}/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`, {
          method: "POST",
          headers: { "content-type": "application/json", "x-goog-api-key": requireKey() },
          body: JSON.stringify({
            systemInstruction: system ? { parts: [{ text: system }] } : undefined,
            contents: geminiContents(messages),
            tools: geminiTools(tools),
            generationConfig,
          }),
          signal,
        });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new LLMError(`Couldn't reach the Gemini API: ${e.message}`);
    }
    await checkResponse(res, "Gemini");

    // Gemini streams whole parts rather than deltas of a block: text arrives in
    // fragments across chunks, but a functionCall part always arrives complete, so
    // there is no partial-JSON accumulation to do here.
    const content = [];
    let stopReason = null;
    let callSeq = 0;
    await readSSE(res, (ev) => {
      if (ev.error) throw new LLMError(ev.error.message || "stream error");
      const cand = ev.candidates?.[0];
      if (!cand) return;
      stopReason = cand.finishReason ?? stopReason;
      for (const part of cand.content?.parts ?? []) {
        // A part flagged `thought: true` is Gemini's reasoning summary. Same rule as
        // Anthropic's thinking blocks: it must not reach the transcript. We don't carry
        // it in history either — Gemini has no replay requirement for it.
        if (part.thought) continue;
        if (part.text) {
          const last = content[content.length - 1];
          if (last?.type === "text") last.text += part.text;
          else content.push({ type: "text", text: part.text });
          onText?.(part.text);
        } else if (part.functionCall) {
          onToolStart?.();
          content.push({
            type: "tool_use",
            // Gemini has no call ids; synthesise stable ones so tool_result blocks in
            // our own history have something to point at.
            id: part.functionCall.id || `call_${Date.now()}_${callSeq++}`,
            name: part.functionCall.name,
            input: part.functionCall.args ?? {},
          });
        }
      }
    });
    return { content, stopReason };
  },
};

// ---- OpenAI-compatible ------------------------------------------------------------------
// Speaks the OpenAI Chat Completions wire format rather than a bespoke one, so it works
// against any OpenAI-compatible host — the Base URL field in ⚙ (backed by
// getBaseUrl/setBaseUrl above) selects it per provider id, e.g. a local server. Defaults
// to DeepInfra (docs.deepinfra.com), which hosts open-weight models behind this API and
// needs no local setup.
const OPENAI_DEFAULT_BASE = "https://api.deepinfra.com/v1/openai";

function openaiTools(tools) {
  if (!tools?.length) return undefined;
  return tools.map(t => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.input_schema },
  }));
}

// Anthropic-shaped history -> OpenAI `messages`. Tool results become their own `role:
// "tool"` messages keyed by `tool_call_id` rather than staying attached to the turn that
// produced them, so one Anthropic `user` message holding several tool_result blocks
// expands into several OpenAI messages here.
function openaiMessages(system, messages) {
  const out = [];
  if (system) out.push({ role: "system", content: system });
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      out.push({ role: msg.role, content: msg.content });
      continue;
    }
    if (msg.role === "user") {
      // Our history never mixes tool_result blocks with plain text in one user turn
      // (agent.js pushes them as separate messages), so this array is tool results only.
      for (const block of msg.content) {
        out.push({
          role: "tool",
          tool_call_id: block.tool_use_id,
          content: block.is_error ? `Error: ${block.content}` : String(block.content),
        });
      }
      continue;
    }
    let content = null;
    const tool_calls = [];
    for (const block of msg.content) {
      if (block.type === "text") content = (content ?? "") + block.text;
      else if (block.type === "tool_use") {
        tool_calls.push({
          id: block.id,
          type: "function",
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
        });
      }
    }
    const m = { role: "assistant", content };
    if (tool_calls.length) m.tool_calls = tool_calls;
    out.push(m);
  }
  return out;
}

// Some open-weight "reasoning" models served through generic OpenAI-compatible APIs
// inline their chain-of-thought as literal <think>...</think> text in `delta.content`,
// rather than a separate structured field the way DeepSeek's own API does (a
// `reasoning_content` field would just go unread by the code below — nothing to filter
// there). This is the same context-protection principle the rest of the agent already
// enforces for tracebacks: the model may think out loud, the user never sees it. Tags
// can straddle chunk boundaries, so this is a small streaming state machine rather than
// a regex run once over the full text.
export function thinkFilter() {
  const OPEN = "<think>", CLOSE = "</think>";
  let mode = "visible", pending = "";
  const findCI = (s, needle) => s.toLowerCase().indexOf(needle);
  // How many trailing characters of `s` could still grow into `needle` — i.e. the longest
  // suffix of `s` that is a prefix of `needle`. Usually zero: ordinary prose almost never
  // ends mid-tag. This must NOT be a fixed `needle.length - 1` window; holding a constant
  // six characters back withheld the last few characters of *every* message until the
  // stream ended, and when a turn ends in a tool call the stream doesn't end until the
  // whole tool-argument JSON has finished streaming — seconds of the user staring at a
  // sentence missing its last word, which then popped in as the tool started.
  const partialTail = (s, needle) => {
    const lower = s.toLowerCase();
    for (let n = Math.min(s.length, needle.length - 1); n > 0; n--)
      if (lower.endsWith(needle.slice(0, n))) return n;
    return 0;
  };
  function feed(chunk) {
    pending += chunk;
    let out = "";
    for (;;) {
      const needle = mode === "visible" ? OPEN : CLOSE;
      const i = findCI(pending, needle);
      if (i === -1) {
        const hold = partialTail(pending, needle);
        if (mode === "visible") out += pending.slice(0, pending.length - hold);
        pending = pending.slice(pending.length - hold);
        return out;
      }
      if (mode === "visible") out += pending.slice(0, i);
      pending = pending.slice(i + needle.length);
      mode = mode === "visible" ? "thinking" : "visible";
    }
  }
  // Release whatever is still held back, at either boundary where no more text can
  // arrive: the end of the stream, or the start of a tool call. Neither can complete a
  // tag, so in visible mode it's just ordinary trailing text; in thinking mode it's an
  // unterminated <think> block, which stays hidden rather than leaking half of it.
  // Clearing `pending` keeps this idempotent, so both callers can use it safely.
  const flush = () => {
    const out = mode === "visible" ? pending : "";
    pending = "";
    return out;
  };
  return { feed, flush };
}

const openai = {
  id: "openai",
  label: "OpenAI-compatible",
  defaultModel: "deepseek-ai/DeepSeek-V3",
  defaultBaseUrl: OPENAI_DEFAULT_BASE,
  keyPlaceholder: "di-…",
  keyUrl: "https://deepinfra.com/dash/api_keys",
  host: "api.deepinfra.com",
  // Local/self-hosted OpenAI-compatible servers typically don't check auth, so unlike
  // Anthropic and Gemini this provider works with no key at all.
  keyOptional: true,

  headers() {
    const key = getKey();
    const h = { "content-type": "application/json" };
    if (key) h.authorization = `Bearer ${key}`;
    return h;
  },

  async listModels() {
    const res = await fetch(`${getBaseUrl()}/models`, { headers: this.headers() });
    await checkResponse(res, "OpenAI-compatible endpoint");
    const { data } = await res.json();
    return data.map(m => ({ id: m.id, label: m.id }));
  },

  async send({ system, messages, tools, onText, onToolStart, signal, maxTokens, effort }) {
    // `reasoning_effort` is the OpenAI-compatible reasoning knob. Non-reasoning models
    // ignore it, and any inline <think> a reasoning model emits is already stripped by
    // thinkFilter() below, so there's nothing further to hide. "off" omits it entirely.
    const body = {
      model: getModel(),
      messages: openaiMessages(system, messages),
      tools: openaiTools(tools),
      max_tokens: maxTokens,
      stream: true,
    };
    if (effort !== "off") body.reasoning_effort = effort;
    let res;
    try {
      res = await fetch(`${getBaseUrl()}/chat/completions`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal,
      });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new LLMError(`Couldn't reach ${getBaseUrl()}: ${e.message}`);
    }
    await checkResponse(res, "OpenAI-compatible endpoint");

    // Chunks stream `delta.content` fragments and, separately, `delta.tool_calls`
    // fragments keyed by index — id/name arrive once, on a call's first chunk, and
    // `function.arguments` accumulates as a JSON string across the rest.
    let text = "";
    const calls = new Map();
    let stopReason = null;
    const think = thinkFilter();
    await readSSE(res, (ev) => {
      if (ev.error) throw new LLMError(ev.error.message || "stream error");
      const choice = ev.choices?.[0];
      if (!choice) return;
      stopReason = choice.finish_reason ?? stopReason;
      const delta = choice.delta ?? {};
      if (delta.content) {
        const visible = think.feed(delta.content);
        if (visible) { text += visible; onText?.(visible); }
      }
      for (const tc of delta.tool_calls ?? []) {
        let call = calls.get(tc.index);
        if (!call) {
          // Prose is over the moment tool-call arguments begin, so release anything the
          // think-filter is still holding *before* the chip goes up — otherwise the tail
          // lands after it, which both truncates the sentence for the whole
          // argument-streaming window and re-fires clearThinking() on arrival.
          const held = think.flush();
          if (held) { text += held; onText?.(held); }
          calls.set(tc.index, call = { id: tc.id, name: tc.function?.name, args: "" });
          // As with Anthropic, `function.arguments` streams as its own run of chunks
          // after this one with no visible signal — surface the call starting instead
          // of leaving the UI silent until it's fully assembled.
          onToolStart?.();
        }
        if (tc.function?.arguments) call.args += tc.function.arguments;
      }
    });
    const tail = think.flush();
    if (tail) { text += tail; onText?.(tail); }
    const content = [];
    if (text) content.push({ type: "text", text });
    for (const call of calls.values()) {
      // A truncated arguments string is recoverable the same way Anthropic's malformed
      // tool_use blocks are: report it back as a tool error and let the model retry.
      let input = {}, malformed = false;
      try { input = call.args ? JSON.parse(call.args) : {}; } catch { malformed = true; }
      const block = { type: "tool_use", id: call.id, name: call.name, input };
      if (malformed) block._malformed = true;
      content.push(block);
    }
    return { content, stopReason };
  },
};

export const PROVIDERS = { anthropic, gemini, openai };

// ---- dispatch ---------------------------------------------------------------------------
export async function send({ system, messages, tools, onText, onToolStart, signal, maxTokens = 4096, effort = getEffort() }) {
  return provider().send({ system, messages, tools, onText, onToolStart, signal, maxTokens, effort });
}

export async function listModels() {
  return provider().listModels();
}
