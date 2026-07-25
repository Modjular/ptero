// Model providers, called directly from the page.
//
// Two are supported: Anthropic and Google Gemini. Both allow browser-origin calls
// with a user-supplied key — Anthropic needs an explicit opt-in header, Gemini serves
// CORS by default. That keeps the "serve a folder and open a page" story intact, at
// the cost of the key being readable by anything on this origin. Two consequences are
// handled rather than merely noted:
//   * keys are read from localStorage at call time and never parked on `window`;
//   * pandas HTML output is sanitised before it reaches the DOM (notebook/cells.js),
//     because a DataFrame built from an untrusted file is otherwise a path to them.
//
// The conversation format used everywhere above this module is Anthropic's — content
// blocks, `tool_use`, `tool_result`. It is the more expressive of the two, so Gemini
// translates on the way out and normalises its responses back into the same shape.
// agent.js never learns which provider is in use.

export class LLMError extends Error {}

const PROVIDER_STORE = "ptero-provider";
const keyStore = (id) => `ptero-key-${id}`;
const modelStore = (id) => `ptero-model-${id}`;

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

  async send({ system, messages, tools, onText, signal, maxTokens }) {
    let res;
    try {
      res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          model: getModel(), max_tokens: maxTokens, system, messages, tools, stream: true,
        }),
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
        blocks[ev.index] = b.type === "tool_use"
          ? { type: "tool_use", id: b.id, name: b.name, _json: "" }
          : { type: "text", text: "" };
      } else if (ev.type === "content_block_delta") {
        const b = blocks[ev.index];
        if (!b) return;
        if (ev.delta.type === "text_delta") { b.text += ev.delta.text; onText?.(ev.delta.text); }
        else if (ev.delta.type === "input_json_delta") b._json += ev.delta.partial_json;
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
    return { content: blocks.filter(Boolean), stopReason };
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

  async send({ system, messages, tools, onText, signal, maxTokens }) {
    const model = getModel();
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
            generationConfig: { maxOutputTokens: maxTokens },
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
        if (part.text) {
          const last = content[content.length - 1];
          if (last?.type === "text") last.text += part.text;
          else content.push({ type: "text", text: part.text });
          onText?.(part.text);
        } else if (part.functionCall) {
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

export const PROVIDERS = { anthropic, gemini };

// ---- dispatch ---------------------------------------------------------------------------
export async function send({ system, messages, tools, onText, signal, maxTokens = 4096 }) {
  return provider().send({ system, messages, tools, onText, signal, maxTokens });
}

export async function listModels() {
  return provider().listModels();
}
