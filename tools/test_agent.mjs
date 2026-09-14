// Agent-harness tests. Drives the real chat UI, agent loop, tool layer and streaming
// parsers against a stubbed provider endpoint, so no API key is needed and no tokens
// are spent.
//
//   python3 -m http.server 8765 &
//   node tools/test_agent.mjs                # all providers
//   node tools/test_agent.mjs --only gemini  # one of: anthropic, gemini, openai
//
// What it pins down, none of which is observable from unit tests:
//   * the SSE parsers reassemble text and tool-call JSON split across chunk boundaries
//   * ask_user blocks the loop and resumes on an answer
//   * push_to_ui lands a cell marked author=agent
//   * CONTEXT PROTECTION — no traceback text ever reaches the chat transcript
//   * RECURSION CAP — 3 failed scratch tests, then a re-prompt with tools stripped
//   * the Anthropic->Gemini and Anthropic->OpenAI translations (schemas, tool-result
//     naming/roles, argument stringification)
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const only = args[args.indexOf("--only") + 1];
const BASE = "http://localhost:8765";
const CHROME = process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

let failures = 0;
const check = (name, ok, extra = "") => {
  console.log(`  ${ok ? "✓" : "✗"} ${name}${extra ? "  " + extra : ""}`);
  if (!ok) failures++;
};

// ---- stub streams ------------------------------------------------------------------
// Anthropic: content blocks arrive as start/delta/stop, with text and tool-input JSON
// both fragmented across deltas.
const anthropicSSE = (blocks, stop = "tool_use") => {
  const out = [`data: ${JSON.stringify({ type: "message_start", message: { id: "m", content: [] } })}`, ""];
  blocks.forEach((b, i) => {
    if (b.text !== undefined) {
      out.push(`data: ${JSON.stringify({ type: "content_block_start", index: i, content_block: { type: "text", text: "" } })}`, "");
      for (const c of b.text.match(/.{1,7}/gs) || [])
        out.push(`data: ${JSON.stringify({ type: "content_block_delta", index: i, delta: { type: "text_delta", text: c } })}`, "");
    } else {
      out.push(`data: ${JSON.stringify({ type: "content_block_start", index: i, content_block: { type: "tool_use", id: b.id, name: b.name } })}`, "");
      for (const c of JSON.stringify(b.input).match(/.{1,11}/gs) || [])
        out.push(`data: ${JSON.stringify({ type: "content_block_delta", index: i, delta: { type: "input_json_delta", partial_json: c } })}`, "");
    }
    out.push(`data: ${JSON.stringify({ type: "content_block_stop", index: i })}`, "");
  });
  out.push(`data: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stop } })}`, "");
  return out.join("\n");
};

// Gemini: one whole GenerateContentResponse per chunk; functionCall parts arrive
// complete, text still spans chunks.
const geminiSSE = (blocks, finish = "STOP") =>
  blocks.map((b, i) => `data: ${JSON.stringify({
    candidates: [{
      content: { role: "model", parts: [b.text !== undefined ? { text: b.text } : { functionCall: { name: b.name, args: b.input } }] },
      ...(i === blocks.length - 1 ? { finishReason: finish } : {}),
    }],
  })}\n\n`).join("");

// OpenAI-compatible (DeepInfra): chunks stream `delta.content` fragments and, separately,
// `delta.tool_calls` fragments keyed by index — id/name arrive once, on a call's first
// chunk, and `function.arguments` accumulates as a JSON string across the rest.
const openaiSSE = (blocks, finish = "tool_calls") => {
  const out = [];
  let idx = 0;
  for (const b of blocks) {
    if (b.text !== undefined) {
      for (const c of b.text.match(/.{1,7}/gs) || [])
        out.push(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: c }, finish_reason: null }] })}`, "");
    } else {
      const i = idx++;
      out.push(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [
        { index: i, id: b.id, type: "function", function: { name: b.name, arguments: "" } },
      ] }, finish_reason: null }] })}`, "");
      for (const c of JSON.stringify(b.input).match(/.{1,11}/gs) || [])
        out.push(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [
          { index: i, function: { arguments: c } },
        ] }, finish_reason: null }] })}`, "");
    }
  }
  out.push(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: finish }] })}`, "");
  out.push("data: [DONE]", "");
  return out.join("\n");
};

// The one place the three encodings need a shared "the turn is over, no more tool calls"
// finish value, since it differs by provider and none of the three treat the others' spelling as valid.
const finishFor = (enc) => enc === anthropicSSE ? "end_turn" : enc === geminiSSE ? "STOP" : "stop";

// The same conversation, expressed once, rendered per provider.
const WORKING = (enc) => [
  enc([{ text: "Let me see what's loaded." }, { id: "t1", name: "inspect_user_kernel", input: {} }]),
  enc([{ id: "t2", name: "ask_user", input: { question: "Which channel is the nuclear stain?", options: ["Channel 1", "Channel 2"] } }]),
  // a draft that fails, to prove the failure is caught and hidden
  enc([{ id: "t3", name: "test_in_scratch", input: { code: "import numpy as np\nimg.shape[7]", vars: [{ name: "img", shape: [128, 128] }] } }]),
  enc([{ id: "t4", name: "test_in_scratch", input: { code: [
    "import numpy as np, pandas as pd",
    "from skimage import measure",
    "from stardist.models import StarDist2D",
    "labels, details = StarDist2D.from_pretrained('2D_versatile_fluo').predict_instances(img)",
    "df = pd.DataFrame(measure.regionprops_table(labels, intensity_image=img, properties=['label','area','intensity_mean']))",
    "f'{len(df)} objects'",
  ].join("\n"), vars: [{ name: "img", shape: [256, 256] }] } }]),
  enc([{ id: "t5", name: "push_to_ui", input: { code: "# count nuclei\nlabels.max()" } }]),
  // capture_view is the one tool result that must reach the transcript (as a thumbnail);
  // the expression is evaluated in the real kernel, so this tests the whole path.
  enc([{ id: "t6", name: "capture_view", input: { expression: "__import__('numpy').arange(4096).reshape(64, 64)", title: "Test preview" } }]),
  enc([{ text: "Added a cell that counts nuclei. Press ▶ to run it." }], finishFor(enc)),
];

const CAP = (enc) => [
  enc([{ id: "f1", name: "test_in_scratch", input: { code: "1/0" } }]),
  enc([{ id: "f2", name: "test_in_scratch", input: { code: "1/0" } }]),
  enc([{ id: "f3", name: "test_in_scratch", input: { code: "1/0" } }]),
  enc([{ text: "I can't get this working — something keeps coming out as zero. What should it be?" }],
      finishFor(enc)),
];

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "POST, GET, OPTIONS",
};

const PROVIDERS = {
  anthropic: {
    host: "api.anthropic.com",
    enc: anthropicSSE,
    storage: { "ptero-provider": "anthropic", "ptero-key-anthropic": "sk-ant-test" },
  },
  gemini: {
    host: "generativelanguage.googleapis.com",
    enc: geminiSSE,
    storage: { "ptero-provider": "gemini", "ptero-key-gemini": "AIza-test",
               "ptero-model-gemini": "gemini-2.5-flash" },
  },
  openai: {
    host: "api.deepinfra.com",
    enc: openaiSSE,
    storage: { "ptero-provider": "openai", "ptero-key-openai": "di-test",
               "ptero-model-openai": "deepseek-ai/DeepSeek-V3" },
  },
};

// ---- one provider's run ---------------------------------------------------------------
async function run(browser, name) {
  const p = PROVIDERS[name];
  console.log(`\n=== ${name} ===`);
  const page = await browser.newPage();
  page.on("pageerror", (e) => { console.log("  [pageerror]", e.message.slice(0, 200)); failures++; });

  let script = WORKING(p.enc), turn = 0;
  const sent = [];
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    if (!req.url().includes(p.host)) return req.continue();
    if (req.method() === "OPTIONS") return req.respond({ status: 204, headers: CORS });
    if (req.postData()) sent.push(JSON.parse(req.postData()));
    req.respond({ status: 200, contentType: "text/event-stream", headers: CORS,
                  body: script[Math.min(turn++, script.length - 1)] });
  });

  await page.goto(`${BASE}/notebook.html`, { waitUntil: "domcontentloaded" });
  await page.evaluate((store) => {
    localStorage.clear();
    for (const [k, v] of Object.entries(store)) localStorage.setItem(k, v);
  }, p.storage);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.getElementById("stat")?.textContent === "ready",
                             { timeout: 240000 });

  const cellsBefore = await page.evaluate(() => document.querySelectorAll(".cell").length);

  // --- working flow ---
  await page.type("#chatinput", "Count the nuclei in this image");
  await page.click("#chatsend");
  await page.waitForSelector(".ask-opts button", { timeout: 120000 });
  check("ask_user blocks the loop and renders options", true);
  await page.click(".ask-opts button");
  await page.waitForFunction(() => !document.getElementById("chatsend").disabled, { timeout: 240000 });

  const log = await page.evaluate(() =>
    [...document.querySelectorAll("#chatlog > *")].map(e => `${e.className}: ${e.textContent.trim()}`));
  const cells = await page.evaluate(() =>
    [...document.querySelectorAll(".cell")].map(el => el.dataset.author));

  check("streamed prose reassembled", log.some(l => l.includes("Let me see what's loaded.")));
  check("scratch failure reported as a chip", log.some(l => l.startsWith("chip") && /failed/.test(l)));
  check("scratch retry passed", log.some(l => l.startsWith("chip") && /passed/.test(l)));
  check("push_to_ui added an agent cell",
        cells.length === cellsBefore + 1 && cells.filter(a => a === "agent").length === 1);
  // The capture preview is the deliberate exception to "no tool results in the
  // transcript": the user should see the same image the agent looked at.
  const capture = await page.evaluate(() => {
    const img = document.querySelector(".capture-thumb");
    return img && { src: img.src.slice(0, 22), alt: img.alt, cap: !!document.querySelector(".capture-cap") };
  });
  check("capture_view renders a thumbnail in the transcript",
        !!capture && capture.src === "data:image/png;base64,", capture ? capture.src : "none");
  check("capture thumbnail is labelled", !!capture && capture.alt === "Test preview");
  // The one that matters most: debugging noise must never reach the user.
  check("context protection: no traceback in the transcript",
        !log.some(l => /Traceback|IndexError|File "|line \d+, in/.test(l)));

  // --- recursion cap ---
  script = CAP(p.enc); turn = 0;
  await page.click("#chatreset");
  await page.type("#chatinput", "Do the impossible thing");
  await page.click("#chatsend");
  await page.waitForFunction(() => !document.getElementById("chatsend").disabled, { timeout: 240000 });
  const capLog = await page.evaluate(() =>
    [...document.querySelectorAll("#chatlog > *")].map(e => `${e.className}: ${e.textContent.trim()}`));

  check("stops after exactly 3 scratch attempts",
        capLog.filter(l => /testing in scratch/.test(l)).length === 3);
  check("explains itself in plain language", capLog.some(l => /can't get this working/.test(l)));
  check("no traceback leaked from the failures",
        !capLog.some(l => /ZeroDivisionError|Traceback/.test(l)));
  const last = sent.at(-1);
  const toolsStripped = name === "anthropic" ? last.tools?.length === 0 : !last.tools;
  check("final re-prompt strips tools so it cannot keep grinding", toolsStripped);

  // --- provider-specific wire format ---
  const first = sent[0];
  if (name === "anthropic") {
    check("system prompt carries the model routing table", /stardist-fluo/.test(first.system));
    check("tools sent in Anthropic schema form", !!first.tools?.[0]?.input_schema);
  } else if (name === "gemini") {
    const decls = first.tools[0].functionDeclarations;
    check("system prompt carries the model routing table",
          /stardist-fluo/.test(first.systemInstruction.parts[0].text));
    check("tool schemas stripped of unsupported keywords",
          !/additionalProperties|\$schema/.test(JSON.stringify(first.tools)));
    check("no-arg tool omits an empty `parameters`",
          !("parameters" in decls.find(d => d.name === "inspect_user_kernel")));
    check("nested schemas survive translation",
          !!decls.find(d => d.name === "test_in_scratch").parameters.properties.vars.items.properties.shape);
    const responses = last.contents.flatMap(c => c.parts.filter(x => x.functionResponse));
    check("tool results resolved to function names (Gemini keys by name, not id)",
          responses.length > 0 && responses.every(r => r.functionResponse.name !== "unknown"));
    check("roles alternate user/model (Gemini rejects consecutive same-role turns)",
          last.contents.every((c, i) => c.role === (i % 2 === 0 ? "user" : "model")),
          last.contents.map(c => c.role).join(","));
  } else {
    check("system prompt carries the model routing table",
          first.messages[0].role === "system" && /stardist-fluo/.test(first.messages[0].content));
    check("tools sent in OpenAI function-call schema form",
          first.tools[0].type === "function" && !!first.tools[0].function.parameters);
    check("nested schemas survive translation",
          !!first.tools.find(t => t.function.name === "test_in_scratch")
            .function.parameters.properties.vars.items.properties.shape);
    const toolMsgs = last.messages.filter(m => m.role === "tool");
    check("tool results become role:\"tool\" messages keyed by tool_call_id",
          toolMsgs.length > 0 && toolMsgs.every(m => !!m.tool_call_id));
    const withCalls = sent.flatMap(r => r.messages).find(m => m.role === "assistant" && m.tool_calls);
    check("assistant tool_use blocks become tool_calls with stringified arguments",
          !!withCalls && typeof withCalls.tool_calls[0].function.arguments === "string");

    // The <think>-tag filter is on this path only, and it is the one place text can be
    // held back mid-sentence. It must hold back *only* a real partial tag: a fixed-width
    // window meant the last few characters of every reply sat invisible until the stream
    // ended — which, on a turn ending in a tool call, is after the whole tool-argument
    // JSON has streamed. Run it in the page, where llm.js's localStorage use is valid.
    const tf = await page.evaluate(async () => {
      const { thinkFilter } = await import("/src/agent/llm.js");
      const emitted = (chunks) => {
        const f = thinkFilter();
        return chunks.map(c => f.feed(c)).concat(f.flush());
      };
      return {
        // no holdback: the sentence is complete the instant its last chunk arrives
        immediate: emitted(["Alright, time to ", "start coding."]).slice(0, -1).join(""),
        // a tag split across chunks is still caught and hidden
        split: emitted(["say <thi", "nk>secret</thi", "nk>done"]).join(""),
        // a genuine partial tag is held, then released by flush()
        heldThenFlushed: emitted(["ok <thi"]),
      };
    });
    check("think-filter holds back nothing when no tag is in flight",
          tf.immediate === "Alright, time to start coding.", JSON.stringify(tf.immediate));
    check("think-filter still hides a <think> block split across chunks",
          tf.split === "say done", JSON.stringify(tf.split));
    check("think-filter releases a real partial tag on flush",
          tf.heldThenFlushed.join("") === "ok <thi" && tf.heldThenFlushed[0] === "ok ",
          JSON.stringify(tf.heldThenFlushed));
  }

  await page.close();
}

// ---- main --------------------------------------------------------------------------------
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"],
});
try {
  for (const name of Object.keys(PROVIDERS)) {
    if (only && only !== name) continue;
    await run(browser, name);
  }
} finally {
  await browser.close();
}
console.log(failures ? `\n${failures} FAILED` : "\nOK");
process.exit(failures ? 1 : 0);
