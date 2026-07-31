// The agent loop. One agent, one conversation, no orchestration.
//
// Two constraints from harness_v1 are enforced here rather than merely hoped for:
//
//   Recursion cap — at most MAX_REPAIRS consecutive failed scratch tests. After that
//   the agent must stop and explain itself in plain language instead of grinding.
//
//   Context protection — failed drafts and stack traces stay in the model's working
//   context (it needs them to fix the bug) but never reach the chat transcript the
//   user reads. The user sees "testing…" and then either working code or a plain
//   explanation. Debugging noise is the agent's problem, not theirs.
import { send, LLMError } from "./llm.js";
import { SCHEMAS, runTool } from "./tools.js";
import { systemPrompt } from "./prompt.js";

const MAX_REPAIRS = 3;
// A backstop against a model that ping-pongs between tools without converging. The
// repair cap covers the common failure; this covers the pathological one.
const MAX_STEPS = 24;

export class Agent {
  constructor(ui) {
    this.ui = ui;              // { note, ask, stream, thinking, done, error }
    this.messages = [];        // full history, including tool results
    this.busy = false;
    this.controller = null;
  }

  stop() { this.controller?.abort(); }

  /** Handle one user turn, running tools until the model has nothing left to do. */
  async turn(userText) {
    if (this.busy) return;
    this.busy = true;
    this.controller = new AbortController();
    this.messages.push({ role: "user", content: userText });

    let failedTests = 0;
    let steps = 0;

    try {
      while (steps++ < MAX_STEPS) {
        this.ui.thinking();
        const stream = this.ui.stream();
        let reply;
        try {
          reply = await send({
            system: systemPrompt(),
            messages: this.messages,
            tools: SCHEMAS,
            onText: (t) => stream.push(t),
            // A tool call's arguments (e.g. a scratch-test draft) stream invisibly after
            // any preceding prose — nothing else marks that gap, so without this the chat
            // looks frozen the moment the model stops talking and starts drafting.
            onToolStart: () => this.ui.thinking(),
            signal: this.controller.signal,
          });
        } finally {
          stream.close();
        }

        this.messages.push({ role: "assistant", content: reply.content });

        const calls = reply.content.filter(b => b.type === "tool_use");
        if (!calls.length) break;   // plain prose: the turn is over

        const results = [];
        for (const call of calls) {
          if (call._malformed) {
            results.push({
              type: "tool_result", tool_use_id: call.id, is_error: true,
              content: "Your tool call was cut off mid-JSON. Send it again, more compactly.",
            });
            continue;
          }
          try {
            const out = await runTool(call.name, call.input, this.ui);
            if (call.name === "test_in_scratch") {
              // Count consecutive failures. A pass resets the budget, so a long
              // legitimate session isn't punished for one early stumble.
              failedTests = /^ERROR:|\nERROR:/m.test(out) ? failedTests + 1 : 0;
            }
            results.push({ type: "tool_result", tool_use_id: call.id, content: out });
          } catch (e) {
            if (e.name === "AbortError") throw e;
            results.push({
              type: "tool_result", tool_use_id: call.id, is_error: true,
              content: String(e.message || e),
            });
          }
        }
        this.messages.push({ role: "user", content: results });

        if (failedTests >= MAX_REPAIRS) {
          // Ask for the explanation in the same conversation — the model has the
          // failures in context and can say what actually blocked it — but forbid it
          // from continuing to try.
          this.messages.push({
            role: "user",
            content:
              `[system] You have now failed ${MAX_REPAIRS} scratch tests in a row. Stop ` +
              `trying to fix it. Reply with prose only — no tool calls — explaining to ` +
              `the user in plain language what is blocking you and what you would need ` +
              `from them to get past it. Do not include stack traces or Python error ` +
              `text; describe the problem in their terms.`,
          });
          this.ui.thinking();
          const stream = this.ui.stream();
          try {
            const final = await send({
              system: systemPrompt(),
              messages: this.messages,
              tools: [],                    // no tools: it cannot keep grinding
              onText: (t) => stream.push(t),
              signal: this.controller.signal,
            });
            this.messages.push({ role: "assistant", content: final.content });
          } finally {
            stream.close();
          }
          this.ui.note("gave up after 3 attempts");
          break;
        }
      }

      if (steps >= MAX_STEPS) {
        this.ui.error("Stopped after too many steps without finishing. Try narrowing " +
                      "the request into a smaller piece.");
      }
    } catch (e) {
      if (e.name === "AbortError") this.ui.note("stopped");
      else if (e instanceof LLMError) this.ui.error(e.message);
      else { console.error(e); this.ui.error(`Something went wrong: ${e.message}`); }
    } finally {
      this.busy = false;
      this.controller = null;
      this.ui.done();
    }
  }

  /** Drop the conversation but keep the notebook and kernels as they are. */
  reset() { this.messages = []; }
}
