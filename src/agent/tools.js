// The agent's tools: what it can actually do to the notebook and the kernels.
//
// harness_v1 specified three (test_in_scratch, push_to_ui, ask_user). Three more are
// here because that spec's own workflow implies them: without `inspect_user_kernel`
// the mock shapes handed to test_in_scratch are guesses, without `inspect_file` the
// agent has to load a whole file into the kernel just to learn its shape and dtype,
// and without `read_cell_result` the agent is blind to the single most likely next
// event — the person pressing ▶ on a pushed cell and getting an error.
import * as cellsMod from "../notebook/cells.js";
import * as kernel from "../kernel.js";
import { catalogue } from "../registry.js";
import { testInScratch } from "./scratch.js";

export const SCHEMAS = [
  {
    name: "inspect_user_kernel",
    description:
      "List the variables currently defined in the user's notebook kernel (names, " +
      "types, array shapes, dtypes, DataFrame columns) and the files available to it. " +
      "Call this before writing code that builds on what the user has already run, and " +
      "before choosing mock shapes for test_in_scratch — guessing an image's shape or " +
      "channel count is the most common cause of code that fails on the user's data.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "inspect_file",
    description:
      "Get metadata for one file in the workspace — byte size, and where the format " +
      "allows it, image shape/dtype (TIFF, PNG/JPEG/BMP/GIF, .npy/.npz) or column names " +
      "(CSV/TSV) — without loading it into a notebook variable. Use this after " +
      "inspect_user_kernel lists a file whose shape or dtype you need, e.g. to pick " +
      "mock shapes for test_in_scratch or to tell channel-first from channel-last data.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Filename as it appears in the workspace listing." },
      },
      required: ["name"],
    },
  },
  {
    name: "test_in_scratch",
    description:
      "Run code in a hidden scratch kernel to check it works, before the user ever " +
      "sees it. The workspace's real files are readable here by their real names — " +
      "prefer `io.imread(\"Composite.tif\")` etc. over a mock var whenever the file " +
      "already exists, it catches real shape/dtype mistakes that a synthetic array " +
      "can't. The one thing still fake is segmentation: it returns synthetic label maps " +
      "of the right shape and dtype, so this verifies your analysis code, not the " +
      "biology. Never adjust segmentation thresholds based on object counts from here; " +
      "they are not real. Returns stdout, the result, any error, and the resulting " +
      "variables.",
    input_schema: {
      type: "object",
      properties: {
        code: { type: "string", description: "Python to execute in the scratch kernel." },
        vars: {
          type: "array",
          description:
            "Stand-in inputs to create before running — only for data that doesn't " +
            "already exist as a file, e.g. an intermediate array the real code would " +
            "compute. A file already in the workspace needs no entry here; read it by " +
            "name instead. Use shapes taken from inspect_user_kernel where possible.",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              shape: { type: "array", items: { type: "integer" },
                       description: "e.g. [960,1280] or [960,1280,3]" },
              kind: { type: "string", enum: ["image", "labels"],
                      description: "'image' = intensity data, 'labels' = a label map." },
              dtype: { type: "string", description: "optional numpy dtype, e.g. uint16" },
            },
            required: ["name", "shape"],
          },
        },
      },
      required: ["code"],
    },
  },
  {
    name: "push_to_ui",
    description:
      "Put finished, tested code into the user's notebook as a cell. Does NOT run it — " +
      "the user decides when to run. Only push code that passed test_in_scratch. Push " +
      "one coherent step per cell, the way a person would organise a notebook.",
    input_schema: {
      type: "object",
      properties: {
        code: { type: "string" },
        index: {
          type: "integer",
          description: "Position to insert at (0 = top). Omit to append at the end.",
        },
        replace: {
          type: "boolean",
          description: "Replace the cell at `index` instead of inserting before it.",
        },
      },
      required: ["code"],
    },
  },
  {
    name: "read_cell_result",
    description:
      "Read what happened the last time a notebook cell was run — its output, or the " +
      "error it raised. Use this when the user says something went wrong.",
    input_schema: {
      type: "object",
      properties: { index: { type: "integer", description: "Cell position, 0-based." } },
      required: ["index"],
    },
  },
  {
    name: "ask_user",
    description:
      "Ask the user a question and stop until they answer. Use it for anything about " +
      "their biology or their intent that you cannot determine from the data: which " +
      "channel is which, expected cell size, what counts as a positive. Do not use it " +
      "for things you could find out with inspect_user_kernel.",
    input_schema: {
      type: "object",
      properties: {
        question: { type: "string" },
        options: {
          type: "array", items: { type: "string" },
          description: "Optional suggested answers, offered as buttons.",
        },
      },
      required: ["question"],
    },
  },
];

/**
 * Execute a tool call. `ui` supplies the two things that need the chat surface:
 * `ask(question, options)` resolving to the user's reply, and `note(text)` for the
 * activity chips.
 *
 * Returns a string (what the model sees). Throwing is fine — the loop reports the
 * message back as a tool error rather than aborting.
 */
export async function runTool(name, input, ui) {
  switch (name) {
    case "inspect_user_kernel": {
      const { vars, files } = kernel.describeGlobals();
      ui.note("looked at your kernel");
      const varLines = vars.length
        ? vars.map(v => {
            const bits = [v.type, v.shape && `shape=${v.shape}`, v.dtype && `dtype=${v.dtype}`,
                          v.columns && `columns=${v.columns}`, v.len != null && `len=${v.len}`];
            return `  ${v.name}: ${bits.filter(Boolean).join(" ")}`;
          }).join("\n")
        : "  (nothing defined yet — the user hasn't run any cells)";
      const cellList = cellsMod.cells
        .map((c, i) => `  [${i}] ${c.author}, ${c.wrapEl.dataset.state}: ` +
                       `${c.view.state.doc.toString().split("\n")[0].slice(0, 70)}`)
        .join("\n") || "  (no cells)";
      return `Variables:\n${varLines}\n\nFiles available:\n  ` +
             `${files.length ? files.join(", ") : "(none — no workspace folder chosen)"}` +
             `\n\nNotebook cells:\n${cellList}` +
             `\n\nAvailable models:\n` +
             catalogue().map(m => `  ${m.id} — ${m.good_for}`).join("\n");
    }

    case "inspect_file": {
      if (!input.name?.trim()) throw new Error("inspect_file needs a name");
      ui.note(`inspecting ${input.name}`);
      const info = await kernel.describeFile(input.name);
      if (info.error && !info.bytes) return `${input.name}: ${info.error}`;
      const bits = [
        info.ext && `type=${info.ext}`,
        info.bytes != null && `size=${info.bytes} bytes`,
        info.shape && `shape=${info.shape}`,
        info.dtype && `dtype=${info.dtype}`,
        info.axes && `axes=${info.axes}`,
        info.pages != null && `pages=${info.pages}`,
        info.format && `format=${info.format}`,
        info.columns && `columns=[${info.columns.join(", ")}]`,
        info.rows != null && `rows=${info.rows}`,
        info.members && `members=${JSON.stringify(info.members)}`,
        info.kind && info.kind,
      ].filter(Boolean).join(" ");
      return info.error ? `${input.name}: ${bits} (${info.error})` : `${input.name}: ${bits}`;
    }

    case "test_in_scratch": {
      if (!input.code?.trim()) throw new Error("test_in_scratch needs code");
      ui.note("testing in scratch…");
      const r = await testInScratch({ code: input.code, vars: input.vars });
      const parts = [];
      if (r.stdout) parts.push(`stdout:\n${r.stdout}`);
      if (r.ok) {
        if (r.result) parts.push(`result: ${r.result}`);
        parts.push("Ran without error.");
      } else {
        parts.push(`ERROR:\n${r.error}`);
      }
      if (r.vars?.length) {
        parts.push("Variables now defined:\n" + r.vars.map(v => {
          const bits = [v.type, v.shape && `shape=${v.shape}`, v.dtype && `dtype=${v.dtype}`,
                        v.columns && `columns=[${v.columns.join(", ")}]`];
          return `  ${v.name}: ${bits.filter(Boolean).join(" ")}`;
        }).join("\n"));
      }
      ui.note(r.ok ? "scratch test passed" : "scratch test failed");
      return parts.join("\n\n");
    }

    case "push_to_ui": {
      if (!input.code?.trim()) throw new Error("push_to_ui needs code");
      let cell, at;
      if (input.replace && Number.isInteger(input.index) && cellsMod.cells[input.index]) {
        cell = cellsMod.cells[input.index];
        cellsMod.setCellSource(cell, input.code);
        at = input.index;
      } else {
        at = Number.isInteger(input.index) ? input.index : cellsMod.cells.length;
        cell = cellsMod.insertCell(at, input.code, "agent");
      }
      cellsMod.highlight(cell);
      ui.note(`${input.replace ? "updated" : "added"} cell ${at + 1}`, cell);
      return `Done — the code is in cell ${at} of the user's notebook (they have not run ` +
             `it yet). Tell them briefly what it does and let them press ▶.`;
    }

    case "read_cell_result": {
      const cell = cellsMod.cells[input.index];
      if (!cell) throw new Error(`there is no cell ${input.index} (the notebook has ` +
                                 `${cellsMod.cells.length})`);
      ui.note(`read cell ${input.index + 1}`);
      const src = cell.view.state.doc.toString();
      if (!cell.lastResult) return `Cell ${input.index} has not been run yet. Its code is:\n${src}`;
      return cell.lastResult.ok
        ? `Cell ${input.index} ran successfully.\nOutput:\n${cell.lastResult.text || "(no output)"}`
        : `Cell ${input.index} FAILED.\nCode:\n${src}\n\nError:\n${cell.lastResult.error}`;
    }

    case "ask_user": {
      if (!input.question?.trim()) throw new Error("ask_user needs a question");
      const answer = await ui.ask(input.question, input.options);
      return `The user replied: ${answer}`;
    }

    default:
      throw new Error(`unknown tool ${name}`);
  }
}
