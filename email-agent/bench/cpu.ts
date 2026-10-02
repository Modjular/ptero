// CPU per run of the harness path the Durable Object takes, measured under Node: workerd's clock does not advance
// during compute, so it cannot time itself. node:sqlite stands in for DO SQLite (both are synchronous SQLite behind
// durable's async facade). The faux model streams at a realistic rate so partial-answer commits happen as in production.
//   node --experimental-strip-types bench/cpu.ts
import { BACKGROUND_CONTEXT as context } from "@earendil-works/chord/context";
import { Type } from "@earendil-works/pi-ai";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxText, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { createRegistry, defineExtension, defineTool, Harness, section } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

const cpuMs = (start: NodeJS.CpuUsage) => {
	const used = process.cpuUsage(start);
	return (used.user + used.system) / 1000;
};
const answer = "Thanks! I see your files. ".repeat(60); // ~360 tokens, a long-ish email

const faux = fauxProvider({ tokensPerSecond: 150, tokenSize: { min: 3, max: 6 } });
const models = createModels();
models.setProvider(faux.provider);
const registry = createRegistry();
registry.install(
	defineExtension({
		name: "mail",
		sections: [section("preamble", () => "You are ptero's email assistant. ".repeat(40), { tag: false })],
		tools: [
			defineTool({
				name: "upload_link",
				description: "Get the upload link",
				parameters: Type.Object({}),
				execute: async () => ({ content: [{ type: "text", text: "https://ptero.example/u/x/y" }] }),
			}),
		],
	}),
);

let start = process.cpuUsage();
const harness = await Harness.open(await openNodeSqliteStorage(":memory:"), { models, registry }, context);
const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
console.log(`cold open + root: ${cpuMs(start).toFixed(1)} ms CPU`);

const rows: string[] = [];
for (let run = 1; run <= 30; run++) {
	faux.setResponses([
		fauxAssistantMessage([fauxToolCall("upload_link", {}, { id: `t${run}` })], { stopReason: "toolUse" }),
		fauxAssistantMessage([fauxText(answer)]),
	]);
	start = process.cpuUsage();
	const wall = performance.now();
	const settled = await (await root.submit({ type: "input", content: `email ${run}: ${"lorem ipsum ".repeat(40)}`, requestId: `m${run}` }, context)).wait(context);
	if (settled.status !== "done") throw new Error(`run ${run}: ${settled.status}`);
	if ([1, 2, 5, 10, 20, 30].includes(run)) {
		rows.push(`run ${String(run).padStart(2)}: ${cpuMs(start).toFixed(1).padStart(6)} ms CPU over ${((performance.now() - wall) / 1000).toFixed(1)} s wall`);
	}
}
console.log(rows.join("\n"));
await harness.close(context);
