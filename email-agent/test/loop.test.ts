// The mocked transcript, end to end, on a faux model: email in → durable run in the customer's Durable Object →
// email out; then an upload notice, a queued analysis, and the runner's callback, each answered by mail.
import { abortAllDurableObjects, env, runDurableObjectAlarm, runInDurableObject } from "cloudflare:test";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxText, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { beforeEach, expect, it } from "vitest";
import { testing } from "../src/agent.ts";
import worker from "../src/worker.ts";

const faux = fauxProvider();
const models = createModels();
models.setProvider(faux.provider);
testing.models = models;

type Sent = { key: string; to: string; subject: string; inReplyTo: string | null; text: string };
let sent: Sent[] = [];
testing.onSend = (mail) => {
	sent.push(mail);
};

const testEnv = { ...env, RUNNER_TOKEN: "runner-secret" } as typeof env & { RUNNER_TOKEN: string };

beforeEach(() => {
	sent = [];
	faux.setResponses([]);
});

function inbound(from: string, messageId: string, subject: string, body: string): ForwardableEmailMessage {
	const raw = [
		`From: ${from}`,
		"To: help@ptero.example",
		`Subject: ${subject}`,
		`Message-ID: ${messageId}`,
		"Content-Type: text/plain; charset=utf-8",
		"",
		body,
	].join("\r\n");
	return {
		from,
		to: "help@ptero.example",
		headers: new Headers({ "Message-ID": messageId, Subject: subject }),
		raw: new Blob([raw]).stream(),
		rawSize: raw.length,
		setReject() {},
		forward: async () => ({ messageId: "" }),
		reply: async () => ({ messageId: "" }),
	} as unknown as ForwardableEmailMessage;
}

const stubFor = (address: string) => env.MAIL_AGENT.get(env.MAIL_AGENT.idFromName(address));
const tool = (name: string, args: Parameters<typeof fauxToolCall>[1]) =>
	fauxAssistantMessage([fauxToolCall(name, args, { id: `${name}-${Math.random()}` })], { stopReason: "toolUse" });
const say = (text: string) => fauxAssistantMessage([fauxText(text)]);
/** The text of the last user message the model was given. */
const lastUser = (ctx: { messages: readonly { role: string; content: unknown }[] }) => {
	const message = [...ctx.messages].reverse().find((m) => m.role === "user");
	const content = message?.content;
	return typeof content === "string" ? content : JSON.stringify(content);
};

it("runs the mocked transcript: link, upload notice, queued analysis, runner callback", async () => {
	const user = "pi@lab.example";
	const stub = stubFor(user);

	// 1. "I need help segmenting… at least 10 tifs" → the agent fetches an upload link and replies with it.
	faux.setResponses([tool("upload_link", {}), (ctx) => say(`Upload them here: ${toolText(ctx)}`)]);
	await worker.email(
		inbound(user, "<m1@lab.example>", "Segmenting cells", "Hi, I have ~10 tifs.\n\nOn Mon, someone wrote:\n> old"),
		testEnv,
	);
	await expect.poll(() => sent.length).toBe(1);
	expect(sent[0]).toMatchObject({ to: user, subject: "Re: Segmenting cells", inReplyTo: "<m1@lab.example>" });
	const link = sent[0]!.text.match(/https:\/\/ptero\.example\/u\/\S+/)?.[0];
	expect(link).toBeDefined();

	// 2. The upload page PUTs each file to R2, then reports them → a system notice starts a run → "I see 3 files".
	let notice = "";
	faux.setResponses([
		(ctx) => {
			notice = lastUser(ctx);
			return say("I see 001.tif, 002.tif, 005.tif. Look right?");
		},
	]);
	for (const name of ["001.tif", "002.tif", "005.tif"]) {
		const put = await worker.fetch(new Request(`${link}/files/${name}`, { method: "PUT", body: `pixels of ${name}` }), testEnv);
		expect(put.status).toBe(200);
	}
	const traversal = await worker.fetch(new Request(`${link}/files/..`, { method: "PUT", body: "x" }), testEnv);
	expect(traversal.status).toBe(404);
	const done = await worker.fetch(
		new Request(`${link}/done`, { method: "POST", body: JSON.stringify({ files: ["001.tif", "002.tif", "005.tif"] }) }),
		testEnv,
	);
	expect(done.status).toBe(200);
	await expect.poll(() => sent.length).toBe(2);
	expect(notice).toContain("[upload] 3 files arrived");
	expect(sent[1]!.text).toContain("005.tif");

	// 3. "Skip 005, it has a bubble" → the agent queues notebook cells and says it will report back.
	const cells = ["from skimage import io\nimg = io.imread('001.tif')", "img.shape"];
	faux.setResponses([
		tool("run_analysis", { files: ["001.tif", "002.tif"], cells, plan: "cyto3 on GFAP, stardist on DAPI" }),
		say("On it, I'll email you when the analysis finishes."),
	]);
	await worker.email(inbound(user, "<m2@lab.example>", "Re: Segmenting cells", "Skip 005.tif, it has a bubble."), testEnv);
	await expect.poll(() => sent.length).toBe(3);
	expect(sent[2]).toMatchObject({ inReplyTo: "<m2@lab.example>", subject: "Re: Segmenting cells" });
	expect((await stub.mailState())!.jobs).toEqual([
		expect.objectContaining({ files: ["001.tif", "002.tif"], status: "queued" }),
	]);

	// 4. The runner claims the job, reads its inputs, stores a figure, and reports → the preview email.
	expect((await runner("POST", "/runner/claim", undefined, "wrong-token")).status).toBe(401);
	const claim = (await (await runner("POST", "/runner/claim")).json()) as { id: string; lease: string; cells: string[] };
	expect(claim.cells).toEqual(cells);
	const job = `/runner/jobs/${encodeURIComponent(claim.id)}`;
	expect(await (await runner("GET", `${job}/files/001.tif`, undefined, undefined, claim.lease)).text()).toBe("pixels of 001.tif");
	// Only the job's own files, and only under its lease.
	expect((await runner("GET", `${job}/files/005.tif`, undefined, undefined, claim.lease)).status).toBe(404);
	expect((await runner("GET", `${job}/files/001.tif`, undefined, undefined, "stale")).status).toBe(404);
	expect((await runner("PUT", `${job}/artifacts/fig-2-1.png`, "png bytes", undefined, claim.lease)).status).toBe(200);
	expect(await (await env.FILES.get(`results/${claim.id}/fig-2-1.png`))!.text()).toBe("png bytes");

	let report = "";
	faux.setResponses([
		(ctx) => {
			report = lastUser(ctx);
			return say("All done: 412 cells across 2 images. Preview attached.");
		},
	]);
	const body = JSON.stringify({
		ok: true,
		cells: [
			{ state: "done", text: "", figures: [] },
			{ state: "done", text: "(512, 512, 3)", figures: ["fig-2-1.png"] },
		],
		artifacts: ["fig-2-1.png"],
	});
	expect((await runner("POST", `${job}/report`, body, undefined, claim.lease)).status).toBe(200);
	await expect.poll(() => sent.length).toBe(4);
	expect(report).toContain("finished");
	expect(report).toContain("(512, 512, 3)");
	expect(sent[3]).toMatchObject({ inReplyTo: "<m2@lab.example>", text: expect.stringContaining("412 cells") });
	expect((await stub.mailState())!.jobs[0]).toMatchObject({ status: "done", artifacts: ["fig-2-1.png"] });
	// The lease closed with the report: a second report is refused, and the job is not handed out again.
	expect((await runner("POST", `${job}/report`, body, undefined, claim.lease)).status).toBe(409);
	expect((await runner("POST", "/runner/claim")).status).toBe(204);

	// Every email went out once, under its own idempotency key, and nothing is left owed.
	expect(new Set(sent.map((m) => m.key)).size).toBe(4);
	const state = (await stub.mailState())!;
	expect(state.pending).toEqual([]);
	expect(state.sent).toHaveLength(4);
	expect(faux.getPendingResponseCount()).toBe(0);
	// The quoted history was stripped before it reached the model.
	expect(JSON.stringify(state)).not.toContain("> old");
});

it("treats a redelivered email as the same submission", async () => {
	const user = "dup@lab.example";
	const stub = stubFor(user);
	faux.setResponses([say("Hello!")]);
	await worker.email(inbound(user, "<d1@lab.example>", "Hi", "Hello"), testEnv);
	await expect.poll(() => sent.length).toBe(1);
	await worker.email(inbound(user, "<d1@lab.example>", "Hi", "Hello"), testEnv);
	// Give a wrongly admitted second run every chance to reach the mailer.
	await new Promise((resolve) => setTimeout(resolve, 200));
	expect(sent).toHaveLength(1);
	expect((await stub.mailState())!.pending).toEqual([]);
	expect(faux.state.callCount).toBeGreaterThan(0);
	expect(faux.getPendingResponseCount()).toBe(0);
});

it("finishes and mails exactly once after the object dies mid-model-call", async () => {
	const user = "crash@lab.example";
	let hung = false;
	faux.setResponses([
		() => {
			hung = true;
			return new Promise(() => {}); // the model call that never comes back: the process dies under it
		},
		say("Recovered answer."),
	]);
	await worker.email(inbound(user, "<c1@lab.example>", "Crash", "Please answer"), testEnv);
	// The alarm set by receive() fires on its own and reaches the model call that hangs.
	await expect.poll(() => hung).toBe(true);
	await abortAllDurableObjects();
	expect(sent).toHaveLength(0);
	const stub = stubFor(user); // stubs to an aborted object stay broken

	// The watchdog alarm set at the top of alarm() survived the abort; running it resumes the Harness.
	const scheduled = await runInDurableObject(stub, (_, state) => state.storage.getAlarm());
	expect(scheduled).not.toBeNull();
	expect(await runDurableObjectAlarm(stub)).toBe(true);
	expect(sent).toEqual([expect.objectContaining({ text: "Recovered answer.", inReplyTo: "<c1@lab.example>" })]);
	expect(await runInDurableObject(stub, (_, state) => state.storage.getAlarm())).toBeNull();
});

function runner(method: string, path: string, body?: string, token = "runner-secret", lease?: string) {
	const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
	if (lease !== undefined) headers["X-Lease"] = lease;
	return worker.fetch(new Request(`https://ptero.example${path}`, { method, headers, body }), testEnv);
}

/** The upload_link tool result the model was just given. */
function toolText(ctx: { messages: readonly { role: string; content: unknown }[] }): string {
	const result = [...ctx.messages].reverse().find((m) => m.role === "toolResult");
	const content = result?.content as { type: string; text: string }[] | undefined;
	return content?.find((block) => block.type === "text")?.text ?? "";
}
