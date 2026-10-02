import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { Type } from "@earendil-works/pi-ai";
import { createModels, type Models } from "@earendil-works/pi-ai/models";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import { type FauxResponseStep, fauxAssistantMessage, fauxProvider } from "@earendil-works/pi-ai/providers/faux";
import {
	AssistantEntry,
	createRegistry,
	defineDoc,
	defineExtension,
	defineTool,
	Harness,
	section,
} from "@earendil-works/pi-durable";
import { DurableObject } from "cloudflare:workers";
import { openDurableObjectStorage } from "./do-sqlite.ts";
import { type JobReport, jobBoard } from "./jobs.ts";

const context = BACKGROUND_CONTEXT;
const WATCHDOG_MS = 60_000;

/** Declared in worker-configuration.d.ts, where the test pool's `env` picks it up too. */
export type Env = Cloudflare.Env;

export type InboundMail = { from: string; messageId: string; subject: string; text: string };

// A submission whose answer still owes the customer an email.
type Pending = { submissionId: string; requestId: string; inReplyTo: string | null };
type SentMail = { key: string; to: string; subject: string; inReplyTo: string | null; text: string };
type Job = { id: string; files: string[]; plan: string; status: "queued" | "done" | "failed"; artifacts: string[] };

/** Everything the mail side needs that is not the transcript. Committed like any other durable state. */
const MailState = defineDoc<{
	customer: string;
	subject: string;
	lastMessageId: string | null;
	uploadToken: string | null;
	pending: Pending[];
	sent: SentMail[];
	jobs: Job[];
}>({
	kind: "app.mail",
	version: 1,
	scope: "session",
	initial: () => ({
		customer: "",
		subject: "",
		lastMessageId: null,
		uploadToken: null,
		pending: [],
		sent: [],
		jobs: [],
	}),
});

/** Tests swap in a faux provider here; production builds Anthropic from the environment. */
export const testing: { models?: Models; onSend?: (mail: SentMail) => void } = {};

const PROMPT = `You are ptero's email assistant. You help biologists segment and analyse microscopy images.

Every run ends with your final message, which is sent verbatim as the email reply. Write it as a short,
friendly plain-text email body: no subject line, no signature block, no markdown headings.

Messages that start with [upload] or [runner] come from the system, not the customer. Tell the customer
what happened in your reply to them.

Data never travels by email: give the customer an upload link with the upload_link tool. Analyses run
asynchronously: queue one with run_analysis, tell the customer you will email when it finishes, and end
your turn. A [runner] message arrives when it does.

run_analysis runs notebook cells in ptero: Python under Pyodide in a browser with WebGPU. numpy, pandas,
scikit-image, scipy and matplotlib are available, and so are cellpose, stardist and instanseg under their
upstream APIs (CellposeModel(model_type='cyto3').eval(img, diameter=...) returns masks, flows, styles;
StarDist2D.from_pretrained('2D_versatile_fluo').predict_instances(img); always pass a cellpose diameter).
Uploaded files sit in the working directory under their own names. Figures left open by a cell and files a
cell writes (csv, tif, png) come back as artifacts. Cells share one namespace and run top to bottom, stopping
at the first error. If a job fails, fix the cells and queue it again; do not tell the customer about
tracebacks.`;

function mailExtension(env: Env, doName: string) {
	const uploadLink = defineTool({
		name: "upload_link",
		description: "Get the customer's private upload link for microscope images and CSVs.",
		parameters: Type.Object({}),
		replay: "safe",
		execute: async (_args, api, ctx) => {
			const token = await api.commit(async (tx) => {
				const state = await tx.doc(MailState);
				state.uploadToken ??= crypto.randomUUID();
				return state.uploadToken;
			}, ctx);
			return { content: [{ type: "text", text: `${env.PUBLIC_URL}/u/${doName}/${token}` }] };
		},
	});

	const runAnalysis = defineTool({
		name: "run_analysis",
		description:
			"Queue notebook cells to run on the runner against uploaded files. Returns at once; a [runner] message arrives when it finishes.",
		parameters: Type.Object({
			files: Type.Array(Type.String(), { description: "Uploaded file names the cells read" }),
			cells: Type.Array(Type.String(), { description: "Python source of each notebook cell, in order" }),
			plan: Type.String({ description: "One line on what the notebook does, for the record" }),
		}),
		// Keyed by the tool task, so a rerun after a crash finds the job it already queued.
		replay: "safe",
		execute: async (args, api, ctx) => {
			// Job IDs are global on the board, so qualify the task ID with the customer.
			const id = `${doName}:${api.taskId}`;
			await api.commit(async (tx) => {
				const state = await tx.doc(MailState);
				if (!state.jobs.some((job) => job.id === id)) {
					state.jobs.push({ id, files: args.files, plan: args.plan, status: "queued", artifacts: [] });
				}
			}, ctx);
			await jobBoard(env).enqueue({ id, customer: doName, files: args.files, cells: args.cells });
			return { content: [{ type: "text", text: `Queued job ${id}.` }] };
		},
	});

	return defineExtension({
		name: "mail",
		sections: [section("preamble", () => PROMPT, { tag: false })],
		tools: [uploadLink, runAnalysis],
	});
}

const clip = (text: string, max: number) => (text.length <= max ? text : `…${text.slice(-max)}`);

/** What the agent reads about a finished job: enough to judge it, never a wall of output. */
export function summarize(id: string, report: JobReport): string {
	const lines = [`[runner] job ${id} ${report.ok ? "finished" : "FAILED"}.`];
	report.cells.forEach((cell, i) => {
		const figures = cell.figures.length ? ` (figures: ${cell.figures.join(", ")})` : "";
		const text = cell.text.trim();
		lines.push(`Cell ${i + 1} [${cell.state}]${figures}${text ? `: ${clip(text, cell.state === "error" ? 1500 : 400)}` : ""}`);
	});
	if (report.error) lines.push(`Error: ${clip(report.error, 1500)}`);
	if (report.artifacts.length) lines.push(`Artifacts: ${report.artifacts.join(", ")}`);
	return lines.join("\n");
}

/**
 * MODEL=echo: a model that answers every message by quoting it back. For local runs without an API key
 * (`wrangler dev`, the runner end-to-end script), where what matters is what reached the agent.
 */
function echoProvider() {
	const faux = fauxProvider();
	const echo: FauxResponseStep = (ctx) => {
		faux.appendResponses([echo]);
		const last = [...ctx.messages].reverse().find((m) => m.role === "user");
		const content = last?.content;
		const text = typeof content === "string" ? content : JSON.stringify(content);
		return fauxAssistantMessage(`Echo: ${text}`);
	};
	faux.setResponses([echo]);
	return faux.provider;
}

function textOf(entry: { model?: readonly unknown[] } | undefined): string {
	const message = entry?.model?.[0] as { role?: string; content?: unknown } | undefined;
	if (message?.role !== "assistant" || !Array.isArray(message.content)) return "";
	return message.content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text")
		.map((block) => block.text)
		.join("\n")
		.trim();
}

/**
 * One customer, one Durable Object, one durable Harness over the object's own SQLite.
 *
 * Inbound mail and system notices become submissions keyed by a request ID, so a redelivered email or webhook
 * finds its existing submission. An alarm drives the work: it resumes the Harness, waits for each pending
 * submission, and mails the answer. A crash or eviction anywhere in that leaves the alarm unfinished, and
 * Cloudflare retries it; the retry resumes the Harness from its last commit.
 */
export class MailAgent extends DurableObject<Env> {
	#harness: Promise<Harness> | undefined;
	#delivering: Promise<void> | undefined;

	#open(): Promise<Harness> {
		this.#harness ??= (async () => {
			const registry = createRegistry();
			registry.install(mailExtension(this.env, this.#name()));
			let models: Models | undefined = testing.models;
			if (models === undefined) {
				const built = createModels();
				built.setProvider(this.env.MODEL === "echo" ? echoProvider() : anthropicProvider());
				models = built;
			}
			return Harness.open(await openDurableObjectStorage(this.ctx.storage), { models, registry }, context);
		})();
		return this.#harness;
	}

	#name(): string {
		const name = this.ctx.id.name;
		if (name === undefined) throw new Error("MailAgent must be addressed with idFromName()");
		return name;
	}

	async #root(harness: Harness) {
		const model =
			testing.models || this.env.MODEL === "echo"
				? { provider: "faux", modelId: "faux-1" }
				: { provider: "anthropic", modelId: this.env.MODEL ?? "claude-sonnet-5-5" };
		return harness.root(context, { agent: { model } });
	}

	/** Customer email in. */
	async receive(mail: InboundMail): Promise<void> {
		const harness = await this.#open();
		const root = await this.#root(harness);
		await root.commit(async (tx) => {
			const state = await tx.doc(MailState);
			state.customer = mail.from;
			if (state.subject === "") state.subject = mail.subject;
			state.lastMessageId = mail.messageId;
		}, context);
		await this.#submit(harness, `mail:${mail.messageId}`, mail.text, mail.messageId);
	}

	/** System notice in: an upload finished, a runner job finished. */
	async notify(requestId: string, text: string): Promise<void> {
		const harness = await this.#open();
		const state = await harness.snapshot(MailState, context);
		await this.#submit(harness, requestId, text, state?.lastMessageId ?? null);
	}

	async #submit(harness: Harness, requestId: string, content: string, inReplyTo: string | null) {
		const root = await this.#root(harness);
		const submission = await root.submit({ type: "input", content, requestId }, context);
		// Not atomic with the submit: a crash between them is healed by the sender's retry, which finds the same
		// submission by request ID and lands here again.
		await root.commit(async (tx) => {
			const state = await tx.doc(MailState);
			const known =
				state.pending.some((p) => p.submissionId === String(submission.id)) ||
				state.sent.some((s) => s.key === `reply:${submission.id}`);
			if (!known) state.pending.push({ submissionId: String(submission.id), requestId, inReplyTo });
		}, context);
		await this.ctx.storage.setAlarm(Date.now());
	}

	async alarm(): Promise<void> {
		// One delivery loop per object: an overlapping alarm (a retry, a watchdog) joins the running one instead of
		// sending the same reply twice.
		this.#delivering ??= this.#drain().finally(() => {
			this.#delivering = undefined;
		});
		return this.#delivering;
	}

	async #drain(): Promise<void> {
		// A watchdog first: if this run dies (eviction, deploy, crash), the next alarm picks the work up. Cloudflare
		// also retries a failed alarm, but only a few times with backoff; this keeps trying.
		await this.ctx.storage.setAlarm(Date.now() + WATCHDOG_MS);
		const harness = await this.#open();
		harness.resume();
		for (;;) {
			const state = await harness.snapshot(MailState, context);
			const next = state?.pending[0];
			if (next === undefined) break;
			await this.#deliver(harness, next);
		}
		await this.ctx.storage.deleteAlarm();
		// #submit records pending work before it sets the alarm, so work admitted after the empty check above is
		// either visible here or sets its alarm after this delete.
		if ((await harness.snapshot(MailState, context))?.pending.length) await this.ctx.storage.setAlarm(Date.now());
	}

	async #deliver(harness: Harness, pending: Pending): Promise<void> {
		const submission = await harness.submission(pending.submissionId as never, context);
		const settled = submission ? await submission.wait(context) : undefined;
		let text = "Sorry, I hit a problem working on that. I'll take another look and get back to you.";
		if (settled?.status === "done" && settled.type === "input") {
			const root = await this.#root(harness);
			const answer = await root.commit((tx) => tx.entry(AssistantEntry, settled.answer), context);
			text = textOf(answer) || text;
		}
		const state = (await harness.snapshot(MailState, context))!;
		const mail: SentMail = {
			key: `reply:${pending.submissionId}`,
			to: state.customer,
			subject: state.subject.startsWith("Re:") ? state.subject : `Re: ${state.subject}`,
			inReplyTo: pending.inReplyTo,
			text,
		};
		// Send before recording: a crash in between sends again on the retried alarm, with the same key, which
		// the email provider deduplicates. Recording first would risk never sending.
		await this.#send(mail);
		const root = await this.#root(harness);
		await root.commit(async (tx) => {
			const draft = await tx.doc(MailState);
			draft.pending = draft.pending.filter((p) => p.submissionId !== pending.submissionId);
			if (!draft.sent.some((s) => s.key === mail.key)) draft.sent.push(mail);
		}, context);
	}

	async #send(mail: SentMail): Promise<void> {
		testing.onSend?.(mail);
		if (this.env.RESEND_API_KEY === undefined) return;
		const headers: Record<string, string> = {};
		if (mail.inReplyTo !== null) {
			headers["In-Reply-To"] = mail.inReplyTo;
			headers.References = mail.inReplyTo;
		}
		const response = await fetch("https://api.resend.com/emails", {
			method: "POST",
			headers: {
				Authorization: `Bearer ${this.env.RESEND_API_KEY}`,
				"Content-Type": "application/json",
				"Idempotency-Key": mail.key,
			},
			body: JSON.stringify({ from: this.env.FROM_ADDRESS, to: mail.to, subject: mail.subject, text: mail.text, headers }),
		});
		if (!response.ok) throw new Error(`Resend ${response.status}: ${await response.text()}`);
	}

	/** The runner finished (or gave up on) a job: record it and tell the agent. Idempotent by job ID. */
	async jobReport(id: string, report: JobReport): Promise<void> {
		const harness = await this.#open();
		const root = await this.#root(harness);
		await root.commit(async (tx) => {
			const job = (await tx.doc(MailState)).jobs.find((j) => j.id === id);
			if (job === undefined) return;
			job.status = report.ok ? "done" : "failed";
			job.artifacts = report.artifacts;
		}, context);
		await this.notify(`job:${id}`, summarize(id, report));
	}

	/** Mail state, for the runner and for tests. */
	async mailState() {
		const harness = await this.#open();
		return harness.snapshot(MailState, context);
	}

	async usage() {
		const harness = await this.#open();
		return harness.usage(context);
	}
}
