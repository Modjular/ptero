import PostalMime from "postal-mime";
import type { Env } from "./agent.ts";
import { type JobReport, jobBoard } from "./jobs.ts";

export { MailAgent } from "./agent.ts";
export { JobBoard } from "./jobs.ts";

/** Drop the quoted history mail clients append; the transcript already has it, and it costs tokens. */
export function stripQuoted(text: string): string {
	const lines = text.replace(/\r\n/g, "\n").split("\n");
	const cut = lines.findIndex((line) => /^On .+wrote:\s*$/.test(line) || /^-+ ?Original Message ?-+$/i.test(line));
	return (cut === -1 ? lines : lines.slice(0, cut))
		.filter((line) => !line.startsWith(">"))
		.join("\n")
		.trim();
}

/** One Durable Object per customer address. */
function agentFor(env: Env, address: string) {
	return env.MAIL_AGENT.get(env.MAIL_AGENT.idFromName(address.trim().toLowerCase()));
}

// R2 layout: what customers upload, and what the runner produced per job.
const uploadKey = (customer: string, name: string) => `uploads/${customer}/${name}`;
const resultKey = (job: string, name: string) => `results/${job}/${name}`;

/** One path segment of a file name: no traversal into another customer's prefix. */
const safeName = (name: string) => name.length > 0 && name.length <= 200 && !/[/\\]|^\.\.?$/.test(name);

type Route = (request: Request, env: Env, params: string[]) => Promise<Response>;

const notFound = () => new Response("not found", { status: 404 });

const routes: [string, RegExp, Route][] = [
	// The upload page: one PUT per file, then one POST listing what arrived.
	[
		"PUT",
		/^\/u\/([^/]+)\/([^/]+)\/files\/([^/]+)$/,
		async (request, env, [customer, token, name]) => {
			if (!safeName(name!) || (await agentFor(env, customer!).mailState())?.uploadToken !== token) return notFound();
			await env.FILES.put(uploadKey(customer!, name!), request.body);
			return new Response("ok");
		},
	],
	[
		"POST",
		/^\/u\/([^/]+)\/([^/]+)\/done$/,
		async (request, env, [customer, token]) => {
			const stub = agentFor(env, customer!);
			if ((await stub.mailState())?.uploadToken !== token) return notFound();
			const { files } = (await request.json()) as { files: string[] };
			await stub.notify(`upload:${token}:${files.join(",")}`, `[upload] ${files.length} files arrived: ${files.join(", ")}`);
			return new Response("ok");
		},
	],

	// The runner. Every route below needs the runner token; every job route also needs the job's live lease.
	[
		"POST",
		/^\/runner\/claim$/,
		async (_request, env) => {
			const claim = await jobBoard(env).claim();
			return claim ? Response.json(claim) : new Response(null, { status: 204 });
		},
	],
	[
		"GET",
		/^\/runner\/jobs\/([^/]+)\/files\/([^/]+)$/,
		async (request, env, [id, name]) => {
			const job = await jobBoard(env).leased(id!, lease(request));
			if (job === null || !job.files.includes(name!)) return notFound();
			const object = await env.FILES.get(uploadKey(job.customer, name!));
			return object ? new Response(object.body) : notFound();
		},
	],
	[
		"PUT",
		/^\/runner\/jobs\/([^/]+)\/artifacts\/([^/]+)$/,
		async (request, env, [id, name]) => {
			if (!safeName(name!) || (await jobBoard(env).leased(id!, lease(request))) === null) return notFound();
			await env.FILES.put(resultKey(id!, name!), request.body);
			return new Response("ok");
		},
	],
	[
		"POST",
		/^\/runner\/jobs\/([^/]+)\/report$/,
		async (request, env, [id]) => {
			const report = (await request.json()) as JobReport;
			const accepted = await jobBoard(env).complete(id!, lease(request), report);
			return accepted ? new Response("ok") : new Response("lease expired", { status: 409 });
		},
	],
];

const lease = (request: Request) => request.headers.get("X-Lease") ?? "";

export default {
	async email(message: ForwardableEmailMessage, env: Env): Promise<void> {
		const parsed = await PostalMime.parse(message.raw);
		const messageId = parsed.messageId ?? message.headers.get("Message-ID") ?? crypto.randomUUID();
		await agentFor(env, message.from).receive({
			from: message.from,
			messageId,
			subject: parsed.subject ?? "",
			text: stripQuoted(parsed.text ?? ""),
		});
	},

	async fetch(request: Request, env: Env): Promise<Response> {
		const path = new URL(request.url).pathname;
		for (const [method, pattern, route] of routes) {
			const match = path.match(pattern);
			if (match === null || request.method !== method) continue;
			if (path.startsWith("/runner/")) {
				if (env.RUNNER_TOKEN === undefined || request.headers.get("Authorization") !== `Bearer ${env.RUNNER_TOKEN}`) {
					return new Response("unauthorized", { status: 401 });
				}
			}
			return route(request, env, match.slice(1).map(decodeURIComponent));
		}
		return notFound();
	},
} satisfies ExportedHandler<Env>;
