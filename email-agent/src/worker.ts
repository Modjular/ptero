import PostalMime from "postal-mime";
import type { Env } from "./agent.ts";

export { MailAgent } from "./agent.ts";

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
		const url = new URL(request.url);
		// POST /u/:customer/:token/done  {files: string[]}: the upload page reports finished uploads.
		const upload = url.pathname.match(/^\/u\/([^/]+)\/([^/]+)\/done$/);
		if (upload && request.method === "POST") {
			const [, customer, token] = upload;
			const stub = agentFor(env, decodeURIComponent(customer!));
			if ((await stub.mailState())?.uploadToken !== token) return new Response("not found", { status: 404 });
			const { files } = (await request.json()) as { files: string[] };
			await stub.notify(`upload:${token}:${files.join(",")}`, `[upload] ${files.length} files arrived: ${files.join(", ")}`);
			return new Response("ok");
		}
		// POST /jobs/:customer/:id/done  {summary: string}: the runner reports a finished job.
		const job = url.pathname.match(/^\/jobs\/([^/]+)\/([^/]+)\/done$/);
		if (job && request.method === "POST") {
			if (env.RUNNER_TOKEN === undefined || request.headers.get("Authorization") !== `Bearer ${env.RUNNER_TOKEN}`) {
				return new Response("unauthorized", { status: 401 });
			}
			const [, customer, id] = job;
			const { summary } = (await request.json()) as { summary: string };
			await agentFor(env, decodeURIComponent(customer!)).notify(`job:${id}`, `[runner] job ${id} finished: ${summary}`);
			return new Response("ok");
		}
		return new Response("not found", { status: 404 });
	},
} satisfies ExportedHandler<Env>;
