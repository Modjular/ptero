// This Worker's bindings, as `Cloudflare.Env`: the Worker reads them as `Env`, and the test pool's `env` gets the same type.
declare namespace Cloudflare {
	interface Env {
		MAIL_AGENT: DurableObjectNamespace<import("./src/agent.ts").MailAgent>;
		JOBS: DurableObjectNamespace<import("./src/jobs.ts").JobBoard>;
		FILES: R2Bucket;
		PUBLIC_URL: string;
		FROM_ADDRESS: string;
		MODEL?: string;
		RUNNER_TOKEN?: string;
		RESEND_API_KEY?: string;
		ANTHROPIC_API_KEY?: string;
	}
}
