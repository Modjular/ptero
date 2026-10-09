// This Worker's bindings. Add one line here each time wrangler.jsonc gains a var, secret or binding,
// so both the Worker and the tests see its type.
declare namespace Cloudflare {
	interface Env {
		FROM_ADDRESS: string;
		// Homework 3: MODEL: string; ALLOWED_SENDERS: string; ANTHROPIC_API_KEY: string;
		// Homework 5: EMAIL: SendEmail;
		// Homework 4: AGENT: DurableObjectNamespace<import("./src/index.ts").Agent>;
		// Homework 8: FILES: R2Bucket;
	}
}
