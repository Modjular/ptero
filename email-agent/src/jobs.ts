import { DurableObject } from "cloudflare:workers";

export type JobSpec = { id: string; customer: string; files: string[]; cells: string[] };
export type Claim = JobSpec & { lease: string; attempt: number };

const LEASE_MS = 20 * 60_000;
/** A job whose lease expired this many times is given up on: the runner keeps dying on it. */
const MAX_ATTEMPTS = 3;

/**
 * The one queue the runner polls. Jobs are born in a customer's MailAgent, which has no way to be found by the
 * runner, so they are mirrored here. A claim is a lease: a runner that dies mid-job loses it when the lease
 * expires, and the next claim hands the job out again.
 */
export class JobBoard extends DurableObject<Cloudflare.Env> {
	constructor(ctx: DurableObjectState, env: Cloudflare.Env) {
		super(ctx, env);
		ctx.storage.sql.exec(`CREATE TABLE IF NOT EXISTS jobs (
			id TEXT PRIMARY KEY,
			customer TEXT NOT NULL,
			spec TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'queued',
			lease TEXT,
			lease_until INTEGER,
			attempts INTEGER NOT NULL DEFAULT 0,
			created INTEGER NOT NULL
		)`);
	}

	/** Idempotent by job ID, so a replayed run_analysis call does not queue twice. */
	enqueue(spec: JobSpec): void {
		this.ctx.storage.sql.exec(
			"INSERT OR IGNORE INTO jobs (id, customer, spec, created) VALUES (?, ?, ?, ?)",
			spec.id,
			spec.customer,
			JSON.stringify(spec),
			Date.now(),
		);
	}

	/** The oldest job that is queued or whose lease ran out. Synchronous, so two claims cannot interleave. */
	claim(now = Date.now()): Claim | null {
		const sql = this.ctx.storage.sql;
		for (;;) {
			const row = sql
				.exec<{ id: string; customer: string; spec: string; attempts: number }>(
					`SELECT id, customer, spec, attempts FROM jobs
					 WHERE status = 'queued' OR (status = 'leased' AND lease_until < ?)
					 ORDER BY created LIMIT 1`,
					now,
				)
				.toArray()[0];
			if (row === undefined) return null;
			if (row.attempts >= MAX_ATTEMPTS) {
				sql.exec("UPDATE jobs SET status = 'failed', lease = NULL WHERE id = ?", row.id);
				this.ctx.waitUntil(
					this.#customer(row.customer).jobReport(row.id, {
						ok: false,
						error: `the runner lost this job ${MAX_ATTEMPTS} times (crash or timeout)`,
						cells: [],
						artifacts: [],
					}),
				);
				continue;
			}
			const lease = crypto.randomUUID();
			sql.exec(
				"UPDATE jobs SET status = 'leased', lease = ?, lease_until = ?, attempts = attempts + 1 WHERE id = ?",
				lease,
				now + LEASE_MS,
				row.id,
			);
			return { ...(JSON.parse(row.spec) as JobSpec), lease, attempt: row.attempts + 1 };
		}
	}

	/** The job behind a live lease, or null: a stale runner cannot read, write, or report for a re-leased job. */
	leased(id: string, lease: string, now = Date.now()): JobSpec | null {
		const row = this.ctx.storage.sql
			.exec<{ spec: string }>(
				"SELECT spec FROM jobs WHERE id = ? AND status = 'leased' AND lease = ? AND lease_until >= ?",
				id,
				lease,
				now,
			)
			.toArray()[0];
		return row ? (JSON.parse(row.spec) as JobSpec) : null;
	}

	/** Close the lease and hand the report to the customer's agent. False when the lease is no longer live. */
	async complete(id: string, lease: string, report: JobReport): Promise<boolean> {
		const spec = this.leased(id, lease);
		if (spec === null) return false;
		this.ctx.storage.sql.exec("UPDATE jobs SET status = ?, lease = NULL WHERE id = ?", report.ok ? "done" : "failed", id);
		await this.#customer(spec.customer).jobReport(id, report);
		return true;
	}

	status(id: string): { status: string; attempts: number } | null {
		return (
			this.ctx.storage.sql
				.exec<{ status: string; attempts: number }>("SELECT status, attempts FROM jobs WHERE id = ?", id)
				.toArray()[0] ?? null
		);
	}

	#customer(name: string) {
		return this.env.MAIL_AGENT.get(this.env.MAIL_AGENT.idFromName(name));
	}
}

export type CellReport = { state: string; text: string; figures: string[] };
export type JobReport = { ok: boolean; error?: string; cells: CellReport[]; artifacts: string[] };

export function jobBoard(env: Cloudflare.Env) {
	return env.JOBS.get(env.JOBS.idFromName("board"));
}
