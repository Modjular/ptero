// The runner's queue: leases, re-delivery after a runner dies, and giving up on a job that keeps killing it.
import { env, runInDurableObject } from "cloudflare:test";
import { expect, it } from "vitest";
import type { JobBoard } from "../src/jobs.ts";

const board = () => env.JOBS.get(env.JOBS.idFromName(crypto.randomUUID()));
const spec = (id: string) => ({ id, customer: "x@lab.example", files: ["a.tif"], cells: ["1 + 1"] });
const MINUTE = 60_000;

it("hands jobs out oldest first, once each, and ignores a repeated enqueue", async () => {
	await runInDurableObject(board(), async (jobs: JobBoard) => {
		jobs.enqueue(spec("j1"));
		jobs.enqueue(spec("j1"));
		jobs.enqueue(spec("j2"));
		expect(jobs.claim()?.id).toBe("j1");
		expect(jobs.claim()?.id).toBe("j2");
		expect(jobs.claim()).toBeNull();
	});
});

it("re-leases a job whose runner went quiet, and locks the old lease out", async () => {
	await runInDurableObject(board(), async (jobs: JobBoard) => {
		const now = Date.now();
		jobs.enqueue(spec("j1"));
		const first = jobs.claim(now)!;
		expect(jobs.claim(now + 5 * MINUTE)).toBeNull();
		const second = jobs.claim(now + 21 * MINUTE)!;
		expect(second).toMatchObject({ id: "j1", attempt: 2 });
		expect(second.lease).not.toBe(first.lease);
		expect(jobs.leased("j1", first.lease)).toBeNull();
		expect(jobs.leased("j1", second.lease)).not.toBeNull();
	});
});

it("gives up after three lost leases and tells the customer's agent", async () => {
	const reports: unknown[] = [];
	await runInDurableObject(board(), async (jobs: JobBoard, state) => {
		// Stand in for the MailAgent: the board must report the failure rather than drop the job silently.
		Object.assign(jobs, {
			env: { MAIL_AGENT: { idFromName: () => "x", get: () => ({ jobReport: async (...args: unknown[]) => reports.push(args) }) } },
		});
		const now = Date.now();
		jobs.enqueue(spec("j1"));
		for (let attempt = 0; attempt < 3; attempt++) expect(jobs.claim(now + attempt * 21 * MINUTE)?.attempt).toBe(attempt + 1);
		expect(jobs.claim(now + 3 * 21 * MINUTE)).toBeNull();
		expect(jobs.status("j1")).toEqual({ status: "failed", attempts: 3 });
	});
	await expect.poll(() => reports.length).toBe(1);
	expect(reports[0]).toEqual(["j1", expect.objectContaining({ ok: false, error: expect.stringContaining("3 times") })]);
});
