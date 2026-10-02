// The DO SQL behaviour the adapter in src/do-sqlite.ts relies on. If workerd changes any of it, this fails first.
import { env, runInDurableObject } from "cloudflare:test";
import { expect, it } from "vitest";

const fresh = () => env.MAIL_AGENT.get(env.MAIL_AGENT.idFromName(crypto.randomUUID()));
const tick = () => Promise.resolve();

it("rejects SQL-level transactions", async () => {
	await runInDurableObject(fresh(), (_, state) => {
		expect(() => state.storage.sql.exec("BEGIN")).toThrow(/transaction\(\)/);
		expect(() => state.storage.sql.exec("SAVEPOINT a")).toThrow(/transaction\(\)/);
	});
});

it("keeps one transaction open across awaits and commits it", async () => {
	await runInDurableObject(fresh(), async (_, { storage }) => {
		storage.sql.exec("CREATE TABLE t (v TEXT)");
		await storage.transaction(async () => {
			storage.sql.exec("INSERT INTO t VALUES ('a')");
			await tick();
			storage.sql.exec("INSERT INTO t VALUES ('b')");
		});
		expect(storage.sql.exec("SELECT v FROM t").toArray()).toEqual([{ v: "a" }, { v: "b" }]);
	});
});

it("rolls back across awaits and rejects with the callback's error", async () => {
	await runInDurableObject(fresh(), async (_, { storage }) => {
		storage.sql.exec("CREATE TABLE t (v TEXT)");
		const boom = new Error("boom");
		const failed = storage.transaction(async () => {
			storage.sql.exec("INSERT INTO t VALUES ('x')");
			await tick();
			throw boom;
		});
		await expect(failed).rejects.toBe(boom);
		expect(storage.sql.exec("SELECT v FROM t").toArray()).toEqual([]);
	});
});

it("runs several statements in one exec, returns blobs as ArrayBuffer, rejects bigint", async () => {
	await runInDurableObject(fresh(), (_, { storage }) => {
		storage.sql.exec("CREATE TABLE a (x); CREATE TABLE b (v BLOB);");
		storage.sql.exec("INSERT INTO b VALUES (?)", new Uint8Array([1, 2, 3]));
		expect(storage.sql.exec("SELECT v FROM b").one().v).toBeInstanceOf(ArrayBuffer);
		expect(() => storage.sql.exec("INSERT INTO a VALUES (?)", 5n as unknown as number)).toThrow(/BigInt/);
	});
});
