import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { env, runInDurableObject } from "cloudflare:test";
import { registerStorageConformance } from "@earendil-works/pi-durable/testing";
import { describe, expect, it } from "vitest";
import { openDurableObjectStorage } from "../src/do-sqlite.ts";

// Every case gets a fresh Durable Object, so its SQLite file starts empty.
registerStorageConformance({ describe, expect, it }, "Durable Object SQLite", async (use) => {
	const stub = env.MAIL_AGENT.get(env.MAIL_AGENT.idFromName(crypto.randomUUID()));
	await runInDurableObject(stub, async (_instance, state) => {
		const storage = await openDurableObjectStorage(state.storage);
		try {
			await use(storage);
		} finally {
			await storage.close(BACKGROUND_CONTEXT);
		}
	});
});
