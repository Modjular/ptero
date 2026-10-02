import type { SqliteDatabase, SqliteExecutor, SqliteValue } from "@earendil-works/pi-durable/storage/sqlite";
import { SqliteStorage } from "@earendil-works/pi-durable/storage/sqlite";

// DO SQL binds number | string | ArrayBuffer | null. pi-durable never binds bigint or blobs today,
// but the facade allows both, so convert rather than let workerd throw "Cannot convert a BigInt".
function bind(value: SqliteValue): SqlStorageValue {
	if (typeof value === "bigint") {
		if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
			throw new RangeError(`bigint ${value} does not fit a DO SQL number`);
		}
		return Number(value);
	}
	if (value instanceof Uint8Array) return value.slice().buffer;
	return value;
}

// Blobs come back as ArrayBuffer; the facade promises Uint8Array.
function row<T extends object>(raw: Record<string, SqlStorageValue>): T {
	for (const key in raw) {
		const value = raw[key];
		if (value instanceof ArrayBuffer) (raw as Record<string, unknown>)[key] = new Uint8Array(value);
	}
	return raw as T;
}

class Executor implements SqliteExecutor {
	constructor(
		protected readonly sql: SqlStorage,
		private readonly gate: () => Promise<void>,
	) {}

	async exec(text: string): Promise<void> {
		await this.gate();
		this.sql.exec(text);
	}

	async run(text: string, ...params: SqliteValue[]): Promise<void> {
		await this.gate();
		this.sql.exec(text, ...params.map(bind));
	}

	async get<T extends object>(text: string, ...params: SqliteValue[]): Promise<T | undefined> {
		await this.gate();
		const first = this.sql.exec(text, ...params.map(bind)).next();
		return first.done ? undefined : row<T>(first.value);
	}

	async all<T extends object>(text: string, ...params: SqliteValue[]): Promise<T[]> {
		await this.gate();
		return this.sql
			.exec(text, ...params.map(bind))
			.toArray()
			.map((raw) => row<T>(raw));
	}
}

/**
 * `SqliteDatabase` over a SQLite-backed Durable Object's `ctx.storage.sql`.
 *
 * DO SQL rejects BEGIN/SAVEPOINT, so transactions go through `ctx.storage.transaction()`, which
 * (verified in workerd) keeps one SQLite transaction open across awaits and rolls back when the
 * callback rejects, rejecting with the callback's own error. Everything shares one connection, so
 * operations from outside a running transaction must wait for it, or they would land inside it.
 */
export class DurableObjectSqliteDatabase extends Executor implements SqliteDatabase {
	private tail: Promise<void> = Promise.resolve();
	private closed = false;

	constructor(private readonly storage: DurableObjectStorage) {
		super(storage.sql, () => this.idle());
	}

	private idle(): Promise<void> {
		if (this.closed) return Promise.reject(new Error("SQLite database is closed"));
		return this.tail;
	}

	async transaction<T>(callback: (transaction: SqliteExecutor) => Promise<T>): Promise<T> {
		const previous = this.idle();
		const { promise: done, resolve: release } = Promise.withResolvers<void>();
		this.tail = previous.then(() => done);
		await previous;
		const scope = { active: true };
		const handle = new Executor(this.sql, async () => {
			if (!scope.active) throw new Error("SQLite transaction handle is no longer active");
		});
		try {
			return await this.storage.transaction(() => callback(handle));
		} finally {
			scope.active = false;
			release();
		}
	}

	async close(): Promise<void> {
		await this.idle();
		this.closed = true;
	}
}

export function openDurableObjectStorage(storage: DurableObjectStorage): Promise<SqliteStorage> {
	return SqliteStorage.open(new DurableObjectSqliteDatabase(storage));
}
