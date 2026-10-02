# email-agent (spike)

Test of the email front end for ptero. Each customer gets one Cloudflare Durable Object, which runs a
[`@earendil-works/pi-durable`](https://www.npmjs.com/package/@earendil-works/pi-durable) Harness on the
object's own SQLite.

```
inbound email ─▶ email() ─▶ MailAgent DO (idFromName(sender))
                               receive(): commit mail state, submit(requestId = "mail:<Message-ID>"), setAlarm(now)
upload page   ─▶ POST /u/:customer/:token/done ─▶ notify("upload:…")   ─┐ also submissions,
runner        ─▶ POST /jobs/:customer/:id/done  ─▶ notify("job:<id>")   ─┘ answered by mail
                               alarm(): watchdog alarm → harness.resume() → wait for each pending submission
                                        → send reply (Idempotency-Key = reply:<submission>) → record as sent
```

The model's final message in each run is the email body. Anything asynchronous, like an upload finishing or a
runner job finishing, comes back as a new submission, so nothing has to stay alive to wait for it.

## What the test answered

| Question | Result |
|---|---|
| Does durable's SQLite storage run on DO SQLite? | **Yes.** `src/do-sqlite.ts` is a ~100-line `SqliteDatabase` adapter. All 23 cases of durable's storage conformance suite pass inside workerd (`test/conformance.test.ts`). |
| DO SQL rejects `BEGIN`/`SAVEPOINT`. How do transactions work? | `ctx.storage.transaction(async …)` keeps one SQLite transaction open across awaits, rolls back when the callback throws, and rejects with the callback's own error. That's what durable's facade requires. Pinned in `test/do-sql.test.ts`. |
| Other DO SQL differences? | Blobs come back as `ArrayBuffer`, and bigint bindings throw. The adapter converts both, though durable uses neither today. |
| Email → run → reply loop | Works. `test/loop.test.ts` plays the mocked transcript on a faux model: upload link, `[upload]` notice, queued analysis, then the runner callback, giving 4 threaded replies. |
| Redelivered email | Returns the same submission by request ID, so no second run and no second reply. |
| Object dies mid-model-call | The watchdog alarm resumes the Harness, the generation retries, and exactly one reply is sent. |
| Bundle size | 1.7 MB raw / **320 KiB gzipped** (`wrangler deploy --dry-run`). The free plan limit is 3 MB. |
| CPU per run (`npm run bench`) | See below. |

### CPU: the free tier probably doesn't fit

Measured under Node with `node:sqlite` as a stand-in, since workerd's clock doesn't advance during compute. One run is
an email, a tool call, and a ~360-token answer:

| | CPU |
|---|---|
| Cold open of the Harness | ~10–12 ms |
| One run, answer streamed at ~150 tok/s (realistic) | **~50–90 ms** |
| One run, no streaming delay (fewer partial commits) | ~15–25 ms (50 ms on the first) |

CPU per run doesn't grow with transcript length over 30 runs. Each run executes inside a single alarm
invocation, so this has to fit the per-invocation CPU limit. Workers Free allows 10 ms of CPU per invocation,
and I couldn't confirm from this sandbox whether Durable Object alarms on the free plan get more. Even without
streaming, a run is 2–5× over 10 ms. **Plan on Workers Paid ($5/month), where the default CPU limit is 30 s.**
At that point the control plane is effectively free, and the LLM bill is the real cost.

## Not built (yet)

- Outbound email: `#send` posts to Resend when `RESEND_API_KEY` is set. Workers' `send_email` binding can only
  reach verified addresses.
- The upload page (a ptero page that writes thumbnails and `meta.json` to R2), R2 itself, and the runner
  (`tools/drive.mjs` in a polling loop).
- Upload tokens are random UUIDs, not signed and not expiring. Sender authentication (SPF/DKIM), a sender
  allowlist, and per-sender rate limits are also missing.
- `receive()` submits and then records the pending reply in two commits. If it crashes between them, the
  sender's retry heals it (same request ID). A crash with no retry would lose that one reply.

## Run

```bash
npm install              # .npmrc sets legacy-peer-deps: npm 10's resolver crashes on vitest 4's peer set
npm test                 # conformance + DO SQL behaviour + the email loop, inside workerd
npm run typecheck
npm run bench            # CPU per run under Node
```

`compatibility_date` is pinned to the newest date the bundled workerd supports. Bump it when you bump wrangler.
