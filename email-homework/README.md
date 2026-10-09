# email-homework

The starter project for **Building an Email Agent on Cloudflare: A Homework Book**
(https://claude.ai/code/artifact/4fd28d95-9216-4e3c-8922-68b56491409c). It holds no solutions. It deploys as-is,
and it fails Homework 1's test until you write the handler.

```bash
npm install
npm test            # fails: "email() never called message.reply()"; that's Homework 1
npm run typecheck
npm run dev         # wrangler dev, locally
npm run email:local # in a second terminal: send the local Worker a test email (/cdn-cgi/local/email)
npm run deploy      # wrangler deploy, once Homework 0 is done
npm run tail        # live logs from the deployed Worker
```

| File | What it's for |
|---|---|
| `src/index.ts` | Your Worker: the `email()` handler (Homework 1 on), `fetch()` (Homework 8 on) |
| `wrangler.jsonc` | Config; commented sections show where Homeworks 3, 4 and 8 add vars and bindings |
| `worker-configuration.d.ts` | The types of those bindings; add a line whenever `wrangler.jsonc` gains one |
| `test/hw1.test.ts` | Homework 1's test, with a fake message; write `hw2.test.ts` and on yourself |
| `scripts/local-email.mjs` | `npm run email:local [subject] [body]`: a real email event for `wrangler dev`, no domain needed |

**Answer key:** `../email-agent/` (the spike), plus `../tools/runner.mjs` and `../tools/notebook.mjs` for Homework 9.
Try first, then compare.

**Suggested habit:** commit and tag at the end of each homework (`git tag hw-1`), so `git diff hw-3 hw-4` shows
exactly what one idea cost you.

**Before your first deploy:** set `FROM_ADDRESS` to an address on your own domain (`reply()` refuses any
other), and uncomment `addresses` with the address that should reach this Worker. The deploy creates that
Email Routing rule.

**Gotchas already handled here:**

- `.npmrc` sets `legacy-peer-deps`, because npm 10 crashes resolving vitest 4's peer dependencies.
- `compatibility_date` is pinned to the newest date the bundled workerd supports. Bump it when you bump wrangler.
