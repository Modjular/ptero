// Send a test email to your Worker running under `npm run dev` (wrangler dev), through the local endpoint
// Cloudflare provides for this. Wrangler prints what your handler did; a reply is saved as an .eml file.
//
//   npm run email:local                       # "Hello" from you@example.com
//   npm run email:local -- "Segmenting cells" "I have about 10 tifs."
const [subject = "Hello", body = "Hi bot, are you there?"] = process.argv.slice(2);
const port = process.env.PORT ?? "8787";
const from = "you@example.com";
const to = "bot@example.com";

// The endpoint wants a raw RFC 5322 message, and it must carry a Message-ID.
const raw = [
	`From: ${from}`,
	`To: ${to}`,
	`Subject: ${subject}`,
	`Message-ID: <${crypto.randomUUID()}@example.com>`,
	`Date: ${new Date().toUTCString()}`,
	"Content-Type: text/plain; charset=utf-8",
	"",
	body,
].join("\r\n");

const url = new URL(`http://localhost:${port}/cdn-cgi/local/email`);
url.searchParams.set("from", from);
url.searchParams.set("to", to);
const response = await fetch(url, { method: "POST", body: raw });
console.log(`${response.status} ${await response.text()}`);
console.log("Now look at the wrangler dev terminal.");
