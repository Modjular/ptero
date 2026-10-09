// Homework 1 starts here. The book: "Building an Email Agent on Cloudflare: A Homework Book".
//
// Cloudflare calls email() once per message that Email Routing sends to this Worker.
// What you can do with `message` (see the Homework 1 diagram):
//   message.from, message.to, message.headers, message.raw   read it          (Homework 2)
//   message.reply({ from, subject, text })                    answer it        (Homework 1)
//   message.forward(address)                                  pass it on
//   message.setReject(reason)                                 bounce it        (Homework 3)
//
// reply() throws unless: the incoming mail passed DMARC, it's the first reply to it, it goes back to the
// sender, it comes from the domain that received the mail, and the thread has at most 100 References.
// Try it locally: `npm run dev`, then `npm run email:local` in a second terminal.

export type Env = Cloudflare.Env;

export default {
	async email(message, env, ctx) {
		// TODO(Homework 1): reply "Got it!" to the sender, from env.FROM_ADDRESS,
		// with the subject "Re: " + the original subject (message.headers.get("Subject")).
	},

	// Not used until Homework 8 (the upload page). Every HTTP request is a 404 until then.
	async fetch() {
		return new Response("not found", { status: 404 });
	},
} satisfies ExportedHandler<Env>;
