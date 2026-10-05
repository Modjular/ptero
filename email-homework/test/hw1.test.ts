// Homework 1's "done when", before you touch real email: call the handler with a fake message and check the reply.
// From Homework 2 on, write the tests yourself: one file per homework, named like this one.
import { env } from "cloudflare:test";
import { expect, it } from "vitest";
import worker from "../src/index.ts";

/** Just enough of a ForwardableEmailMessage for Homework 1, recording what the handler does with it. */
function fakeMessage(subject: string) {
	const replies: unknown[] = [];
	const message = {
		from: "you@example.com",
		to: "bot@example.com",
		headers: new Headers({ Subject: subject, "Message-ID": "<test-1@example.com>" }),
		raw: new Blob([`Subject: ${subject}\r\n\r\nHello bot`]).stream(),
		rawSize: 0,
		setReject() {},
		forward: async () => ({ messageId: "" }),
		reply: async (reply: unknown) => {
			replies.push(reply);
			return { messageId: "<reply-1@example.com>" };
		},
	};
	return { message: message as unknown as ForwardableEmailMessage, replies };
}

it("Homework 1: replies 'Got it!' in the same thread", async () => {
	const { message, replies } = fakeMessage("Hello");
	await worker.email(message, env, {} as ExecutionContext);
	expect(replies, "email() never called message.reply()").toHaveLength(1);
	expect(replies[0]).toMatchObject({
		from: env.FROM_ADDRESS,
		subject: "Re: Hello",
		text: expect.stringContaining("Got it"),
	});
});
