// End to end: the email-agent Worker (in Miniflare) ↔ tools/runner.mjs ↔ a real notebook.html with WebGPU.
//
//   python3 -m http.server 8765 --directory ..  &      # ptero, served from the repo root
//   node scripts/runner-e2e.mjs                         # add runner flags after --, e.g. -- --swiftshader
//
// Queues two jobs on the board, one that segments tools/test_image.tif with StarDist and one that fails on
// purpose, runs the runner until the queue is empty, and checks what came back: the board's verdicts, the
// artifacts in R2, and the [runner] reports reaching the customer's agent (MODEL=echo quotes them back as its
// email). RUNNER_LAUNCHER, if set, is a Node script the runner is started through.
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createTestHarness } from "wrangler";

const HERE = dirname(fileURLToPath(import.meta.url));
const AGENT = join(HERE, "..");
const PTERO = join(AGENT, "..");
const TOKEN = "e2e-runner-token";
const CUSTOMER = "e2e@lab.example";
const runnerArgs = process.argv.includes("--") ? process.argv.slice(process.argv.indexOf("--") + 1) : [];

const segment = {
	id: `${CUSTOMER}:segment`,
	customer: CUSTOMER,
	files: ["test_image.tif"],
	cells: [
		`import numpy as np
import pandas as pd
from skimage import io, measure
from stardist.models import StarDist2D
from csbdeep.utils import normalize

img = io.imread("test_image.tif")
img.shape`,
		`labels, details = StarDist2D.from_pretrained('2D_versatile_fluo').predict_instances(normalize(img[..., 2]))
f"{labels.max()} nuclei"`,
		`import matplotlib.pyplot as plt
from skimage.color import label2rgb

props = pd.DataFrame(measure.regionprops_table(labels, intensity_image=img[..., 1],
                                               properties=['label', 'area', 'intensity_mean']))
props.to_csv("nuclei.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(label2rgb(labels, bg_label=0))
ax.axis('off')
len(props)`,
	],
};
const broken = { id: `${CUSTOMER}:broken`, customer: CUSTOMER, files: [], cells: ["x = 1", "x / 0", "never_runs()"] };

const check = (ok, what) => {
	console.log(`${ok ? "✓" : "✗"} ${what}`);
	if (!ok) process.exitCode = 1;
};

const server = createTestHarness({
	workers: [{ configPath: join(AGENT, "wrangler.jsonc"), vars: { MODEL: "echo" }, secrets: { RUNNER_TOKEN: TOKEN } }],
});
try {
	const api = (await server.listen()).url.href.replace(/\/$/, "");
	const env = await server.getWorker().getEnv();
	const files = env.FILES;
	const board = env.JOBS.get(env.JOBS.idFromName("board"));
	const agent = env.MAIL_AGENT.get(env.MAIL_AGENT.idFromName(CUSTOMER));

	await files.put(`uploads/${CUSTOMER}/test_image.tif`, await readFile(join(PTERO, "tools", "test_image.tif")));
	await board.enqueue(segment);
	await board.enqueue(broken);
	console.log(`→ Worker at ${api}; 2 jobs queued; starting the runner`);

	const launcher = process.env.RUNNER_LAUNCHER ? [process.env.RUNNER_LAUNCHER] : [];
	const runner = spawn(
		process.execPath,
		[...launcher, join(PTERO, "tools", "runner.mjs"), "--api", api, "--once", "--poll", "1", ...runnerArgs],
		{ cwd: PTERO, env: { ...process.env, RUNNER_TOKEN: TOKEN }, stdio: "inherit" },
	);
	const code = await new Promise((resolve) => runner.on("exit", resolve));
	check(code === 0, `runner exited cleanly (${code})`);

	check((await board.status(segment.id))?.status === "done", "segmentation job: done");
	check((await board.status(broken.id))?.status === "failed", "broken job: failed");

	const stored = (await files.list({ prefix: `results/${segment.id}/` })).objects.map((o) => o.key.split("/").pop());
	check(stored.includes("nuclei.csv"), `artifacts stored: ${stored.join(", ")}`);
	check(stored.some((name) => /^cell-3-fig-\d+\.png$/.test(name)), "the figure came back as a PNG");
	const csv = await (await files.get(`results/${segment.id}/nuclei.csv`)).text();
	const rows = csv.trim().split("\n").length - 1;
	check(rows > 50, `nuclei.csv has ${rows} nuclei`);

	// Both reports reach the agent, which answers each by email; MODEL=echo makes the answer the report itself.
	let sent = [];
	for (let i = 0; i < 120 && sent.length < 2; i++) {
		sent = (await agent.mailState())?.sent ?? [];
		if (sent.length < 2) await new Promise((resolve) => setTimeout(resolve, 500));
	}
	const texts = sent.map((mail) => mail.text);
	const ok = texts.find((t) => t.includes(`job ${segment.id} finished`));
	const failed = texts.find((t) => t.includes(`job ${broken.id} FAILED`));
	check(ok !== undefined && /Cell 2 \[done\]: \d+ nuclei/.test(ok), "the agent got the segmentation report");
	check(ok !== undefined && !/already loaded|Loading micropip/.test(ok), "the report carries no package-loader noise");
	check(failed !== undefined && failed.includes("ZeroDivisionError") && failed.includes("Cell 3 [not run]"), "the agent got the failure, with the error and the cell that never ran");
	if (process.exitCode) for (const text of texts) console.log(`\n--- agent email ---\n${text}`);
	else console.log(`\n--- the segmentation report, as the agent saw it ---\n${ok.replace(/^Echo: /, "")}`);
} finally {
	await server.close();
}
