// Analytical cost of every dispatch the Cellpose engine issues.
//
// This is the piece that substitutes for hardware counters. WebGPU will not tell us how
// many bytes a kernel moved or how many warps were resident, but every kernel in this
// repo is hand-written, so its FLOP count and its byte traffic are *derivable exactly*
// from the shader source. That is a strictly better position than an `ncu` profile of an
// opaque cuDNN kernel for the architectural questions, and strictly worse for the
// implementation ones, because it says nothing about what the cache actually absorbed.
//
// Two traffic figures are reported for each dispatch and the distinction is the whole
// point:
//
//   cioIdeal      compulsory traffic — each input read once, each output written once.
//                 A property of the *architecture*. This is HarDNet's CIO, and it is
//                 what AI and MoC are computed from.
//   cioRequested  what this kernel's loop structure actually asks the memory hierarchy
//                 for, before any cache absorbs it. A property of the *implementation*.
//
// Their ratio (`amplification`) separates "this layer is memory-hungry" from "this
// kernel re-reads its input 32 times", which is exactly the confound Phase 0 exists to
// resolve. The conv kernel, for instance, loops over all Cin channels inside every
// z-block, so it requests the input once per ceil(Cout/8) block plus an 18/16 halo
// margin — an amplification of ~40x at the bottleneck that the SLC mostly hides.
//
// Labels are produced by the ops in src/cellpose.js; parseLabel() below is the other
// half of that contract and the two must be changed together.

const BLK = 8;   // output channels per conv workgroup   (src/cellpose.js)
const TS = 16;   // conv/pool/up spatial tile side
const TW = TS + 2;  // incl. halo, for K=3
const F32 = 4;

/** `conv|down.1.conv2|64->64|112x112|k3|relu|resid` -> a structured descriptor. */
export function parseLabel(label) {
  if (!label) return null;
  const p = label.split("|");
  const kind = p[0];
  const num = (s, pre) => Number(String(s).replace(pre, ""));
  try {
    if (kind === "conv") {
      const [cin, cout] = p[2].split("->").map(Number);
      const [h, w] = p[3].split("x").map(Number);
      return {
        kind, name: p[1], Cin: cin, Cout: cout, H: h, W: w, K: num(p[4], "k"),
        relu: p.includes("relu"), add: p.includes("add"), resid: p.includes("resid"),
      };
    }
    if (kind === "pool" || kind === "up") {
      const [from, to] = p[1].split("->");
      const [hi, wi] = from.split("x").map(Number);
      const [ho, wo] = to.split("x").map(Number);
      return { kind, Hi: hi, Wi: wi, Ho: ho, Wo: wo, C: num(p[2], "C") };
    }
    if (kind === "gap") {
      const [h, w] = p[1].split("x").map(Number);
      return { kind, H: h, W: w, C: num(p[2], "C") };
    }
    if (kind === "flowdiff" || kind === "flowgrad") {
      const [h, w] = p[1].split("x").map(Number);
      return { kind, H: h, W: w };
    }
    if (kind === "normstyle") return { kind, C: num(p[1], "C") };
    if (kind === "styleproj") return { kind, name: p[1], Cout: num(p[2], "C"), S: 256 };
    if (kind === "add") return { kind, N: num(p[1], "N") };
    if (kind === "dynamics") return { kind, npts: num(p[1], "npts"), niter: num(p[2], "niter") };
  } catch { /* fall through to null — an unparseable label is reported, not guessed at */ }
  return null;
}

const ceilDiv = (a, b) => Math.ceil(a / b);

/**
 * FLOPs and byte traffic for one parsed dispatch.
 *
 * Returns `trafficModelled: false` where compulsory traffic genuinely isn't a
 * meaningful quantity — the flow-dynamics kernel gathers along trajectories, so its
 * traffic is data-dependent and no static count would mean anything.
 */
export function costOf(d) {
  if (!d) return null;
  switch (d.kind) {
    case "conv": {
      const { H, W, Cin, Cout, K } = d;
      const hw = H * W;
      const macs = hw * Cout * Cin * K * K;
      // Per input element per channel: scale-multiply + shift-add, plus the optional
      // skip add and relu. Applied on the way into the shared-memory tile.
      const perIn = 2 + (d.add ? 1 : 0) + (d.relu ? 1 : 0);
      const flops = 2 * macs + hw * Cin * perIn + (d.resid ? hw * Cout : 0);
      const weights = Cin * Cout * K * K + 3 * Cout;   // w + bias/scale/shift
      const cioIdeal = F32 * (
        hw * Cin + (d.add ? hw * Cin : 0) + weights + (d.resid ? hw * Cout : 0) + hw * Cout);
      // The z dimension of the grid re-walks every input channel, and each 16x16 tile
      // pulls an 18x18 halo. Weights are re-read once per spatial tile.
      const zBlocks = ceilDiv(Cout, BLK);
      const halo = (TW * TW) / (TS * TS);
      const spatialTiles = ceilDiv(W, TS) * ceilDiv(H, TS);
      const cioRequested = F32 * (
        zBlocks * hw * Cin * halo * (d.add ? 2 : 1)
        + spatialTiles * weights
        + (d.resid ? hw * Cout : 0) + hw * Cout);
      const covered = ceilDiv(W, TS) * TS * ceilDiv(H, TS) * TS * zBlocks * BLK;
      return {
        flops, macs, cioIdeal, cioRequested,
        tileWaste: 1 - (hw * Cout) / covered,
        note: null,
      };
    }
    case "pool": {
      const { Hi, Wi, Ho, Wo, C } = d;
      const flops = 3 * C * Ho * Wo;                       // three max() per output
      const cio = F32 * (C * Hi * Wi + C * Ho * Wo);
      const covered = ceilDiv(Wo, TS) * TS * ceilDiv(Ho, TS) * TS * C;
      return { flops, macs: 0, cioIdeal: cio, cioRequested: cio,
               tileWaste: 1 - (Ho * Wo * C) / covered, note: null };
    }
    case "up": {
      const { Hi, Wi, Ho, Wo, C } = d;
      const cioIdeal = F32 * (C * Hi * Wi + C * Ho * Wo);
      // Nearest-neighbour: every output thread issues its own load, so the input is
      // requested four times over even though only a quarter of it is distinct.
      const cioRequested = F32 * (C * Ho * Wo + C * Ho * Wo);
      const covered = ceilDiv(Wo, TS) * TS * ceilDiv(Ho, TS) * TS * C;
      return { flops: 0, macs: 0, cioIdeal, cioRequested,
               tileWaste: 1 - (Ho * Wo * C) / covered, note: "pure data movement" };
    }
    case "gap": {
      const { H, W, C } = d;
      const cio = F32 * (C * H * W + C);
      return { flops: C * H * W + C, macs: 0, cioIdeal: cio, cioRequested: cio,
               tileWaste: 0,
               // One thread per channel, each looping serially over H*W. The parallelism
               // available is C, not C*H*W.
               note: "one thread per channel" };
    }
    case "flowdiff": {
      // One iteration of the label-masked 9-point diffusion over the whole image.
      // Compulsory traffic is one read of the field, one of the label map and one write;
      // what the kernel actually requests is nine of each, which the cache mostly
      // absorbs — the gap between the two is the amplification column.
      const { H, W } = d, n = H * W;
      const cio = F32 * 3 * n;
      return { flops: 10 * n, macs: 0, cioIdeal: cio, cioRequested: F32 * 19 * n,
               tileWaste: 0, note: "stencil; background pixels exit early" };
    }
    case "flowgrad": {
      const { H, W } = d, n = H * W;
      const cio = F32 * 4 * n;
      return { flops: 2 * n, macs: 0, cioIdeal: cio, cioRequested: F32 * 10 * n,
               tileWaste: 0, note: "central difference, label-masked" };
    }
    case "normstyle": {
      const cio = F32 * (2 * d.C);
      return { flops: 3 * d.C, macs: 0, cioIdeal: cio, cioRequested: cio, tileWaste: 0,
               note: "single workgroup" };
    }
    case "styleproj": {
      const { Cout, S } = d;
      const flops = 2 * Cout * S + 2 * Cout;
      const cio = F32 * (Cout * S + S + 3 * Cout + Cout);
      return { flops, macs: Cout * S, cioIdeal: cio, cioRequested: cio, tileWaste: 0, note: null };
    }
    case "add": {
      const cio = F32 * 3 * d.N;
      return { flops: d.N, macs: 0, cioIdeal: cio, cioRequested: cio, tileWaste: 0, note: null };
    }
    case "dynamics": {
      // Per point per iteration: two bilinear samples (4 weights, 4 muls, 3 adds each,
      // plus the weight algebra) and the position update with clamping. ~24 FLOP is a
      // deliberate order-of-magnitude estimate, not a derivation.
      const flops = d.npts * d.niter * 24;
      return { flops, macs: 0, cioIdeal: null, cioRequested: null, tileWaste: 0,
               trafficModelled: false,
               note: "gathers along trajectories; compulsory traffic is data-dependent" };
    }
    default: return null;
  }
}

/**
 * Join a measured dispatch record (from timing.js) to its analytical cost, and derive
 * every rate the roofline needs.
 *
 * `waveCapacity` is the number of workgroups this GPU can hold concurrently, measured
 * by roofs.js's waveProbe rather than assumed — WebGPU exposes no SM/core count, and
 * assuming one would put a fabricated number at the centre of the report's sharpest
 * argument.
 */
export function annotate(record, { waveCapacity = null } = {}) {
  const desc = parseLabel(record.label);
  const cost = costOf(desc);
  const seconds = record.ns / 1e9;
  const wg = record.workgroups
    ? record.workgroups[0] * record.workgroups[1] * record.workgroups[2]
    : null;

  const out = {
    ...record,
    kind: desc?.kind ?? "unknown",
    name: desc?.name ?? null,
    desc, workgroups: wg, workgroupsXYZ: record.workgroups,
    flops: cost?.flops ?? null,
    macs: cost?.macs ?? null,
    cioIdeal: cost?.cioIdeal ?? null,
    cioRequested: cost?.cioRequested ?? null,
    tileWaste: cost?.tileWaste ?? null,
    note: cost?.note ?? null,
    trafficModelled: cost ? cost.trafficModelled !== false : false,
  };

  if (cost && cost.flops != null && seconds > 0) {
    out.gflops = cost.flops / record.ns;              // FLOP/ns == GFLOP/s
  }
  if (cost && cost.cioIdeal) {
    out.ai = cost.flops / cost.cioIdeal;              // HarDNet CIO-based intensity
    out.moc = cost.macs / cost.cioIdeal;              // HarDNet's density objective
    out.amplification = cost.cioRequested / cost.cioIdeal;
    // The same kernel has two intensities, and which one binds is the whole question.
    // `ai` is the architecture's — what a perfect implementation would see. `aiRequested`
    // is what this kernel's loop nest actually asks for, and a conv that re-reads its
    // input once per output-channel block can be memory-bound at `aiRequested` while
    // looking comfortably compute-bound at `ai`. Reported so the two can be told apart
    // rather than one silently standing in for the other.
    out.aiRequested = cost.flops / cost.cioRequested;
    if (seconds > 0) {
      out.gbpsIdeal = cost.cioIdeal / record.ns;
      out.gbpsRequested = cost.cioRequested / record.ns;
    }
  }
  if (wave(waveCapacity, wg)) Object.assign(out, wave(waveCapacity, wg));
  return out;
}

function wave(capacity, wg) {
  if (!capacity || !wg) return null;
  const waves = wg / capacity;
  const full = Math.ceil(waves) * capacity;
  return {
    waves,
    // Fraction of the machine idle during the final partial wave. A dispatch of 4
    // workgroups on a device that holds 40 wastes 90% of the wave it occupies.
    waveWaste: 1 - wg / full,
    partialWave: waves < 1,
  };
}

/** Roll annotated dispatches up by kind and by name, weighted by measured time. */
export function summarise(annotated) {
  const total = annotated.reduce((s, r) => s + r.ns, 0);
  const by = (keyFn) => {
    const m = new Map();
    for (const r of annotated) {
      const k = keyFn(r);
      const e = m.get(k) || { key: k, ns: 0, count: 0, flops: 0, cioIdeal: 0 };
      e.ns += r.ns; e.count++; e.flops += r.flops || 0; e.cioIdeal += r.cioIdeal || 0;
      m.set(k, e);
    }
    return [...m.values()]
      .map((e) => ({ ...e, share: total ? e.ns / total : 0, gflops: e.ns ? e.flops / e.ns : 0 }))
      .sort((a, b) => b.ns - a.ns);
  };
  return { totalNs: total, byKind: by((r) => r.kind), byName: by((r) => r.name || r.kind) };
}
