// Four-way kernel taxonomy, time-weighted roofline attainment, and the Phase 0 gates.
//
// The report asks for every kernel to be sorted into compute-bound-and-efficient /
// memory-bound-at-roof / memory-bound-below-roof / launch-or-wave-limited, with the last
// two being "your entire opportunity", and for the aggregate to be weighted by duration
// rather than treating every kernel as one dot on a scatter plot.
//
// One deliberate deviation: a fifth bucket, `compute-bound-below-roof`. The report's four
// have no home for a kernel that sits above the ridge and still runs slowly, and folding
// it into memory-bound-below-roof would be simply wrong. It also happens to be where the
// confound this whole phase exists to isolate lives — a hand-written WGSL conv that is
// not CUTLASS lands here, and that is a statement about the kernel, not about the U-Net.
// Keeping it separate is what stops it being counted as architectural headroom.

export const CLASSES = [
  "launch-or-wave-limited",
  "memory-bound-below-roof",
  "memory-bound-at-roof",
  "compute-bound-below-roof",
  "compute-bound-efficient",
];

// Classes whose time is plausibly recoverable by restructuring rather than by writing a
// better kernel. Gate B is measured against these.
export const RECOVERABLE = new Set(["launch-or-wave-limited", "memory-bound-below-roof"]);

/**
 * @param annotated dispatches from cost.js annotate()
 * @param roofs     from roofs.js measureRoofs()
 * @param floor     from timing.js launchFloor()
 * @param atRoof    fraction of a roof that counts as "at" it
 */
export function classify(annotated, roofs, floor, { atRoof = 0.7 } = {}) {
  const computeRoof = roofs.sweep.peakGflops;      // swept plateau, not the FMA probe:
                                                   // same kernel shape across the sweep
  const bwRoof = roofs.bandwidth.gbps;
  const ridge = roofs.ridgeFromSweep ?? roofs.ridgeFromRatio;
  const floorNs = floor?.medianNs ?? 0;

  const rows = annotated.map((r) => {
    // Attainable roof for *this* kernel's intensity — the whole point of the report's
    // §2 objection to normalising against peak FLOP/s.
    const attainableGflops = r.ai != null
      ? Math.min(computeRoof, bwRoof * r.ai)
      : null;
    const attainment = attainableGflops && r.gflops != null
      ? r.gflops / attainableGflops
      : null;

    let cls;
    // Launch and wave limits are checked first and dominate: a dispatch that cannot fill
    // one wave is limited by that fact whatever its arithmetic intensity says.
    if (r.partialWave || (floorNs && r.ns < floorNs * 3)) {
      cls = "launch-or-wave-limited";
    } else if (r.ai == null) {
      cls = "unclassified";
    } else if (r.ai < ridge) {
      cls = (r.gbpsIdeal ?? 0) >= bwRoof * atRoof ? "memory-bound-at-roof" : "memory-bound-below-roof";
    } else {
      cls = (r.gflops ?? 0) >= computeRoof * atRoof ? "compute-bound-efficient" : "compute-bound-below-roof";
    }

    // Sub-diagnosis for anything below its roof: is the kernel starved of arithmetic
    // throughput, or is it saturating memory with traffic it did not need to generate?
    // A conv that re-reads its input once per output-channel block can be pinned to the
    // bandwidth roof at its *requested* intensity while its ideal intensity says it
    // should be comfortably compute-bound. That distinction decides whether the fix is
    // kernel blocking (Phase 1) or topology (Phase 3+), so it must not be collapsed.
    let cause = null;
    if (cls === "compute-bound-below-roof" || cls === "memory-bound-below-roof") {
      cause = (r.gbpsRequested ?? 0) >= bwRoof * atRoof
        ? "traffic-amplification"   // memory-bound as implemented, not as designed
        : "kernel-throughput";      // genuinely not issuing enough arithmetic
    }
    return { ...r, attainableGflops, attainment, class: cls, cause };
  });

  const totalNs = rows.reduce((s, r) => s + r.ns, 0);
  const buckets = {};
  for (const c of [...CLASSES, "unclassified"]) {
    const rs = rows.filter((r) => r.class === c);
    const ns = rs.reduce((s, r) => s + r.ns, 0);
    buckets[c] = { ns, share: totalNs ? ns / totalNs : 0, dispatches: rs.length };
  }

  // Time-weighted, attainable-roof-normalised attainment — the deliverable the report
  // says does not exist in the literature for a production segmentation U-Net.
  let wsum = 0, wns = 0;
  for (const r of rows) {
    if (r.attainment == null) continue;
    wsum += r.attainment * r.ns; wns += r.ns;
  }

  const recoverableNs = [...RECOVERABLE].reduce((s, c) => s + buckets[c].ns, 0);

  // Where the below-roof time actually goes, independent of which bucket it landed in.
  const causes = {};
  for (const c of ["traffic-amplification", "kernel-throughput"]) {
    const ns = rows.filter((r) => r.cause === c).reduce((s, r) => s + r.ns, 0);
    causes[c] = { ns, share: totalNs ? ns / totalNs : 0 };
  }

  return {
    rows, buckets, causes, totalNs,
    roofs: { computeGflops: computeRoof, bandwidthGbps: bwRoof, ridge, launchFloorNs: floorNs },
    timeWeightedAttainment: wns ? wsum / wns : null,
    coverage: totalNs ? wns / totalNs : 0,   // share of GPU time we could attribute at all
    recoverableShare: totalNs ? recoverableNs / totalNs : 0,
  };
}

/**
 * The two Phase 0 gates.
 *
 * Gate A is Amdahl: if the network's GPU time is a small slice of wall clock, no
 * architectural work on the network is justified yet, whatever the roofline says. On
 * this substrate that is a live possibility rather than a formality — the tile blending
 * and mask assembly in cellpose.js are single-threaded JS.
 *
 * Gate B is headroom: of the GPU time, how much sits in the classes that restructuring
 * could plausibly recover.
 *
 * Thresholds are arguments, not literals, and the caller is expected to have fixed them
 * before seeing the numbers.
 */
export function gates(stages, gpuBusyNs, classification, { gateA = 0.30, gateB = 0.40 } = {}) {
  const wallMs = stages.total ?? 0;
  const gpuMs = gpuBusyNs / 1e6;
  const gpuShare = wallMs ? gpuMs / wallMs : 0;

  // Everything the CPU did, and everything spent waiting on a GPU that wasn't busy.
  const cpuMs = ["preprocess", "pad", "tile_extract", "blend", "crop", "resize_back",
                 "dyn_setup", "getmasks"].reduce((s, k) => s + (stages[k] || 0), 0);
  const waitMs = (stages.forward_wait || 0) + (stages.dyn_wait || 0);
  const stallMs = Math.max(0, waitMs - gpuMs);

  return {
    wallMs, gpuBusyMs: gpuMs, cpuMs, stallMs,
    gpuShare, cpuShare: wallMs ? cpuMs / wallMs : 0, stallShare: wallMs ? stallMs / wallMs : 0,
    // The per-tile serialization price: time spent inside forward_wait that the GPU was
    // not actually computing. Tiles are independent, so this is recoverable by
    // pipelining alone — no architecture change required.
    tileStallMs: Math.max(0, (stages.forward_wait || 0) - gpuMs),
    tiles: stages.tiles ?? null,
    gateA: {
      threshold: gateA, value: gpuShare, pass: gpuShare >= gateA,
      verdict: gpuShare >= gateA
        ? "network GPU time is a meaningful share of wall clock — proceed to Gate B"
        : "network is not the bottleneck; the higher-value work is CPU-side or pipelining",
    },
    gateB: {
      threshold: gateB, value: classification.recoverableShare,
      pass: classification.recoverableShare >= gateB,
      verdict: classification.recoverableShare >= gateB
        ? "enough GPU time sits in recoverable classes to justify restructuring"
        : "little recoverable GPU time on this substrate — revisit scope before Phase 1",
    },
  };
}
