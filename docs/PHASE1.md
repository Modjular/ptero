# Phase 1 — exhausting the implementation space

Phase 0 ([PHASE0.md](PHASE0.md)) found `conv` at 99.8% of GPU time running at 3.5% of the
machine's attainable roof, with the loss attributed to arithmetic throughput rather than
bandwidth, occupancy or topology, and refused to open architectural work until that was
fixed. Phase 1 fixed it.

Reproduce with `node tools/convbench.mjs` (kernel variants) and `node tools/profile.mjs`
(end-to-end). Method in `src/profile/convbench.js`; the variants and what each tests in
`src/profile/conv-variants.js`.

## The waterfall

Each step is a single isolated change, measured on the real cyto3 layer shapes weighted
by their real share of conv time. Attainment is against a roof measured in the same
session (~3100 GFLOP/s).

| step | GFLOP/s | attainment | cumulative | what changed |
|---|---|---|---|---|
| baseline | 104 | 3.3% | 1.00× | the kernel Phase 0 measured |
| scalar accumulators | 532 | 17.1% | **5.15×** | H1 — no dynamically-indexed private array |
| + compile-time K | 733 | 23.6% | **7.08×** | nine taps unroll; weight addressing folds |
| + 2×2 register block | 1152 | 37.1% | **11.02×** | one weight read feeds four FMAs |
| ~~+ 2×4 register block~~ | 896 | 28.8% | 8.05× | rejected — register pressure costs more than it saves |
| ~~+ f16 shared memory~~ | 1419 | 45.7% | 13.53× | rejected on numerics — see below |

**H1 was the big one and Phase 0 called it.** The old kernel accumulated into
`var acc: array<f32, BLK>` indexed by a loop whose bound came from a uniform. A
dynamically-indexed private array does not stay in registers; on Metal it spills to
thread-local memory, which is device-backed. Replacing it with eight named scalars — no
other change, same arithmetic, same order — was **5.15×** on its own.

The remaining two are the classic GEMM levers: make the loop bounds compile-time so the
taps unroll, then block in registers so shared-memory reads amortise. The 2×4 result is
the useful negative: 64 live accumulators cost more occupancy than they save in traffic,
so the block size is a measured interior optimum, not something to maximise. This is
DeFiNES's "over-fusing is counterproductive" showing up one level down.

**The landed kernel is bit-identical to the one it replaces** — maxRel 0.0 against the
frozen reference on every layer shape. The FMA order per output element never changed;
only which thread performs it. That is why a 10× rewrite of the hottest kernel in a
weights-validated engine could be landed at all.

## End to end

| workload | wall before | wall after | speedup | attainment |
|---|---|---|---|---|
| single_tile 208² | 346 ms | 71 ms | 4.87× | 3.5% → 35.0% |
| composite 1280×960 d100 | 3069 ms | 1340 ms | 2.29× | 3.5% → 33.3% |
| cellpose_020 881×1001 | 9806 ms | 1427 ms | **6.87×** | 3.5% → 34.8% |
| cellpose_020 d15 440² | 7955 ms | 992 ms | **8.02×** | 3.5% → 35.1% |
| **total** | **21177 ms** | **3830 ms** | **5.53×** | |

Correctness held at every level: 190 cells / 183 nuclei / 173 kept unchanged, all three
demo pages unchanged (cellpose 9594 ms → 1445 ms for an identical 179 masks), shims
unaffected.

## What rejecting f16 cost, and why

f16 shared memory with f32 accumulation measured **13.53× and 45.7% attainment** — a
further 1.24× over what shipped. It was rejected because the output is not equivalent:
~2.8e-4 relative, ~1e-2 absolute error against the reference.

That is ordinary f16 rounding, not a bug, and in many settings it would be fine. Here the
flow field feeds a 200-iteration Euler integration whose trajectories decide instance
boundaries, and the engine's entire claim is that it reproduces the desktop reference. A
1e-2 absolute perturbation of a flow vector is not obviously safe under 200 integration
steps, and "the cell count happened to match on four images" is not evidence that it is.

This is worth revisiting deliberately: the right test is cell-count and IoU agreement
against the PyTorch reference across a real image set, not a tolerance on one conv. If it
passes that, 1.24× is there for the taking. It is logged, not lost.

## The gates, re-run

Gate B still **fails** — 5.9%, up from 2.3%, against a 40% threshold. Architectural work
is still not justified, and the reason is unchanged in kind: 92.6% of GPU time remains
`compute-bound-below-roof`. At 37% of roof there is still ~2.7× of kernel work available
before topology becomes the binding constraint.

But the composition of that shortfall has shifted in an informative way. Below-roof time
by cause was 99.8% `kernel-throughput` / 0.0% `traffic-amplification` before Phase 1; it
is now **75.8% / 21.9%**. As the arithmetic got faster, the kernel's own redundant traffic
— it re-reads the input once per output-channel block, 4–14× amplification — started to
matter. That is the signpost for Phase 1b.

**Gate A now fails on `composite`** (15.0% GPU, threshold 30%), and this is the headline
result. The bottleneck has left the GPU:

| workload | GPU | CPU | `getmasks` alone |
|---|---|---|---|
| single_tile | 42% | 49% | 46% |
| composite | **15%** | **83%** | **79%** |
| cellpose_020 | 64% | 33% | 28% |
| cellpose_020 d15 | 75% | 19% | 16% |

`getmasks` — the single-threaded JS that builds label maps from the integrated flow
trajectories: histogram, seed finding, 5-iteration dilation per seed, flow-error filter,
min-size filter, relabel — was 5% of wall clock before Phase 1 and is up to 79% now. It
did not get slower. Everything else got 11× faster around it.

This is exactly the Amdahl warning the report raises in §4.7 for Cellpose
post-processing, arriving one phase later than it predicted and on the CPU rather than
the GPU.

## Phase 1b — the flow-consistency QC

Acting on item 1 below, which the numbers above made unavoidable.

Sub-stage timing inside `getmasks` put **95% of it in one place**: `_maskFlowErrors`, the
flow-consistency check. Histogram, seed finding and seed growth were 13 ms combined
against 1097 ms for the flow reconstruction.

It works by reconstructing each mask's flow field — a 9-point diffusion from a heat source
at the mask's centre, run for `2*(ly+lx)` iterations over the mask's own footprint — and
comparing the gradient against the network's prediction. For 190 masks at ~100 px that is
on the order of a billion stencil evaluations in single-threaded JS.

Two changes:

| step | flow QC | what changed |
|---|---|---|
| before | 1097 ms | |
| precomputed flat indices | 852 ms | **1.29×**, bit-exact — nine taps off one base by addition instead of three multiplies and two typed-array loads per pixel per iteration |
| GPU diffusion | 101 ms | **8.2×** on top; every mask diffused at once over the whole image |

The global formulation is equivalent to the per-mask one given two details, both of which
are easy to get wrong and are commented in `FLOWDIFF_WGSL`:

- **Neighbours are label-masked.** Per-mask, positions outside the footprint stay zero
  forever. Globally, a neighbour belonging to a *different* mask holds that mask's heat,
  so it must read as zero — otherwise adjacent cells bleed into each other.
- **The heat source is folded into the read.** `T[med] += 1` at the top of each iteration
  accumulates into the field; adding 1 to the centre pixel's value as it is read is
  algebraically the same thing.

Each mask keeps its own iteration count and freezes individually once it reaches it, so
the dispatch loop runs to the maximum without changing any mask's result. The diffusion
runs in f32, but normalisation and the per-mask error stay on the CPU in f64 — the
threshold comparison that decides how many masks survive is never made on f32 sums.

**Verified by direct comparison, not by a tolerance:** both implementations were run on
identical copies of the same raw label map. cyto channel — 190 masks vs 190, **0 of
1,228,800 pixels differ**, 826 ms → 101 ms. Nuclear channel — 183 vs 183, **0 pixels
differ**, 82 ms → 56 ms. The nuclear channel gains less because its masks are small and
the kernel still dispatches over the whole image regardless.

## Where it ended up

| workload | original | after conv | after flow QC | total |
|---|---|---|---|---|
| single_tile 208² | 346 ms | 71 ms | 48 ms | **7.24×** |
| composite 1280×960 d100 | 3069 ms | 1340 ms | 366 ms | **8.38×** |
| cellpose_020 881×1001 | 9806 ms | 1427 ms | 1048 ms | **9.36×** |
| cellpose_020 d15 440² | 7955 ms | 992 ms | 850 ms | **9.36×** |
| **total** | **21177 ms** | 3830 ms | **2312 ms** | **9.16×** |

**Gate A passes on all four workloads again** — `composite` went from 15.0% GPU back to
68.2%, and `getmasks` from 79% of wall clock to 28%. The bottleneck is back on the GPU,
which is where the remaining levers are.

Correctness unchanged throughout: 190 / 183 / 173, all three demo pages, shims.

## Next, in order

1. **StarDist and InstanSeg still carry the old kernel** and the old 4.6% attainment. The
   Phase 1 findings transfer directly — same 16×16 / BLK=8 shape, same dynamically-indexed
   accumulator — so this is mechanical work with a known ~10× waiting at the end of it.
   Cheapest remaining win by a wide margin.
2. **Traffic amplification in the conv.** 12–22% of below-roof time and rising as the
   kernel improves. The fix is standard: block over input channels so the activation tile
   is loaded once per output-channel group rather than once per group per channel.
3. **The flow-QC kernel dispatches over the whole image** regardless of how much of it is
   masked, which is why the nuclear channel only gained 1.5× against the cyto channel's
   8.2×. Bounding the dispatch to the union of mask bounding boxes, or compacting mask
   pixels into a dense list, would recover that.
4. **`subgroup-matrix`** — Metal simdgroup matmul, available on this adapter and still
   unused. This is the lever that would take the conv from 37% toward the roof, and it is
   a real rewrite (implicit GEMM with im2col in shared memory), so it deserves its own
   phase rather than being squeezed in here.
5. **Re-test f16** properly, against reference agreement rather than a tolerance.

Architectural work (H1/H2/H3 in the report) remains gated behind Gate B and is now fifth
in line behind four cheaper, better-evidenced targets.

## Caveats

- On `cellpose_020_diam15` the pass-overlap guard fired: median sum/span 1.00 but max
  2.00, meaning at least one command encoder had compute passes running concurrently.
  With the faster kernel, dispatches are short enough that Metal can overlap adjacent
  passes. Per-dispatch attribution on that workload is weaker than on the other three;
  the aggregate and the wall-clock numbers are unaffected.
- Attainment is against a roof measured per session, and the machine is a passively
  cooled laptop. Run-to-run spread is reported per workload (2.8–7.9%).
- The 11.02× kernel figure is time-share-weighted across the real layer shapes; per-shape
  speedups range 10.3–12.2×.
