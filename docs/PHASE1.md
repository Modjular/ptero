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

## Phase 1c — tuning the conv's three budgets jointly

With the flow QC fixed, `conv` was back to 95.7% of GPU time on `cellpose_020` and the
per-shape diagnostics said something specific: the K=3 shapes (86% of conv time) were
`kernel-throughput` at 29–48% with requested bandwidth only 34–66% of roof — so *not*
traffic-bound even counting re-reads. The `traffic-amplification` flag the previous
section pointed at turned out to come almost entirely from the K=1 shapes, which are ~5%
of conv time. Chasing it would have been chasing the wrong thing.

What did matter was that attainment fell as channels grew — 46.7% at 32ch, 29.4% at 256ch
— and that the 256ch shape is the single largest line item. Three parameters control that,
and they cannot be tuned separately because they trade against two shared budgets:

| | divides | costs |
|---|---|---|
| `BLK` output channels per workgroup | activation re-reads from global memory | registers |
| `RBY`×`RBX` output pixels per thread | shared-memory weight reads per FMA | registers |
| `CB` input channels staged per barrier | barriers (512 per workgroup at Cin=256) | threadgroup memory |

Swept as a grid, holding the register product fixed so the *split* is what varies:

| config | GFLOP/s | attainment | vs shipping |
|---|---|---|---|
| BLK=8, 2×2, CB=1 (previous) | 1028 | 33.5% | 1.00× |
| BLK=16, 2×2, CB=1 | 1153 | 37.6% | 1.12× |
| BLK=16, 2×2, CB=2 | 1189 | 38.7% | 1.16× |
| **BLK=16, 2×2, CB=4** | **1227** | **40.0%** | **1.19×** |
| BLK=32, 1×2, CB=2 | 1190 | 38.8% | 1.16× |
| BLK=32, 2×2, CB=2 | 503 | 16.4% | 0.49× |
| BLK=64, 1×1, CB=1 | 644 | 20.8% | 0.63× |

Both budgets bite hard and in opposite directions, which is why the optimum is interior on
every axis: past 64 accumulators the register file gives out (BLK=32 at 2×2 loses *half*
its throughput), and past CB=4 the threadgroup allocation exceeds even the raised limit.

Two supporting changes:

- **`maxComputeWorkgroupStorageSize` is now requested at the adapter's maximum.** The
  WebGPU default is 16 KB; this adapter offers 32 KB. CB=4 at 2×2 needs ~21 KB, so the
  default was silently capping the sweep. It is a ceiling, not an allocation, and a kernel
  that exceeds it fails pipeline creation loudly rather than degrading — nothing to lose.
- The benchmark now reports unbuildable variants instead of crashing on them. Pipeline
  limits fail through *asynchronous* validation, so `createComputePipeline` returns an
  invalid pipeline rather than throwing and the first symptom was an unrelated
  `getBindGroupLayout` error. It is caught with an error scope now.

**This step is not bit-exact**, and that is a deliberate break from the rest of Phase 1.
Staging CB input channels reorders the accumulation over `ci`, so results differ by
~5e-7 relative — f32 summation-order noise, given f32 epsilon is 1.2e-7 and the reduction
runs over 256 channels. For scale, that is ~500× smaller than the f16 error rejected
above, and unlike f16 it is not a loss of precision but a different, equally valid order.
Verified end-to-end rather than on the tolerance alone: 190 / 183 / 173 unchanged, demo
pages unchanged (179 / 278 / 240 masks).

Per-shape config selection was measured and rejected: BLK=16/2×2/CB=4 wins on 5 of 7
shapes, and picking the best per shape would add only ~1.7% for the complexity of
compiling and dispatching several pipelines per kernel size.

End to end this is 1.06×, much less than the kernel's 1.19× — because conv is no longer
99.8% of wall clock. That gap *is* the result: the conv is into diminishing returns, and
the next real lever on it is `subgroup-matrix`, not more tuning.

| workload | original | before 1c | after 1c | total |
|---|---|---|---|---|
| single_tile | 346 ms | 48 ms | 44 ms | **7.92×** |
| composite | 3069 ms | 366 ms | 354 ms | **8.66×** |
| cellpose_020 | 9806 ms | 1048 ms | 982 ms | **9.98×** |
| cellpose_020 d15 | 7955 ms | 850 ms | 798 ms | **9.97×** |
| **total** | **21177 ms** | 2312 ms | **2178 ms** | **9.72×** |

Attainment is now 38–55% depending on workload.

## Next, in order

1. **StarDist and InstanSeg still carry the old kernel** and the old 4.5% attainment. The
   Phase 1 findings transfer directly — same 16×16 shape, same dynamically-indexed
   accumulator — so this is mechanical work with a known ~10× waiting at the end of it.
   Cheapest remaining win by a wide margin, and deferred only because the current focus is
   Cellpose.
2. **`subgroup-matrix`** — Metal simdgroup matmul, available on this adapter and still
   unused. Now the only substantial lever left on the conv: parameter tuning is exhausted
   (Phase 1c found an interior optimum on all three axes), and 40% of roof means ~2.5×
   still on the table. It is a real rewrite — implicit GEMM with im2col in shared memory —
   so it deserves its own phase.
3. **The flow-QC kernel dispatches over the whole image** regardless of how much of it is
   masked, which is why the nuclear channel only gained 1.5× against the cyto channel's
   8.2×. Bounding the dispatch to the union of mask bounding boxes, or compacting mask
   pixels into a dense list, would recover that.
4. **Re-test f16** properly, against reference agreement rather than a tolerance.
5. **`getmasks` is back to 26% of wall clock on `composite`** now that everything around it
   shrank again. The remainder after the flow QC is the seed-growth and label-assignment
   work, which is a different (and smaller) problem than the one Phase 1b solved.

Architectural work (H1/H2/H3 in the report) remains gated behind Gate B — still failing at
4–12% against a 40% threshold — and sits behind all five of these.

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
