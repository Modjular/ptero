# Phase 0 — the accounting

Where U-Net inference time actually goes, measured on WebGPU. This is the gate the
throughput research programme has to pass before any architectural work starts.

Reproduce with `node tools/profile.mjs`; read the numbers at `profile.html`. Raw results
in `results/phase0-*.json`. Method notes live in the headers of `src/profile/*.js`.

## What was measured, and on what

Apple M5, 10 GPU cores, 32 GB unified, Chrome + Metal. Cellpose cyto3 through this repo's
hand-written WGSL engine, four workloads spanning the single-tile fast path (1 tile) and
the multi-tile blend path (6–30 tiles), medians of 7 runs, spread ±2.8–7.9%.

This is **not** the report's CUDA Phase 0 and does not inherit its 36% anchor, which came
from a PyTorch measurement on NVIDIA hardware. WebGPU exposes per-pass timestamps and
nothing else — no DRAM counters, no occupancy — so byte traffic and FLOPs are derived
analytically from the shaders, which is exact here only because every kernel in this repo
is hand-written. Ideal traffic is not actual traffic; the cache is invisible to us.

### The machine's roofs, measured rather than looked up

| | |
|---|---|
| compute roof (swept plateau) | **3108 GFLOP/s** |
| compute roof (independent FMA probe) | 2503 GFLOP/s |
| bandwidth | **104 GB/s** |
| ridge point | **32 FLOP/byte** (swept), 24 (ratio) |
| concurrent workgroup capacity | **10** — exactly the core count |
| launch floor (empty dispatch) | 1000 ns |

The two compute probes agree within 1.24×, which is the cross-check that the FMA loop
wasn't folded by the compiler; an earlier version failed it at 3×.

**The ridge point is the first result that matters.** At 32 FLOP/byte, against ~150 on an
A100 and ~295 on an H100, the report's §3.1 intensity barbell does not survive the move.
Its own table puts a C=16 stem at AI≈72 and a C=32 encoder level at ≈144 — both
comfortably *compute-bound* here. The memory-roof half of the barbell does not exist on
this machine, and nothing downstream should be read as though it does.

## The result

| workload | tiles | wall | GPU | CPU | stalled | attainment | Gate A | Gate B |
|---|---|---|---|---|---|---|---|---|
| single_tile 208² | 1 | 346 ms | 87% | 11% | 2.0% | 3.48% | PASS | **FAIL** 2.3% |
| composite 1280×960 d100 | 6 | 3069 ms | 60% | 38% | 1.5% | 3.46% | PASS | **FAIL** 2.4% |
| cellpose_020 881×1001 | 30 | 9806 ms | 92% | 6% | 1.4% | 3.49% | PASS | **FAIL** 2.3% |
| cellpose_020 d15 440² | 25 | 7955 ms | 95% | 3% | 2.4% | 3.49% | PASS | **FAIL** 2.4% |

**Gate A passes.** The GPU is busy 60–95% of wall clock, so the network is genuinely
where the time goes and the work is not misdirected — with one caveat below.

**Gate B fails, and not narrowly.** 2.3–2.4% of GPU time sits in the classes the report
calls "your entire opportunity", against a 40% threshold set in advance.

The reason is a single number: **time-weighted attainment is 3.5%**, and 97.6% of GPU
time is in dispatches that are compute-bound by intensity — well right of the ridge — yet
running at ~3.5% of the roof they could reach. That is not a topology problem. The
sub-diagnosis is unambiguous: **99.8% of below-roof time is `kernel-throughput`, 0.0% is
`traffic-amplification`.** The convolutions are not starved of bandwidth and are not
re-reading themselves into a memory bound. They are simply not issuing arithmetic.

### Three of the report's mechanisms do not fire here

The value of a measurement gate is that it can say no, and this one says no three times.

**Wave quantization — refuted on this substrate.** This is the report's "sharpest single
argument" (§3.2). It does not hold here. The bottleneck convolutions at 28×28×256 launch
**128 workgroups against a capacity of 10** — 12.8 full waves, 2% waste in the tail. Not
one convolution dispatch is partial-wave. The dispatches that *are* partial-wave
(`styleproj`, `gap`, `normstyle` — 420 of them, all tiny) account for **0.09% of GPU
time**. The 224px tile is small enough that the report's bottleneck-starvation argument
would need a much larger machine or a much smaller tile to bite.

**Non-convolutional operators — already solved here.** §3.3 predicts elementwise and
concat traffic rivalling the convolutions. In this engine BN scale/shift, ReLU, the skip
add and the residual add are all fused into the conv kernel already, and skips are
additive rather than concatenative. Everything that is not a convolution — pool, upsample,
GAP, style projection, normalisation, and the flow-dynamics kernel — comes to **0.19% of
GPU time combined**. `conv` is 99.81%.

**Per-tile serialization — real but negligible.** Reading `runNet` suggests an obvious
target: every tile is a separate `submit()` + `mapAsync()`, draining the pipeline between
tiles that are entirely independent. Measured, it costs 134 ms of 9806 ms (1.4%) on the
30-tile workload. The GPU work per tile is so large that the drain amortises away.
Worth knowing before spending a week on tile pipelining.

Tile quantization does register: the 28×28 bottleneck wastes **23%** of every 16×16
spatial tile it launches. Real, second-order, and dwarfed by the 28× implementation gap.

### It is not specific to cyto3

StarDist, a different U-shaped network with a **separately written** conv kernel of the
same 16×16 / BLK=8 shape, lands in the same place: **4.7% attainment, 95.9% of GPU time
compute-bound-below-roof**. The finding is about this family of hand-written kernels, not
about one model.

### The one place Amdahl bites

`composite` is the exception worth flagging. At diameter 100 the network runs at 0.3×
rescale while the flow dynamics and mask assembly run at full 1280×960, so `getmasks` —
single-threaded JS — takes **1124 ms of 3069 ms, 37% of wall clock**, more than half of
what the GPU spends. This is the WebGPU form of the report's §4.7 warning that Cellpose
post-processing can dominate inference. Large-diameter workloads have a CPU problem
before they have a GPU one.

## Decision

**Do not proceed to architectural work (H1, H2, H3).** Gate B fails by a factor of ~17,
and it fails for a reason that would invalidate the measurements anyway: with the
convolution running at 3.5% of its attainable roof, implementation loss (~28×) is more
than an order of magnitude larger than any architectural effect being hunted. Restructure
the U on this substrate today and the result would be buried in kernel noise. This is
precisely the confound Phase 0 exists to detect, and it detected it.

**Go to Phase 1 — the configuration and implementation space — with the conv kernel as
the sole target.** It is 99.8% of GPU time; nothing else is worth touching. Phase 1's
exit condition should be a conv that is a meaningful fraction of the 3108 GFLOP/s roof.
Only then does re-running Phase 0 produce numbers in which architecture is visible.

Reading the kernel, three untested hypotheses for where the throughput goes, in the order
I would test them:

1. `acc` is a dynamically-indexed `array<f32, BLK>` in the inner loop. Dynamic indexing of
   a private array commonly forces it out of registers into thread-local memory, which is
   device-backed. If that is happening, it alone would explain the result.
2. One shared-memory read per FMA (`ws[j * KK + k]`), with no register blocking in the
   spatial dimension, caps throughput at shared-memory bandwidth rather than ALU rate.
3. `subgroups` and `chromium-experimental-subgroup-matrix` — Metal simdgroup matmul — are
   both available on this adapter and entirely unused.

None of these is architecture. All of them are Phase 1.

## What this does and does not say about the report

It does not refute the report's hypotheses for well-optimised stacks. cuDNN on an A100 is
a different regime, and the 36% measured there remains the interesting number. What Phase
0 establishes is narrower and firmer: **on WebGPU/M5, the binding constraint is the
convolution kernel, and the U-Net's shape is not measurable until that is fixed.** The
ridge-point shift (32 vs 150 vs 295) means the barbell argument was never going to
transfer here unchanged, and the wave-quantization argument demonstrably does not.

## Caveats

- Ideal traffic ≠ actual traffic. Unified memory and a large system-level cache are
  invisible to this method; the `amplification` column shows what each kernel *requests*
  (4–14× for the convolutions) but not what the cache absorbed.
- Attainment is measured against a hand-written kernel. That is the finding, not a flaw,
  but it means these numbers describe this engine, not U-Nets in general.
- Single machine, single browser. Pin the Chrome version — a browser update silently
  invalidates comparisons.
- Passively cooled laptop; run-to-run spread is reported per workload and reached 7.9% on
  the shortest one.
- `tools/test_shims.py` has one pre-existing failure (`normalize -> ~[0,1] float32`),
  present before this work and unrelated to it.
