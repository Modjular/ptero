# vendor/webgpu-cellseg

The three segmentation engines and their measurement tooling, vendored from
`webgpu-cellseg`. **Do not edit anything under this directory in ptero.** Changes go
upstream and come back through a re-sync, which is the point: kernel optimisation has its
own review, its own reference tests, and its own history there rather than churning
ptero's.

Vendored from commit `6d696a1` (branch `perf/conv-kernel-and-flow-qc`).

## What is here, and what is deliberately not

Vendored: `src/{cellpose,stardist,instanseg}.js` (the engines), `src/conv-kernel.js` (the
generated conv shader), and `src/device.js`. That is the runtime and nothing else.

Not vendored:

- **The measurement library and its drivers** (`src/profile/`, `tools/convbench.mjs`,
  the profiling page) and **`docs/PHASE0.md` / `docs/PHASE1.md`**, which explain why the
  kernel's constants have the values they do. All of it lives upstream, because that is
  where the reference fixtures and the fidelity harnesses are and therefore where
  performance work can actually be validated. Profiling ptero's own pipeline would mean
  maintaining a second driver against a moving library; if you need it, do the work
  upstream and re-sync.

Also not vendored, because ptero supplies its own:

- **Weights.** `registry.js` passes an explicit base URL, so the engines load from
  ptero's `weights/` and the ~47 MB is not duplicated.
- **The shared device.** ptero's `src/gpu.js` memoises one `GPUDevice` across all three
  engines, because the notebook can reach for any of them on one page; upstream's
  `src/device.js` requests a device per model, which is right for a page that uses one.
  `registry.js` passes `{ device }` into `load()` so the upstream path is bypassed.
  **Both must request the same limits** — in particular `maxComputeWorkgroupStorageSize`
  at the adapter maximum, without which the conv kernel fails pipeline creation outright
  (WebGPU's 16 KB default is below what it needs). If you touch `gpu.js`, check it
  against `vendor/webgpu-cellseg/src/device.js`.
- **Tests.** Upstream has the PyTorch reference dumps and the fidelity harnesses
  (AP@0.5 against reference masks). That is where engine correctness is established;
  ptero's `tools/drive.mjs` checks the integration, not the numerics.

## Re-syncing

```bash
cd ../webgpu-cellseg && git pull        # or check out the commit you want
cd ../ptero
cp ../webgpu-cellseg/src/{cellpose,stardist,instanseg,conv-kernel,device}.js vendor/webgpu-cellseg/src/
# update the commit hash above, then:
node tools/drive.mjs --stage demo/images/Composite.tif    # expect 190 / 183 / 173
```

If a re-sync changes segmentation output, that is a bug upstream — every engine change
there is held to unchanged mask counts and AP@0.5 = 1.000 against the reference.

## Performance, for context

Against the pre-optimisation kernel, measured on an Apple M5 (upstream `docs/PHASE1.md`
has the method and the full waterfall):

| | |
|---|---|
| Cellpose end-to-end | ~10× |
| Cellpose conv kernel | ~11×, 3.5% → 40% of the machine's attainable roof |
| Cellpose flow-consistency QC | ~8× (moved to GPU) |
| StarDist forward | 3.0× |
| InstanSeg forward | 5.2× |

The conv kernel's four constants (`BLK`, `RBY`/`RBX`, `CB`) are a **joint** optimum,
interior on every axis — raising any one of them loses, sometimes by half. They are not
tunable by inspection; re-run `tools/convbench.mjs` upstream.
