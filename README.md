# ptero

Bio-image segmentation and analysis that runs entirely in a browser tab. Three
microscopy segmentation models ported to WebGPU (no PyTorch, no TensorFlow,
no CUDA, no ML framework at inference) plus a Python notebook running the real
scientific stack under Pyodide, plus an agent that writes the analysis code for you.

Your images never leave your machine.

## Run it

```bash
python3 -m http.server 8765
open http://localhost:8765/
```

> [!IMPORTANT]
> Chrome or Edge (or Safari 18+) for WebGPU is required. `file://` won't work: ES modules
and the filesystem mount both need a real origin.

## What's here

| | |
|---|---|
| **`notebook.html`** | numpy · pandas · scikit-image · matplotlib in the tab, with all four model checkpoints callable from Python under their real library names. Point it at a folder on your disk. |
| **`demo/*.html`** | One standalone interactive demo per model, with threshold sliders. |

### Models

| Model | Good for | Size |
|---|---|---|
| `cellpose-cyto3` | generalist cells and cytoplasm | 26 MB |
| `stardist-fluo` | fluorescent nuclei (DAPI/Hoechst) | 5.7 MB |
| `stardist-he` | H&E histology nuclei | 5.7 MB |
| `instanseg-brightfield` | brightfield / unstained nuclei | 15 MB |

Weights load lazily, per model, on first use — a session downloads only what it uses.
Cold start is about 5 MB.

## Writing Python

The models are exposed under their upstream names, so code from the real docs works:

```python
from stardist.models import StarDist2D
labels, details = StarDist2D.from_pretrained('2D_versatile_fluo').predict_instances(img)

from cellpose import models
masks, flows, styles = models.CellposeModel(model_type='cyto3').eval(img, diameter=100)

from instanseg import InstanSeg
labels, _ = InstanSeg('brightfield_nuclei').eval_small_image(rgb)
```

There are a handful of deliberate deviations (no torch tensors, no SizeModel, no
polygon output) — each is documented in the module's docstring, and
`docs/ARCHITECTURE.md` lists them all. To see what's installed at runtime:

```python
import ptero
ptero.models.list()
```

## The assistant

The pane on the right turns "count the nuclei in the DAPI channel" into notebook cells.
It picks the segmentation model, asks you about anything biological it can't determine,
tests its code in a hidden kernel first, and puts working cells in your notebook for you
to run. It never runs them for you.

Open ⚙ to choose a provider — **Anthropic** or **Google Gemini** — and paste a key.
Keys are stored in this browser only and sent straight from the page to the provider;
each provider's key and model choice are remembered separately, and "List models" asks
the provider what your key can actually reach rather than offering a stale list.

## Development

```bash
npm install
node tools/drive.mjs --stage demo/images/Composite.tif   # headless regression run
```

Architecture, design decisions and the gotchas worth not rediscovering:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Licence

The ported weights carry their upstream licences — see `weights/<model>/LICENSE` and
`NOTICE`.
