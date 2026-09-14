# ptero

Run Cellpose, Stardist, and Instanseg in the browser inside a real Python notebook. There are many better, more powerful, more performant solutions in the cloud, but ptero is completely local.

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

Weights load lazily, per model, on first use. Cold start is about 5 MB.

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

It's not a perfect 1:1 port. The minor deviations are documented in the module's docstring, and
`docs/ARCHITECTURE.md` lists them all. To see what's installed at runtime:

```python
import ptero
ptero.models.list()
```

## Optional AI Assistance

Bring your own provider. Compatible with Anthropic and any OpenAI compliant spec (so third party or Ollama endpoints work too). 

Open ⚙ to choose a provider. **Keys are stored in this browser only and sent straight from the page to the provider**. You can even open the devtools to confirm this. Each provider's key and model choice are only stored via `localStorage`, and "List models" directly asks the provider what your key can access.

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
