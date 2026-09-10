# VisualML — Build Neural Networks Visually

A modular, visual workbench for creating neural networks: drag blocks (Input, Conv,
Activation, Pool, Flatten, Linear, Dropout, Output) onto a canvas, wire them together,
configure them in an inspector, press **Train**, and watch live loss curves stream back.
Save the result (`.pt` / `.pth` / `.pkl`, weights-only or full checkpoint), export the
whole architecture as a clean standalone `.py` / `.ipynb` file, or save the canvas itself
as a `visualML v1 JSON` graph (`visualml-graph.json`) — which can be re-imported later or
produced by an AI assistant from the in-app format guide. PyTorch is the compute
engine, React is the view.

```
visualML/
├── engine/   # FastAPI + PyTorch backend  (port 8000)
└── view/     # React + Vite + React Flow frontend (port 5173)
```

---

## 1) Overall idea

### What this project does

VisualML turns neural-network construction into a **LEGO-like visual activity**. Instead of
writing PyTorch boilerplate, the user:

1. **Drags blocks** from a searchable palette onto a canvas (unlimited instances of any block).
2. **Wires them** with edges that represent tensor flow, in any DAG shape — chains, branches,
   skip connections.
3. **Configures** each block in an inspector (channels, kernels, activations, batch size,
   dataset path, learning rate, …).
4. **Groups** blocks into named, collapsible sub-assemblies (e.g. “Encoder”, “Classifier head”)
   that appear as one tidy card at the root but can be double-clicked open — including nested
   groups — so large architectures stay readable.
5. **Trains in the UI**: sets epochs, picks save format (`.pt` / `.pth` / `.pkl`) and save mode
   (weights-only vs full training state), presses **▶ Train**, and watches per-epoch
   train/val loss, accuracy, and log lines stream in live.
6. **Takes the model with them**: downloads the trained file, or exports the architecture as a
   clean, runnable `model.py` / `model.ipynb` with all dimensions resolved to concrete numbers.
7. **Saves and reuses graphs as JSON**: exports the full canvas (blocks, positions, groups,
   wires) to `visualml-graph.json` and re-imports it later — see §2.6.
8. **Builds graphs with AI help**: presses the **i** guide button, copies the format guide
   (+ per-block init snippets from the Available Nodes tab), asks any LLM for a graph, and
   imports the returned JSON — see §2.6.

### What it is trying to solve

- **Barrier to entry.** Building even a small CNN in PyTorch requires knowing datasets,
  dataloaders, shape bookkeeping (`in_features` after every flatten), training loops,
  checkpointing, and device handling. VisualML absorbs all of that: conv/linear layers
  **infer their input sizes automatically** from whatever flows into them, so users think in
  _blocks and data flow_, not tensor algebra.
- **Architecture comprehension.** Code hides structure; a graph _shows_ it. Grouping adds
  zoom levels (root overview → named sub-assembly → raw blocks), which is how engineers
  actually reason about big models.
- **Iteration speed.** Change a kernel size, rewire a branch, retrain, compare loss curves —
  without touching code. Validation errors name the exact block and problem
  (e.g. `Linear block 'fc1' got a 4D tensor … add a Flatten block before Linear`).
- **No lock-in.** Everything the UI knows can leave the UI: the trained weights _and_ a
  human-readable, dependency-light source export that runs anywhere PyTorch runs.

### Current scope (working prototype)

Image classification with a dynamic CNN/MLP DAG (the original “LLM” framing is the north
star; the engine abstraction — typed blocks + DAG executor — is what a transformer-block
extension would plug into). Single input, single output, CPU training, `cross_entropy`
loss, `adam`/`sgd`.

---

## 2) Architecture

### 2.1 Tech stack

| Layer    | Technology                                       | Role                                                    |
| -------- | ------------------------------------------------ | ------------------------------------------------------- |
| Engine   | **Python 3.10, PyTorch ≥ 2.0, FastAPI, Uvicorn** | Model building, datasets, training, export, persistence |
| View     | **React 19, TypeScript, Vite, React Flow v11**   | Drag-drop canvas, inspector, training dashboard         |
| Protocol | **REST (JSON) + WebSocket**                      | REST for control plane, WS for live training telemetry  |
| Storage  | Filesystem (`engine/runs/`, `engine/data/`)      | Job checkpoints, cached MNIST, uploaded datasets        |

No database, no auth, no GPU requirement, no `torchvision` dependency (image decoding is
done with Pillow + NumPy so the install stays lean).

### 2.2 Methodology — how a graph becomes a trained model

**a) The view sends a flat graph.** The canvas state is a node list and an edge list:

```json
{
  "nodes": [
    {
      "id": "conv1",
      "type": "conv",
      "params": {
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
      }
    }
  ],
  "edges": [{ "id": "e2", "source": "conv1", "target": "act1" }]
}
```

**b) The engine validates the DAG** (`engine/graph.py`).
Topological sort (Kahn's algorithm; cycles and self-loops rejected), then structural rules:
exactly one `input` and one `output`; every block must lie on a path from input to output
(forward + backward reachability; orphans are reported by `type(id)`); input/output must
have wires. Legacy PoC names still work (`cnn` → `conv`; `classifier` → expanded
`adaptive-pool → flatten → [linear → relu] → linear`, mirroring the original fixed head).

**c) Lazy shape inference builds the model** (`engine/model_builder.py`, `GraphExecutor`).
Conv and Linear layers are created _during a dry-run trace_ with a dummy batch
`(2, in_channels, image_size, image_size)`, reading the actual incoming tensor shape — so
users never declare `in_channels`/`in_features`. Execution is topological; a block with
several incoming wires merges by element-wise sum (shape mismatches fail with an explicit
message). A final check enforces `output_features == num_classes` and tells the user how
to fix it. The trace also records per-block `in → out` shapes, which power the Validate
response and the code exporter.

**d) Class count is resolved before building** (`engine/datasets.py::resolve_num_classes`):
`mnist` → 10; `imagefolder` → number of class subfolders; `synthetic` → `out_features` of
the last Linear block (the architecture defines its own demo problem).

**e) Data loading** (`engine/datasets.py`).

- `synthetic` — random images with a learnable label pattern (mean-pixel bins), zero setup.
- `mnist` — raw IDX files downloaded once to `engine/data/mnist` (offline → synthetic fallback).
- `imagefolder` — `<dataset_path>/<class>/*.png|jpg|jpeg|bmp|webp`, PIL-resized to
  `image_size`, 80/20 train/val split. `POST /api/upload` accepts a zip of class folders
  and returns the server path to paste into the input block.

**f) Training as a background job with WS fan-out** (`engine/trainer.py`).
`POST /api/train` creates a job id and schedules `run_training` as an `asyncio` task
(epochs of SGD/Adam over the loaders, train loss + val loss/acc per epoch). Each job owns a
set of subscriber queues; the `init`/`log`/`progress`/`ping`/`final` messages are pushed to
`WS /ws/jobs/{id}`. On completion the model is saved to `engine/runs/{job_id}/model.{pt,pth,pkl}`:
`weights_only` → `state_dict`; `full` → `{arch, train_cfg, graph, history, model_state,
optimizer_state}` (`.pkl` via `pickle`, `.pt`/`.pth` via `torch.save`).

**g) Export resolves everything to concrete numbers** (`engine/exporter.py`).
The server rebuilds + traces the graph, then emits code: pure chains become an
`nn.Sequential` model; branched DAGs become a small explicit executor. Both include config,
demo data, training loop, and save — verified to run standalone.

**h) Groups are view-only.** Grouping sets `parentId` membership on the master node list;
edges are _never rewritten_. The current scope renders direct children; wires crossing a
collapsed group render as computed dashed “portal” stubs. Train/Validate/Export serialize
the untouched master graph, so grouping cannot corrupt an architecture. Wiring _onto_ a
collapsed card auto-resolves when the group has a single entry/exit block, otherwise the UI
asks you to open the group and wire the exact block.

**i) The canvas itself persists as visualML v1 JSON** (flat React Flow state, distinct
from the slimmed-down engine graph in (a)).
`{version: 1, app: "visualML", exportedAt: <ISO>, nodes: [...], edges: [...]}` where each
node is `{id, type: "ml"|"group", position: {x, y}, parentId?, data: {kind, params, name?}}`
(`kind` is an open-ended block-type string; `parentId` preserves group membership) and
each edge is `{id, source, target}`. Import runs a structural check only
(`App.tsx::parseGraphFile`): top-level `nodes`/`edges` arrays, string ids, numeric
`position`, object `data`, non-empty `kind` for `type === "ml"` (missing `type` coerces
to `"ml"`). There is deliberately **no allowlist** — unknown future kinds load as-is
(the card falls back to Linear styling via `MlNode`). Failures surface as inline notices
(`Import failed: <reason>`), never a silent partial load.

### 2.3 Block reference

| Block        | Key params                                                                                           | Notes                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `input`      | `dataset` (synthetic\|mnist\|imagefolder), `dataset_path`, `image_size`, `batch_size`, `in_channels` | Exactly one per graph                                   |
| `conv`       | `out_channels`, `kernel_size`, `stride`, `padding`                                                   | `in_channels` inferred                                  |
| `activation` | `type` (relu\|sigmoid\|tanh\|leaky_relu\|gelu)                                              | —                                                       |
| `pool`       | `pool` (max\|avg\|adaptive), `kernel_size`, `stride`, `adaptive_size`                                | —                                                       |
| `flatten`    | —                                                                                                    | Required before `linear` on image tensors               |
| `linear`     | `out_features`                                                                                       | `in_features` inferred; last one must equal class count |
| `dropout`    | `p`                                                                                                  | Active only in training mode                            |
| `output`     | `loss` (cross_entropy), `optimizer` (adam\|sgd), `lr`                                                | Exactly one per graph                                   |

### 2.4 API endpoints (`engine/main.py`)

| Method | Path                   | Body / params                                                                   | Returns                                                                                        |
| ------ | ---------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| GET    | `/api/health`          | —                                                                               | `{ok, service}`                                                                                |
| POST   | `/api/validate`        | `{nodes, edges}`                                                                | `{ok, order, params, config, shapes}` or 400 with reason                                       |
| POST   | `/api/train`           | `{graph, epochs 1–100, save_format pt\|pth\|pkl, save_mode weights_only\|full}` | `{job_id, …status}` (job runs in background)                                                   |
| GET    | `/api/jobs`            | —                                                                               | All job summaries                                                                              |
| GET    | `/api/jobs/{id}`       | —                                                                               | Status, `current_epoch`, `history[]`, `logs[]`, `save_path`, `params`, `dataset_info`, `error` |
| WS     | `/ws/jobs/{id}`        | —                                                                               | `init` → `log`/`progress` stream → `final`; `ping` keepalive replays state                     |
| GET    | `/api/download/{id}`   | —                                                                               | The saved model file (404 until training finishes)                                             |
| POST   | `/api/export/python`   | `{graph, epochs}`, `?as_file=false` for raw text                                | `model.py` download / code                                                                     |
| POST   | `/api/export/notebook` | `{graph, epochs}`                                                               | `model.ipynb` download (markdown + code cells)                                                 |
| POST   | `/api/upload`          | multipart `file` (.zip), optional `target`                                      | `{path, uploads}` server-side dataset path                                                     |
| GET    | `/api/uploads`         | —                                                                               | Previously uploaded dataset paths                                                              |

### 2.5 View internals (`view/src/`)

- `App.tsx` — studio shell: master node/edge state, scope navigation (breadcrumb), grouping,
  portal-stub derivation (`useMemo`), train/export orchestration, top bar + notices.
  JSON persistence lives here too: `exportJson` (dumps full master state as v1 JSON),
  `onImportFile` (hidden file input → `parseGraphFile` → replaces master state, resets scope
  to root), and the `i` guide modal (`showGuide`/`guideTab`, `GUIDE_EXAMPLE`/`GUIDE_PROMPT`/
  `GUIDE_COPY`, see §2.6).
- `graph.ts` — block kinds, default params, inspector field specs, `JobState`/`WireInfo` types.
  `PALETTE` is the source of truth for available blocks; `KIND_META` (label/color/desc),
  `defaultParams()` (per-kind initial params; `{}` for unknowns), `FIELDS` (inspector specs),
  `canonKind()` (legacy `cnn`→`conv`, `classifier`→`linear` aliases).
- `api.ts` — `API`/`WS` base URLs, `serialize()` (drops group proxies, maps to
  `{id, type: kind, params}` + `{id, source, target}`), `postJSON()` (throws `detail` on
  error), `downloadBlob()` (POSTs to the engine and saves the response), `downloadText()`
  (saves client-side text — used for the JSON graph export).
- `components/MlNode.tsx` / `GroupNode.tsx` — canvas cards (params summary; name + member count).
- `components/ConfigPanel.tsx` — inspector: per-block fields, zip upload, group rename/open/ungroup,
  selected-wire deletion.
- `components/LogsPanel.tsx` — collapsible dock: SVG loss curve, epoch table, live log pane.

Two hard-won implementation notes: all React Flow callbacks/options are referentially stable
(`useCallback`/module constants) because RF v11’s `SelectionListener` re-fires its effect on
callback identity change — inline callbacks caused an infinite update loop (blank screen).
RF’s light-theme chrome (MiniMap, Controls) is force-overridden to dark in `App.css`.

### 2.6 Graph persistence & AI-assisted import (`App.tsx` + `api.ts`)

Top-bar buttons (left to right after the train controls): **Validate**, **▶ Train**,
**⇩ .py**, **⇩ .ipynb**, model download link (appears after training), **⇩ JSON**
(save canvas as `visualml-graph.json`), **⇧ Import** (load a v1 JSON graph),
**i** (format guide modal).

- **Format tab** — the contract, unchanged since introduction: schema text (top-level
  `{version, app, exportedAt, nodes, edges}`; node/edge fields; `kind` is an open-ended
  block-type string with a `params` object; unknown kinds load as-is), a tiny
  input→linear→output example (`GUIDE_EXAMPLE`), a one-line prompt template
  (`GUIDE_PROMPT`: "Create a visualML v1 JSON graph … Only output JSON"), and a
  **Copy guide** button (copies `GUIDE_COPY` to clipboard).
- **Available Nodes tab** — a catalog derived dynamically from `PALETTE` (no hardcoded
  kind list, so future blocks appear automatically). Per kind: the kind name, the
  one-line `KIND_META` description, and a minimal single-node init snippet
  `{"id":"<kind>-1","type":"ml","position":{"x":0,"y":0},"data":{"kind":"<kind>",
  "params":<defaultParams(kind)>}}`, rendered as a compact scrollable list of `<pre>`
  snippets. A **Copy all** button copies the whole catalog as text.
- **Import validation** (`parseGraphFile`): the file must parse as JSON with top-level
  `nodes`/`edges` arrays; every node needs a string `id`, a numeric
  `position: {x, y}`, and an object `data`; missing `type` coerces to `"ml"`;
  `type === "ml"` additionally requires a non-empty string `data.kind` (no PALETTE
  check); every edge needs string `id`/`source`/`target`. Success replaces the master
  nodes/edges, resets scope to root, and posts a notice with counts
  (`Imported <file> (N nodes, M edges)`); failure posts `Import failed: <reason>` and
  the input value is always cleared so the same file can be retried.
- **Round-trip workflow**: arrange canvas → `⇩ JSON` → modify or close → `⇧ Import` →
  identical canvas (positions and group membership preserved, unlike `.py`/`.ipynb`
  export which is architecture-only and one-way).
- **AI workflow**: `i` → copy guide (+ Copy all from Available Nodes) → paste into any
  LLM with a request ("Only output JSON") → save reply as `.json` → `⇧ Import` →
  **Validate** to check the engine accepts it.

### 2.7 Run it

```powershell
# engine (from engine/; use the Python that has torch/fastapi/uvicorn)
python -m uvicorn main:app --reload --port 8000   # -> http://127.0.0.1:8000

# view (from view/)
npm install
npm run dev                                      # -> http://localhost:5173
```

First run: leave the input block on `synthetic`, press **▶ Train** — no files needed.
`engine/requirements.txt` pins the backend deps (`fastapi, uvicorn, torch, numpy, pillow,
pydantic, python-multipart`).

### 2.8 Limits & next steps

CPU-only training; one input / one output per graph; branch merges require equal shapes;
`cross_entropy` loss only. Canvas persistence already exists (v1 JSON export/import, §2.6).
Natural extensions: transformer blocks (the executor already
supports arbitrary DAGs — new block types plug into `GraphExecutor` + `FIELDS` and then
appear automatically in the palette, the Available Nodes tab, and the importer), GPU
selection, run comparison, and a model zoo of shareable JSON graphs.

# Home screen preview

![preview](previews/home.png)

## Grouped model view

![preview](previews/grouped.png)
