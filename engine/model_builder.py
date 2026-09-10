"""Build any user-wired DAG into a PyTorch model.

Lego principle: conv/linear layers infer their input size from whatever flows
into them (a dry-run "trace" pass), so users can stack blocks freely without
declaring channel/dim bookkeeping.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from graph import Edge, Node, ValidationError, canon, check_semantic_ranks, expand_classifier_edges, is_legacy_classifier, normalize_nodes, validate_dynamic

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "sigmoid": lambda: nn.Sigmoid(),
    "tanh": lambda: nn.Tanh(),
    "leaky_relu": lambda: nn.LeakyReLU(),
    "gelu": lambda: nn.GELU(),
    "softmax": lambda: nn.Softmax(dim=1),
}


def _act(name: str) -> nn.Module:
    fn = ACTIVATIONS.get((name or "relu").lower())
    if fn is None:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(ACTIVATIONS)}")
    return fn()


def _pool(params: Dict) -> nn.Module:
    kind = str(params.get("pool", params.get("type", "max"))).lower()
    k = int(params.get("kernel_size", params.get("kernel", 2)))
    s = int(params.get("stride", k))
    if kind == "adaptive":
        return nn.AdaptiveAvgPool2d(int(params.get("adaptive_size", 4)))
    if kind == "avg":
        return nn.AvgPool2d(k, s)
    if kind == "max":
        return nn.MaxPool2d(k, s)
    raise ValueError(f"Unknown pool '{kind}' (block id params). Choose max | avg | adaptive")


class GraphExecutor(nn.Module):
    """Executes the wired DAG in topological order; lazy conv/linear creation."""

    def __init__(self, ordered: List[Node], edges: List[Edge]):
        super().__init__()
        self.order_ids = [n.id for n in ordered]
        self.kinds = {n.id: canon(n.type) for n in ordered}
        self.params = {n.id: dict(n.params) for n in ordered}
        self.preds: Dict[str, List[str]] = {n.id: [] for n in ordered}
        for e in edges:
            self.preds[e.target].append(e.source)
        self.entry = next(i for i in self.order_ids if self.kinds[i] == "input")
        self.exit = next(i for i in self.order_ids if self.kinds[i] == "output")

        self.key_of = {nid: f"m{idx}" for idx, nid in enumerate(self.order_ids)}
        self.created = nn.ModuleDict()  # lazy conv / linear with real weights
        self.fixed: Dict[str, nn.Module] = {}
        for nid in self.order_ids:
            k = self.kinds[nid]
            if k == "activation":
                self.fixed[nid] = _act(str(self.params[nid].get("type", "relu")))
            elif k == "pool":
                self.fixed[nid] = _pool(self.params[nid])
        self.last_shapes: Dict[str, Dict[str, List[int]]] = {}

    def _merge(self, tensors: List[torch.Tensor], nid: str) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise ValidationError(
                f"Block '{nid}' merges {len(tensors)} branches by element-add, "
                f"which requires identical shapes — got branch shapes "
                f"{sorted(list(s) for s in shapes)} "
                "(add Linear/Pool/Flatten on a branch to align them)",
                stage="trace", node_id=nid, op=self.kinds.get(nid, ""),
                expected="identical branch shapes for element-add",
                got=str(sorted(list(s) for s in shapes)),
                reason="branches joining a block have different shapes",
                hint="Align branch shapes (Linear/Pool/Flatten) before they join",
            )
        return sum(tensors)

    def _conv(self, nid: str, x: torch.Tensor) -> nn.Module:
        key = self.key_of[nid]
        if key not in self.created:
            p = self.params[nid]
            self.created[key] = nn.Conv2d(
                x.shape[1], int(p.get("out_channels", 16)),
                kernel_size=int(p.get("kernel_size", p.get("kernel", 3))),
                stride=int(p.get("stride", 1)), padding=int(p.get("padding", 1)),
            )
        return self.created[key]

    def _linear(self, nid: str, x: torch.Tensor) -> nn.Module:
        if x.dim() != 2:
            raise ValidationError(
                f"Linear block '{nid}' expects 2D input [B, F] but got a "
                f"{x.dim()}D tensor {list(x.shape)} — "
                "add a Flatten block before Linear",
                stage="trace", node_id=nid, op="linear",
                expected="2D [B, F]", got=f"{x.dim()}D {list(x.shape)}",
                reason="linear layers only accept flattened features",
                hint=f"Add Flatten before '{nid}'",
            )
        key = self.key_of[nid]
        if key not in self.created:
            self.created[key] = nn.Linear(x.shape[1], int(self.params[nid].get("out_features", 64)))
        return self.created[key]

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        vals: Dict[str, torch.Tensor] = {}
        for nid in self.order_ids:
            k = self.kinds[nid]
            if k == "input":
                vals[nid] = x
                continue
            if k == "output":
                vals[nid] = self._merge([vals[p] for p in self.preds[nid]], nid)
                continue
            t = self._merge([vals[p] for p in self.preds[nid]], nid)
            in_shape = list(t.shape)
            try:
                if k == "conv":
                    t = self._conv(nid, t)(t)
                elif k == "activation":
                    t = self.fixed[nid](t)
                elif k == "pool":
                    t = self.fixed[nid](t)
                elif k == "flatten":
                    t = t.flatten(1)
                elif k == "linear":
                    t = self._linear(nid, t)(t)
                elif k == "dropout":
                    t = nn.functional.dropout(t, p=float(self.params[nid].get("p", 0.5)), training=self.training)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"{k.capitalize()} block '{nid}' failed on input shape "
                    f"{in_shape}: {e}",
                    stage="trace", node_id=nid, op=k,
                    expected="valid spatial/feature dims for this block",
                    got=str(in_shape), reason=str(e),
                    hint="Check kernel/stride/padding vs. the incoming feature size",
                ) from e
            if trace:
                self.last_shapes[nid] = {"in": in_shape, "out": list(t.shape)}
            vals[nid] = t
        return vals[self.exit]


def _first(nodes: List[Node], kind: str) -> Node:
    for n in nodes:
        if canon(n.type) == kind:
            return n
    raise ValueError(f"Missing block '{kind}'")


def build_model(
    raw_nodes: List[Node], raw_edges: List[Edge], num_classes: int
) -> Tuple[GraphExecutor, Dict, List[Dict]]:
    """Build + trace. Returns (model, train_cfg, specs per block)."""
    nodes = normalize_nodes(raw_nodes, num_classes)
    # rewire legacy classifier edges if expansion happened
    from graph import Graph as _G

    edges = expand_classifier_edges(nodes, raw_edges) if any(
        is_legacy_classifier(n.type) for n in raw_nodes
    ) else raw_edges
    ordered = validate_dynamic(_G(nodes=nodes, edges=edges))

    inp = _first(ordered, "input").params
    outp = _first(ordered, "output").params
    dataset = str(inp.get("dataset", "synthetic")).lower()
    in_channels = int(inp.get("in_channels", 1 if dataset in ("mnist", "synthetic") else 3))
    image_size = int(inp.get("image_size", 28))

    model = GraphExecutor(ordered, edges)
    dummy = torch.randn(2, in_channels, image_size, image_size)
    model.eval()
    final = model(dummy, trace=True)
    if final.dim() != 2:
        raise ValidationError(
            f"Output block expects 2D logits [B, classes] but got "
            f"{final.dim()}D {list(final.shape)} — end the graph with "
            "Flatten + Linear before output (no softmax: CrossEntropyLoss applies it)",
            stage="trace", node_id=model.exit, op="output",
            expected="2D [B, classes]", got=f"{final.dim()}D {list(final.shape)}",
            reason="graph output is not per-class logits",
            hint="Add Flatten, then Linear with out_features = class count, before output",
        )
    if final.shape[1] != num_classes:
        last_linears = [i for i in model.order_ids if model.kinds[i] == "linear"]
        hint = (
            f" (last Linear outputs {final.shape[1]} — set its out_features to {num_classes})"
            if last_linears else " (add a Linear block with out_features="
            f"{num_classes} before output)"
        )
        raise ValidationError(
            f"Architecture outputs {final.shape[1]} features but the dataset has "
            f"{num_classes} classes (Expected {num_classes}, Actual {final.shape[1]})"
            f"{hint}",
            stage="trace",
            node_id=last_linears[-1] if last_linears else model.exit,
            op="linear" if last_linears else "output",
            expected=str(num_classes), got=str(final.shape[1]),
            reason="final Linear out_features != dataset class count",
            hint=f"Set the last Linear out_features to {num_classes}",
        )
    model.train()

    cfg = {
        "loss": str(outp.get("loss", "cross_entropy")).lower(),
        "optimizer": str(outp.get("optimizer", "adam")).lower(),
        "lr": float(outp.get("lr", 1e-3)),
        "dataset": dataset,
        "dataset_path": str(inp.get("dataset_path", "")),
        "image_size": image_size,
        "batch_size": int(inp.get("batch_size", 64)),
        "in_channels": in_channels,
        "num_classes": num_classes,
    }
    specs = []
    for nid in model.order_ids:
        if model.kinds[nid] in ("input", "output"):
            continue
        key = model.key_of[nid]
        mod = model.created[key] if key in model.created else None
        n_params = sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        specs.append({"id": nid, "kind": model.kinds[nid], "params": model.params[nid],
                      "num_params": n_params,
                      **model.last_shapes.get(nid, {})})
    return model, cfg, specs


def validate_pipeline(raw_nodes: List[Node], raw_edges: List[Edge]) -> Dict:
    """Staged validation shared by /api/validate, /api/train and training.

    Stages: schema -> structural -> semantic -> params -> trace -> dataset.
    Returns {"model", "cfg", "specs", "warnings", "order", "num_classes",
    "total_params"}. Raises ValidationError (a ValueError, so str(e) stays
    backward-compatible) carrying stage/node_id/op/expected/got/reason/hint.
    """
    from datasets import resolve_num_classes
    from graph import Graph as _G

    # schema + structural (on the raw graph)
    ordered_raw = validate_dynamic(_G(nodes=raw_nodes, edges=raw_edges))

    # dataset stage: class count before building
    try:
        inp = next(n for n in ordered_raw if canon(n.type) == "input").params
        num_classes = resolve_num_classes(
            str(inp.get("dataset", "synthetic")),
            str(inp.get("dataset_path", "")),
            ordered_raw,
        )
    except ValidationError:
        raise
    except ValueError as e:
        raise ValidationError(str(e), stage="dataset", op="input",
                              reason="dataset configuration invalid",
                              hint="Check dataset / dataset_path on the input block") from e

    # normalize legacy blocks, re-check structure on the expanded graph
    nodes_n = normalize_nodes(raw_nodes, num_classes)
    edges_n = (expand_classifier_edges(nodes_n, raw_edges)
               if any(is_legacy_classifier(n.type) for n in raw_nodes)
               else raw_edges)
    ordered = validate_dynamic(_G(nodes=nodes_n, edges=edges_n))

    # semantic + params (one static pass, no torch)
    _, warnings = check_semantic_ranks(ordered, edges_n)

    # trace stage only from here (build_model dry-runs the graph)
    model, cfg, specs = build_model(raw_nodes, raw_edges, num_classes)

    for s in specs:
        out = s.get("out") or []
        if s["kind"] in ("conv", "pool") and len(out) == 4 and out[2] == 1 and out[3] == 1:
            warnings.append({"node_id": s["id"], "code": "resolution_1x1",
                             "msg": f"Block '{s['id']}' outputs 1x1 spatial resolution {out} — "
                                    "further conv/pool blocks can no longer learn spatial features"})
        if s["kind"] == "activation" and str(s["params"].get("type", "")).lower() == "softmax":
            warnings.append({"node_id": s["id"], "code": "softmax_logits",
                             "msg": f"Softmax block '{s['id']}': CrossEntropyLoss expects raw logits — "
                                    "a Softmax here will distort training"})

    return {"model": model, "cfg": cfg, "specs": specs, "warnings": warnings,
            "order": ordered, "num_classes": num_classes,
            "total_params": count_params(model)}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_loss(name: str):
    name = (name or "cross_entropy").lower()
    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss()
    if name in ("nll", "nll_loss"):
        return nn.NLLLoss()
    raise ValueError(f"Unsupported loss '{name}' (available: cross_entropy)")


def make_optimizer(name: str, params, lr: float):
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{name}' (available: adam | sgd)")
