"""Build any user-wired DAG into a PyTorch model.

Lego principle: conv/linear layers infer their input size from whatever flows
into them (a dry-run "trace" pass), so users can stack blocks freely without
declaring channel/dim bookkeeping.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from graph import Edge, Node, canon, expand_classifier_edges, is_legacy_classifier, normalize_nodes, validate_dynamic

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
            raise ValueError(
                f"Block '{nid}' has {len(tensors)} incoming wires with different shapes "
                f"{sorted(shapes)} — branches joining a block must match (or add Linear/Pool to align them)"
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
            raise ValueError(
                f"Linear block '{nid}' got a {x.dim()}D tensor {list(x.shape)} — "
                "add a Flatten block before Linear"
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
    if final.shape[1] != num_classes:
        last_linears = [i for i in model.order_ids if model.kinds[i] == "linear"]
        hint = (
            f" (last Linear outputs {final.shape[1]} — set its out_features to {num_classes})"
            if last_linears else " (add a Linear block with out_features="
            f"{num_classes} before output)"
        )
        raise ValueError(
            f"Architecture outputs {final.shape[1]} features but the dataset has "
            f"{num_classes} classes{hint}"
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
    specs = [
        {"id": nid, "kind": model.kinds[nid], "params": model.params[nid],
         **model.last_shapes.get(nid, {})}
        for nid in model.order_ids if model.kinds[nid] not in ("input", "output")
    ]
    return model, cfg, specs


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
