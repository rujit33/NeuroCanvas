"""Export any user-wired graph to clean standalone .py / .ipynb code.

Strategy: build + trace the graph server-side to resolve all lazy dims
(channels, flattened features), then emit code with concrete numbers.
Pure chains become nn.Sequential; branched DAGs get a small executor.
"""
from __future__ import annotations

import json
from typing import Dict, List

from datasets import resolve_num_classes
from graph import Edge, Node, canon, expand_classifier_edges, is_legacy_classifier, normalize_nodes
from model_builder import build_model


def _act_code(t: str) -> str:
    return {
        "relu": "nn.ReLU()", "sigmoid": "nn.Sigmoid()", "tanh": "nn.Tanh()",
        "leaky_relu": "nn.LeakyReLU()", "gelu": "nn.GELU()",
        "softmax": "nn.Softmax(dim=1)",
    }.get((t or "relu").lower(), "nn.ReLU()")


def _layer_code(spec: Dict) -> str:
    p, k = spec["params"], spec["kind"]
    if k == "conv":
        ci, co = spec["in"][1], spec["out"][1]
        ks = int(p.get("kernel_size", p.get("kernel", 3)))
        return f"nn.Conv2d({ci}, {co}, kernel_size={ks}, stride={int(p.get('stride', 1))}, padding={int(p.get('padding', 1))})"
    if k == "activation":
        return _act_code(str(p.get("type", "relu")))
    if k == "pool":
        kind = str(p.get("pool", p.get("type", "max"))).lower()
        if kind == "adaptive":
            return f"nn.AdaptiveAvgPool2d(({int(p.get('adaptive_size', 4))}, {int(p.get('adaptive_size', 4))}))"
        kk = int(p.get("kernel_size", p.get("kernel", 2)))
        st = int(p.get("stride", kk))
        return (f"nn.MaxPool2d({kk}, {st})" if kind == "max" else f"nn.AvgPool2d({kk}, {st})")
    if k == "flatten":
        return "nn.Flatten()"
    if k == "linear":
        return f"nn.Linear({spec['in'][1]}, {spec['out'][1]})"
    if k == "dropout":
        return f"nn.Dropout({float(p.get('p', 0.5))})"
    return f"# unknown block {k}"


def _is_chain(order_ids: List[str], preds: Dict[str, List[str]]) -> bool:
    return all(len(preds[i]) <= 1 for i in order_ids)


def generate_python(nodes: List[Node], edges: List[Edge], epochs: int = 5) -> str:
    inp = next((n for n in nodes if canon(n.type) == "input"), None)
    num_classes = resolve_num_classes(
        str((inp.params.get("dataset", "synthetic")) if inp else "synthetic"),
        str(inp.params.get("dataset_path", "") if inp else ""),
        nodes,
    )
    model, cfg, specs = build_model(nodes, edges, num_classes)
    nodes_n = normalize_nodes(nodes, num_classes)
    edges_n = (expand_classifier_edges(nodes_n, edges)
               if any(is_legacy_classifier(n.type) for n in nodes) else edges)
    n_params = sum(p.numel() for p in model.parameters())
    preds: Dict[str, List[str]] = {s["id"]: [] for s in specs}
    preds[model.entry] = []
    preds[model.exit] = []
    for e in edges_n:
        preds.setdefault(e.target, []).append(e.source)

    chain = _is_chain(list(preds), preds)
    layers = "\n".join(f"        {_layer_code(s)}," for s in specs)
    arch_comment = " -> ".join([model.kinds[i] for i in model.order_ids])
    optim_line = (
        "    opt = torch.optim.Adam(model.parameters(), lr=LR)"
        if cfg["optimizer"] == "adam" else
        "    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)"
    )

    if chain:
        model_code = f'''class VisualNet(nn.Module):
    """{arch_comment} ({n_params} params)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
{layers}
        )

    def forward(self, x):
        return self.net(x)'''
    else:
        layer_defs = "\n".join(f"        self.l_{i} = {_layer_code(s)}" for i, s in enumerate(specs))
        fwd_lines = []
        for i, s in enumerate(specs):
            pre = [e.source for e in edges_n if e.target == s["id"]]
            if len(pre) == 1 and pre[0] == model.entry:
                fwd_lines.append(f"        v_{i} = self.l_{i}(x)")
            elif not pre:
                fwd_lines.append(f"        v_{i} = self.l_{i}(x)")
            else:
                terms = " + ".join(f"vals['{p}']" for p in pre)
                fwd_lines.append(f"        v_{i} = self.l_{i}({terms})")
            fwd_lines.append(f"        vals['{s['id']}'] = v_{i}")
        fwd = "\n".join(fwd_lines)
        model_code = f'''class VisualNet(nn.Module):
    """Branched DAG: {arch_comment} ({n_params} params)."""
    def __init__(self):
        super().__init__()
{layer_defs}

    def forward(self, x):
        vals = {{'{model.entry}': x}}
{fwd}
        return v_{len(specs) - 1}'''

    return f'''"""Exported from VisualML -- architecture: {arch_comment}."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- config (mirrors your visual graph) ----
IMAGE_SIZE = {cfg["image_size"]}
IN_CHANNELS = {cfg["in_channels"]}
NUM_CLASSES = {num_classes}
BATCH_SIZE = {cfg["batch_size"]}
EPOCHS = {epochs}
LR = {cfg["lr"]}


{model_code}


def demo_data(n=2000):
    X = torch.randn(n, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    s = X.mean(dim=(1, 2, 3))
    y = ((s - s.min()) / (s.max() - s.min() + 1e-6) * NUM_CLASSES).long().clamp(0, NUM_CLASSES - 1)
    return TensorDataset(X, y)


def main():
    torch.manual_seed(0)
    model = VisualNet()
    loader = DataLoader(demo_data(), batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()  # loss: {cfg["loss"]}
{optim_line}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for X, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(X)
        print(f"epoch {{epoch}}/{{EPOCHS}} loss={{total / len(loader.dataset):.4f}}")
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


if __name__ == "__main__":
    main()
'''


def generate_notebook(nodes: List[Node], edges: List[Edge], epochs: int = 5) -> dict:
    code = generate_python(nodes, edges, epochs)
    cells = [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# VisualML Export\n", "Run each cell top to bottom.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": [l + "\n" for l in code.splitlines()]},
    ]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": cells}


def notebook_to_json(nb: dict) -> str:
    return json.dumps(nb, indent=1)
