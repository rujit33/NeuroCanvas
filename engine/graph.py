"""Graph schema + validation for dynamic architectures.

Any DAG is allowed: exactly one `input`, exactly one `output`, every other
block must lie on a path from input to output. Groups are a frontend-only
concept — the view flattens them before sending, so the engine always sees
a complete flat graph.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str
    type: str  # input | conv | activation | pool | flatten | linear | dropout | output
    params: Dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    id: str = ""
    source: str
    target: str


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


# Legacy PoC names -> current block names
ALIASES = {"cnn": "conv", "classifier": "linear"}

COMPUTE_KINDS = {"conv", "activation", "pool", "flatten", "linear", "dropout"}


def canon(kind: str) -> str:
    return ALIASES.get((kind or "").lower(), (kind or "").lower())


def is_legacy_classifier(kind: str) -> bool:
    return (kind or "").lower() == "classifier"


def order_graph(graph: Graph) -> List[Node]:
    """Topologically sort nodes following edges. Raises ValueError on cycles."""
    by_id = {n.id: n for n in graph.nodes}
    indeg = {n.id: 0 for n in graph.nodes}
    adj: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        if e.source not in by_id or e.target not in by_id:
            raise ValueError(f"Edge references unknown node: {e.source} -> {e.target}")
        if e.source == e.target:
            raise ValueError(f"Node '{e.source}' is wired to itself")
        adj[e.source].append(e.target)
        indeg[e.target] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    ordered_ids: List[str] = []
    while queue:
        nid = queue.pop(0)
        ordered_ids.append(nid)
        for m in adj[nid]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(ordered_ids) != len(graph.nodes):
        raise ValueError("Graph has a cycle — data must flow in one direction (input -> ... -> output)")
    return [by_id[nid] for nid in ordered_ids]


def normalize_nodes(nodes: List[Node], num_classes: int) -> List[Node]:
    """Apply aliases; expand legacy `classifier` into pool+flatten+linear(s),
    mirroring the original fixed head (AdaptiveAvgPool -> FC)."""
    out: List[Node] = []
    for n in nodes:
        if is_legacy_classifier(n.type):
            hidden = int(n.params.get("hidden_dim", 0) or 0)
            out.append(Node(id=f"{n.id}__pool", type="pool",
                            params={"pool": "adaptive", "adaptive_size": 4}))
            out.append(Node(id=f"{n.id}__flat", type="flatten", params={}))
            if hidden > 0:
                out.append(Node(id=f"{n.id}__fc1", type="linear", params={"out_features": hidden}))
                out.append(Node(id=f"{n.id}__act", type="activation", params={"type": "relu"}))
            out.append(Node(id=f"{n.id}__fc", type="linear", params={"out_features": num_classes}))
        else:
            out.append(Node(id=n.id, type=canon(n.type), params=dict(n.params)))
    return out


def _classifier_chain(nid: str, hidden: int) -> List[str]:
    chain = [f"{nid}__pool", f"{nid}__flat"]
    if hidden > 0:
        chain += [f"{nid}__fc1", f"{nid}__act"]
    return chain + [f"{nid}__fc"]


def expand_classifier_edges(nodes: List[Node], edges: List[Edge]) -> List[Edge]:
    """Rewire edges that touched a legacy classifier through its expansion chain."""
    chains: Dict[str, List[str]] = {}
    for n in nodes:
        if n.id.endswith("__fc"):
            base = n.id[: -len("__fc")]
            hid = 1 if f"{base}__fc1" in {m.id for m in nodes} else 0
            chains[base] = _classifier_chain(base, hid)
    new_edges: List[Edge] = []
    for e in edges:
        s, t = e.source, e.target
        s_new = chains[s][-1] if s in chains else s
        t_new = chains[t][0] if t in chains else t
        new_edges.append(Edge(id=e.id, source=s_new, target=t_new))
    for base, chain in chains.items():
        for a, b in zip(chain, chain[1:]):
            new_edges.append(Edge(id=f"{base}__link_{a}_{b}", source=a, target=b))
    return new_edges


def validate_dynamic(graph: Graph) -> List[Node]:
    """Dynamic validation: single input/output, all blocks on the data path."""
    if not graph.nodes:
        raise ValueError("Graph is empty — drag blocks onto the canvas")
    kinds = [canon(n.type) for n in graph.nodes]
    if kinds.count("input") != 1:
        raise ValueError(f"Need exactly one input block (found {kinds.count('input')})")
    if kinds.count("output") != 1:
        raise ValueError(f"Need exactly one output block (found {kinds.count('output')})")
    for n in graph.nodes:
        k = canon(n.type)
        if k not in COMPUTE_KINDS | {"input", "output"}:
            raise ValueError(
                f"Unknown block '{n.type}' (id={n.id}). "
                "Available: input, conv, activation, pool, flatten, linear, dropout, output"
            )
    ordered = order_graph(graph)
    by_id = {n.id: n for n in graph.nodes}
    succ: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        succ[e.source].append(e.target)
    entry = next(n.id for n in graph.nodes if canon(n.type) == "input")
    exit = next(n.id for n in graph.nodes if canon(n.type) == "output")

    # forward reachability from input
    seen, stack = set(), [entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(succ[cur])
    # backward reachability to output
    pred: Dict[str, List[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        pred[e.target].append(e.source)
    back, stack = set(), [exit]
    while stack:
        cur = stack.pop()
        if cur in back:
            continue
        back.add(cur)
        stack.extend(pred[cur])

    orphans = [f"{by_id[n].type}({n})" for n in by_id if n not in seen or n not in back]
    if orphans:
        raise ValueError(
            "These blocks are not on the data path input -> ... -> output: "
            + ", ".join(orphans) + ". Wire every block between input and output."
        )
    if not pred[exit]:
        raise ValueError("Output block has no incoming wire")
    if not succ[entry]:
        raise ValueError("Input block has no outgoing wire")
    return ordered
