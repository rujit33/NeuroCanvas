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

STAGES = ("schema", "structural", "semantic", "params", "trace", "dataset")


class ValidationError(ValueError):
    """ValueError with structured context.

    Backward-compat: str() is the plain human message, so existing
    ``HTTPException(400, str(e))`` / ``j.detail`` clients keep working.
    New clients can read ``to_dict()`` for stage/node/op details.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str = "",
        node_id: str = "",
        op: str = "",
        expected: str = "",
        got: str = "",
        reason: str = "",
        hint: str = "",
    ):
        super().__init__(message)
        self.stage = stage
        self.node_id = node_id
        self.op = op
        self.expected = expected
        self.got = got
        self.reason = reason
        self.hint = hint

    def to_dict(self) -> Dict[str, str]:
        return {
            "stage": self.stage,
            "node_id": self.node_id,
            "op": self.op,
            "expected": self.expected,
            "got": self.got,
            "reason": self.reason,
            "hint": self.hint,
            "message": str(self),
        }


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
    """Dynamic validation: single input/output, all blocks on the data path.

    Covers schema (node counts/kinds) + structural (duplicate wires, bad
    refs via order_graph, self-loops, reachability) stages.
    """
    if not graph.nodes:
        raise ValidationError(
            "Graph is empty — drag blocks onto the canvas", stage="schema")
    seen_ids: Dict[str, int] = {}
    for n in graph.nodes:
        seen_ids[n.id] = seen_ids.get(n.id, 0) + 1
    dup_ids = sorted(i for i, c in seen_ids.items() if c > 1)
    if dup_ids:
        raise ValidationError(
            f"Duplicate block id(s): {', '.join(dup_ids)} — block ids must be unique",
            stage="schema", node_id=dup_ids[0], reason="duplicate block id",
            hint="Delete or rename the duplicated block",
        )
    kinds = [canon(n.type) for n in graph.nodes]
    if kinds.count("input") != 1:
        raise ValidationError(
            f"Need exactly one input block (found {kinds.count('input')})",
            stage="schema", op="input", expected="exactly 1 input block",
            got=str(kinds.count("input")),
            hint="Add one input block (or delete extras)",
        )
    if kinds.count("output") != 1:
        raise ValidationError(
            f"Need exactly one output block (found {kinds.count('output')})",
            stage="schema", op="output", expected="exactly 1 output block",
            got=str(kinds.count("output")),
            hint="Add one output block (or delete extras)",
        )
    for n in graph.nodes:
        k = canon(n.type)
        if k not in COMPUTE_KINDS | {"input", "output"}:
            raise ValidationError(
                f"Unknown block '{n.type}' (id={n.id}). "
                "Available: input, conv, activation, pool, flatten, linear, dropout, output",
                stage="schema", node_id=n.id, op=str(n.type),
                expected="input | conv | activation | pool | flatten | linear | dropout | output",
                got=str(n.type),
            )
    seen_wires: Dict[tuple, str] = {}
    for e in graph.edges:
        key = (e.source, e.target)
        if key in seen_wires:
            raise ValidationError(
                f"Duplicate wire '{e.source}' -> '{e.target}' — remove one of the duplicate wires",
                stage="structural", node_id=e.target, op="edge",
                expected="at most one wire per source -> target",
                got=f"duplicate wire {e.source} -> {e.target}",
                reason="duplicate edge", hint="Delete one of the duplicate wires",
            )
        seen_wires[key] = e.id
    try:
        ordered = order_graph(graph)
    except ValidationError:
        raise
    except ValueError as e:
        msg = str(e)
        nid = ""
        if "wired to itself" in msg:
            nid = msg.split("'")[1] if "'" in msg else ""
        raise ValidationError(msg, stage="structural", node_id=nid, op="edge",
                              reason="cycle | self-loop | unknown ref",
                              hint="Data must flow one way (input -> ... -> output); "
                                   "remove loops and fix dangling wires")
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
        raise ValidationError(
            "These blocks are not on the data path input -> ... -> output: "
            + ", ".join(orphans) + ". Wire every block between input and output.",
            stage="structural", reason="block not on input -> output path",
            hint="Wire every block between input and output",
        )
    if not pred[exit]:
        raise ValidationError("Output block has no incoming wire",
                              stage="structural", node_id=exit, op="output",
                              hint="Wire a block into output")
    if not succ[entry]:
        raise ValidationError("Input block has no outgoing wire",
                              stage="structural", node_id=entry, op="input",
                              hint="Wire input into the first block")
    return ordered


# ---- semantic (rank) + params checks: one static pass, no torch ----

_FOUR_D = "4D [B,C,H,W]"
_TWO_D = "2D [B,F]"

_ACT_KINDS = {"relu", "sigmoid", "tanh", "leaky_relu", "gelu", "softmax"}
_POOL_KINDS = {"max", "avg", "adaptive"}


def _int_param(params: Dict[str, Any], nid: str, op: str, *keys: str,
               default: int, minimum: int, label: str) -> int:
    raw = default
    for k in keys:
        if k in params and params[k] is not None:
            raw = params[k]
            break
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{op} block '{nid}': {label} must be an integer (got {raw!r})",
            stage="params", node_id=nid, op=op,
            expected=f"{label} integer", got=repr(raw),
            hint=f"Set {label} to a valid integer",
        )
    if val < minimum:
        raise ValidationError(
            f"{op} block '{nid}': {label} must be "
            f"{'>= 0' if minimum == 0 else '> 0'} (got {val})",
            stage="params", node_id=nid, op=op,
            expected=f"{label} {'>= 0' if minimum == 0 else '> 0'}",
            got=str(val), hint=f"Set {label} to a valid value",
        )
    return val


def _float_param(params: Dict[str, Any], nid: str, op: str, key: str,
                 default: float, lo: float, hi: float | None, label: str) -> float:
    raw = params.get(key, default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{op} block '{nid}': {label} must be a number (got {raw!r})",
            stage="params", node_id=nid, op=op,
            expected=f"{label} number", got=repr(raw),
        )
    if val < lo or (hi is not None and val >= hi):
        raise ValidationError(
            f"{op} block '{nid}': {label} must satisfy {lo} <= {label} < {hi} (got {val})",
            stage="params", node_id=nid, op=op,
            expected=f"{lo} <= {label} < {hi}", got=str(val),
            hint=f"Set {label} within range",
        )
    return val


def check_semantic_ranks(ordered: List[Node], edges: List[Edge]):
    """Static rank + param validation in a single pass (no torch).

    Ranks — input:4D, conv:4D->4D, pool:4D->4D, flatten:4D->2D,
    linear:2D->2D, activation/dropout:passthrough, output:2D.
    Enforces the flatten boundary (once 2D, never back to 4D);
    conv->conv stays valid. Returns (ranks, warnings).
    """
    kinds = {n.id: canon(n.type) for n in ordered}
    preds: Dict[str, List[str]] = {n.id: [] for n in ordered}
    for e in edges:
        if e.target in preds and e.source in kinds:
            preds[e.target].append(e.source)
    ranks: Dict[str, int | None] = {}
    warnings: List[Dict[str, str]] = []

    for n in ordered:
        nid, op = n.id, kinds[n.id]
        p = n.params or {}
        pre = preds[n.id]
        pre_kinds = sorted({kinds[i] for i in pre if i in kinds})
        in_rank: int | None = None
        if op == "input":
            in_rank = 4
        elif pre:
            pre_ranks = {ranks[i] for i in pre if i in ranks}
            pre_ranks.discard(None)
            if len(pre_ranks) == 1:
                in_rank = next(iter(pre_ranks))
            # mixed/unknown branch ranks: defer to the trace-stage merge check

        # ---- semantic rank rules ----
        if op == "input":
            ranks[nid] = 4
        elif op in ("conv", "pool"):
            if in_rank == 2:
                src = "/".join(pre_kinds) or "flatten/linear"
                raise ValidationError(
                    f"{op.capitalize()} block '{nid}' expects {_FOUR_D} but got "
                    f"{_TWO_D} — flatten boundary: once flattened, data cannot "
                    f"flow back into {op} (incoming from {src})",
                    stage="semantic", node_id=nid, op=op,
                    expected=_FOUR_D, got=_TWO_D,
                    reason=f"incoming {src} output is 2D",
                    hint=f"Move this {op} block before Flatten "
                         f"(after {pre[0] if pre else 'input'}) or drop the Flatten",
                )
            ranks[nid] = 4
        elif op == "flatten":
            if in_rank == 2 and any(kinds.get(i) == "flatten" for i in pre):
                warnings.append({"node_id": nid, "code": "flatten_flatten",
                                 "msg": f"Flatten block '{nid}' follows another Flatten — redundant, safe to delete"})
            ranks[nid] = 2
        elif op == "linear":
            if in_rank == 4:
                src = "/".join(pre_kinds) or "input/conv"
                raise ValidationError(
                    f"Linear block '{nid}' expects {_TWO_D} but got {_FOUR_D} "
                    f"(incoming from {src}) — add a Flatten block before Linear",
                    stage="semantic", node_id=nid, op=op,
                    expected=_TWO_D, got=_FOUR_D,
                    reason=f"incoming {src} output is 4D",
                    hint=f"Add Flatten between {pre[0] if pre else 'input'} and '{nid}'",
                )
            ranks[nid] = 2
        elif op in ("activation", "dropout"):
            if op == "activation" and any(kinds.get(i) == "activation" for i in pre):
                warnings.append({"node_id": nid, "code": "act_act",
                                 "msg": f"Activation block '{nid}' follows another activation — usually redundant"})
            ranks[nid] = in_rank
        elif op == "output":
            if in_rank == 4:
                raise ValidationError(
                    f"Output block '{nid}' expects 2D logits [B, classes] but got "
                    f"{_FOUR_D} — add Flatten + Linear(out_features=<classes>) before output",
                    stage="semantic", node_id=nid, op=op,
                    expected="2D [B, classes]", got=_FOUR_D,
                    reason="graph never flattens to per-class logits",
                    hint="Add Flatten, then Linear with out_features = class count, before output",
                )
            ranks[nid] = 2
        else:
            ranks[nid] = in_rank

        # ---- param rules (same loop, no silent clamps) ----
        if op == "input":
            _int_param(p, nid, op, "image_size", default=28, minimum=1, label="image_size")
            _int_param(p, nid, op, "batch_size", default=64, minimum=1, label="batch_size")
            _int_param(p, nid, op, "in_channels", default=1, minimum=1, label="in_channels")
        elif op == "conv":
            _int_param(p, nid, op, "out_channels", default=16, minimum=1, label="out_channels")
            _int_param(p, nid, op, "kernel_size", "kernel", default=3, minimum=1, label="kernel_size")
            _int_param(p, nid, op, "stride", default=1, minimum=1, label="stride")
            _int_param(p, nid, op, "padding", default=1, minimum=0, label="padding")
        elif op == "pool":
            kind = str(p.get("pool", p.get("type", "max"))).lower()
            if kind not in _POOL_KINDS:
                raise ValidationError(
                    f"Pool block '{nid}': unknown pool '{kind}' — choose max | avg | adaptive",
                    stage="params", node_id=nid, op=op,
                    expected="max | avg | adaptive", got=kind)
            if kind == "adaptive":
                _int_param(p, nid, op, "adaptive_size", default=4, minimum=1, label="adaptive_size")
            else:
                ks = _int_param(p, nid, op, "kernel_size", "kernel", default=2, minimum=1, label="kernel_size")
                _int_param(p, nid, op, "stride", default=ks, minimum=1, label="stride")
        elif op == "linear":
            _int_param(p, nid, op, "out_features", default=64, minimum=1, label="out_features")
        elif op == "dropout":
            pv = _float_param(p, nid, op, "p", default=0.5, lo=0.0, hi=1.0, label="p")
            if pv == 0:
                warnings.append({"node_id": nid, "code": "dropout_zero",
                                 "msg": f"Dropout block '{nid}' has p=0 — no-op, safe to delete"})
        elif op == "activation":
            t = str(p.get("type", "relu")).lower()
            if t not in _ACT_KINDS:
                raise ValidationError(
                    f"Activation block '{nid}': unknown activation '{t}'",
                    stage="params", node_id=nid, op=op,
                    expected=f"one of {sorted(_ACT_KINDS)}", got=t)
        elif op == "output":
            loss = str(p.get("loss", "cross_entropy")).lower()
            if loss not in ("cross_entropy", "ce"):
                raise ValidationError(
                    f"Output block '{nid}': unsupported loss '{loss}' (available: cross_entropy)",
                    stage="params", node_id=nid, op=op,
                    expected="cross_entropy", got=loss)
            opt = str(p.get("optimizer", "adam")).lower()
            if opt not in ("adam", "sgd"):
                raise ValidationError(
                    f"Output block '{nid}': unsupported optimizer '{opt}' (available: adam | sgd)",
                    stage="params", node_id=nid, op=op,
                    expected="adam | sgd", got=opt)
            try:
                lr = float(p.get("lr", 1e-3))
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Output block '{nid}': lr must be a number (got {p.get('lr')!r})",
                    stage="params", node_id=nid, op=op,
                    expected="lr number", got=repr(p.get("lr")))
            if not lr > 0:
                raise ValidationError(
                    f"Output block '{nid}': lr must be > 0 (got {p.get('lr')})",
                    stage="params", node_id=nid, op=op,
                    expected="lr > 0", got=str(p.get("lr")))

    return ranks, warnings
