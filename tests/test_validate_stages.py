"""Validation pipeline + round-trip + export regression tests.

Reuses the engine as-is (no new deps beyond stdlib/pytest/torch, no new node
types): graph.validate_dynamic/check_semantic_ranks/ValidationError,
model_builder.validate_pipeline/build_model, datasets.resolve_num_classes,
exporter.generate_python/generate_notebook. The pipeline entry point is
validate_pipeline — the same staged pipeline POST /api/validate and
POST /api/train run (schema -> structural -> semantic -> params -> trace).
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from datasets import resolve_num_classes  # noqa: E402
from exporter import generate_notebook, generate_python, notebook_to_json  # noqa: E402
import exporter as _exporter_mod  # noqa: E402  (hasattr probe for notebook support)
from graph import (  # noqa: E402
    Edge,
    Graph,
    Node,
    ValidationError,
    canon,
    validate_dynamic,
)
from model_builder import build_model, validate_pipeline  # noqa: E402

N_CLASSES = 10
IMG = 28


# ---------- minimal fixtures inline ----------
def N(nid, kind, **params):  # noqa: N802
    return Node(id=nid, type=kind, params=params)


def chain_edges(*ids):
    return [Edge(id=f"e{i}", source=a, target=b) for i, (a, b) in enumerate(zip(ids, ids[1:]))]


def input_node(nid="in", **kw):
    d = {"dataset": "synthetic", "image_size": IMG, "in_channels": 1, "batch_size": 8}
    d.update(kw)
    return N(nid, "input", **d)


def conv_node(nid="conv", out=16, **kw):
    d = {"out_channels": out, "kernel_size": 3, "stride": 1, "padding": 1}
    d.update(kw)
    return N(nid, "conv", **d)


def relu_node(nid="relu"):
    return N(nid, "activation", type="relu")


def pool_node(nid="pool"):
    return N(nid, "pool", pool="max", kernel_size=2, stride=2)


def flat_node(nid="flat"):
    return N(nid, "flatten")


def lin_node(nid="fc", out=N_CLASSES):
    return N(nid, "linear", out_features=out)


def out_node(nid="out", **kw):
    d = {"loss": "cross_entropy", "optimizer": "adam", "lr": 1e-3}
    d.update(kw)
    return N(nid, "output", **d)


def fwd_ok(model, n=N_CLASSES):
    model.eval()
    with torch.no_grad():
        assert list(model(torch.randn(2, 1, IMG, IMG)).shape) == [2, n]


# ---------- representative valid graphs ----------
def g_simple():
    nodes = [input_node(), conv_node(), relu_node(), pool_node(), flat_node(),
             lin_node(), out_node()]
    return nodes, chain_edges(*[n.id for n in nodes])


def g_cnn():
    nodes = [input_node(), conv_node("c1"), relu_node("r1"), pool_node("p1"),
             conv_node("c2", out=32), relu_node("r2"), pool_node("p2"),
             flat_node(), lin_node("fc1", out=64), relu_node("r3"),
             lin_node("fc2"), out_node()]
    return nodes, chain_edges(*[n.id for n in nodes])


def g_branched():
    nodes = [input_node(), conv_node("ca", out=8), conv_node("cb", out=8),
             relu_node("m"), pool_node(), flat_node(), lin_node(), out_node()]
    edges = [Edge(id="e0", source="in", target="ca"), Edge(id="e1", source="in", target="cb"),
             Edge(id="e2", source="ca", target="m"), Edge(id="e3", source="cb", target="m"),
             *chain_edges("m", "pool", "flat", "fc", "out")]
    return nodes, edges


def g_conv_conv():
    nodes = [input_node(), conv_node("c1"), conv_node("c2"), flat_node(),
             lin_node(), out_node()]
    return nodes, chain_edges(*[n.id for n in nodes])


# ---------- valid: full pipeline ok + same forward semantics ----------
def test_valid_simple_chain():
    res = validate_pipeline(*g_simple())
    assert res["num_classes"] == N_CLASSES
    fwd_ok(res["model"])


def test_valid_cnn_deep():
    fwd_ok(validate_pipeline(*g_cnn())["model"])


def test_valid_branched_merge_identical_shapes():
    fwd_ok(validate_pipeline(*g_branched())["model"])


def test_valid_conv_conv():
    fwd_ok(validate_pipeline(*g_conv_conv())["model"])


def test_resolve_num_classes_reuse():
    nodes, edges = g_cnn()
    assert resolve_num_classes("mnist", "", validate_dynamic(Graph(nodes=nodes, edges=edges))) == 10
    assert resolve_num_classes("synthetic", "", g_simple()[0]) == N_CLASSES
    assert build_model(*g_simple(), N_CLASSES)[1]["num_classes"] == N_CLASSES


# ---------- invalid: each asserts node + reason via structured ValidationError ----------
def test_invalid_linear_before_flatten():
    nodes = [input_node(), lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in", "fc", "out"))
    assert (e.value.stage, e.value.node_id) == ("semantic", "fc")
    assert "flatten" in e.value.reason.lower() or "flatten" in str(e.value).lower()


def test_invalid_flatten_then_conv():
    # mnist input: class count resolves without Linear, exact chain preserved
    nodes = [input_node(dataset="mnist"), flat_node(), conv_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in", "flat", "conv", "out"))
    assert (e.value.stage, e.value.node_id) == ("semantic", "conv")
    assert "flatten boundary" in str(e.value)


def test_invalid_conv_then_linear_no_flatten():
    nodes = [input_node(), conv_node(), lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in", "conv", "fc", "out"))
    assert (e.value.stage, e.value.node_id) == ("semantic", "fc")
    assert "flatten" in e.value.hint.lower()


def test_invalid_conv_flatten_conv():
    nodes = [input_node(dataset="mnist"), conv_node("c1"), flat_node(), conv_node("c2"), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in", "c1", "flat", "c2", "out"))
    assert (e.value.stage, e.value.node_id) == ("semantic", "c2")
    assert "flatten boundary" in str(e.value)


def test_invalid_spatial_collapse():
    nodes = [input_node(), conv_node()] + [N(f"p{i}", "pool", pool="max", kernel_size=2, stride=2)
                                           for i in range(8)] + [flat_node(), lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges(*[n.id for n in nodes]))
    assert e.value.stage == "trace" and e.value.node_id.startswith("p")
    assert "failed on input shape" in str(e.value)


def test_invalid_wrong_final_linear_size():
    nodes = [input_node(dataset="mnist"), conv_node(), relu_node(), pool_node(),
             flat_node(), lin_node(out=7), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges(*[n.id for n in nodes]))
    assert (e.value.stage, e.value.node_id) == ("trace", "fc")
    assert e.value.reason == "final Linear out_features != dataset class count"


def test_invalid_two_inputs():
    nodes = [input_node("in1"), input_node("in2"), lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in1", "in2", "fc", "out"))
    assert e.value.stage == "schema" and "exactly one input" in str(e.value)


def test_invalid_two_outputs():
    nodes = [input_node(), conv_node(), flat_node(), lin_node(), out_node("o1"), out_node("o2")]
    edges = chain_edges("in", "conv", "flat", "fc", "o1") + [Edge(id="ex", source="fc", target="o2")]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, edges)
    assert e.value.stage == "schema" and "exactly one output" in str(e.value)


def test_invalid_cycle():
    nodes = [input_node(), conv_node("a"), conv_node("b"), conv_node("c"), flat_node(),
             lin_node(), out_node()]
    edges = chain_edges("in", "a", "b", "c", "flat", "fc", "out") + [Edge(id="back", source="c", target="b")]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, edges)
    assert e.value.stage == "structural" and "cycle" in str(e.value).lower()


def test_invalid_orphan():
    nodes = [input_node(), conv_node("c1"), flat_node(), lin_node(), out_node(),
             conv_node("orphan")]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges("in", "c1", "flat", "fc", "out"))
    assert e.value.stage == "structural" and "conv(orphan)" in str(e.value)


def test_invalid_branch_merge_mismatch():
    nodes = [input_node(), conv_node("ca", out=8), conv_node("cb", out=16),
             relu_node("m"), pool_node(), flat_node(), lin_node(), out_node()]
    edges = [Edge(id="e0", source="in", target="ca"), Edge(id="e1", source="in", target="cb"),
             Edge(id="e2", source="ca", target="m"), Edge(id="e3", source="cb", target="m"),
             *chain_edges("m", "pool", "flat", "fc", "out")]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, edges)
    assert (e.value.stage, e.value.node_id) == ("trace", "m")
    assert "identical shapes" in str(e.value)


@pytest.mark.parametrize("p", [1, -0.1])
def test_invalid_dropout_p(p):
    nodes = [input_node(), conv_node(), N("do", "dropout", p=p), relu_node(),
             pool_node(), flat_node(), lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges(*[n.id for n in nodes]))
    assert (e.value.stage, e.value.node_id) == ("params", "do")
    assert "0.0 <= p < 1.0" in str(e.value)


@pytest.mark.parametrize("lr", [-0.5, 0])
def test_invalid_lr(lr):
    nodes = [input_node(), conv_node(), relu_node(), pool_node(), flat_node(),
             lin_node(), out_node(lr=lr)]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges(*[n.id for n in nodes]))
    assert (e.value.stage, e.value.node_id) == ("params", "out")
    assert "lr must be > 0" in str(e.value)


@pytest.mark.parametrize("kw", [{"kernel_size": 0}, {"stride": 0}, {"out_channels": 0}])
def test_invalid_conv_params(kw):
    nodes = [input_node(), conv_node(**kw), relu_node(), pool_node(), flat_node(),
             lin_node(), out_node()]
    with pytest.raises(ValidationError) as e:
        validate_pipeline(nodes, chain_edges(*[n.id for n in nodes]))
    assert (e.value.stage, e.value.node_id) == ("params", "conv")


# ---------- round-trip: export JSON v1 -> parse/import -> validate, same semantics ----------
def roundtrip_envelope(nodes, edges):
    env = {"version": 1, "app": "visualML", "graph": Graph(nodes=nodes, edges=edges).model_dump()}
    return json.loads(json.dumps(env))


@pytest.mark.parametrize("builder", [g_simple, g_cnn, g_branched])
def test_roundtrip_json_v1_same_semantics(builder):
    nodes, edges = builder()
    env = roundtrip_envelope(nodes, edges)
    assert env["version"] == 1 and env["app"] == "visualML"
    g2 = Graph.model_validate(env["graph"])
    assert [canon(n.type) for n in g2.nodes] == [canon(n.type) for n in nodes]
    assert [(e.source, e.target) for e in g2.edges] == [(e.source, e.target) for e in edges]
    r1, r2 = validate_pipeline(nodes, edges), validate_pipeline(g2.nodes, g2.edges)
    assert r1["num_classes"] == r2["num_classes"]
    shapes = lambda r: [(s["kind"], s["in"], s["out"]) for s in r["specs"]]  # noqa: E731
    assert shapes(r1) == shapes(r2)
    fwd_ok(r2["model"], r2["num_classes"])


# ---------- export regression: validate -> generate -> code compiles/runs standalone ----------
@pytest.mark.parametrize("builder", [g_simple, g_cnn, g_branched])
def test_export_python_compiles_and_runs(builder):
    nodes, edges = builder()
    validate_pipeline(nodes, edges)
    code = generate_python(nodes, edges)
    compile(code, "<visualML_export>", "exec")
    ns = {"__name__": "visualML_export_test"}  # guard: main() must not run on import
    exec(compile(code, "<visualML_export>", "exec"), ns)
    net = ns["VisualNet"]()
    net.eval()
    with torch.no_grad():
        assert list(net(torch.randn(2, 1, IMG, IMG)).shape) == [2, N_CLASSES]


def test_export_notebook():
    if not hasattr(_exporter_mod, "generate_notebook"):
        pytest.skip("notebook export not supported by exporter")
    nodes, edges = g_cnn()
    validate_pipeline(nodes, edges)
    nb = generate_notebook(nodes, edges)
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    compile("".join(code_cells[0]["source"]), "<visualML_notebook>", "exec")
    assert "nbformat" in notebook_to_json(nb)
