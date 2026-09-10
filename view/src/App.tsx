import type { Edge, Node } from "reactflow";
import { API, WS, downloadBlob, downloadText, postJSON, serialize } from "./api";
import ConfigPanel from "./components/ConfigPanel";
import LogsPanel from "./components/LogsPanel";
import GroupNode from "./components/GroupNode";
import MlNode from "./components/MlNode";
import { KIND_META, defaultParams, type JobState, type NodeKind, type WireInfo } from "./graph";

const nodeTypes = { ml: MlNode, group: GroupNode };
const DEFAULT_EDGE_OPTIONS = { interactionWidth: 28 };
let seq = 100;
let groupCount = 1;

/* ---------- master state helpers (groups = parentId membership) ---------- */

function kidsOf(nodes: Node[], scope: string | null): Node[] {
  const anchor = scope ?? undefined;
  return nodes.filter((n) => (n.parentId ?? undefined) === anchor);
}

/** Walk up to the direct child of `scope` that contains `id` (or null if outside). */
function directChild(nodes: Node[], id: string, scope: string | null): string | null {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  let cur = byId.get(id);
  if (!cur) return null;
  const anchor = scope ?? undefined;
  let guard = 0;
  while ((cur.parentId ?? undefined) !== anchor && guard++ < 1000) {
    const p = byId.get(cur.parentId!);
    if (!p) return null; // outside current scope tree
    cur = p;
  }
  return cur.id;
}

function countMembers(nodes: Node[], groupId: string): number {
  let n = 0;
  const walk = (gid: string) => {
    for (const c of nodes) {
      if ((c.parentId ?? undefined) === gid) {
        if (c.type === "group") walk(c.id);
        else n++;
      }
    }
  };
  walk(groupId);
  return n;
}

/* ---------- JSON graph save/load + format guide (v1, kind is open-ended) ---------- */

const GUIDE_EXAMPLE = `{
  "version": 1, "app": "visualML",
  "nodes": [
    {"id": "input", "type": "ml", "position": {"x": 40, "y": 180},
     "data": {"kind": "input", "params": {}}},
    {"id": "fc1", "type": "ml", "position": {"x": 260, "y": 180},
     "data": {"kind": "linear", "params": {"out_features": 10}}},
    {"id": "output", "type": "ml", "position": {"x": 480, "y": 180},
     "data": {"kind": "output", "params": {}}}
  ],
  "edges": [
    {"id": "e1", "source": "input", "target": "fc1"},
    {"id": "e2", "source": "fc1", "target": "output"}
  ]
}`;

const GUIDE_PROMPT =
  `Create a visualML v1 JSON graph with nodes/edges matching the schema above (kind is an open-ended block-type string with a params object). Only output JSON.`;

const GUIDE_COPY = `visualML v1 JSON graph format
Top level: {version: 1, app: "visualML", exportedAt: ISO-string, nodes: [...], edges: [...]}.
Node: {id: string, type: "ml" | "group", position: {x: number, y: number}, parentId?: string (group membership), data: {kind: string (open-ended block type), params: object, name?: string (groups)}}.
Edge: {id: string, source: node-id, target: node-id}.
Unknown kinds load as-is (no allowlist). Minimal example:
${GUIDE_EXAMPLE}
Prompt template: ${GUIDE_PROMPT}`;

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

// ponytail: structural guard only, unknown kinds pass (future blocks must load).
function parseGraphFile(text: string): { nodes: Node[]; edges: Edge[] } {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    throw new Error("not valid JSON");
  }
  if (!isRecord(doc) || !Array.isArray(doc.nodes) || !Array.isArray(doc.edges)) {
    throw new Error("expected {nodes: [...], edges: [...]}");
  }
  const nodes = (doc.nodes as unknown[]).map((v, i) => {
    if (!isRecord(v) || typeof v.id !== "string") throw new Error(`nodes[${i}].id must be a string`);
    const pos = v.position as unknown;
    if (!isRecord(pos) || typeof pos.x !== "number" || typeof pos.y !== "number") {
      throw new Error(`nodes[${i}] (${v.id}).position must be {x: number, y: number}`);
    }
    if (!isRecord(v.data)) throw new Error(`nodes[${i}] (${v.id}).data must be an object`);
    const type = v.type === undefined ? "ml" : v.type;
    if (type !== "ml" && type !== "group") throw new Error(`nodes[${i}] (${v.id}).type must be "ml" or "group"`);
    if (type === "ml") {
      const kind = (v.data as Record<string, unknown>).kind;
      if (typeof kind !== "string" || !kind) throw new Error(`nodes[${i}] (${v.id}).data.kind must be a non-empty string`);
    }
    return { ...v, type } as Node;
  });
  const edges = (doc.edges as unknown[]).map((v, i) => {
    if (!isRecord(v) || typeof v.id !== "string" || typeof v.source !== "string" || typeof v.target !== "string") {
      throw new Error(`edges[${i}] needs string id/source/target`);
    }
    return v as unknown as Edge;
  });
  return { nodes, edges };
}

/* ------------------------------- studio ---------------------------------- */

function initialNodes(): Node[] {
  const mk = (id: string, kind: NodeKind, x: number, params?: Record<string, string | number>): Node => ({
    id,
    type: "ml",
    position: { x, y: 180 },
    data: { kind, params: params ?? defaultParams(kind) },
  });
  return [
    mk("input", "input", 40),
    mk("conv1", "conv", 260),
    mk("act1", "activation", 480),
    mk("pool1", "pool", 700),
    mk("flat1", "flatten", 920),
    mk("fc1", "linear", 1140, { out_features: 10 }),
    mk("output", "output", 1360),
  ];
}

function initialEdges(): Edge[] {
  const ids = ["input", "conv1", "act1", "pool1", "flat1", "fc1", "output"];
  return ids.slice(1).map((t, i) => ({
    id: `e${i + 1}`, source: ids[i], target: t, animated: true,
  }));
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type EdgeChange,
  type NodeChange,
} from "reactflow";
import "reactflow/dist/style.css";
import "./App.css";

function Studio() {
  const [masterNodes, setMasterNodes] = useState<Node[]>(initialNodes);
  const [masterEdges, setMasterEdges] = useState<Edge[]>(initialEdges());
  const [scope, setScope] = useState<string | null>(null);
  const [path, setPath] = useState<{ id: string; name: string }[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<string[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>("input");
  const [epochs, setEpochs] = useState(5);
  const [saveFormat, setSaveFormat] = useState("pt");
  const [saveMode, setSaveMode] = useState("weights_only");
  const [job, setJob] = useState<JobState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [logsOpen, setLogsOpen] = useState(true);
  const [notice, setNotice] = useState("");
  const [palQuery, setPalQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [showGuide, setShowGuide] = useState(false);

  const byId = useMemo(() => new Map(masterNodes.map((n) => [n.id, n])), [masterNodes]);

  /* ----- derived view for current scope (portal stubs computed, never stored) ----- */
  const viewNodes: Node[] = useMemo(() => {
    return kidsOf(masterNodes, scope).map((n) => {
      const { parentId: _p, ...rest } = n as Node & { parentId?: string };
      if (n.type === "group") {
        return {
          ...rest, parentId: undefined,
          data: { ...n.data, count: countMembers(masterNodes, n.id) },
        };
      }
      return { ...rest, parentId: undefined };
    });
  }, [masterNodes, scope]);

  const viewEdges: Edge[] = useMemo(() => {
    const out: Edge[] = [];
    for (const e of masterEdges) {
      const a = directChild(masterNodes, e.source, scope);
      const b = directChild(masterNodes, e.target, scope);
      if (!a || !b) continue; // endpoint outside this view
      if (a === e.source && b === e.target) {
        out.push(e); // fully visible real edge
      } else if (a !== b) {
        out.push({
          ...e, id: `portal:${e.id}`, source: a, target: b,
          style: { strokeDasharray: "6 4" }, data: { ...(e.data ?? {}), realId: e.id, portal: true },
        });
      }
      // a === b but not the raw endpoints => both ends hidden inside one collapsed group
    }
    return out;
  }, [masterNodes, masterEdges, scope]);

  const selected = selectedId ? byId.get(selectedId) ?? null : null;
  const selectedForPanel = selected && selected.type === "group"
    ? { ...selected, data: { ...selected.data, count: countMembers(masterNodes, selected.id) } }
    : selected;

  /* ----- selected wires (portal stubs map back to the real edge) ----- */
  const nameOf = useCallback((id: string): string => {
    const n = byId.get(id);
    if (!n) return id;
    if (n.type === "group") return `🗂 ${(n.data.name as string) || id}`;
    const kind = (n.data as { kind: string }).kind;
    return `${KIND_META[kind]?.label ?? kind} (${id})`;
  }, [byId]);

  const selectedWires: WireInfo[] = useMemo(() => {
    const seen = new Set<string>();
    const out: WireInfo[] = [];
    for (const vid of selectedEdgeIds) {
      const rid = vid.startsWith("portal:") ? vid.slice("portal:".length) : vid;
      if (seen.has(rid)) continue;
      seen.add(rid);
      const e = masterEdges.find((m) => m.id === rid);
      if (!e) continue;
      out.push({ id: e.id, source: e.source, target: e.target, sourceName: nameOf(e.source), targetName: nameOf(e.target) });
    }
    return out;
  }, [selectedEdgeIds, masterEdges, nameOf]);

  const deleteEdge = useCallback((id: string) => {
    setMasterEdges((es) => es.filter((e) => e.id !== id));
    setSelectedEdgeIds((ids) => ids.filter((v) => v !== id && v !== `portal:${id}`));
  }, []);

  /* ----- stable RF callbacks (new identities retrigger RF store effects) ----- */
  const onNodeClick = useCallback((_e: unknown, n: Node) => setSelectedId(n.id), []);
  const onPaneClick = useCallback(() => setSelectedId(null), []);
  const onSelectionChange = useCallback((p: { nodes: Node[]; edges: Edge[] }) => {
    setSelectedIds(p.nodes.map((n) => n.id));
    setSelectedEdgeIds(p.edges.map((e) => e.id));
  }, []);

  /* ----- RF change handlers: positions sync to master, deletes cascade ----- */
  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setMasterNodes((ns) => {
      let next = applyNodeChanges(changes, ns);
      const removed = changes.filter((c) => c.type === "remove").map((c) => (c as { id: string }).id);
      if (removed.length) {
        const gone = new Set<string>();
        const collect = (gid: string) => {
          gone.add(gid);
          next.filter((n) => n.parentId === gid).forEach((c) => collect(c.id));
        };
        removed.forEach(collect);
        next = next.filter((n) => !gone.has(n.id));
        setMasterEdges((es) => es.filter((e) => !gone.has(e.source) && !gone.has(e.target)));
      }
      return next;
    });
  }, []);

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setMasterEdges((es) => {
      const real = (id: string) => {
        const stub = es.find((e) => `portal:${e.id}` === id);
        return stub ? stub.id : id;
      };
      const mapped = changes.map((c) =>
        c.type === "remove" ? { ...c, id: real(c.id) } : c
      );
      return applyEdgeChanges(mapped, es);
    });
  }, []);

  /** Boundary of a collapsed group in the current view: unique entry/exit blocks. */
  const boundary = useCallback((groupId: string, side: "in" | "out"): string | null => {
    const kids = new Set(kidsOf(masterNodes, groupId).map((n) => n.id));
    const pts = new Set<string>();
    for (const e of masterEdges) {
      const sIn = kids.has(e.source) || directChild(masterNodes, e.source, groupId) !== null;
      const tIn = kids.has(e.target) || directChild(masterNodes, e.target, groupId) !== null;
      if (side === "in" && !sIn && tIn) {
        const dc = directChild(masterNodes, e.target, groupId);
        if (dc) pts.add(dc);
      }
      if (side === "out" && sIn && !tIn) {
        const dc = directChild(masterNodes, e.source, groupId);
        if (dc) pts.add(dc);
      }
    }
    return pts.size === 1 ? [...pts][0] : null;
  }, [masterNodes, masterEdges]);

  const onConnect = useCallback((c: Connection) => {
    if (!c.source || !c.target) return;
    let { source, target } = c;
    const sn = byId.get(source), tn = byId.get(target);
    if (sn?.type === "group") {
      const exit = boundary(source, "out");
      if (!exit) {
        setNotice(`"${sn.data.name}" has no single exit block — open it and wire a specific block.`);
        return;
      }
      source = exit;
    }
    if (tn?.type === "group") {
      const entry = boundary(target, "in");
      if (!entry) {
        setNotice(`"${tn.data.name}" has no single entry block — open it and wire a specific block.`);
        return;
      }
      target = entry;
    }
    if (source === target) return;
    const id = `e${Date.now().toString(36)}`;
    setMasterEdges((es) =>
      es.some((e) => e.source === source && e.target === target)
        ? es : [...es, { id, source, target, animated: true }]
    );
  }, [byId, boundary]);

  /* ----- drag new blocks from palette ----- */
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const kind = e.dataTransfer.getData("application/visualml") as NodeKind;
    if (!kind) return;
    const bounds = wrapRef.current?.getBoundingClientRect();
    const id = `${kind}-${seq++}`;
    const node: Node = {
      id, type: "ml",
      position: { x: e.clientX - (bounds?.left ?? 0) - 80, y: e.clientY - (bounds?.top ?? 0) - 40 },
      data: { kind, params: defaultParams(kind) },
    };
    if (scope) (node as Node & { parentId?: string }).parentId = scope;
    setMasterNodes((ns) => [...ns, node]);
    setSelectedId(id);
  }, [scope]);

  /* ----- grouping ----- */
  const groupSelected = () => {
    const inView = selectedIds.filter((id) => {
      const n = byId.get(id);
      return n && (n.parentId ?? undefined) === (scope ?? undefined);
    });
    if (inView.length === 0) {
      setNotice("Select one or more blocks (Shift+click or drag a box), then Group.");
      return;
    }
    const pts = inView.map((id) => byId.get(id)!.position);
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = Math.min(...pts.map((p) => p.y)) - 110;
    const gid = `g-${seq++}`;
    const name = `Group ${groupCount++}`;
    const proxy: Node = {
      id: gid, type: "group", position: { x: cx, y: cy },
      data: { kind: "group", groupId: gid, name },
    };
    if (scope) (proxy as Node & { parentId?: string }).parentId = scope;
    setMasterNodes((ns) =>
      ns.map((n) => (inView.includes(n.id) ? { ...n, parentId: gid } : n)).concat(proxy)
    );
    setSelectedId(gid);
    setNotice(`Created "${name}" — rename it in the inspector, double-click to open.`);
  };

  const ungroup = (gid: string) => {
    const proxy = byId.get(gid);
    if (!proxy) return;
    const up = (proxy.parentId ?? undefined) as string | undefined;
    setMasterNodes((ns) =>
      ns.filter((n) => n.id !== gid).map((n) =>
        (n.parentId ?? undefined) === gid ? { ...n, parentId: up } : n
      )
    );
    if (scope === gid) {
      // we are inside the group being dissolved -> go up
      setScope(up ?? null);
      setPath((p) => p.slice(0, -1));
    }
    setSelectedId(null);
  };

  const openGroup = (gid: string) => {
    const g = byId.get(gid);
    if (!g) return;
    setScope(gid);
    setPath((p) => [...p, { id: gid, name: (g.data.name as string) || gid }]);
    setSelectedId(null);
  };

  const goTo = (idx: number) => {
    // idx -1 = root, else path index
    if (idx < 0) {
      setScope(null);
      setPath([]);
    } else {
      setScope(path[idx].id);
      setPath(path.slice(0, idx + 1));
    }
    setSelectedId(null);
  };

  /* ----- inspector edits ----- */
  const patchParams = (id: string, params: Record<string, string | number>) =>
    setMasterNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, params } } : n)));

  const renameGroup = (id: string, name: string) =>
    setMasterNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, name } } : n)));

  const deleteNode = (id: string) => {
    const gone = new Set<string>([id]);
    const n = byId.get(id);
    if (n?.type === "group") {
      const collect = (gid: string) => {
        masterNodes.filter((m) => m.parentId === gid).forEach((c) => {
          gone.add(c.id);
          if (c.type === "group") collect(c.id);
        });
      };
      collect(id);
    }
    setMasterNodes((ns) => ns.filter((m) => !gone.has(m.id)));
    setMasterEdges((es) => es.filter((e) => !gone.has(e.source) && !gone.has(e.target)));
    setSelectedId(null);
  };

  /* ----- backend ----- */
  const graph = () => serialize(masterNodes, masterEdges);

  useEffect(() => () => wsRef.current?.close(), []);

  const validate = async () => {
    setBusy(true);
    try {
      const r = await postJSON<{ ok: boolean; order: string[]; params: number }>(`/api/validate`, graph());
      setNotice(`Valid ✓ ${(r.order as string[]).join("  →  ")} · ${r.params} params`);
    } catch (e) {
      setNotice(`Invalid: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const train = async () => {
    setBusy(true);
    setLogs([]);
    try {
      const r = await postJSON<{ job_id: string }>(`/api/train`, {
        graph: graph(), epochs, save_format: saveFormat, save_mode: saveMode,
      });
      const jobId = r.job_id;
      setNotice(`Training started: ${jobId}`);
      wsRef.current?.close();
      const ws = new WebSocket(`${WS}/ws/jobs/${jobId}`);
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.type === "init" || m.type === "final" || m.type === "ping") {
          if (m.job) setJob(m.job);
          if (m.job?.logs) setLogs(m.job.logs);
        } else if (m.type === "log") {
          setLogs((l) => [...l.slice(-400), m.line]);
        } else if (m.type === "progress") {
          setJob((j) => {
            if (!j || j.job_id !== jobId) return j;
            const hist = m.epoch
              ? [...j.history.filter((h) => h.epoch !== m.epoch),
                 { epoch: m.epoch, train_loss: +m.train_loss.toFixed(4), val_loss: +m.val_loss.toFixed(4), val_acc: +m.val_acc.toFixed(4) }]
              : j.history;
            return { ...j, status: m.status ?? j.status, current_epoch: m.epoch ?? j.current_epoch, history: hist, save_path: m.save_path ?? j.save_path };
          });
        }
      };
      const jr = await fetch(`${API}/api/jobs/${jobId}`).then((x) => x.json());
      setJob(jr);
    } catch (e) {
      setNotice(`Train failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const doExport = async (kind: "python" | "notebook") => {
    try {
      await downloadBlob(
        `/api/export/${kind}`, { graph: graph(), epochs },
        kind === "python" ? "model.py" : "model.ipynb"
      );
      setNotice(`Exported ${kind === "python" ? "model.py" : "model.ipynb"}`);
    } catch (e) {
      setNotice(`Export failed: ${(e as Error).message}`);
    }
  };

  const exportJson = () => {
    const payload = {
      version: 1, app: "visualML",
      exportedAt: new Date().toISOString(),
      nodes: masterNodes, edges: masterEdges,
    };
    downloadText("visualml-graph.json", JSON.stringify(payload, null, 2));
    setNotice(`Exported visualml-graph.json (${masterNodes.length} nodes, ${masterEdges.length} edges)`);
  };

  const onImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    try {
      const { nodes, edges } = parseGraphFile(await f.text());
      setMasterNodes(nodes);
      setMasterEdges(edges);
      setScope(null);
      setPath([]);
      setSelectedId(null);
      setNotice(`Imported ${f.name} (${nodes.length} nodes, ${edges.length} edges)`);
    } catch (err) {
      setNotice(`Import failed: ${(err as Error).message}`);
    }
  };

  const scopeName = scope ? byId.get(scope)?.data.name ?? scope : null;
  const inputNode = masterNodes.find((n) => n.type === "ml" && (n.data as { kind: string }).kind === "input");
  const batchSize = (inputNode?.data as { params?: Record<string, unknown> } | undefined)?.params?.batch_size ?? 64;

  const palItems = (Object.keys(KIND_META) as NodeKind[]).filter((k) => {
    if (k === "group") return false;
    const q = palQuery.trim().toLowerCase();
    if (!q) return true;
    const m = KIND_META[k];
    return k.includes(q) || m.label.toLowerCase().includes(q) || m.desc.toLowerCase().includes(q);
  });

  return (
    <div className="studio">
      <header className="topbar">
        <div className="brand">◈ VisualML <span className="poc">dynamic</span></div>
        <div className="controls">
          <label>epochs <input type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} /></label>
          <label title="Batch size lives on the input block">batches <input type="number" value={String(batchSize)} readOnly /></label>
          <select value={saveFormat} onChange={(e) => setSaveFormat(e.target.value)} title="Save file type">
            <option value="pt">.pt</option>
            <option value="pth">.pth</option>
            <option value="pkl">.pkl</option>
          </select>
          <select value={saveMode} onChange={(e) => setSaveMode(e.target.value)} title="Save weights only or full checkpoint">
            <option value="weights_only">weights only</option>
            <option value="full">full training data</option>
          </select>
          <button onClick={validate} disabled={busy}>Validate</button>
          <button className="primary" onClick={train} disabled={busy}>▶ Train</button>
          <button onClick={() => doExport("python")}>⇩ .py</button>
          <button onClick={() => doExport("notebook")}>⇩ .ipynb</button>
          <button onClick={exportJson} title="Save canvas as visualML v1 JSON">⇩ JSON</button>
          <button onClick={() => fileRef.current?.click()} title="Load a visualML v1 JSON graph">⇧ Import</button>
          <button onClick={() => setShowGuide(true)} title="JSON format guide">i</button>
          <input ref={fileRef} type="file" accept=".json,application/json" hidden onChange={onImportFile} />
          {job?.save_path && (
            <a className="btn" href={`${API}/api/download/${job.job_id}`}>⇩ model.{saveFormat}</a>
          )}
        </div>
      </header>

      {notice && (
        <div className="notice">
          <span title={notice}>{notice}</span>
          <button onClick={() => setNotice("")} title="Dismiss">×</button>
        </div>
      )}

      <div className="crumbs">
        <button className={scope === null ? "active" : ""} onClick={() => goTo(-1)}>Root</button>
        {path.map((p, i) => (
          <span key={p.id}>
            <span className="sep">/</span>
            <button className={i === path.length - 1 ? "active" : ""} onClick={() => goTo(i)}>
              {byId.get(p.id)?.data.name ?? p.name}
            </button>
          </span>
        ))}
        {scope && (
          <button className="danger-link" onClick={() => ungroup(scope)} title="Dissolve this group">
            Ungroup "{scopeName}"
          </button>
        )}
        <span className="spacer" />
        <button onClick={groupSelected} disabled={selectedIds.length === 0} title="Group selected blocks (Shift+click / box-select)">
          ⧉ Group ({selectedIds.length})
        </button>
      </div>

      <div className="main">
        <aside className="palette">
          <div className="palette-search">
            <input
              type="text"
              placeholder="Search blocks… (conv, pool, …)"
              value={palQuery}
              onChange={(e) => setPalQuery(e.target.value)}
            />
          </div>
          <div className="palette-list">
            {palItems.map((k) => (
              <div
                key={k}
                className="palette-item"
                draggable
                onDragStart={(e) => e.dataTransfer.setData("application/visualml", k)}
                style={{ borderLeftColor: KIND_META[k].color }}
                title="Drag onto the canvas"
              >
                <b style={{ color: KIND_META[k].color }}>{KIND_META[k].label}</b>
                <small>{KIND_META[k].desc}</small>
              </div>
            ))}
            {palItems.length === 0 && (
              <div className="palette-empty">No blocks match “{palQuery}”.</div>
            )}
          </div>
          <div className="palette-hint">
            Drag blocks onto the canvas, wire them freely. Click a wire to select it, then press
            Delete or remove it from the inspector. Shift+click or box-select, then ⧉ Group.
          </div>
        </aside>

        <div className="canvas" ref={wrapRef} onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
          <ReactFlow
            nodes={viewNodes}
            edges={viewEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onNodeDoubleClick={(_, n) => { if (n.type === "group") openGroup(n.id); }}
            onPaneClick={onPaneClick}
            onSelectionChange={onSelectionChange}
            nodeTypes={nodeTypes}
            defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
            fitView
            multiSelectionKeyCode="Shift"
            deleteKeyCode="Delete"
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>

        <ConfigPanel
          node={selectedForPanel}
          wires={selectedWires}
          onDeleteWire={deleteEdge}
          onChange={patchParams}
          onDelete={deleteNode}
          onRename={renameGroup}
          onOpen={openGroup}
          onUngroup={ungroup}
        />
      </div>

      {showGuide && (
        <div
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}
          onClick={() => setShowGuide(false)}
        >
          <div
            className="panel"
            style={{ maxWidth: 560, width: "100%", maxHeight: "85dvh", overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3>visualML v1 JSON format</h3>
            <p className="muted">Top level <code>{"{version, app, exportedAt, nodes, edges}"}</code>. Node: <code>{"{id, type: ml|group, position: {x,y}, parentId?, data: {kind, params, name?}}"}</code> — <code>kind</code> is an open-ended block-type string with a <code>params</code> object. Edge: <code>{"{id, source, target}"}</code>. Unknown kinds load as-is.</p>
            <pre className="logs-pre" style={{ height: "auto", marginBottom: 8 }}>{GUIDE_EXAMPLE}</pre>
            <p className="muted">{GUIDE_PROMPT}</p>
            <button
              onClick={async () => { await navigator.clipboard.writeText(GUIDE_COPY); setNotice("Format guide copied"); }}
            >Copy guide</button>
            <button onClick={() => setShowGuide(false)}>Close</button>
          </div>
        </div>
      )}

      <LogsPanel job={job} logs={logs} open={logsOpen} onToggle={() => setLogsOpen((v) => !v)} />
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Studio />
    </ReactFlowProvider>
  );
}
