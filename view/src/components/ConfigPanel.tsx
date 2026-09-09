import type { Node } from "reactflow";
import { API } from "../api";
import { FIELDS, KIND_META, canonKind, type WireInfo } from "../graph";

interface Props {
  node: Node | null;
  wires: WireInfo[];
  onDeleteWire: (id: string) => void;
  onChange: (id: string, params: Record<string, string | number>) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onOpen: (id: string) => void;
  onUngroup: (id: string) => void;
}

function WiresSection({ wires, onDeleteWire }: { wires: WireInfo[]; onDeleteWire: (id: string) => void }) {
  if (wires.length === 0) return null;
  return (
    <div className="wires">
      <h3>⌁ Selected wire{wires.length === 1 ? "" : "s"}</h3>
      {wires.map((w) => (
        <div key={w.id} className="wire-row" title={`${w.source} → ${w.target}`}>
          <span className="wire-label">{w.sourceName} → {w.targetName}</span>
          <button className="danger wire-del" onClick={() => onDeleteWire(w.id)} title="Delete this wire">
            ×
          </button>
        </div>
      ))}
      <p className="muted">Tip: the Delete key removes selected wires too.</p>
    </div>
  );
}

export default function ConfigPanel({ node, wires, onDeleteWire, onChange, onDelete, onRename, onOpen, onUngroup }: Props) {
  if (!node) {
    return (
      <aside className="panel">
        <WiresSection wires={wires} onDeleteWire={onDeleteWire} />
        <h3>Inspector</h3>
        <p className="muted">Select a block to edit it — or click a wire to remove it. Drag new blocks from the palette and wire them freely.</p>
        <div className="hint">
          <b>Grouping:</b> Shift+click or box-select blocks, then ⧉ Group. The group shows as one tidy card —
          double-click to open it, rename it here, nest groups inside groups.
        </div>
      </aside>
    );
  }

  if (node.type === "group") {
    const name = (node.data.name as string) || "Group";
    const count = (node.data.count as number) ?? 0;
    return (
      <aside className="panel">
        <WiresSection wires={wires} onDeleteWire={onDeleteWire} />
        <h3 style={{ color: "#e879f9" }}>🗂 Group</h3>
        <label className="field">
          <span>Name this group however you like</span>
          <input type="text" value={name} onChange={(e) => onRename(node.id, e.target.value)} />
        </label>
        <p className="muted">{count} block{count === 1 ? "" : "s"} inside.</p>
        <button className="primary-btn" onClick={() => onOpen(node.id)}>Open group →</button>
        <button onClick={() => onUngroup(node.id)}>Ungroup (keep blocks)</button>
        <button className="danger" onClick={() => onDelete(node.id)}>Delete group + contents</button>
      </aside>
    );
  }

  const kind = canonKind((node.data as { kind: string }).kind);
  const params = (node.data as { params: Record<string, string | number> }).params ?? {};
  const meta = KIND_META[kind] ?? { label: kind, color: "#fff", desc: "" };

  const set = (key: string, raw: string, fkind: string) => {
    let v: string | number = raw;
    if (fkind === "number") v = raw === "" ? 0 : Number(raw);
    onChange(node.id, { ...params, [key]: v });
  };

  const uploadZip = async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
    if (!res.ok) {
      alert("Upload failed: " + (await res.text()));
      return;
    }
    const j = await res.json();
    onChange(node.id, { ...params, dataset: "imagefolder", dataset_path: j.path });
  };

  const fields = FIELDS[kind] ?? [];

  return (
    <aside className="panel">
      <WiresSection wires={wires} onDeleteWire={onDeleteWire} />
      <h3 style={{ color: meta.color }}>{meta.label} block</h3>
      {fields.length === 0 && <p className="muted">No parameters — just wire it in.</p>}
      {fields.map((f) => (
        <label key={f.key} className="field">
          <span>{f.label}</span>
          {f.kind === "select" ? (
            <select value={String(params[f.key] ?? "")} onChange={(e) => set(f.key, e.target.value, "text")}>
              {f.options!.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          ) : (
            <input
              type={f.kind === "number" ? "number" : "text"}
              step={f.step}
              value={String(params[f.key] ?? "")}
              placeholder={f.hint}
              onChange={(e) => set(f.key, e.target.value, f.kind)}
            />
          )}
          {f.hint && <small>{f.hint}</small>}
        </label>
      ))}
      {kind === "input" && (
        <label className="field">
          <span>Upload dataset zip</span>
          <input
            type="file"
            accept=".zip"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) uploadZip(f);
            }}
          />
          <small>zip of class folders → fills file path automatically</small>
        </label>
      )}
      <button className="danger" onClick={() => onDelete(node.id)}>Delete block</button>
    </aside>
  );
}
