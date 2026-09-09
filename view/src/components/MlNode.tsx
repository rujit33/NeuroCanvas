import { Handle, Position, type NodeProps } from "reactflow";
import { KIND_META, canonKind } from "../graph";

export default function MlNode({ data, selected }: NodeProps) {
  const kind = canonKind(data.kind as string);
  const meta = KIND_META[kind] ?? KIND_META.linear;
  const params = data.params as Record<string, string | number>;
  const keys = Object.keys(params ?? {}).slice(0, 3);
  const summary = keys.map((k) => `${k}=${params[k]}`).join("  ") || "no params";
  return (
    <div className={`ml-node${selected ? " sel" : ""}`} style={{ borderTopColor: meta.color }}>
      <Handle type="target" position={Position.Left} />
      <div className="ml-node-title" style={{ color: meta.color }}>
        {meta.label}
      </div>
      <div className="ml-node-desc">{meta.desc}</div>
      <div className="ml-node-params">{summary}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
