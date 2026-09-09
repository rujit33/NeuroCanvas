import { Handle, Position, type NodeProps } from "reactflow";

export default function GroupNode({ data, selected }: NodeProps) {
  const name = (data.name as string) || "Group";
  const count = (data.count as number) ?? 0;
  return (
    <div className={`group-node${selected ? " sel" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="group-node-folder">🗂</div>
      <div className="group-node-name">{name}</div>
      <div className="group-node-sub">
        {count} block{count === 1 ? "" : "s"} · double-click to open
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
