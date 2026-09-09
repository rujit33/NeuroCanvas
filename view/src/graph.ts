export type NodeKind =
  | "input" | "conv" | "activation" | "pool"
  | "flatten" | "linear" | "dropout" | "output"
  | "group";

export interface NodeParams {
  [key: string]: string | number;
}

/** Legacy PoC names -> current blocks */
export function canonKind(k: string): string {
  const t = (k || "").toLowerCase();
  if (t === "cnn") return "conv";
  if (t === "classifier") return "linear";
  return t;
}

export const KIND_META: Record<string, { label: string; color: string; desc: string }> = {
  input: { label: "Input", color: "#38bdf8", desc: "Dataset source" },
  conv: { label: "Conv", color: "#a78bfa", desc: "Conv2d (in-ch auto)" },
  activation: { label: "Activation", color: "#fbbf24", desc: "Non-linearity" },
  pool: { label: "Pool", color: "#22d3ee", desc: "Downsample" },
  flatten: { label: "Flatten", color: "#a3e635", desc: "-> vector" },
  linear: { label: "Linear", color: "#34d399", desc: "FC (in-dim auto)" },
  dropout: { label: "Dropout", color: "#fb923c", desc: "Regularize" },
  output: { label: "Output", color: "#f472b6", desc: "Loss + optim" },
  group: { label: "Group", color: "#e879f9", desc: "Named sub-assembly" },
};

export const PALETTE: NodeKind[] = [
  "input", "conv", "activation", "pool", "flatten", "linear", "dropout", "output",
];

export function defaultParams(kind: string): NodeParams {
  switch (canonKind(kind)) {
    case "input":
      return { dataset: "synthetic", dataset_path: "", image_size: 28, batch_size: 64, in_channels: 1 };
    case "conv":
      return { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 };
    case "activation":
      return { type: "relu" };
    case "pool":
      return { pool: "max", kernel_size: 2, stride: 2, adaptive_size: 4 };
    case "flatten":
      return {};
    case "linear":
      return { out_features: 64 };
    case "dropout":
      return { p: 0.5 };
    case "output":
      return { loss: "cross_entropy", optimizer: "adam", lr: 0.001 };
    default:
      return {};
  }
}

/** Field specs drive the config panel. select options or numeric/text input. */
export interface FieldSpec {
  key: string;
  label: string;
  kind: "select" | "number" | "text";
  options?: string[];
  step?: number;
  hint?: string;
}

export const FIELDS: Record<string, FieldSpec[]> = {
  input: [
    { key: "dataset", label: "Dataset", kind: "select", options: ["synthetic", "mnist", "imagefolder"], hint: "synthetic = zero-setup demo" },
    { key: "dataset_path", label: "File path", kind: "text", hint: "Server folder for imagefolder, or upload a zip" },
    { key: "image_size", label: "Image size", kind: "number" },
    { key: "batch_size", label: "Batch size", kind: "number" },
    { key: "in_channels", label: "Channels", kind: "select", options: ["1", "3"] },
  ],
  conv: [
    { key: "out_channels", label: "Out channels", kind: "number" },
    { key: "kernel_size", label: "Kernel", kind: "number" },
    { key: "stride", label: "Stride", kind: "number" },
    { key: "padding", label: "Padding", kind: "number" },
  ],
  activation: [
    { key: "type", label: "Function", kind: "select", options: ["relu", "sigmoid", "tanh", "leaky_relu", "gelu"] },
  ],
  pool: [
    { key: "pool", label: "Type", kind: "select", options: ["max", "avg", "adaptive"] },
    { key: "kernel_size", label: "Kernel", kind: "number" },
    { key: "stride", label: "Stride", kind: "number" },
    { key: "adaptive_size", label: "Adaptive out size", kind: "number" },
  ],
  flatten: [],
  linear: [
    { key: "out_features", label: "Out features", kind: "number", hint: "Last linear must equal class count" },
  ],
  dropout: [
    { key: "p", label: "Drop prob", kind: "number", step: 0.05 },
  ],
  output: [
    { key: "loss", label: "Loss", kind: "select", options: ["cross_entropy"] },
    { key: "optimizer", label: "Optimizer", kind: "select", options: ["adam", "sgd"] },
    { key: "lr", label: "LR", kind: "number", step: 0.0001 },
  ],
};
export interface EpochPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_acc: number;
}

export interface WireInfo {
  id: string;
  source: string;
  target: string;
  sourceName: string;
  targetName: string;
}

export interface JobState {
  job_id: string;
  status: string;
  epochs: number;
  current_epoch: number;
  history: EpochPoint[];
  logs: string[];
  save_path?: string | null;
  params?: number | null;
  dataset_info?: Record<string, unknown> | null;
  error?: string | null;
}
