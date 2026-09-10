import type { Edge, Node } from "reactflow";

export const API = "http://127.0.0.1:8000";
export const WS = "ws://127.0.0.1:8000";

export function serialize(nodes: Node[], edges: Edge[]) {
  // Group proxies are view-only; the engine sees the flat block graph.
  const blocks = nodes.filter((n) => n.type !== "group");
  return {
    nodes: blocks.map((n) => ({
      id: n.id,
      type: (n.data as { kind: string }).kind,
      params: (n.data as { params: Record<string, unknown> }).params ?? {},
    })),
    edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
  };
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text();
    try {
      const j = JSON.parse(txt);
      throw new Error(j.detail ?? txt);
    } catch (e) {
      if (e instanceof Error && (e as Error).message !== txt) throw e;
      throw new Error(txt || `HTTP ${res.status}`);
    }
  }
  return res.json() as Promise<T>;
}

export function downloadText(filename: string, text: string, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function downloadBlob(path: string, body: unknown, filename: string) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
