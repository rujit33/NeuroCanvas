import type { JobState, ValidateShape } from "../graph";
import { fmtDims } from "../graph";

function Curve({ history }: { history: JobState["history"] }) {
  if (history.length === 0)
    return (
      <p className="muted">Loss curve will appear here once training starts.</p>
    );
  const W = 520,
    H = 150,
    P = 28;
  const vals = history.flatMap((h) => [h.train_loss, h.val_loss]);
  const max = Math.max(...vals, 1e-6),
    min = Math.min(...vals, 0);
  const X = (i: number) =>
    P + (i / Math.max(history.length - 1, 1)) * (W - 2 * P);
  const Y = (v: number) => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P);
  const line = (pick: (h: JobState["history"][number]) => number) =>
    history
      .map(
        (h, i) =>
          `${i === 0 ? "M" : "L"}${X(i).toFixed(1)},${Y(pick(h)).toFixed(1)}`,
      )
      .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="curve">
      <path
        d={line((h) => h.train_loss)}
        fill="none"
        stroke="#38bdf8"
        strokeWidth="2"
      />
      <path
        d={line((h) => h.val_loss)}
        fill="none"
        stroke="#f472b6"
        strokeWidth="2"
        strokeDasharray="5 3"
      />
      {history.map((h, i) => (
        <circle key={i} cx={X(i)} cy={Y(h.train_loss)} r="2.5" fill="#38bdf8" />
      ))}
      <text x={P} y={14} fill="#94a3b8" fontSize="11">
        — train_loss
      </text>
      <text x={P + 90} y={14} fill="#94a3b8" fontSize="11">
        - - val_loss
      </text>
    </svg>
  );
}

export interface ModelSummary {
  order: string[];
  params: number | null;
  shapes: ValidateShape[];
  config?: Record<string, unknown> | null;
  warnings: string[];
}

export default function LogsPanel({
  job,
  logs,
  open,
  onToggle,
  summary,
}: {
  job: JobState | null;
  logs: string[];
  open: boolean;
  onToggle: () => void;
  summary?: ModelSummary | null;
}) {
  return (
    <section className={`logs${open ? "" : " closed"}`}>
      <div
        className="logs-head"
        onClick={onToggle}
        title={open ? "Collapse" : "Expand"}
      >
        <span className="chev">{open ? "▾" : "▸"}</span>
        <h3>Training</h3>
        {job && (
          <span className={`badge ${job.status}`}>
            {job.status} {job.current_epoch}/{job.epochs}
          </span>
        )}
        {job?.params != null && (
          <span className="muted small">{job.params} params</span>
        )}
        {job?.dataset_info && (
          <span className="muted small">
            {String((job.dataset_info as { source?: string }).source)}
          </span>
        )}
      </div>
      <div className="logs-body">
        <div className="logs-left">
          <Curve history={job?.history ?? []} />
          {job && job.history.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>epoch</th>
                  <th>train</th>
                  <th>val</th>
                  <th>acc</th>
                </tr>
              </thead>
              <tbody>
                {job.history.slice(-8).map((h) => (
                  <tr key={h.epoch}>
                    <td>{h.epoch}</td>
                    <td>{h.train_loss}</td>
                    <td>{h.val_loss}</td>
                    <td>{h.val_acc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {summary && (
            <div className="summary">
              <b>Model Summary</b>
              <span className="muted small">
                {" "}
                {summary.order.join(" → ") || "—"}
              </span>
              {summary.params != null && (
                <span className="muted small"> · {summary.params} params</span>
              )}
              {summary.config && (
                <span className="muted small">
                  {" "}
                  ·{" "}
                  {String(
                    (summary.config as { dataset?: string }).dataset ?? "",
                  )}
                </span>
              )}
              <table>
                <thead>
                  <tr>
                    <th>block</th>
                    <th>in → out</th>
                    <th>params</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.shapes.map((s) => (
                    <tr key={s.id}>
                      <td>
                        {s.kind}({s.id})
                      </td>
                      <td>
                        {fmtDims(s.in) ?? "?"} → {fmtDims(s.out) ?? "?"}
                      </td>
                      <td>
                        {typeof s.nodeParams === "number" ? s.nodeParams : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {summary.warnings.length > 0 && (
                <ul className="warns">
                  {summary.warnings.map((w, i) => (
                    <li key={i}>⚠ {w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}{" "}
        </div>
        <pre className="logs-pre">
          {logs.length
            ? logs.join("\n")
            : "No money for NOICE server so used free server only 512 MB RAM so u no train here okay!!!U can clone GitHub repo and run locally to train model with more RAM and GPU. "}
        </pre>
      </div>
    </section>
  );
}
