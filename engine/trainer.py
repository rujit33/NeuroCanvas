"""Background training jobs with WebSocket fan-out."""
from __future__ import annotations

import asyncio
import pickle
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Set

import torch

from datasets import get_dataloaders, resolve_num_classes
from graph import Graph, canon, order_graph
from model_builder import build_model, count_params, make_loss, make_optimizer

RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

jobs: Dict[str, Dict] = {}
subscribers: Dict[str, Set[asyncio.Queue]] = {}


def _log(job_id: str, msg: str):
    job = jobs[job_id]
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    job["logs"].append(line)
    for q in list(subscribers.get(job_id, set())):
        try:
            q.put_nowait({"type": "log", "line": line})
        except asyncio.QueueFull:
            pass


def _progress(job_id: str, payload: dict):
    for q in list(subscribers.get(job_id, set())):
        try:
            q.put_nowait({"type": "progress", **payload})
        except asyncio.QueueFull:
            pass


def create_job(graph_dict: dict, epochs: int, save_format: str, save_mode: str) -> str:
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "epochs": epochs,
        "current_epoch": 0,
        "history": [],
        "logs": [],
        "save_format": save_format,
        "save_mode": save_mode,
        "graph": graph_dict,
        "save_path": None,
        "params": None,
        "dataset_info": None,
        "error": None,
    }
    subscribers[job_id] = set()
    return job_id


@torch.no_grad()
def _evaluate(model, loader, loss_fn) -> Tuple[float, float]:
    import torch as _t  # local alias

    model.eval()
    total, correct, loss_sum, n = 0, 0, 0.0, 0
    for X, y in loader:
        out = model(X)
        loss_sum += loss_fn(out, y).item() * len(X)
        correct += (out.argmax(1) == y).sum().item()
        n += len(X)
    return loss_sum / max(n, 1), correct / max(n, 1)


from typing import Tuple  # noqa: E402  (kept at bottom to avoid clutter up top)


async def run_training(job_id: str):
    from graph import Graph as _Graph  # local import for worker clarity

    job = jobs[job_id]
    job["status"] = "running"
    try:
        graph = _Graph(**job["graph"])
        ordered_pre = order_graph(graph)  # topo first so errors name cycles early
        kinds = " -> ".join(canon(n.type) for n in ordered_pre)
        _log(job_id, f"Graph wired: {kinds}")
        inp = next(n for n in ordered_pre if canon(n.type) == "input").params
        num_classes = await asyncio.to_thread(
            resolve_num_classes, str(inp.get("dataset", "synthetic")),
            str(inp.get("dataset_path", "")), ordered_pre,
        )
        model, cfg, specs = await asyncio.to_thread(
            build_model, graph.nodes, graph.edges, num_classes)
        job["params"] = count_params(model)
        _log(job_id, f"Model built: {count_params(model)} params | cfg={cfg}")
        for s in specs:
            _log(job_id, f"  {s['kind']}({s['id']}): {s.get('in')} -> {s.get('out')}")

        train_loader, val_loader, info = await asyncio.to_thread(
            get_dataloaders,
            cfg["dataset"], cfg["dataset_path"], cfg["image_size"],
            cfg["batch_size"], cfg["in_channels"], cfg["num_classes"],
        )
        job["dataset_info"] = info
        _log(job_id, f"Dataset ready: {info}")

        loss_fn = make_loss(cfg["loss"])
        opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])
        epochs = int(job["epochs"])

        for epoch in range(1, epochs + 1):
            model.train()
            running, n = 0.0, 0
            for X, y in train_loader:
                opt.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                running += loss.item() * len(X)
                n += len(X)
            train_loss = running / max(n, 1)
            val_loss, val_acc = await asyncio.to_thread(_evaluate, model, val_loader, loss_fn)
            job["current_epoch"] = epoch
            job["history"].append(
                {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
            )
            _log(job_id, f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            _progress(job_id, {"epoch": epoch, "epochs": epochs,
                               "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
            await asyncio.sleep(0)  # yield to WS loop

        # ---- save ----
        fmt = (job["save_format"] or "pt").lower().lstrip(".")
        if fmt not in ("pt", "pth", "pkl"):
            fmt = "pt"
        mode = (job["save_mode"] or "weights_only").lower()
        job_dir = RUNS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        save_path = job_dir / f"model.{fmt}"
        payload = {
            "arch": [{"id": s["id"], "kind": s["kind"], "params": s["params"]} for s in specs],
            "train_cfg": cfg,
            "graph": job["graph"],
            "history": job["history"],
        }
        if mode == "full":
            payload["model_state"] = model.state_dict()
            payload["optimizer_state"] = opt.state_dict()
            to_save = {"checkpoint": payload}
        else:
            to_save = model.state_dict()
        await asyncio.to_thread(_save_file, to_save, save_path, fmt, mode)
        job["save_path"] = str(save_path)
        job["status"] = "done"
        _log(job_id, f"Done. Saved {mode} -> {save_path}")
        _progress(job_id, {"status": "done", "save_path": str(save_path)})
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        _log(job_id, f"ERROR: {job['error']}\n{traceback.format_exc(limit=3)}")
        _progress(job_id, {"status": "error", "error": job["error"]})


def _save_file(obj, path: Path, fmt: str, mode: str):
    if fmt == "pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    else:  # pt / pth via torch.save (zip archive)
        torch.save(obj, path)


def job_summary(job_id: str) -> Dict:
    j = jobs[job_id]
    return {k: v for k, v in j.items() if k != "graph"}


def list_jobs() -> List[Dict]:
    return [job_summary(jid) for jid in sorted(jobs)]
