"""VisualML engine — FastAPI + WebSocket backend (PyTorch).

Run:  uvicorn main:app --reload --port 8000
View expects this at http://localhost:8000
"""
from __future__ import annotations

import asyncio
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from datasets import DATA_DIR, list_uploads
from exporter import generate_notebook, generate_python, notebook_to_json
from graph import Graph, ValidationError, canon, validate_dynamic
from model_builder import validate_pipeline
from trainer import create_job, job_summary, jobs, list_jobs, run_training, subscribers

app = FastAPI(title="VisualML Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    graph: Graph
    epochs: int = Field(default=5, ge=1, le=100)
    save_format: str = Field(default="pt")   # pt | pth | pkl
    save_mode: str = Field(default="weights_only")  # weights_only | full


class ExportRequest(BaseModel):
    graph: Graph
    epochs: int = 5


@app.get("/api/health")
def health():
    return {"ok": True, "service": "visualml-engine"}


def _validation_error_response(e: ValueError):
    """Backward-compat envelope: detail stays a plain string; structured
    stage/node/op context rides alongside for new clients."""
    if isinstance(e, ValidationError):
        d = e.to_dict()
        content = {"detail": str(e), "error": d}
        content.update({k: d[k] for k in
                        ("stage", "node_id", "op", "expected", "got", "reason", "hint")})
        return JSONResponse(status_code=400, content=content)
    return JSONResponse(status_code=400, content={"detail": str(e)})


@app.post("/api/validate")
def validate(graph: Graph):
    try:
        res = validate_pipeline(graph.nodes, graph.edges)
    except ValueError as e:
        return _validation_error_response(e)
    cfg, specs = res["cfg"], res["specs"]
    return {
        "ok": True,
        "order": [f"{canon(n.type)}({n.id})" for n in res["order"]],
        "params": res["total_params"],
        "config": cfg,
        "shapes": [{k: s[k] for k in ("id", "kind", "in", "out") if k in s
                     } | {"num_params": s.get("num_params", 0)} for s in specs],
        "warnings": res["warnings"],
    }


@app.post("/api/train")
async def start_train(req: TrainRequest):
    try:
        validate_pipeline(req.graph.nodes, req.graph.edges)
    except ValueError as e:
        return _validation_error_response(e)
    fmt = req.save_format.lower().lstrip(".")
    if fmt not in ("pt", "pth", "pkl"):
        raise HTTPException(400, "save_format must be pt | pth | pkl")
    if req.save_mode not in ("weights_only", "full"):
        raise HTTPException(400, "save_mode must be weights_only | full")
    job_id = create_job(req.graph.model_dump(), req.epochs, fmt, req.save_mode)
    asyncio.create_task(run_training(job_id))
    return {"job_id": job_id, **job_summary(job_id)}


@app.get("/api/jobs")
def all_jobs():
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    return job_summary(job_id)


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    if job_id not in jobs:
        await ws.send_json({"type": "error", "error": "unknown job"})
        await ws.close()
        return
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    subscribers.setdefault(job_id, set()).add(q)
    # replay history so late joiners see everything
    j = jobs[job_id]
    await ws.send_json({"type": "init", "job": job_summary(job_id)})
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20)
                await ws.send_json(msg)
                if msg.get("status") in ("done", "error") and msg.get("type") == "progress":
                    pass
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping", "job": job_summary(job_id)})
                if jobs[job_id]["status"] in ("done", "error"):
                    await ws.send_json({"type": "final", "job": job_summary(job_id)})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.get(job_id, set()).discard(q)


@app.get("/api/download/{job_id}")
def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "unknown job")
    path = jobs[job_id].get("save_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "model not ready yet")
    return FileResponse(path, filename=Path(path).name)


@app.post("/api/export/python")
def export_python(req: ExportRequest, as_file: bool = True):
    try:
        validate_dynamic(req.graph)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        code = generate_python(req.graph.nodes, req.graph.edges, req.epochs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if as_file:
        p = Path("runs") / "_export_model.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(code, encoding="utf-8")
        return FileResponse(str(p), filename="model.py", media_type="text/x-python")
    return PlainTextResponse(code)


@app.post("/api/export/notebook")
def export_notebook(req: ExportRequest):
    try:
        validate_dynamic(req.graph)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        nb = generate_notebook(req.graph.nodes, req.graph.edges, req.epochs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    p = Path("runs") / "_export_model.ipynb"
    p.parent.mkdir(exist_ok=True)
    p.write_text(notebook_to_json(nb), encoding="utf-8")
    return FileResponse(str(p), filename="model.ipynb")


@app.post("/api/upload")
async def upload_dataset(file: UploadFile, target: Optional[str] = None):
    """Upload a zip of <class>/*.png images; returns extracted server path for the input node."""
    dest_dir = DATA_DIR / "uploads" / (target or Path(file.filename or "data").stem)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir.with_suffix(".zip.tmp")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        with zipfile.ZipFile(tmp) as z:
            z.extractall(dest_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "file must be a .zip of class folders")
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(dest_dir.resolve()), "uploads": list_uploads()}


@app.get("/api/uploads")
def uploads():
    return {"uploads": list_uploads(), "hint": "paste one of these paths into the input node's dataset_path"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
