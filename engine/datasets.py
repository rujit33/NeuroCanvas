"""Dataset loaders for the PoC — no torchvision dependency.

Sources:
  - synthetic : random images, zero setup (default, for first Train click)
  - mnist     : downloads raw IDX files once, caches in ./data/mnist
  - imagefolder: <dataset_path>/<class_name>/*.png|jpg|jpeg|bmp, PIL-resized
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DATA_DIR = Path(__file__).parent / "data"
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def _mnist_tensors(image_size: int) -> Tuple[TensorDataset, TensorDataset]:
    d = DATA_DIR / "mnist"
    for k, url in MNIST_URLS.items():
        _download(url, d / f"{k}.gz")
    Xtr = _read_idx_images(d / "train_images.gz").astype(np.float32) / 255.0
    ytr = _read_idx_labels(d / "train_labels.gz")
    Xte = _read_idx_images(d / "test_images.gz").astype(np.float32) / 255.0
    yte = _read_idx_labels(d / "test_labels.gz")

    def prep(X: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(X).unsqueeze(1)  # N,1,28,28
        if image_size != 28:
            t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return t

    return (
        TensorDataset(prep(Xtr), torch.from_numpy(ytr).long()),
        TensorDataset(prep(Xte), torch.from_numpy(yte).long()),
    )


def _synthetic(n: int, channels: int, image_size: int, num_classes: int) -> TensorDataset:
    X = torch.randn(n, channels, image_size, image_size)
    # Learnable pattern so loss visibly drops: label from mean pixel sign bins
    s = X.mean(dim=(1, 2, 3))
    y = ((s - s.min()) / (s.max() - s.min() + 1e-6) * num_classes).long().clamp(0, num_classes - 1)
    return TensorDataset(X, y)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _imagefolder(path: str, image_size: int, grayscale: bool) -> TensorDataset:
    if not HAS_PIL:
        raise RuntimeError("Pillow is required for imagefolder datasets (pip install pillow)")
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f"dataset_path is not a directory: {root}")
    classes = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not classes:
        raise ValueError(f"No class subfolders found in {root}. Expected <path>/<class>/*.png")
    idx = {c.name: i for i, c in enumerate(classes)}
    Xs, ys = [], []
    for c in classes:
        for f in sorted(c.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            img = Image.open(f)
            img = img.convert("L" if grayscale else "RGB").resize((image_size, image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = arr[None, ...]
            else:
                arr = arr.transpose(2, 0, 1)
            Xs.append(arr)
            ys.append(idx[c.name])
    if not Xs:
        raise ValueError(f"No images found under {root}")
    X = torch.from_numpy(np.stack(Xs))
    y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(X, y)


def get_dataloaders(
    dataset: str,
    dataset_path: str,
    image_size: int,
    batch_size: int,
    in_channels: int,
    num_classes: int,
) -> Tuple[DataLoader, DataLoader, Dict]:
    dataset = (dataset or "synthetic").lower()
    if dataset == "mnist":
        try:
            train_ds, val_ds = _mnist_tensors(image_size)
            info = {"source": "mnist", "train_n": len(train_ds), "val_n": len(val_ds)}
        except Exception as e:
            train_ds = _synthetic(2000, 1, image_size, 10)
            val_ds = _synthetic(500, 1, image_size, 10)
            info = {"source": f"synthetic-mnist-fallback ({e})", "train_n": len(train_ds), "val_n": len(val_ds)}
    elif dataset == "imagefolder":
        full = _imagefolder(dataset_path, image_size, grayscale=(in_channels == 1))
        n_val = max(1, int(0.2 * len(full)))
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        info = {"source": f"imagefolder:{dataset_path}", "train_n": len(train_ds), "val_n": len(val_ds)}
    else:  # synthetic demo
        train_ds = _synthetic(2000, in_channels, image_size, num_classes)
        val_ds = _synthetic(500, in_channels, image_size, num_classes)
        info = {"source": "synthetic", "train_n": len(train_ds), "val_n": len(val_ds)}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(256, batch_size), shuffle=False)
    return train_loader, val_loader, info


def list_uploads() -> list:
    up = DATA_DIR / "uploads"
    if not up.is_dir():
        return []
    return [str(p) for p in sorted(up.iterdir())]


def _canon_kind(t: str) -> str:
    t = (t or "").lower()
    return {"cnn": "conv", "classifier": "linear"}.get(t, t)


def resolve_num_classes(dataset: str, dataset_path: str, ordered_nodes) -> int:
    """Figure out class count before building: mnist=10, imagefolder=folder
    count, synthetic=out_features of the last linear block."""
    dataset = (dataset or "synthetic").lower()
    if dataset == "mnist":
        return 10
    if dataset == "imagefolder":
        root = Path(str(dataset_path or "")).expanduser()
        if not root.is_dir():
            raise ValueError(f"dataset_path is not a directory: {root}")
        classes = [p for p in root.iterdir() if p.is_dir()]
        if not classes:
            raise ValueError(f"No class subfolders in {root}. Expected <path>/<class>/*.png")
        return len(classes)
    # synthetic demo: classes defined by the architecture itself
    legacy = [n for n in ordered_nodes if (n.type or "").lower() == "classifier"]
    linears = [n for n in ordered_nodes if _canon_kind(n.type) == "linear"]
    if legacy:
        return int(legacy[-1].params.get("num_classes", 10))
    if not linears:
        raise ValueError("Synthetic demo needs a Linear block (it defines the class count)")
    return int(linears[-1].params.get("out_features", 10))
