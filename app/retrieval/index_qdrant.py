#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as m


def jprint(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False))


STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
CONFIG_PATH = Path(os.environ.get("RETRIEVAL_CONFIG", STACK_ROOT / "app" / "retrieval" / "config.yaml"))
cfg = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

chip_cfg = cfg.get("runtime", {})
index_cfg = cfg.get("index", {})

EMB_PATH = Path(os.environ.get("EMBEDDINGS_PATH", chip_cfg.get("embeddings", DATA_ROOT / "chips" / "embeddings.npy")))
META_PATH = Path(os.environ.get("METADATA_PATH", chip_cfg.get("metadata", DATA_ROOT / "chips" / "metadata.parquet")))
COL = os.environ.get("QDRANT_COLLECTION", index_cfg.get("collection", "sweden_demo_v0"))
HOST = os.environ.get("QDRANT_HOST", "qdrant")
PORT = int(os.environ.get("QDRANT_PORT", "6333"))
BATCH = int(os.environ.get("QDRANT_BATCH", "256"))
VECTOR_SIZE = int(index_cfg.get("vector_size", 768))

emb = np.load(EMB_PATH, mmap_mode="r", allow_pickle=False)
rows, dim = emb.shape
meta = pd.read_parquet(META_PATH)
if len(meta) != rows:
    raise ValueError(f"metadata rows {len(meta)} != embeddings rows {rows}")
if dim != VECTOR_SIZE:
    raise ValueError(f"embedding dim {dim} does not match configured vector size {VECTOR_SIZE}")

client = QdrantClient(host=HOST, port=PORT, timeout=120.0, prefer_grpc=True)
client.recreate_collection(
    collection_name=COL,
    vectors_config=m.VectorParams(size=VECTOR_SIZE, distance=m.Distance.COSINE),
)

for start in range(0, rows, BATCH):
    end = min(rows, start + BATCH)
    vecs = emb[start:end].astype("float32", copy=False)
    meta_slice = meta.iloc[start:end].reset_index(drop=True)

    points = []
    for offset, (vec, row) in enumerate(zip(vecs, meta_slice.itertuples(index=False))):
        pid = start + offset
        lon = float(getattr(row, "lon", float("nan")))
        lat = float(getattr(row, "lat", float("nan")))
        if not (lon == lon and lat == lat):  # NaN check
            try:
                import mercantile
                import re

                if all(hasattr(row, key) for key in ("z", "x", "y")):
                    bounds = mercantile.bounds(int(row.z), int(row.x), int(row.y))
                else:
                    match = re.search(r"(\d+)_(\d+)_(\d+)\.png$", row.png_path)
                    if not match:
                        raise ValueError("could not parse xyz from png_path")
                    z, x, y = map(int, match.groups())
                    bounds = mercantile.bounds(x, y, z)
                lon = (bounds.west + bounds.east) / 2.0
                lat = (bounds.south + bounds.north) / 2.0
            except Exception:
                lon = lat = None

        payload = {
            "png": row.png_path,
            "z": int(getattr(row, "z", -1)),
            "x": int(getattr(row, "x", -1)),
            "y": int(getattr(row, "y", -1)),
            "lon": float(lon) if lon is not None else None,
            "lat": float(lat) if lat is not None else None,
        }
        points.append(m.PointStruct(id=pid, vector=vec.tolist(), payload=payload))

    client.upsert(collection_name=COL, points=points)

info = client.get_collection(COL)
jprint(
    collection=COL,
    points_count=info.points_count,
    dim=dim,
    batches=(rows + BATCH - 1) // BATCH,
    batch_size=BATCH,
)
