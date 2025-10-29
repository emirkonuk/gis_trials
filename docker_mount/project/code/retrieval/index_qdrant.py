#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as m

def jprint(**kw):
    print(json.dumps(kw, ensure_ascii=False))

# Inputs
EMB_PATH  = "/project/data/chips/embeddings.npy"
META_PATH = "/project/data/chips/metadata.parquet"
COL       = "sweden_demo_v0"
HOST, PORT = "qdrant", 6333
BATCH = int(os.environ.get("QDRANT_BATCH", "256"))  # adjust if timeouts

# Load data
emb = np.load(EMB_PATH, mmap_mode="r", allow_pickle=False)
N, D = emb.shape
meta = pd.read_parquet(META_PATH)
if len(meta) != N:
    raise ValueError(f"metadata rows {len(meta)} != embeddings rows {N}")

# Client
cli = QdrantClient(host=HOST, port=PORT, timeout=120.0, prefer_grpc=True)

# Recreate collection from scratch
cli.recreate_collection(
    collection_name=COL,
    vectors_config=m.VectorParams(size=D, distance=m.Distance.COSINE)
)

# Upsert in batches
for s in range(0, N, BATCH):
    e = min(N, s + BATCH)
    # ensure float32
    vecs = emb[s:e].astype("float32", copy=False)
    # payload: keep only what you need; png path is useful for audit
    # payload = [{"png": p} for p in meta["png_path"].iloc[s:e]]
    # slice once per batch
    meta_slice = meta.iloc[s:e].reset_index(drop=True)   # rows align to vecs[0..e-s)
    vecs = emb[s:e].astype("float32", copy=False)

    points = []
    for j, (vec, row) in enumerate(zip(vecs, meta_slice.itertuples(index=False))):
        pid = s + j  # deterministic id 0..N-1 across batches
        # prefer existing lon/lat; if missing, compute from z/x/y
        lon = float(getattr(row, "lon", float("nan")))
        lat = float(getattr(row, "lat", float("nan")))
        if not (lon == lon and lat == lat):  # NaN check
            # expect z,x,y columns; else parse from png_path "z_x_y.png"
            try:
                import mercantile, re
                if all(hasattr(row, k) for k in ("z","x","y")):
                    b = mercantile.bounds(int(row.z), int(row.x), int(row.y))
                else:
                    m = re.search(r"/(\d+)_(\d+)_(\d+)\.png$", row.png_path)
                    z,x,y = map(int, m.groups())
                    b = mercantile.bounds(x, y, z)
                lon = (b.west + b.east) / 2.0
                lat = (b.south + b.north) / 2.0
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

    cli.upsert(collection_name=COL, points=points)

# Report
info = cli.get_collection(COL)  # returns models.CollectionInfo
jprint(collection=COL, points_count=info.points_count, dim=D, batches=(N + BATCH - 1)//BATCH, batch_size=BATCH)
































# #!/usr/bin/env python3
# import json, yaml, numpy as np, pandas as pd
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct

# def log(stage, goal, next_step):
#     print(json.dumps({"stage": stage, "goal": goal, "next_step": next_step}, ensure_ascii=False))

# cfg = yaml.safe_load(open("config.yaml"))
# host, port = cfg["index"]["host"], cfg["index"]["port"]
# collection = cfg["index"]["collection"]
# dist = cfg["index"]["distance"].lower()

# distance = Distance.COSINE if dist=="cosine" else Distance.DOT

# client = QdrantClient(host=host, port=port)

# vecs = np.load("/project/data/chips/embeddings.npy")
# meta = pd.read_parquet("/project/data/chips/metadata.parquet")
# dim = int(vecs.shape[1])

# if collection not in [c.name for c in client.get_collections().collections]:
#     client.create_collection(
#         collection_name=collection,
#         vectors_config=VectorParams(size=dim, distance=distance)
#     )

# points = [
#     PointStruct(id=int(i),
#                 vector=vecs[i].tolist(),
#                 payload={"z": int(row.z), "x": int(row.x), "y": int(row.y),
#                          "lon": float(row.lon), "lat": float(row.lat),
#                          "png_path": row.png_path})
#     for i, row in meta.reset_index(drop=True).iterrows()
# ]
# client.upsert(collection_name=collection, points=points)

# res = client.count(collection_name=collection, exact=True)
# print(json.dumps({"collection": collection, "points": res.count}, ensure_ascii=False))

# log("index", "load vectors into Qdrant", "run search sanity checks")
