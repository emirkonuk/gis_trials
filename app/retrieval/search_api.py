#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))
COL = os.environ.get("QDRANT_COLLECTION", "sweden_demo_v0")
HOST = os.environ.get("QDRANT_HOST", "qdrant")
PORT = int(os.environ.get("QDRANT_PORT", "6333"))
META = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "chips" / "metadata.parquet"))

app = FastAPI(title="retrieval_search_api")
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.environ.get("RETRIEVAL_MODEL", "openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device).eval()
if META.exists():
    meta = (
        pd.read_parquet(META)[["png_path", "lon", "lat"]]
        .reset_index()
        .rename(columns={"index": "row"})
    )
else:
    print(f"[search_api] metadata missing at {META}, starting with empty index", flush=True)
    meta = pd.DataFrame(columns=["row", "png_path", "lon", "lat"])
client = QdrantClient(host=HOST, port=PORT, timeout=60.0, prefer_grpc=True)
print(
    f"[search_api] device={device} cuda_visible={visible} collection={COL}",
    flush=True,
)


def encode_text(query: str) -> list[float]:
    with torch.inference_mode():
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        vector = model.get_text_features(**inputs)
        vector = torch.nn.functional.normalize(vector, dim=-1).detach().cpu().numpy()[0].astype("float32")
        return vector.tolist()


@app.get("/search/text")
def search_text(q: str = Query(...), topk: int = 10):
    vector = encode_text(q)
    results = client.search(collection_name=COL, query_vector=vector, limit=topk, with_payload=True)
    output = []
    for hit in results:
        rid = int(hit.id)
        if 0 <= rid < len(meta):
            row = meta.iloc[rid]
            output.append(
                {
                    "id": rid,
                    "score": float(hit.score),
                    "png": row["png_path"],
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                }
            )
    return {"query": q, "topk": topk, "results": output}
