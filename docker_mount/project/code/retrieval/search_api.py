#!/usr/bin/env python3
import os, json, numpy as np, pandas as pd, torch
from fastapi import FastAPI, Query
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as m

COL, HOST, PORT = "sweden_demo_v0", "qdrant", 6333
META = "/project/data/chips/metadata.parquet"

app = FastAPI(title="retrieval_search_api")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device).eval()
meta = pd.read_parquet(META)[["png_path","lon","lat"]].reset_index().rename(columns={"index":"row"})
cli = QdrantClient(host=HOST, port=PORT, timeout=60.0, prefer_grpc=True)

def encode_text(q: str) -> list[float]:
    with torch.inference_mode():
        inp = processor(text=[q], return_tensors="pt", padding=True).to(device)
        v = model.get_text_features(**inp)
        v = torch.nn.functional.normalize(v, dim=-1).detach().cpu().numpy()[0].astype("float32")
        return v.tolist()

@app.get("/search/text")
def search_text(q: str = Query(...), topk: int = 10):
    v = encode_text(q)
    res = cli.search(collection_name=COL, query_vector=v, limit=topk, with_payload=True)
    out=[]
    for r in res:
        rid = int(r.id)
        if 0 <= rid < len(meta):
            row = meta.iloc[rid]
            out.append({"id": rid, "score": float(r.score), "png": row["png_path"], "lon": float(row["lon"]), "lat": float(row["lat"])})
    return {"query": q, "topk": topk, "results": out}
