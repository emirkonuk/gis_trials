#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer

# --- NEW: Add log function ---
def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)
# --- END NEW ---

STACK_ROOT = Path(os.environ.get("GIS_STACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", STACK_ROOT / "data"))

# --- Qdrant Collections ---
CHIP_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sweden_demo_v0")
LISTING_COLLECTION = "hemnet_listings_v1"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

META_PATH = Path(os.environ.get("METADATA_PATH", DATA_ROOT / "chips" / "metadata.parquet")) 

app = FastAPI(title="Retrieval Search API")
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model 1: CLIP (for chips AND listing images) ---
log("Loading CLIP model: openai/clip-vit-large-patch14")
CHIP_MODEL_NAME = "openai/clip-vit-large-patch14"
clip_processor = CLIPProcessor.from_pretrained(CHIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CHIP_MODEL_NAME).to(device).eval()
log("CLIP model loaded.")

# --- Load Model 2: SentenceTransformer (for listing text) ---
log("Loading Text model: sentence-transformers/all-MiniLM-L6-v2")
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
log("Text model loaded.")

# Load legacy metadata for chip search
if META_PATH.exists():
    meta = (
        pd.read_parquet(META_PATH)[["png_path", "lon", "lat"]]
        .reset_index()
        .rename(columns={"index": "row"})
    )
else:
    print(f"[search_api] chip metadata missing at {META_PATH}", flush=True)
    meta = pd.DataFrame(columns=["row", "png_path", "lon", "lat"])

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0, prefer_grpc=True)
print(
    f"[search_api] device={device} cuda_visible={visible}",
    flush=True,
)

# --- Helper Functions for Encoding ---

def encode_clip_text(query: str) -> list[float]:
    """Encodes text using CLIP for chip search"""
    with torch.inference_mode():
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
        vector = clip_model.get_text_features(**inputs)
        vector = F.normalize(vector, dim=-1).detach().cpu().numpy()[0].astype("float32")
        return vector.tolist()

def encode_sentence_text(query: str) -> list[float]:
    """Encodes text using SentenceTransformer for listing text search"""
    with torch.inference_mode():
        vector = text_model.encode(query, convert_to_tensor=True)
        vector = F.normalize(vector, p=2, dim=0).cpu().numpy()
        return vector.tolist()


# --- API Endpoints ---

@app.get("/search/text", tags=["Satellite Chips (Legacy)"])
def search_chip_text(q: str = Query(...), topk: int = 10):
    """
    SEARCH (SATELLITE CHIPS): Encodes text with CLIP and searches the
    'sweden_demo_v0' collection.
    """
    vector = encode_clip_text(q)
    results = client.search(
        collection_name=CHIP_COLLECTION,
        query_vector=vector, 
        limit=topk, 
        with_payload=True
    )
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
    return {"query": q, "topk": topk, "collection": CHIP_COLLECTION, "results": output}


@app.get("/search/listing_text", tags=["Hemnet Listings (New)"])
def search_listing_text(q: str = Query(...), topk: int = 10):
    """
    SEARCH (LISTING TEXT): Encodes text with SentenceTransformer and
    searches the 'text' vectors in the 'hemnet_listings_v1' collection.
    """
    vector = encode_sentence_text(q)
    results = client.search(
        collection_name=LISTING_COLLECTION,
        query_vector=models.NamedVector(name="text", vector=vector),
        limit=topk,
        with_payload=True
    )
    output = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
    return {"query": q, "topk": topk, "collection": LISTING_COLLECTION, "search_vector": "text", "results": output}


@app.get("/search/listing_image", tags=["Hemnet Listings (New)"])
def search_listing_image(q: str = Query(...), topk: int = 10):
    """
    SEARCH (LISTING IMAGES): Encodes text with CLIP and searches
    the 'image' vectors in the 'hemnet_listings_v1' collection.
    """
    vector = encode_clip_text(q) # Use CLIP to encode text for image search
    results = client.search(
        collection_name=LISTING_COLLECTION,
        query_vector=models.NamedVector(name="image", vector=vector),
        limit=topk,
        with_payload=True
    )
    output = [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
    return {"query": q, "topk": topk, "collection": LISTING_COLLECTION, "search_vector": "image", "results": output}