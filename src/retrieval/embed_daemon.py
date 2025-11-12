#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_daemon.py: Connects to Postgres, pulls unprocessed listings from the
embedding_queue.

- Fetches text, splits into chunks, and embeds using SentenceTransformer (384-dim).
- Fetches JPG images, resizes them, and embeds using CLIP (768-dim).
- Upserts all vectors into separate fields in Qdrant.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

try:
    import psycopg
except ImportError:
    sys.exit("Error: 'psycopg' library not found. Please (re)build container.")

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    sys.exit("Error: 'qdrant-client' library not found. Please (re)build container.")

try:
    # This import is correct and will work now
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Error: 'sentence-transformers' library not found. Please (re)build container.")

# --- Config ---
PG_HOST = os.environ.get('PGHOST', 'db')
PG_DB = os.environ.get('PGDATABASE', 'gis')
PG_USER = os.environ.get('PGUSER', 'gis')
PG_PASS = os.environ.get('PGPASSWORD', 'gis')

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "hemnet_listings_v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Config ---
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TEXT_VECTOR_DIM = 384
TEXT_CHUNK_SIZE = 256  # Max words per chunk
TEXT_CHUNK_OVERLAP = 50 # Overlap in words
MAX_TEXT_CHUNKS = 10     # Max chunks to embed per listing

IMAGE_MODEL_NAME = 'openai/clip-vit-large-patch14'
IMAGE_VECTOR_DIM = 768
IMAGE_RESIZE_DIM = 224 # CLIP's native resolution
MAX_IMAGE_EMBEDS = 50  # Max JPGs to embed per listing

BATCH_SIZE = 16
POLL_INTERVAL = 10 

def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """A simple sliding window text chunker that operates on words."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size # Avoid infinite loops
        
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        # Stop if the last chunk was already a full-sized one
        if i + chunk_size >= len(words):
            break
    
    return chunks[:MAX_TEXT_CHUNKS] # Apply hard limit

def get_db_connection() -> Optional[psycopg.Connection]:
    try:
        conn_str = f"host={PG_HOST} dbname={PG_DB} user={PG_USER} password={PG_PASS}"
        conn = psycopg.connect(conn_str, autocommit=False)
        return conn
    except psycopg.OperationalError as e:
        log("db_connect_fail", error=repr(e))
        return None

def load_models() -> Tuple[SentenceTransformer, AutoModel, AutoProcessor]:
    log("model_load_start", device=DEVICE)
    try:
        text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
        image_model = AutoModel.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True).to(DEVICE).eval()
        image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True)
        log("model_load_success", text_model=TEXT_MODEL_NAME, image_model=IMAGE_MODEL_NAME)
        return text_model, image_model, image_processor
    except Exception as e:
        log("model_load_fail", error=repr(e))
        raise

def ensure_qdrant_collection(client: QdrantClient):
    """Creates the new Qdrant collection with separate vector fields."""
    try:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "text": models.VectorParams(size=TEXT_VECTOR_DIM, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=IMAGE_VECTOR_DIM, distance=models.Distance.COSINE),
            }
        )
        log("qdrant_collection_created", collection=QDRANT_COLLECTION, text_dim=TEXT_VECTOR_DIM, image_dim=IMAGE_VECTOR_DIM)
    except Exception:
        log("qdrant_collection_exists", collection=QDRANT_COLLECTION)

def fetch_unprocessed_listings(cur: psycopg.Cursor, limit: int) -> List[str]:
    cur.execute("""
        SELECT listing_id
        FROM public.embedding_queue
        WHERE processed = false
        LIMIT %s
        FOR UPDATE SKIP LOCKED;
    """, (limit,))
    listing_ids = [row[0] for row in cur.fetchall()]
    
    if listing_ids:
        cur.execute("""
            UPDATE public.embedding_queue
            SET processed = true, processed_at = now()
            WHERE listing_id = ANY(%s);
        """, (listing_ids,))
    
    return listing_ids

def get_data_for_listings(cur: psycopg.Cursor, listing_ids: List[str]) -> Dict[str, Dict]:
    data = {lid: {"id": lid, "texts": [], "images": []} for lid in listing_ids}

    cur.execute("""
        SELECT listing_id, description_short, description_detailed
        FROM public.listings_attrs
        WHERE listing_id = ANY(%s);
    """, (listing_ids,))
    
    for lid, desc_short, desc_long in cur.fetchall():
        if desc_short:
            data[lid]["texts"].append(desc_short)
        if desc_long:
            data[lid]["texts"].append(desc_long)

    cur.execute("""
        SELECT listing_id, local_path
        FROM public.listings_images
        WHERE listing_id = ANY(%s) AND local_path IS NOT NULL;
    """, (listing_ids,))
    
    for lid, local_path in cur.fetchall():
        if not local_path.lower().endswith((".jpg", ".jpeg")):
            continue
            
        img_name = Path(local_path).name
        abs_path = f"/workspace/data/listings_raw/hemnet/snapshots/{lid}/assets/images/{img_name}"
        
        if os.path.exists(abs_path):
             data[lid]["images"].append(abs_path)
        else:
            if os.path.exists(local_path):
                data[lid]["images"].append(local_path)
    return data

def embed_listings(
    listings_data: List[Dict],
    text_model: SentenceTransformer,
    image_model: AutoModel,
    image_processor: AutoProcessor
) -> List[models.PointStruct]:
    
    qdrant_points = []
    
    for item in listings_data:
        listing_id = item["id"]
        
        # --- 1. Embed Text Chunks ---
        text_chunks = []
        for text in item["texts"]:
            chunks = _chunk_text(
                text,
                chunk_size=TEXT_CHUNK_SIZE,
                chunk_overlap=TEXT_CHUNK_OVERLAP
            )
            text_chunks.extend(chunks)
        
        if text_chunks:
            try:
                text_vectors = text_model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=False)
                text_vectors_norm = F.normalize(text_vectors, p=2, dim=1).cpu().numpy()
                
                for i, chunk in enumerate(text_chunks):
                    chunk_id = abs(hash(f"{listing_id}_text_{i}")) % (10**18)
                    qdrant_points.append(models.PointStruct(
                        id=chunk_id,
                        vector={"text": text_vectors_norm[i].tolist()},
                        payload={
                            "listing_id": listing_id, 
                            "type": "text",
                            "text_chunk": chunk
                        }
                    ))
            except Exception as e:
                log("embed_text_fail", id=listing_id, error=repr(e))

        # --- 2. Embed Images ---
        image_paths = item["images"][:MAX_IMAGE_EMBEDS]
        if image_paths:
            try:
                images_pil = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((IMAGE_RESIZE_DIM, IMAGE_RESIZE_DIM))
                    images_pil.append(img)
                
                if images_pil:
                    inputs = image_processor(images=images_pil, return_tensors="pt").to(DEVICE)
                    with torch.inference_mode():
                        img_vectors = image_model.get_image_features(**inputs)
                    img_vectors_norm = F.normalize(img_vectors, dim=1).cpu().numpy()
                    
                    for i, img_path in enumerate(image_paths):
                        img_id = abs(hash(f"{listing_id}_img_{i}")) % (10**18)
                        qdrant_points.append(models.PointStruct(
                            id=img_id,
                            vector={"image": img_vectors_norm[i].tolist()},
                            payload={
                                "listing_id": listing_id,
                                "type": "image",
                                "image_path": img_path
                            }
                        ))
            except Exception as e:
                log("embed_image_fail", id=listing_id, error=repr(e))
            
    return qdrant_points

def main_loop():
    log("daemon_start", device=DEVICE)
    text_model, image_model, image_processor = load_models()
    
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
    ensure_qdrant_collection(qdrant)
    
    db_conn = None
    while True:
        try:
            if not db_conn or db_conn.closed:
                db_conn = get_db_connection()
                if not db_conn:
                    log("daemon_sleep", reason="db connection failed", seconds=POLL_INTERVAL)
                    time.sleep(POLL_INTERVAL)
                    continue

            with db_conn.cursor() as cur:
                listing_ids = fetch_unprocessed_listings(cur, BATCH_SIZE)
                db_conn.commit()

            if not listing_ids:
                log("queue_empty", sleeping_seconds=POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
                continue
                
            log("queue_fetch_batch", count=len(listing_ids))
                
            with db_conn.cursor() as cur:
                listings_data = get_data_for_listings(cur, listing_ids)
            
            qdrant_points = embed_listings(
                list(listings_data.values()), 
                text_model,
                image_model, 
                image_processor
            )

            if qdrant_points:
                qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=qdrant_points,
                    wait=True
                )
                log("qdrant_upsert_batch", count=len(qdrant_points))

        except psycopg.OperationalError as e:
            log("db_error", error=repr(e))
            if db_conn:
                db_conn.close()
            db_conn = None
            time.sleep(POLL_INTERVAL)
            
        except Exception as e:
            log("daemon_loop_error", error=repr(e))
            if db_conn:
                db_conn.close()
            db_conn = None
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()