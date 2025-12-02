#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

try:
    import psycopg
except ImportError:
    sys.exit("Error: 'psycopg' library not found.")

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    sys.exit("Error: 'qdrant-client' library not found.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Error: 'sentence-transformers' library not found.")

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
TEXT_CHUNK_SIZE = 256
TEXT_CHUNK_OVERLAP = 50
MAX_TEXT_CHUNKS = 10

IMAGE_MODEL_NAME = 'openai/clip-vit-large-patch14'
IMAGE_VECTOR_DIM = 768
IMAGE_RESIZE_DIM = 224
MAX_IMAGE_EMBEDS = 50

BATCH_SIZE = 16
POLL_INTERVAL = 10 

def log(event, **kw):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    msg = " ".join(f"{k}={repr(v)}" for k,v in kw.items())
    print(f"{ts} {event} {msg}".strip(), flush=True)

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text: return []
    words = text.split()
    if not words: return []
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0: step = chunk_size
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words): break
    return chunks[:MAX_TEXT_CHUNKS]

def _make_serializable(data: Dict) -> Dict:
    clean = {}
    for k, v in data.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            clean[k] = v.isoformat()
        elif isinstance(v, Decimal):
            clean[k] = int(v) if v % 1 == 0 else float(v)
        elif isinstance(v, dict):
            clean[k] = _make_serializable(v)
        else:
            clean[k] = v
    return clean

def get_db_connection() -> Optional[psycopg.Connection]:
    try:
        conn_str = f"host={PG_HOST} dbname={PG_DB} user={PG_USER} password={PG_PASS}"
        return psycopg.connect(conn_str, autocommit=False)
    except psycopg.OperationalError as e:
        log("db_connect_fail", error=repr(e))
        return None

def load_models() -> Tuple[SentenceTransformer, AutoModel, AutoProcessor]:
    log("model_load_start", device=DEVICE)
    try:
        text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
        image_model = AutoModel.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True).to(DEVICE).eval()
        image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_NAME, trust_remote_code=True)
        return text_model, image_model, image_processor
    except Exception as e:
        log("model_load_fail", error=repr(e))
        raise

def ensure_qdrant_collection(client: QdrantClient):
    """Creates collection and ensures necessary indexes exist."""
    try:
        # 1. Create Collection if missing
        if not client.collection_exists(QDRANT_COLLECTION):
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "text": models.VectorParams(size=TEXT_VECTOR_DIM, distance=models.Distance.COSINE),
                    "image": models.VectorParams(size=IMAGE_VECTOR_DIM, distance=models.Distance.COSINE),
                }
            )
            log("qdrant_collection_created", collection=QDRANT_COLLECTION)
        else:
            log("qdrant_collection_exists", collection=QDRANT_COLLECTION)

        # 2. Ensure Indexes (Idempotent)
        # Grouping Index (Required for 'group_by')
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="listing_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        # Geospatial Indexes (Required for efficient filtering)
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="location", 
            field_schema=models.PayloadSchemaType.GEO
        )
        
        # Legacy/Fallback indexes (if we still use raw lat/lon ranges)
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="latitude",
            field_schema=models.PayloadSchemaType.FLOAT
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="longitude",
            field_schema=models.PayloadSchemaType.FLOAT
        )
        
        log("qdrant_indexes_ensured")

    except Exception as e:
        log("qdrant_init_error", error=repr(e))

def fetch_unprocessed_listings(cur: psycopg.Cursor, limit: int) -> List[str]:
    cur.execute("""
        SELECT listing_id FROM public.embedding_queue
        WHERE processed = false LIMIT %s FOR UPDATE SKIP LOCKED;
    """, (limit,))
    listing_ids = [row[0] for row in cur.fetchall()]
    if listing_ids:
        cur.execute("""
            UPDATE public.embedding_queue SET processed = true, processed_at = now()
            WHERE listing_id = ANY(%s);
        """, (listing_ids,))
    return listing_ids

def get_data_for_listings(cur: psycopg.Cursor, listing_ids: List[str]) -> Dict[str, Dict]:
    data = {lid: {"id": lid, "texts": [], "images": [], "attrs": {}} for lid in listing_ids}

    # 1. Fetch ALL attributes
    cur.execute("SELECT * FROM public.listings_attrs WHERE listing_id = ANY(%s)", (listing_ids,))
    colnames = [desc.name for desc in cur.description]

    for row in cur.fetchall():
        row_dict = dict(zip(colnames, row))
        lid = row_dict.get('listing_id')
        if not lid or lid not in data: continue

        if row_dict.get("description_short"):
            data[lid]["texts"].append(row_dict["description_short"])
        if row_dict.get("description_detailed"):
            data[lid]["texts"].append(row_dict["description_detailed"])

        # --- NEW: Format Geospatial Data for Qdrant ---
        # Qdrant requires: "location": { "lat": float, "lon": float }
        lat = row_dict.get("latitude")
        lon = row_dict.get("longitude")
        
        sanitized_attrs = _make_serializable(row_dict)
        
        # Inject the location object if valid coords exist
        if lat is not None and lon is not None and isinstance(lat, (float, int, Decimal)) and isinstance(lon, (float, int, Decimal)):
             sanitized_attrs["location"] = {
                 "lat": float(lat), 
                 "lon": float(lon)
             }
        
        data[lid]["attrs"] = sanitized_attrs
        # -----------------------------------------------

    # 2. Fetch Images
    cur.execute("""
        SELECT listing_id, local_path FROM public.listings_images
        WHERE listing_id = ANY(%s) AND local_path IS NOT NULL
    """, (listing_ids,))
    
    for lid, local_path in cur.fetchall():
        if not local_path.lower().endswith((".jpg", ".jpeg")): continue
        img_name = Path(local_path).name
        # Map local path to container path
        abs_path = f"/workspace/data/listings_raw/hemnet/snapshots/{lid}/assets/images/{img_name}"
        
        if os.path.exists(abs_path): data[lid]["images"].append(abs_path)
        elif os.path.exists(local_path): data[lid]["images"].append(local_path)
    return data

def embed_listings(listings_data: List[Dict], text_model, image_model, image_processor) -> List[models.PointStruct]:
    qdrant_points = []
    for item in listings_data:
        listing_id = item["id"]
        attrs = item["attrs"]
        
        # Embed Text
        text_chunks = []
        for text in item["texts"]:
            text_chunks.extend(_chunk_text(text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP))
        
        if text_chunks:
            try:
                text_vectors = text_model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=False)
                text_vectors_norm = F.normalize(text_vectors, p=2, dim=1).cpu().numpy()
                for i, chunk in enumerate(text_chunks):
                    chunk_id = abs(hash(f"{listing_id}_text_{i}")) % (10**18)
                    qdrant_points.append(models.PointStruct(
                        id=chunk_id,
                        vector={"text": text_vectors_norm[i].tolist()},
                        payload={"listing_id": listing_id, "type": "text", "text_chunk": chunk, **attrs}
                    ))
            except Exception as e:
                log("embed_text_fail", id=listing_id, error=repr(e))

        # Embed Images
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
                            payload={"listing_id": listing_id, "type": "image", "image_path": img_path, **attrs}
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
                    time.sleep(POLL_INTERVAL)
                    continue

            with db_conn.cursor() as cur:
                listing_ids = fetch_unprocessed_listings(cur, BATCH_SIZE)
                db_conn.commit()

            if not listing_ids:
                time.sleep(POLL_INTERVAL)
                continue
                
            log("queue_fetch_batch", count=len(listing_ids))
            with db_conn.cursor() as cur:
                listings_data = get_data_for_listings(cur, listing_ids)
            
            qdrant_points = embed_listings(list(listings_data.values()), text_model, image_model, image_processor)

            if qdrant_points:
                qdrant.upsert(collection_name=QDRANT_COLLECTION, points=qdrant_points, wait=True)
                log("qdrant_upsert_batch", count=len(qdrant_points))

        except Exception as e:
            log("daemon_loop_error", error=repr(e))
            if db_conn: db_conn.close()
            db_conn = None
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()